"""Adaptive depth sweep followed by matched residual experiments, one GPU job at a time."""
import argparse
import contextlib
import fcntl
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from depth_cnn import HERE, atomic_json, build_model, config, digest, implementation_hash, load_manifest


def candidates(m):
    return [("plain", d) for d in m["depths"]] + [("residual", d) for d in m["depths"] if d >= 4]


def key(variant, depth, seed):
    return f"{variant}/conv-{depth}/seed-{seed}"


def completed(directory, expected, m, require_wandb):
    summary_path, checkpoint_path = directory / "summary.json", directory / "last.pt"
    if not summary_path.exists():
        return None
    if not checkpoint_path.exists():
        raise ValueError(f"summary without checkpoint: {directory}")
    summary = json.loads(summary_path.read_text())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if summary["config"] != expected or state["config"] != expected:
        raise ValueError(f"configuration or implementation mismatch: {directory}")
    if summary["epochs"] != m["epochs"] or state["epoch"] != m["epochs"]:
        raise ValueError(f"summary/checkpoint epoch mismatch: {directory}")
    history = state["history"]
    if [r["epoch"] for r in history] != list(range(1, m["epochs"] + 1)):
        raise ValueError(f"invalid checkpoint metric history: {directory}")
    score = statistics.mean(r["eval/validation_accuracy"] for r in history[-m["score_epochs"]:])
    if not math.isfinite(summary["score"]) or not math.isclose(summary["score"], score, abs_tol=1e-10):
        raise ValueError(f"summary scoring window disagrees with checkpoint: {directory}")
    if summary["final"] != history[-1] or state["next_epoch"] != m["epochs"] + 1:
        raise ValueError(f"final metrics disagree with checkpoint: {directory}")
    if require_wandb and (not summary.get("wandb_url") or not state.get("wandb_id")):
        raise ValueError(f"missing W&B identity: {directory}")
    return summary


def decision(m, scores):
    """Replay the adaptive rule from complete depth groups; never inspect test metrics."""
    visited, comparisons, streak = [], [], 0
    for depth in m["depths"]:
        if depth not in scores or set(scores[depth]) != set(m["seeds"]):
            return {"visited": visited, "comparisons": comparisons, "stop": None, "next_depth": depth}
        visited.append(depth)
        if len(visited) >= 2:
            previous = visited[-2]
            diffs = [scores[depth][s] - scores[previous][s] for s in m["seeds"]]
            gain = statistics.mean(diffs)
            streak = streak + 1 if gain < m["plateau_gain_pp"] else 0
            comparisons.append({"from": previous, "to": depth, "paired_differences_pp": diffs,
                                "mean_gain_pp": gain, "plateau_streak": streak})
            if gain <= m["decline_pp"] and sum(d < 0 for d in diffs) >= m["declining_seeds"]:
                return {"visited": visited, "comparisons": comparisons, "stop": "decline", "next_depth": None}
            if streak >= m["plateau_patience"]:
                return {"visited": visited, "comparisons": comparisons, "stop": "plateau", "next_depth": None}
    return {"visited": visited, "comparisons": comparisons, "stop": "cap", "next_depth": None}


def select_winner(m, records):
    groups = []
    for variant, depth in candidates(m):
        rows = [records.get((variant, depth, s)) for s in m["seeds"]]
        if all(rows):
            groups.append((statistics.mean(r["score"] for r in rows), depth, variant))
    if not groups:
        raise ValueError("no complete candidate available")
    score, depth, variant = min(groups, key=lambda r: (-r[0], r[1], r[2] != "plain"))
    return {"variant": variant, "depth": depth, "score": score, "seeds": m["seeds"]}


def records_at(m, root, device, no_wandb):
    if not (root / "split.json").exists():
        return {}
    split = json.loads((root / "split.json").read_text())
    if digest(split["indices"]) != split["hash"]:
        raise ValueError("saved split hash does not match indices")
    records = {}
    for variant, depth in candidates(m):
        for seed in m["seeds"]:
            directory = root / key(variant, depth, seed)
            result = completed(directory, config(m, variant, depth, seed, device, split["hash"]), m, not no_wandb)
            if result:
                records[variant, depth, seed] = result
    return records


def plots(m, root, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = root / "plots"
    figures.mkdir(exist_ok=True)
    colors = {"plain": "#2563eb", "residual": "#e76f51"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in ("plain", "residual"):
        depths, means, spreads = [], [], []
        for v, d in candidates(m):
            rows = [records[v, d, s] for s in m["seeds"] if (v, d, s) in records]
            if v == variant and rows:
                depths.append(d)
                means.append(statistics.mean(r["score"] for r in rows))
                spreads.append(statistics.stdev(r["score"] for r in rows) if len(rows) > 1 else 0)
        if depths:
            ax.errorbar(depths, means, yerr=spreads, marker="o", label=variant, color=colors[variant], capsize=3)
    ax.set(xlabel="Total convolution layers", ylabel="Validation score (%)", title="Mean last-window validation accuracy ± seed SD")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures / "accuracy-vs-depth.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    depths, gains, errors = [], [], []
    for d in m["depths"]:
        paired = [records["residual", d, s]["score"] - records["plain", d, s]["score"] for s in m["seeds"]
                  if ("residual", d, s) in records and ("plain", d, s) in records]
        if paired:
            depths.append(d); gains.append(statistics.mean(paired))
            errors.append(statistics.stdev(paired) if len(paired) > 1 else 0)
    if depths:
        ax.errorbar(depths, gains, yerr=errors, marker="o", capsize=3)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Total convolution layers", ylabel="Residual − plain (percentage points)", title="Paired shortcut gains ± seed SD")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / "residual-gains.svg"); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ("train/eval_accuracy", "eval/validation_accuracy", "train/eval_loss", "eval/validation_loss")
    for variant, depth in candidates(m):
        histories = []
        for seed in m["seeds"]:
            path = root / key(variant, depth, seed) / "metrics.jsonl"
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if rows:
                    histories.append(rows)
        if not histories:
            continue
        length = min(len(h) for h in histories)
        for ax, metric in zip(axes.flat, metrics):
            values = [statistics.mean(h[i][metric] for h in histories) for i in range(length)]
            ax.plot(range(1, length + 1), values, label=f"{variant} {depth}")
            ax.set(xlabel="Epoch", ylabel=metric); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figures / "learning-curves.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, depth in candidates(m):
        rows = [records[variant, depth, s] for s in m["seeds"] if (variant, depth, s) in records]
        if rows:
            seconds = statistics.mean(r["final"]["performance/elapsed_seconds"] for r in rows)
            score = statistics.mean(r["score"] for r in rows)
            ax.scatter(seconds, score, color=colors[variant])
            ax.annotate(f"{variant} {depth}", (seconds, score), fontsize=8)
    ax.set(xlabel="Training + evaluation seconds", ylabel="Validation score (%)", title="Accuracy versus measured compute time")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / "accuracy-vs-runtime.svg"); plt.close(fig)


def report(m, root, device, no_wandb, state=None):
    records = records_at(m, root, device, no_wandb)
    scores = {d: {s: records["plain", d, s]["score"] for s in m["seeds"] if ("plain", d, s) in records} for d in m["depths"]}
    outcome = decision(m, scores)
    aggregates = []
    lines = ["# Depth and residual comparison", "", "Selection uses validation only; score is mean accuracy over the final "
             f"{m['score_epochs']} epochs. Spread is sample standard deviation across seeds. Partial groups are marked.", "",
             f"Plain sweep stopping reason: {outcome['stop'] or 'pending'}; visited depths: {outcome['visited']}.", "",
             "| Variant | Conv layers | Seeds | Validation score, % | Final train, % | Final val loss | Params | Compute seconds | Wall seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def spread(values):
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f} (one seed)"
    for variant, depth in candidates(m):
        rows = [records[variant, depth, s] for s in m["seeds"] if (variant, depth, s) in records]
        if not rows:
            continue
        scores_here = [r["score"] for r in rows]
        final_train = statistics.mean(r["final"]["train/eval_accuracy"] for r in rows)
        final_loss = statistics.mean(r["final"]["eval/validation_loss"] for r in rows)
        seconds = statistics.mean(r["final"]["performance/elapsed_seconds"] for r in rows)
        wall = [state.get("jobs", {}).get(key(variant, depth, s), {}).get("wall_seconds", 0) for s in m["seeds"]
                if (variant, depth, s) in records] if state else []
        lines.append(f"| {variant} | {depth} | {len(rows)}/{len(m['seeds'])} | {spread(scores_here)} | "
                     f"{final_train:.2f} | {final_loss:.4f} | {rows[0]['params']} | {seconds:.1f} | "
                     f"{statistics.mean(wall):.1f} |" if wall else
                     f"| {variant} | {depth} | {len(rows)}/{len(m['seeds'])} | {spread(scores_here)} | {final_train:.2f} | "
                     f"{final_loss:.4f} | {rows[0]['params']} | {seconds:.1f} | pending |")
        paired = [records["residual", depth, s]["score"] - records["plain", depth, s]["score"] for s in m["seeds"]
                  if ("residual", depth, s) in records and ("plain", depth, s) in records]
        aggregates.append({"variant": variant, "depth": depth, "seeds": len(rows), "mean_score": statistics.mean(scores_here),
                           "score_std": statistics.stdev(scores_here) if len(rows) > 1 else None,
                           "paired_residual_gains_pp": paired, "runs": rows})
    if state and state.get("selection"):
        lines.extend(["", f"Selected by validation: {state['selection']}."])
    tests = []
    if state and state.get("selection"):
        selected = state["selection"]
        for seed in m["seeds"]:
            path = root / key(selected["variant"], selected["depth"], seed) / "test-summary.json"
            if path.exists():
                tests.append(json.loads(path.read_text()))
        if tests:
            lines.extend(["", f"Selected model final test accuracy ({len(tests)}/{len(m['seeds'])} seeds): "
                          f"{spread([r['accuracy'] for r in tests])}%."])
    if records:
        plots(m, root, records)
        lines.extend(["", "## Plots", "", "![Validation accuracy by depth](plots/accuracy-vs-depth.svg)",
                      "![Paired residual gains](plots/residual-gains.svg)", "![Learning curves](plots/learning-curves.svg)",
                      "![Accuracy versus runtime](plots/accuracy-vs-runtime.svg)"])
    lines.extend(["", "## Plain depth decisions", "", "```json", json.dumps(outcome, indent=2), "```", ""])
    atomic_json(root / "comparison.json", {"plain_decision": outcome, "candidates": aggregates,
                                           "selection": state.get("selection") if state else None, "test_results": tests})
    temp = root / "comparison.tmp"
    temp.write_text("\n".join(lines)); temp.replace(root / "comparison.md")
    return records, outcome


def wait_for_augmentation(root, state, state_path, timeout_sleep=60):
    """Yield an acquired legacy lock only after all expected jobs are verified complete."""
    from run_augmentation_queue import completed as aug_completed, expected_config as aug_config, jobs as aug_jobs
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "queue.lock").open("a")
    try:
        while True:
            path = root / "queue-state.json"
            aug_state = json.loads(path.read_text()) if path.exists() else None
            failed = [k for k, job in (aug_state or {}).get("jobs", {}).items() if job["status"] == "failed"]
            if failed:
                raise RuntimeError(f"augmentation suite failed: {failed}; resolve it before restarting depth queue")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if state.get("phase") != "waiting-for-augmentation":
                    state["phase"] = "waiting-for-augmentation"
                    atomic_json(state_path, state)
                    print("Waiting for augmentation suite to finish and release its GPU lock.", flush=True)
                time.sleep(timeout_sleep)
                continue
            manifest_path = root / "manifest.json"
            if not manifest_path.exists():
                raise RuntimeError("augmentation queue is absent/incomplete; start or finish it before the depth queue")
            saved = json.loads(manifest_path.read_text())
            m = saved["manifest"]
            if len(aug_jobs(m)) != 45:
                raise RuntimeError("expected the existing 45-job augmentation suite")
            for recipe, seed in aug_jobs(m):
                directory = root / recipe / f"seed-{seed}"
                if not aug_completed(directory, aug_config(m, recipe, seed, saved["device"]), m["epochs"],
                                     require_wandb=not saved["no_wandb"]):
                    raise RuntimeError(f"augmentation runner exited with incomplete job {recipe}/seed-{seed}; resume it first")
            print("All 45 augmentation runs verified; GPU lock acquired.", flush=True)
            return lock
    except BaseException:
        lock.close()
        raise


def command(action, manifest, root, directory, variant, depth, seed, device, no_wandb):
    cmd = [sys.executable, "-u", str(HERE / "depth_cnn.py"), action,
           "--manifest", str(manifest), "--root", str(root), "--directory", str(directory),
           "--variant", variant, "--depth", str(depth), "--seed", str(seed), "--device", device]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd


def validate_architectures(m):
    torch.set_num_threads(1)
    for depth in m["depths"]:
        torch.manual_seed(0)
        plain = build_model(depth)
        assert plain(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
        if depth >= 4:
            torch.manual_seed(0)
            residual = build_model(depth, "residual")
            assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in residual.parameters())
            # Shape validation changes BN buffers; compare untouched parameters rather than those buffers here.
            for a, b in zip(plain.parameters(), residual.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert residual(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def eta(m, state, smoke_results, outcome):
    if outcome.get("stop"):
        allowed = [("plain", d) for d in outcome["visited"]] + [("residual", d) for d in outcome["visited"] if d >= 4]
        upper_bound = False
    else:
        allowed = candidates(m)
        upper_bound = True
    remaining = 0
    for variant, depth in allowed:
        sample = smoke_results[f"{variant}/{depth}"]["estimated_run_seconds"]
        finished = [state["jobs"].get(key(variant, depth, seed), {}) for seed in m["seeds"]]
        observed = [r["wall_seconds"] for r in finished if r.get("status") == "completed" and r.get("wall_seconds")]
        estimate = statistics.mean(observed) if observed else sample
        remaining += sum(estimate for r in finished if r.get("status") != "completed")
    return {"remaining_seconds": remaining, "upper_bound_candidates": upper_bound,
            "note": "Before plain stopping, assumes all candidates. Smoke estimates exclude startup/checkpoint/upload overhead; completed siblings use measured wall time."}


def execute(a, m, root, state, state_path, locks):
    frozen_path = root / "manifest.json"
    frozen = {"manifest": m, "implementation": implementation_hash(), "device": a.device, "no_wandb": a.no_wandb}
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("queue configuration or implementation changed; use a new root")
    atomic_json(frozen_path, frozen)
    state.pop("error", None)
    manifest_path = root / "training-manifest.json"
    atomic_json(manifest_path, m)
    validate_architectures(m)
    state["phase"] = "preflight"; atomic_json(state_path, state)
    smoke_path = root / "smoke-results.json"
    smoke_results = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    for variant, depth in candidates(m):
        if implementation_hash() != frozen["implementation"]:
            raise ValueError("implementation changed during preflight; use the frozen code or a new root")
        candidate = f"{variant}/{depth}"
        if candidate not in smoke_results:
            directory = root / "smoke" / variant / f"conv-{depth}"
            directory.mkdir(parents=True, exist_ok=True)
            cmd = command("smoke", manifest_path, root, directory, variant, depth, m["seeds"][0], a.device, True)
            with (directory / "smoke.log").open("a") as log:
                result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=tuple(lock.fileno() for lock in locks))
            if result.returncode:
                raise RuntimeError(f"preflight failed for {candidate}; inspect {directory / 'smoke.log'}")
            smoke_results[candidate] = json.loads((directory / "smoke.json").read_text())
            atomic_json(smoke_path, smoke_results)
        probe = smoke_results[candidate]
        print(f"Preflight {candidate}: {probe['peak_gpu_memory_mb']:.0f} MB peak; "
              f"estimated {probe['estimated_run_seconds'] / 60:.1f} min/run; resume passed.", flush=True)
    if a.action == "smoke":
        state["phase"] = "preflight-complete"; atomic_json(state_path, state)
        return

    def run_job(variant, depth, seed, action="train"):
        if implementation_hash() != frozen["implementation"]:
            raise ValueError("implementation changed during queue execution; restore the frozen code or use a new root")
        job_key = key(variant, depth, seed)
        directory = root / job_key
        directory.mkdir(parents=True, exist_ok=True)
        split = json.loads((root / "split.json").read_text())
        expected = config(m, variant, depth, seed, a.device, split["hash"])
        job = state["jobs"].setdefault(job_key, {"status": "pending", "wall_seconds": 0})
        if action == "train" and completed(directory, expected, m, not a.no_wandb):
            job["status"] = "completed"; atomic_json(state_path, state)
            return
        cmd = command(action, manifest_path, root, directory, variant, depth, seed, a.device, a.no_wandb)
        job.update(status="running" if action == "train" else "assessing", command=cmd)
        atomic_json(state_path, state)
        print(f"Starting {action}: {job_key}", flush=True)
        started = time.perf_counter()
        try:
            with (directory / ("run.log" if action == "train" else "test.log")).open("a") as log:
                result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=tuple(lock.fileno() for lock in locks))
            if result.returncode:
                raise RuntimeError(f"{action} failed (exit {result.returncode}); inspect {directory}")
            if action == "train" and not completed(directory, expected, m, not a.no_wandb):
                raise RuntimeError(f"training exited without valid completion: {directory}")
        except BaseException as error:
            job.update(status="failed", error=str(error))
            raise
        else:
            job.update(status="completed")
            job.pop("error", None)
        finally:
            field = "wall_seconds" if action == "train" else "assessment_wall_seconds"
            job[field] = job.get(field, 0) + time.perf_counter() - started
            atomic_json(state_path, state)

    while True:
        records, outcome = report(m, root, a.device, a.no_wandb, state)
        state["plain_decision"] = outcome
        state["eta"] = eta(m, state, smoke_results, outcome)
        atomic_json(state_path, state)
        if outcome["stop"]:
            break
        state["phase"] = "plain-sweep"; atomic_json(state_path, state)
        for seed in m["seeds"]:
            run_job("plain", outcome["next_depth"], seed)
            _, updated_outcome = report(m, root, a.device, a.no_wandb, state)
            state["eta"] = eta(m, state, smoke_results, updated_outcome)
            atomic_json(state_path, state)
        print(f"Remaining training estimate (before adaptive stopping): {state['eta']['remaining_seconds'] / 3600:.2f} h", flush=True)
    state["phase"] = "residual-sweep"; atomic_json(state_path, state)
    print(f"Plain sweep stopped: {outcome['stop']} at {outcome['visited'][-1]} convolutions.", flush=True)
    for depth in outcome["visited"]:
        if depth < 4:
            continue
        for seed in m["seeds"]:
            run_job("residual", depth, seed)
            records, _ = report(m, root, a.device, a.no_wandb, state)
            state["eta"] = eta(m, state, smoke_results, outcome)
            atomic_json(state_path, state)
            print(f"Remaining training estimate: {state['eta']['remaining_seconds'] / 3600:.2f} h", flush=True)
    records, _ = report(m, root, a.device, a.no_wandb, state)
    eligible = {k: v for k, v in records.items() if k[1] in outcome["visited"]}
    state["selection"] = select_winner(m, eligible)
    state["phase"] = "final-assessment"; atomic_json(state_path, state)
    selected = state["selection"]
    for seed in m["seeds"]:
        run_job(selected["variant"], selected["depth"], seed, action="test")
        report(m, root, a.device, a.no_wandb, state)
    state["phase"] = "completed"; atomic_json(state_path, state)
    report(m, root, a.device, a.no_wandb, state)
    print(f"Depth queue complete. Report: {root / 'comparison.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("smoke", "run", "status", "report"))
    p.add_argument("--manifest", type=Path, default=HERE / "depth_experiments.json")
    p.add_argument("--root", type=Path, default=HERE / "runs/depth")
    p.add_argument("--augmentation-root", type=Path, default=HERE / "runs/augmentation")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    m = load_manifest(a.manifest)
    root = a.root.resolve()
    state_path = root / "queue-state.json"
    frozen = {"manifest": m, "implementation": implementation_hash(), "device": a.device, "no_wandb": a.no_wandb}
    identity = digest(frozen)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"identity": identity, "jobs": {}, "phase": "not-started"}
    if state["identity"] != identity:
        raise ValueError("queue settings or implementation changed; use a new root")
    if a.action == "status":
        compact = {"phase": state["phase"], "maximum_training_runs": len(candidates(m)) * len(m["seeds"]),
                   "counts": {status: sum(j["status"] == status for j in state["jobs"].values())
                              for status in ("pending", "running", "completed", "failed", "assessing")},
                   "active": {k: v for k, v in state["jobs"].items() if v["status"] in ("running", "failed", "assessing")},
                   "plain_decision": state.get("plain_decision"), "eta": state.get("eta"),
                   "selection": state.get("selection"), "error": state.get("error")}
        print(json.dumps(compact, indent=2)); return
    if a.action == "report":
        root.mkdir(parents=True, exist_ok=True)
        report(m, root, a.device, a.no_wandb, state)
        print(root / "comparison.md"); return
    root.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        locks = []
        # A shared lock spans all depth roots; CPU fixtures only need the per-root lock.
        paths = [root / "queue.lock"] + ([HERE / "runs" / "depth-gpu.lock"] if a.device == "cuda" else [])
        for path in paths:
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        atomic_json(state_path, state)
        try:
            frozen_path = root / "manifest.json"
            if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
                raise ValueError("frozen queue configuration differs; use a new root")
            atomic_json(frozen_path, frozen)
            if a.device == "cuda":
                locks.append(stack.enter_context(wait_for_augmentation(a.augmentation_root.resolve(), state, state_path)))
            execute(a, m, root, state, state_path, locks)
        except BaseException as error:
            state.update(phase="failed", error=str(error)); atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
