"""Wait for the depth winner, compare its augmentations, then train CIFAR ResNet-18."""
import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

import depth_cnn as base
import followup_cnn as trainer
import run_depth_queue as depth_queue

HERE = base.HERE


def load_manifest(path):
    m = json.loads(path.read_text())
    if m["version"] != 1 or m["winner_augmentations"] != ["none", "flip", "crop-flip"]:
        raise ValueError("expected none/flip/crop-flip winner comparison")
    if m["resnet_augmentation"] != "crop-flip" or m["resnet_epochs"] < 10:
        raise ValueError("invalid ResNet recipe")
    if not m["seeds"] or len(set(m["seeds"])) != len(m["seeds"]):
        raise ValueError("invalid seeds")
    return m


def make_specs(request, upstream, selection):
    if request["seeds"] != upstream["seeds"]:
        raise ValueError("follow-up seeds must match the upstream controls")
    specs = []
    for recipe in request["winner_augmentations"][1:]:
        training = {**upstream, "augmentation": recipe}
        specs.append({"id": f"winner-{recipe}", "architecture": "depth-winner",
                      "variant": selection["variant"], "depth": selection["depth"],
                      "augmentation": recipe, "training": training})
    resnet_recipe = dict(upstream)
    resnet_recipe["epochs"] = request["resnet_epochs"]
    resnet_recipe["augmentation"] = request["resnet_augmentation"]
    specs.append({"id": "resnet18-crop-flip", "architecture": "resnet18-cifar",
                  "augmentation": request["resnet_augmentation"], "training": resnet_recipe})
    return specs


def job_key(candidate, seed):
    return f"{candidate}/seed-{seed}"


def wait_for_depth(source, augmentation_root, device, state, state_path, expected_source, interval=60):
    """Only acquire the shared GPU/source locks after the upstream queue finishes."""
    announced = False
    while True:
        path = source / "queue-state.json"
        if not path.exists():
            raise RuntimeError("depth queue has not started; start it before the follow-up queue")
        upstream = json.loads(path.read_text())
        if upstream["phase"] == "failed":
            raise RuntimeError(f"depth queue failed: {upstream.get('error')}; resolve and restart it first")
        if json.loads((source / "manifest.json").read_text()) != expected_source:
            raise ValueError("upstream configuration changed while waiting")
        if upstream["phase"] != "completed":
            if not announced:
                state["phase"] = "waiting-for-depth"
                base.atomic_json(state_path, state)
                print("Waiting for depth sweep, residual comparisons, and final assessment to finish.", flush=True)
                announced = True
            time.sleep(interval)
            continue
        locks = []
        paths = [source / "queue.lock"]
        if device == "cuda":
            paths = [HERE / "runs" / "depth-gpu.lock", source / "queue.lock", augmentation_root / "queue.lock"]
        try:
            for lock_path in paths:
                lock = lock_path.open("a")
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            for lock in locks:
                lock.close()
            time.sleep(interval)
            continue
        except BaseException:
            for lock in locks:
                lock.close()
            raise
        latest = json.loads(path.read_text())
        if latest["phase"] != "completed":
            for lock in locks:
                lock.close()
            continue
        print("Depth queue completed; exclusive GPU and source locks acquired.", flush=True)
        return latest, locks


def prepare(request, source, root, upstream_state, source_frozen):
    if base.implementation_hash() != source_frozen["implementation"]:
        raise ValueError("upstream source code changed; restore it before starting follow-up experiments")
    if upstream_state["phase"] != "completed" or not upstream_state.get("selection"):
        raise ValueError("upstream selection is not complete")
    m = source_frozen["manifest"]
    rows = depth_queue.records_at(m, source, source_frozen["device"], source_frozen["no_wandb"])
    eligible = {k: v for k, v in rows.items() if k[1] in upstream_state["plain_decision"]["visited"]}
    selection = depth_queue.select_winner(m, eligible)
    if selection != upstream_state["selection"]:
        raise ValueError("recorded upstream winner differs from verified validation results")
    controls = []
    for seed in request["seeds"]:
        row = eligible.get((selection["variant"], selection["depth"], seed))
        if row is None:
            raise ValueError("missing completed upstream control")
        controls.append({"seed": seed, "summary": row,
                         "directory": str(source / depth_queue.key(selection["variant"], selection["depth"], seed))})
    split = json.loads((source / "split.json").read_text())
    if base.digest(split["indices"]) != split["hash"]:
        raise ValueError("upstream split hash differs from indices")
    document = {"implementation": trainer.implementation_hash(), "source_root": str(source),
                "source_manifest": source_frozen, "selection": selection, "seeds": request["seeds"],
                "split_hash": split["hash"], "controls": controls, "candidates": make_specs(request, m, selection)}
    path = root / "specs.json"
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError("frozen follow-up specifications differ; use a new root")
    split_path = root / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError("follow-up split differs from source split")
    base.atomic_json(split_path, split)
    base.atomic_json(path, document)
    return document


def read_records(document, root, device, no_wandb):
    records = {("winner-none", r["seed"]): r["summary"] for r in document["controls"]}
    for spec in document["candidates"]:
        for seed in document["seeds"]:
            directory = root / job_key(spec["id"], seed)
            expected = trainer.config(spec, seed, device, document["split_hash"])
            row = depth_queue.completed(directory, expected, spec["training"], not no_wandb)
            if row:
                records[spec["id"], seed] = row
    return records


def select_winner(document, records):
    order = ["winner-none"] + [s["id"] for s in document["candidates"]]
    complete = []
    for rank, candidate in enumerate(order):
        rows = [records.get((candidate, s)) for s in document["seeds"]]
        if all(rows):
            complete.append((statistics.mean(r["score"] for r in rows), rank, candidate))
    if len(complete) != len(order):
        raise ValueError("all candidate seeds must finish before final selection")
    score, _, candidate = min(complete, key=lambda r: (-r[0], r[1]))
    return {"candidate": candidate, "score": score, "seeds": document["seeds"]}


def report(document, root, device, no_wandb, state):
    records = read_records(document, root, device, no_wandb)
    lines = ["# Winner augmentation and CIFAR ResNet-18 comparison", "",
             "Same 45,000/5,000 split and paired seeds. Winner augmentation uses the upstream recipe unchanged; "
             "ResNet-18 uses 200 epochs by default. Scores average the final ten validation epochs. "
             "These are recipe comparisons, not an isolated architecture comparison.", "",
             f"Upstream architecture: {document['selection']['variant']}, {document['selection']['depth']} convolutions.", "",
             "| Candidate | Seeds | Epochs | Validation score, % | Paired gain vs none, pp | Final train, % | Params |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def spread(values):
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f} (one seed)"
    aggregates = []
    order = ["winner-none"] + [s["id"] for s in document["candidates"]]
    for candidate in order:
        seeds = [s for s in document["seeds"] if (candidate, s) in records]
        if not seeds:
            continue
        rows = [records[candidate, s] for s in seeds]
        differences = [records[candidate, s]["score"] - records["winner-none", s]["score"] for s in seeds]
        lines.append(f"| {candidate} | {len(rows)}/{len(document['seeds'])} | {rows[0]['epochs']} | "
                     f"{spread([r['score'] for r in rows])} | {spread(differences)} | "
                     f"{statistics.mean(r['final']['train/eval_accuracy'] for r in rows):.2f} | {rows[0]['params']} |")
        aggregates.append({"candidate": candidate, "seeds": seeds, "mean_score": statistics.mean(r["score"] for r in rows),
                           "score_std": statistics.stdev(r["score"] for r in rows) if len(rows) > 1 else None,
                           "paired_gain_pp": differences, "runs": rows})
    lines.extend(["", "The winner-none control reuses verified completed depth runs; no duplicate training is scheduled."])
    if state.get("selection"):
        lines.extend(["", f"Selected using validation: {state['selection']}."])
    if state.get("test_results"):
        lines.extend(["", f"Selected model final test accuracy: {spread([r['accuracy'] for r in state['test_results']])}%."])
    if len(records) > len(document["controls"]):
        make_plots(document, root, records)
        lines.extend(["", "![Validation comparison](plots/validation-comparison.svg)",
                      "![Learning curves](plots/learning-curves.svg)"])
    base.atomic_json(root / "comparison.json", {"selection": state.get("selection"), "candidates": aggregates,
                                               "test_results": state.get("test_results", [])})
    temp = root / "comparison.tmp"
    temp.write_text("\n".join(lines) + "\n"); temp.replace(root / "comparison.md")
    return records


def make_plots(document, root, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root / "plots"; output.mkdir(exist_ok=True)
    order = ["winner-none"] + [s["id"] for s in document["candidates"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    names, means, deviations = [], [], []
    for candidate in order:
        scores = [records[candidate, s]["score"] for s in document["seeds"] if (candidate, s) in records]
        if scores:
            names.append(candidate); means.append(statistics.mean(scores))
            deviations.append(statistics.stdev(scores) if len(scores) > 1 else 0)
    ax.bar(names, means, yerr=deviations, capsize=4)
    ax.set(ylabel="Validation score (%)", title="Final-window score ± seed SD")
    fig.tight_layout(); fig.savefig(output / "validation-comparison.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for candidate in order:
        histories = []
        for seed in document["seeds"]:
            directory = (Path(next(r["directory"] for r in document["controls"] if r["seed"] == seed))
                         if candidate == "winner-none" else root / job_key(candidate, seed))
            path = directory / "metrics.jsonl"
            if path.exists():
                histories.append([json.loads(line) for line in path.read_text().splitlines()])
        if not histories or any(not h for h in histories):
            continue
        length = min(len(h) for h in histories)
        for ax, metric in zip(axes, ("train/eval_accuracy", "eval/validation_accuracy")):
            ax.plot(range(1, length + 1), [statistics.mean(h[i][metric] for h in histories) for i in range(length)], label=candidate)
            ax.set(xlabel="Epoch", ylabel=metric); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "learning-curves.svg"); plt.close(fig)


def command(action, root, spec, seed, directory, device, no_wandb):
    cmd = [sys.executable, "-u", str(HERE / "followup_cnn.py"), action, "--specs", str(root / "specs.json"),
           "--candidate", spec["id"], "--seed", str(seed), "--root", str(root),
           "--directory", str(directory), "--device", device]
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd


def execute(a, document, root, state, state_path, locks):
    state.pop("error", None)
    def launch(action, spec, seed, directory):
        if document["implementation"] != trainer.implementation_hash():
            raise ValueError("source implementation changed during follow-up queue")
        directory.mkdir(parents=True, exist_ok=True)
        cmd = command(action, root, spec, seed, directory, a.device, a.no_wandb if action != "smoke" else True)
        with (directory / ("run.log" if action == "train" else f"{action}.log")).open("a") as log:
            result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=tuple(lock.fileno() for lock in locks))
        if result.returncode:
            raise RuntimeError(f"{action} failed (exit {result.returncode}); inspect {directory}")
    state["phase"] = "preflight"; base.atomic_json(state_path, state)
    smoke_results = state.setdefault("preflight", {})
    for spec in document["candidates"]:
        if spec["id"] not in smoke_results:
            directory = root / "smoke" / spec["id"]
            launch("smoke", spec, document["seeds"][0], directory)
            smoke_results[spec["id"]] = json.loads((directory / "smoke.json").read_text())
            base.atomic_json(state_path, state)
        probe = smoke_results[spec["id"]]
        print(f"Preflight {spec['id']}: {probe['peak_gpu_memory_mb']:.0f} MB; "
              f"estimated {probe['estimated_run_seconds'] / 60:.1f} min/run; resume passed.", flush=True)
    if a.action == "smoke":
        state["phase"] = "preflight-complete"; base.atomic_json(state_path, state)
        return
    state["phase"] = "training"; base.atomic_json(state_path, state)
    for spec in document["candidates"]:
        for seed in document["seeds"]:
            job_id = job_key(spec["id"], seed)
            directory = root / job_id
            job = state["jobs"].setdefault(job_id, {"status": "pending", "wall_seconds": 0})
            expected = trainer.config(spec, seed, a.device, document["split_hash"])
            if depth_queue.completed(directory, expected, spec["training"], not a.no_wandb):
                job["status"] = "completed"; base.atomic_json(state_path, state)
                continue
            job.update(status="running", command=command("train", root, spec, seed, directory, a.device, a.no_wandb))
            base.atomic_json(state_path, state)
            print(f"Starting {job_id}", flush=True)
            started = time.perf_counter()
            try:
                launch("train", spec, seed, directory)
                if not depth_queue.completed(directory, expected, spec["training"], not a.no_wandb):
                    raise RuntimeError("training exited without verified completion")
            except BaseException as error:
                job.update(status="failed", error=str(error)); raise
            else:
                job.update(status="completed"); job.pop("error", None)
            finally:
                job["wall_seconds"] += time.perf_counter() - started
                base.atomic_json(state_path, state)
            report(document, root, a.device, a.no_wandb, state)
            remaining = 0
            for remaining_spec in document["candidates"]:
                siblings = [state["jobs"].get(job_key(remaining_spec["id"], s), {}) for s in document["seeds"]]
                measured = [j["wall_seconds"] for j in siblings if j.get("status") == "completed" and j.get("wall_seconds")]
                estimate = statistics.mean(measured) if measured else smoke_results[remaining_spec["id"]]["estimated_run_seconds"]
                remaining += sum(estimate for j in siblings if j.get("status") != "completed")
            state["eta_seconds"] = remaining; base.atomic_json(state_path, state)
            print(f"Remaining training estimate: {remaining / 3600:.2f} h", flush=True)
    rows = report(document, root, a.device, a.no_wandb, state)
    state["selection"] = select_winner(document, rows)
    state["phase"] = "final-assessment"; base.atomic_json(state_path, state)
    selected = state["selection"]["candidate"]
    tests = []
    for seed in document["seeds"]:
        if selected == "winner-none":
            control = next(r for r in document["controls"] if r["seed"] == seed)
            row = json.loads((Path(control["directory"]) / "test-summary.json").read_text())
            if row["config"] != control["summary"]["config"]:
                raise ValueError("cached upstream test configuration differs")
        else:
            spec = next(s for s in document["candidates"] if s["id"] == selected)
            directory = root / job_key(selected, seed)
            launch("test", spec, seed, directory)
            row = json.loads((directory / "test-summary.json").read_text())
        tests.append({"seed": seed, "accuracy": row["accuracy"], "loss": row["loss"], "reused_upstream": selected == "winner-none"})
        state["test_results"] = tests; base.atomic_json(state_path, state)
    state["phase"] = "completed"; base.atomic_json(state_path, state)
    report(document, root, a.device, a.no_wandb, state)
    print(f"Follow-up queue completed. Report: {root / 'comparison.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "smoke", "status", "report"))
    p.add_argument("--manifest", type=Path, default=HERE / "followup_experiments.json")
    p.add_argument("--source-root", type=Path, default=HERE / "runs/depth")
    p.add_argument("--root", type=Path, default=HERE / "runs/followup")
    p.add_argument("--augmentation-root", type=Path, default=HERE / "runs/augmentation")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    request = load_manifest(a.manifest)
    root, source = a.root.resolve(), a.source_root.resolve()
    if root == source:
        raise ValueError("follow-up output root must differ from source root")
    source_frozen = json.loads((source / "manifest.json").read_text())
    if source_frozen["device"] != a.device or source_frozen["manifest"]["seeds"] != request["seeds"]:
        raise ValueError("follow-up must use the source device and paired seeds")
    frozen = {"request": request, "source_root": str(source), "source_manifest": source_frozen,
              "device": a.device, "no_wandb": a.no_wandb, "implementation": trainer.implementation_hash()}
    identity = base.digest(frozen)
    state_path = root / "queue-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"identity": identity, "jobs": {}, "phase": "not-started"}
    if state["identity"] != identity:
        raise ValueError("follow-up queue settings or implementation changed; use a new root")
    if a.action == "status":
        print(json.dumps({"phase": state["phase"], "new_training_runs": 3 * len(request["seeds"]),
                          "reused_control_runs": len(request["seeds"]),
                          "counts": {s: sum(j["status"] == s for j in state["jobs"].values()) for s in ("pending", "running", "completed", "failed")},
                          "active": {k: j for k, j in state["jobs"].items() if j["status"] in ("running", "failed")},
                          "selection": state.get("selection"), "eta_seconds": state.get("eta_seconds"), "error": state.get("error")}, indent=2))
        return
    if a.action == "report":
        document = json.loads((root / "specs.json").read_text())
        report(document, root, a.device, a.no_wandb, state)
        print(root / "comparison.md"); return
    root.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        queue_lock = stack.enter_context((root / "queue.lock").open("a"))
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.atomic_json(root / "manifest.json", frozen)
        base.atomic_json(state_path, state)
        try:
            upstream_state, source_locks = wait_for_depth(source, a.augmentation_root.resolve(), a.device,
                                                        state, state_path, source_frozen)
            locks = [queue_lock] + [stack.enter_context(lock) for lock in source_locks]
            if frozen["implementation"] != trainer.implementation_hash():
                raise ValueError("source implementation changed while waiting; restore it or use a new root")
            document = prepare(request, source, root, upstream_state, source_frozen)
            report(document, root, a.device, a.no_wandb, state)
            execute(a, document, root, state, state_path, locks)
        except BaseException as error:
            state.update(phase="failed", error=str(error)); base.atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
