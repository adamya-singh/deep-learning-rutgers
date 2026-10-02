"""Sequential, restartable CIFAR-10 augmentation experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from augmentations import RECIPES

HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest["version"] != 1:
        raise ValueError("unsupported manifest version")
    if not manifest["recipes"] or not manifest["seeds"]:
        raise ValueError("recipes and seeds must be nonempty")
    if len(set(manifest["recipes"])) != len(manifest["recipes"]) or len(set(manifest["seeds"])) != len(manifest["seeds"]):
        raise ValueError("duplicate recipe or seed")
    for recipe in manifest["recipes"]:
        if recipe not in RECIPES:
            raise ValueError(f"unknown recipe: {recipe}")
    for key in ("epochs", "batch_size", "eval_batch_size", "lr"):
        if manifest[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return manifest


def jobs(manifest):
    return [(recipe, seed) for recipe in manifest["recipes"] for seed in manifest["seeds"]]


def command(manifest, recipe, seed, directory, device, no_wandb=False):
    cmd = [sys.executable, "-u", str(HERE / "cifar_cnn.py"),
           "--augmentation", recipe, "--seed", str(seed), "--device", device,
           "--group", "augmentation", "--run-name", f"Augmentation: {recipe}, seed {seed}",
           "--output-dir", str(directory), "--data-dir", str(HERE / "data")]
    for key in ("epochs", "batch_size", "eval_batch_size", "lr", "precision", "layout"):
        cmd.extend(["--" + key.replace("_", "-"), str(manifest[key])])
    if no_wandb:
        cmd.append("--no-wandb")
    return cmd


def expected_config(manifest, recipe, seed, device):
    return {"run_name": f"Augmentation: {recipe}, seed {seed}", "group": "augmentation",
            "augmentation": recipe, "augmentation_params": RECIPES[recipe], "seed": seed,
            "augmentation_backend": "hybrid-v1" if recipe != "none" else "none",
            "batch_size": manifest["batch_size"], "eval_batch_size": manifest["eval_batch_size"],
            "lr": manifest["lr"], "precision": manifest["precision"], "layout": manifest["layout"],
            "device": device, "compile": False, "rng_policy": "isolated-v1",
            "entity": "7adamyasingh-rutgers-university", "project": "cifar-activity"}


def completed(directory, expected, epochs, require_wandb=False):
    import torch
    path = directory / "summary.json"
    if not path.exists() or not (directory / "last.pt").exists():
        return False
    summary = json.loads(path.read_text())
    if summary["config"] != expected or summary["epochs"] != epochs:
        raise ValueError(f"completed output configuration differs: {directory}")
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"] != expected or checkpoint["epoch"] != epochs:
        raise ValueError(f"checkpoint and summary disagree: {directory}")
    if require_wandb and not summary.get("wandb_url"):
        raise ValueError(f"completed output has no W&B run: {directory}")
    return True


def previews(manifest, root, device):
    import torch
    torch.set_num_threads(1)
    from torchvision.datasets import CIFAR10
    from torchvision.utils import save_image
    from augmentations import Augmenter
    from cifar_cnn import MEAN, STD
    dataset = CIFAR10(str(HERE / "data"), train=True, download=False)
    raw = torch.from_numpy(dataset.data[:16]).permute(0, 3, 1, 2).to(device)
    labels = torch.tensor(dataset.targets[:16], device=device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    for recipe in manifest["recipes"]:
        directory = root / "previews"
        directory.mkdir(parents=True, exist_ok=True)
        augmented, targets = Augmenter(recipe, 12345, device, MEAN, STD)(raw, labels)
        images = torch.cat((raw.float() / 255, (augmented * std + mean).clamp(0, 1)))
        save_image(images, directory / f"{recipe}.png", nrow=8)
        atomic_json(directory / f"{recipe}.json", {"preview_seed": 12345,
                    "original_labels": labels.tolist(), "augmented_targets": targets.tolist(),
                    "parameters": RECIPES[recipe], "layout": "first two rows original, last two rows augmented"})


def smoke(manifest, root, device):
    estimates = []
    for recipe in manifest["recipes"]:
        directory = root / "smoke" / recipe
        directory.mkdir(parents=True, exist_ok=True)
        cmd = command(manifest, recipe, manifest["seeds"][0], directory, device, True)
        cmd.extend(["--benchmark", "--benchmark-batches", "2"])
        started = time.perf_counter()
        with (directory / "smoke.log").open("a") as log:
            result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"smoke failed for {recipe}: {directory / 'smoke.log'}")
        benchmark = json.loads((directory / "benchmarks.jsonl").read_text().splitlines()[-1])
        estimate = (50000 / benchmark["train_images_per_second"] + 60000 / benchmark["eval_images_per_second"]) * manifest["epochs"]
        estimates.append({"recipe": recipe, "train_images_per_second": benchmark["train_images_per_second"],
                          "estimated_run_seconds": estimate, "smoke_wall_seconds": time.perf_counter() - started})
        print(f"Smoke passed: {recipe}; estimated training/evaluation {estimate:.1f}s per run", flush=True)
    atomic_json(root / "smoke-results.json", {"manifest": manifest,
                "recipes": estimates, "estimated_queue_seconds": sum(r["estimated_run_seconds"] for r in estimates) * len(manifest["seeds"]),
                "note": "Estimate excludes process startup, previews, checkpointing, W&B, and final diagnostics."})


def report(manifest, root):
    records = {}
    for recipe, seed in jobs(manifest):
        path = root / recipe / f"seed-{seed}" / "summary.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["epochs"] != manifest["epochs"] or row["config"] != expected_config(manifest, recipe, seed, row["config"]["device"]):
                raise ValueError(f"incompatible summary: {path}")
            records[recipe, seed] = row
    lines = ["# CIFAR-10 augmentation comparison", "",
             f"Fixed recipes; batch {manifest['batch_size']}, {manifest['epochs']} epochs, SGD lr {manifest['lr']}. "
             "All 50,000 training images; clean test evaluation at the final epoch. Comparisons are exploratory.", "",
             f"Completed summaries: {len(records)}/{len(jobs(manifest))}. Spread is sample standard deviation across seeds.", "",
             "| Recipe | Seeds | Test accuracy, % | Δ vs none, pp | Δ vs crop+flip, pp | Clean train−test gap, pp | Train images/s | Train/eval time, s |", 
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    aggregate = []
    def mean_spread(values):
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f} (one seed)"
    for recipe in manifest["recipes"]:
        seeds = [s for s in manifest["seeds"] if (recipe, s) in records]
        if not seeds:
            continue
        rows = [records[recipe, s] for s in seeds]
        deltas = {}
        for reference in ("none", "crop-flip"):
            paired = [records[recipe, s]["test_accuracy"] - records[reference, s]["test_accuracy"] for s in seeds if (reference, s) in records]
            deltas[reference] = paired
        accuracy = [r["test_accuracy"] for r in rows]
        gap = [r["train_accuracy"] - r["test_accuracy"] for r in rows]
        seconds = [r["elapsed_seconds"] for r in rows]
        speed = [r["mean_train_images_per_second"] for r in rows]
        lines.append(f"| {recipe} | {len(seeds)} | {mean_spread(accuracy)} | "
                     f"{mean_spread(deltas['none']) if deltas['none'] else 'pending'} | "
                     f"{mean_spread(deltas['crop-flip']) if deltas['crop-flip'] else 'pending'} | "
                     f"{statistics.mean(gap):.2f} | {statistics.mean(speed):.0f} | {mean_spread(seconds)} |")
        aggregate.append({"recipe": recipe, "seeds": seeds, "test_accuracy_mean": statistics.mean(accuracy),
                          "test_accuracy_sample_std": statistics.stdev(accuracy) if len(accuracy) > 1 else None,
                          "paired_differences": deltas, "runs": rows})
    lines.extend(["", "Historical reference: the previous unaugmented seed-0 run reached 67.67% test accuracy. "
                  "Fresh paired controls determine differences in this suite; no checkpoint or augmentation strength was selected using test results.", ""])
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.md").write_text("\n".join(lines))
    atomic_json(root / "comparison.json", {"completed": len(records), "expected": len(jobs(manifest)), "recipes": aggregate})
    return len(records)


def run_queue(manifest, root, device, no_wandb):
    # Prevent two runners from touching the same checkpoints/results on this host.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = hashlib.sha256(json.dumps({"manifest": manifest, "recipes": RECIPES,
                                            "device": device, "no_wandb": no_wandb}, sort_keys=True).encode()).hexdigest()
        path = root / "queue-state.json"
        state = json.loads(path.read_text()) if path.exists() else {"identity": identity, "jobs": {}}
        if state["identity"] != identity:
            raise ValueError("queue settings changed; use the original settings or a separate root")
        for recipe, seed in jobs(manifest):
            state["jobs"].setdefault(f"{recipe}/seed-{seed}", {"status": "pending"})
        atomic_json(path, state)
        atomic_json(root / "manifest.json", {"manifest": manifest,
                    "recipe_parameters": {recipe: RECIPES[recipe] for recipe in manifest["recipes"]},
                    "augmentation_backend": "hybrid-v1", "device": device, "no_wandb": no_wandb})
        previews(manifest, root, device)
        for position, (recipe, seed) in enumerate(jobs(manifest), 1):
            key = f"{recipe}/seed-{seed}"
            directory = root / recipe / f"seed-{seed}"
            expected = expected_config(manifest, recipe, seed, device)
            if completed(directory, expected, manifest["epochs"], require_wandb=not no_wandb):
                state["jobs"][key]["status"] = "completed"
                atomic_json(path, state)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            cmd = command(manifest, recipe, seed, directory, device, no_wandb)
            if (directory / "last.pt").exists():
                cmd.append("--resume")
            state["jobs"][key].update(status="running", command=cmd)
            atomic_json(path, state)
            print(f"[{position}/{len(jobs(manifest))}] Starting {key}", flush=True)
            started = time.perf_counter()
            try:
                with (directory / "run.log").open("a") as log:
                    # A surviving training child retains the lock if its runner exits.
                    result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                            pass_fds=(lock.fileno(),))
                if result.returncode or not completed(directory, expected, manifest["epochs"], require_wandb=not no_wandb):
                    raise RuntimeError(f"job failed (exit {result.returncode}); inspect {directory / 'run.log'}")
            except BaseException as error:
                state["jobs"][key].update(status="failed", error=str(error))
                atomic_json(path, state)
                raise
            job = state["jobs"][key]
            job.update(status="completed", wall_seconds=job.get("wall_seconds", 0) + time.perf_counter() - started)
            job.pop("error", None)
            atomic_json(path, state)
            report(manifest, root)
            print(f"[{position}/{len(jobs(manifest))}] Completed {key}", flush=True)
        count = report(manifest, root)
        if count != len(jobs(manifest)):
            raise RuntimeError("queue finished without all expected summaries")
        print(f"All {count} runs complete. Report: {root / 'comparison.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "smoke", "run", "report", "status"))
    parser.add_argument("--manifest", type=Path, default=HERE / "augmentation_experiments.json")
    parser.add_argument("--root", type=Path, default=HERE / "runs/augmentation")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    root = args.root.resolve()
    if args.action == "run":
        run_queue(manifest, root, args.device, args.no_wandb)
    elif args.action == "smoke":
        smoke(manifest, root, args.device)
    elif args.action == "preview":
        previews(manifest, root, args.device)
    elif args.action == "report":
        report(manifest, root)
        print(root / "comparison.md")
    else:
        path = root / "queue-state.json"
        if path.exists():
            state = json.loads(path.read_text())
            counts = {status: sum(job["status"] == status for job in state["jobs"].values())
                      for status in ("pending", "running", "completed", "failed")}
            print(json.dumps({"counts": counts, "active": {key: job for key, job in state["jobs"].items()
                  if job["status"] in ("running", "failed")}}, indent=2))
        else:
            print(f"{len(jobs(manifest))} jobs planned; queue has not started")


if __name__ == "__main__":
    main()
