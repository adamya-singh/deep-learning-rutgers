"""Standalone normalized plain/residual CIFAR experiment trainer."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10

from cifar_cnn import MEAN, STD, evaluate, save_checkpoint, setup, train_epoch

HERE = Path(__file__).resolve().parent
SOURCES = ("depth_cnn.py", "run_depth_queue.py", "cifar_cnn.py", "augmentations.py")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def implementation_hash():
    h = hashlib.sha256()
    for name in SOURCES:
        h.update(name.encode())
        h.update((HERE / name).read_bytes())
    return h.hexdigest()


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if m["version"] != 1:
        raise ValueError("unsupported depth manifest version")
    if m["depths"][0] != 2 or sorted(set(m["depths"])) != m["depths"]:
        raise ValueError("depths must be unique ascending candidates beginning at 2")
    if any(d != 2 and (d < 4 or d % 4) for d in m["depths"]):
        raise ValueError("deeper candidates must have an even number of pairs per two-stage network")
    if not m["seeds"] or len(set(m["seeds"])) != len(m["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    for key in ("epochs", "score_epochs", "batch_size", "eval_batch_size", "warmup_epochs",
                "validation_per_class", "plateau_patience", "declining_seeds"):
        if not isinstance(m[key], int) or m[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if m["score_epochs"] > m["epochs"] or m["epochs"] <= m["warmup_epochs"] + 1:
        raise ValueError("invalid scoring window or warmup duration")
    if m["declining_seeds"] > len(m["seeds"]) or m["validation_per_class"] >= 5000:
        raise ValueError("invalid seed threshold or validation size")
    for key in ("lr_start", "lr_peak", "lr_min", "momentum", "weight_decay", "plateau_gain_pp", "decline_pp"):
        if not math.isfinite(m[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < m["lr_min"] <= m["lr_start"] <= m["lr_peak"] or not 0 <= m["momentum"] < 1 or m["weight_decay"] < 0:
        raise ValueError("invalid optimizer settings")
    if m["plateau_gain_pp"] < 0 or m["decline_pp"] >= 0:
        raise ValueError("invalid stopping thresholds")
    if (m["augmentation"], m["precision"], m["layout"]) != ("none", "fp32", "contiguous"):
        raise ValueError("depth experiments require none/fp32/contiguous")
    return m


class Block(nn.Module):
    def __init__(self, input_channels, channels, residual):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.residual = residual

    def shortcut(self, x):
        missing = self.conv2.out_channels - x.shape[1]
        return torch.cat((x, x.new_zeros(x.shape[0], missing, *x.shape[2:])), 1) if missing else x

    def forward(self, x):
        z = torch.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return torch.relu(z + self.shortcut(x) if self.residual else z)


def build_model(depth, variant="plain"):
    if variant not in ("plain", "residual") or depth < 2 or (depth != 2 and depth % 4):
        raise ValueError("invalid architecture")
    if depth == 2 and variant == "residual":
        raise ValueError("residual candidates start at depth 4")
    stages = []
    inputs = 3
    for channels in (32, 64):
        if depth == 2:
            stages.extend([nn.Conv2d(inputs, channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.ReLU()])
        else:
            for _ in range(depth // 4):
                stages.append(Block(inputs, channels, variant == "residual"))
                inputs = channels
        stages.append(nn.MaxPool2d(2))
        inputs = channels
    return nn.Sequential(*stages, nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))


def learning_rate(epoch, m):
    if epoch <= m["warmup_epochs"]:
        fraction = (epoch - 1) / max(1, m["warmup_epochs"] - 1)
        return m["lr_start"] + fraction * (m["lr_peak"] - m["lr_start"])
    fraction = (epoch - m["warmup_epochs"] - 1) / (m["epochs"] - m["warmup_epochs"] - 1)
    return m["lr_min"] + (m["lr_peak"] - m["lr_min"]) * (1 + math.cos(math.pi * fraction)) / 2


def split_indices(targets, seed, per_class):
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    train, val = [], []
    for label in range(10):
        positions = np.flatnonzero(labels == label)
        if len(positions) <= per_class:
            raise ValueError("not enough examples for stratified split")
        positions = rng.permutation(positions)
        val.extend(positions[:per_class].tolist())
        train.extend(positions[per_class:].tolist())
    return {"train": train, "validation": val}


def normalize(images, device):
    x = torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous()
    return x.div_(255).sub_(torch.tensor(MEAN, device=device).view(1, 3, 1, 1)).div_(
        torch.tensor(STD, device=device).view(1, 3, 1, 1))


def load_training(m, root, device):
    dataset = CIFAR10(str(HERE / "data"), train=True, download=False)
    indices = split_indices(dataset.targets, m["split_seed"], m["validation_per_class"])
    identity = digest(indices)
    path = root / "split.json"
    document = {"seed": m["split_seed"], "hash": identity, "indices": indices}
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError("saved split differs from requested split")
    atomic_json(path, document)
    labels = np.asarray(dataset.targets)
    return (normalize(dataset.data[indices["train"]], device),
            torch.tensor(labels[indices["train"]], device=device),
            normalize(dataset.data[indices["validation"]], device),
            torch.tensor(labels[indices["validation"]], device=device)), identity


def config(m, variant, depth, seed, device, split_hash):
    return {"manifest": m, "implementation": implementation_hash(), "variant": variant,
            "depth": depth, "seed": seed, "device": str(device), "split_hash": split_hash,
            "architecture": {"channels": [32, 64], "pooling": "max-2-after-each-stage",
                             "head": [4096, 128, 10], "batch_norm": True,
                             "shortcut": "zero-pad-channels" if variant == "residual" else "none"},
            "normalization": {"mean": list(MEAN), "std": list(STD)}, "torch_version": str(torch.__version__),
            "entity": "7adamyasingh-rutgers-university", "project": "cifar-activity"}


def initialize(m, variant, depth, seed, device):
    setup(seed, device, "fp32")
    # Deterministic convolutions make paired seeds and resume comparisons easier to interpret.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(depth, variant).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=m["lr_start"], momentum=m["momentum"],
                                weight_decay=m["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = argparse.Namespace(batch_size=m["batch_size"], eval_batch_size=m["eval_batch_size"],
                              precision="fp32", layout="contiguous", augmenter=None,
                              shuffle_generator=torch.Generator(device=device).manual_seed(seed))
    return model, optimizer, scaler, args


def snapshot(model, optimizer, scaler, args, cfg, epoch, history, wandb_id):
    return {"config": cfg, "epoch": epoch, "next_epoch": epoch + 1,
            "lr_epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "history": history, "wandb_id": wandb_id,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(), "shuffle_rng": args.shuffle_generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None}


def restore(state, cfg, model, optimizer, scaler, args):
    if state["config"] != cfg:
        raise ValueError("checkpoint configuration or implementation differs")
    if state["next_epoch"] != state["epoch"] + 1 or state["lr_epoch"] != state["epoch"]:
        raise ValueError("checkpoint learning-rate progress differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    args.shuffle_generator.set_state(state["shuffle_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def write_metrics(directory, history):
    temp = directory / "metrics.tmp"
    temp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    os.replace(temp, directory / "metrics.jsonl")


def train(m, root, directory, variant, depth, seed, device, no_wandb):
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    data, split_hash = load_training(m, root, device)
    cfg = config(m, variant, depth, seed, device, split_hash)
    model, optimizer, scaler, args = initialize(m, variant, depth, seed, device)
    path = directory / "last.pt"
    state = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    history = state["history"] if state else []
    if state:
        restore(state, cfg, model, optimizer, scaler, args)
    run = None
    if not no_wandb:
        import wandb
        run = wandb.init(entity=cfg["entity"], project=cfg["project"], group=f"{variant}-depth",
                         name=f"{variant}: {depth} conv, seed {seed}",
                         id=state["wandb_id"] if state else None, resume="must" if state else None,
                         config={**cfg, "params": sum(p.numel() for p in model.parameters()),
                                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
                         tags=["depth", variant, "validation-selection"])
        run.define_metric("epoch")
        for pattern in ("train/*", "eval/*", "performance/*", "optimizer/*"):
            run.define_metric(pattern, step_metric="epoch")
        if history and int(run.summary.get("last_epoch", 0)) < history[-1]["epoch"]:
            run.log(history[-1])
            run.summary["last_epoch"] = history[-1]["epoch"]
    elif state and state["wandb_id"]:
        raise ValueError("cannot resume a W&B run with --no-wandb")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps = history[-1]["performance/optimizer_steps"] if history else 0
    elapsed = history[-1]["performance/elapsed_seconds"] if history else 0
    loss_fn = nn.CrossEntropyLoss()
    x, y, vx, vy = data
    try:
        for epoch in range(state["next_epoch"] if state else 1, m["epochs"] + 1):
            lr = learning_rate(epoch, m)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = train_epoch(model, optimizer, scaler, loss_fn, x, y, args, device, check_gradients=True)
            clean = evaluate(model, loss_fn, x, y, args, device)
            val = evaluate(model, loss_fn, vx, vy, args, device)
            if not all(math.isfinite(r[k]) for r in (batch, clean, val) for k in ("loss", "accuracy")):
                raise FloatingPointError("non-finite training/evaluation metrics")
            steps += batch["steps"]
            elapsed += batch["seconds"] + clean["seconds"] + val["seconds"]
            row = {"epoch": epoch, "optimizer/lr": lr, "train/loss": batch["loss"],
                   "train/batch_accuracy": batch["accuracy"], "train/eval_accuracy": clean["accuracy"],
                   "train/eval_loss": clean["loss"], "eval/validation_accuracy": val["accuracy"],
                   "eval/validation_loss": val["loss"], "performance/optimizer_steps": steps,
                   "performance/elapsed_seconds": elapsed, "performance/train_seconds": batch["seconds"],
                   "performance/train_images_per_second": batch["images_per_second"],
                   "performance/eval_seconds": clean["seconds"] + val["seconds"],
                   "performance/peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0}
            history.append(row)
            save_checkpoint(path, snapshot(model, optimizer, scaler, args, cfg, epoch, history, run.id if run else None))
            write_metrics(directory, history)
            if run:
                run.log(row)
                run.summary["last_epoch"] = epoch
            print(f"epoch {epoch}/{m['epochs']} lr={lr:.6f} train={clean['accuracy']:.2f}% "
                  f"val={val['accuracy']:.2f}% loss={batch['loss']:.4f}", flush=True)
        if len(history) != m["epochs"]:
            raise ValueError("checkpoint history length differs from epoch budget")
        write_metrics(directory, history)
        score = sum(r["eval/validation_accuracy"] for r in history[-m["score_epochs"]:]) / m["score_epochs"]
        summary = {"config": cfg, "epochs": m["epochs"], "score": score, "final": history[-1],
                   "params": sum(p.numel() for p in model.parameters()), "process_seconds": time.perf_counter() - started,
                   "wandb_url": run.url if run else ""}
        if run:
            run.summary.update({"final/validation_score": score, "final/validation_accuracy": history[-1]["eval/validation_accuracy"],
                                "final/train_accuracy": history[-1]["train/eval_accuracy"]})
            run.finish()
        atomic_json(directory / "summary.json", summary)
    except BaseException:
        if run:
            run.finish(exit_code=1)
        raise


def smoke(m, root, directory, variant, depth, seed, device):
    directory.mkdir(parents=True, exist_ok=True)
    data, split_hash = load_training(m, root, device)
    cfg = config(m, variant, depth, seed, device, split_hash)
    model, opt, scaler, args = initialize(m, variant, depth, seed, device)
    loss_fn = nn.CrossEntropyLoss()
    x, y, vx, vy = data
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_epoch(model, opt, scaler, loss_fn, x, y, args, device, max_batches=3, check_gradients=True)
    evaluate(model, loss_fn, vx[:1024], vy[:1024], args, device)
    measured = train_epoch(model, opt, scaler, loss_fn, x, y, args, device, max_batches=8, check_gradients=True)
    evaluation = evaluate(model, loss_fn, vx[:2048], vy[:2048], args, device)
    if not all(math.isfinite(r[k]) for r in (measured, evaluation) for k in ("loss", "accuracy")):
        raise FloatingPointError("non-finite preflight metrics")
    # Actual checkpoint serialization and one-step continuation comparison on each architecture.
    save_checkpoint(directory / "probe.pt", snapshot(model, opt, scaler, args, cfg, 1, [], None))
    small_x, small_y = x[:m["batch_size"]], y[:m["batch_size"]]
    train_epoch(model, opt, scaler, loss_fn, small_x, small_y, args, device, check_gradients=True)
    expected = copy.deepcopy(model.state_dict())
    state = torch.load(directory / "probe.pt", map_location="cpu", weights_only=False)
    restored, ropt, rscaler, rargs = initialize(m, variant, depth, seed, device)
    restore(state, cfg, restored, ropt, rscaler, rargs)
    train_epoch(restored, ropt, rscaler, loss_fn, small_x, small_y, rargs, device, check_gradients=True)
    for key, value in expected.items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=1e-3 if device.type == "cuda" else 0,
                                   atol=1e-4 if device.type == "cuda" else 0)
    for a, b in zip(opt.state.values(), ropt.state.values()):
        torch.testing.assert_close(a["momentum_buffer"], b["momentum_buffer"],
                                   rtol=1e-3 if device.type == "cuda" else 0,
                                   atol=1e-4 if device.type == "cuda" else 0)
    torch.testing.assert_close(args.shuffle_generator.get_state(), rargs.shuffle_generator.get_state(), rtol=0, atol=0)
    (directory / "probe.pt").unlink()
    result = {"variant": variant, "depth": depth, "config": cfg, "resume_verified": True,
              "train_images_per_second": measured["images_per_second"],
              "eval_images_per_second": min(2048, len(vx)) / evaluation["seconds"],
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0,
              "params": sum(p.numel() for p in model.parameters())}
    result["estimated_run_seconds"] = m["epochs"] * (len(x) / result["train_images_per_second"] +
                                                        (len(x) + len(vx)) / result["eval_images_per_second"])
    atomic_json(directory / "smoke.json", result)
    print(json.dumps(result), flush=True)


def assess(m, root, directory, variant, depth, seed, device, no_wandb):
    split = json.loads((root / "split.json").read_text())
    cfg = config(m, variant, depth, seed, device, split["hash"])
    output = directory / "test-summary.json"
    if output.exists():
        row = json.loads(output.read_text())
        if row["config"] != cfg:
            raise ValueError("test summary configuration differs")
        state = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        upload_assessment(directory, state, row, no_wandb)
        return
    state = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    if state["config"] != cfg or state["epoch"] != m["epochs"]:
        raise ValueError("final assessment requires a completed compatible checkpoint")
    model, _, _, args = initialize(m, variant, depth, seed, device)
    model.load_state_dict(state["model"])
    dataset = CIFAR10(str(HERE / "data"), train=False, download=False)
    x = normalize(dataset.data, device)
    y = torch.tensor(dataset.targets, device=device)
    result = evaluate(model, nn.CrossEntropyLoss(), x, y, args, device, diagnostics=True)
    row = {"config": cfg, "epoch": state["epoch"], "accuracy": result["accuracy"], "loss": result["loss"],
           "matrix": result["matrix"].tolist()}
    # Persist before network logging, so retries never re-evaluate an already recorded test result.
    atomic_json(output, row)
    upload_assessment(directory, state, row, no_wandb)


def upload_assessment(directory, state, row, no_wandb):
    marker = directory / "test-uploaded.json"
    if no_wandb or marker.exists():
        return
    import wandb
    run = wandb.init(entity=row["config"]["entity"], project=row["config"]["project"],
                     id=state["wandb_id"], resume="must")
    run.summary.update({"final/test_accuracy": row["accuracy"], "final/test_loss": row["loss"]})
    run.finish()
    atomic_json(marker, {"wandb_id": state["wandb_id"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "smoke", "test"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--variant", choices=("plain", "residual"), required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    m = load_manifest(a.manifest)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if a.depth not in m["depths"] or a.seed not in m["seeds"]:
        raise ValueError("candidate is not in the manifest")
    if a.action == "smoke":
        smoke(m, a.root, a.directory, a.variant, a.depth, a.seed, device)
    elif a.action == "train":
        train(m, a.root, a.directory, a.variant, a.depth, a.seed, device, a.no_wandb)
    else:
        assess(m, a.root, a.directory, a.variant, a.depth, a.seed, device, a.no_wandb)


if __name__ == "__main__":
    main()
