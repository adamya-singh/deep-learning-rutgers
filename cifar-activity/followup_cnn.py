"""Augmentation of the depth winner and a CIFAR-adapted ResNet-18, without changing the active depth code."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18

import depth_cnn as base
from augmentations import Augmenter, RECIPES

HERE = base.HERE


def implementation_hash():
    h = hashlib.sha256(base.implementation_hash().encode())
    for name in ("followup_cnn.py", "run_followup_queue.py"):
        h.update(name.encode()); h.update((HERE / name).read_bytes())
    return h.hexdigest()


def build_model(spec):
    if spec["architecture"] == "depth-winner":
        return base.build_model(spec["depth"], spec["variant"])
    if spec["architecture"] != "resnet18-cifar":
        raise ValueError("unknown architecture")
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model


def initialize(spec, seed, device):
    m = spec["training"]
    base.setup(seed, device, "fp32")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(spec).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=m["lr_start"], momentum=m["momentum"],
                                weight_decay=m["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    recipe = spec["augmentation"]
    args = argparse.Namespace(batch_size=m["batch_size"], eval_batch_size=m["eval_batch_size"],
                              precision="fp32", layout="contiguous", raw_train=None,
                              shuffle_generator=torch.Generator(device=device).manual_seed(seed),
                              augmenter=Augmenter(recipe, seed, device, base.MEAN, base.STD) if recipe != "none" else None)
    return model, optimizer, scaler, args


def load_training(spec, root, device):
    data, split_hash = base.load_training(spec["training"], root, device)
    raw = None
    if spec["augmentation"] != "none":
        dataset = base.CIFAR10(str(HERE / "data"), train=True, download=False)
        indices = json.loads((root / "split.json").read_text())["indices"]["train"]
        raw = torch.from_numpy(dataset.data[indices]).permute(0, 3, 1, 2).contiguous().to(device)
    return (*data, raw), split_hash


def config(spec, seed, device, split_hash):
    return {"spec": spec, "seed": seed, "device": str(device), "split_hash": split_hash,
            "implementation": implementation_hash(), "augmentation_params": RECIPES[spec["augmentation"]],
            "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
            "entity": "7adamyasingh-rutgers-university", "project": "cifar-activity"}


def snapshot(model, opt, scaler, args, cfg, epoch, history, wandb_id):
    state = base.snapshot(model, opt, scaler, args, cfg, epoch, history, wandb_id)
    state["augmentation_rng"] = args.augmenter.state_dict() if args.augmenter else None
    return state


def restore(state, cfg, model, opt, scaler, args):
    base.restore(state, cfg, model, opt, scaler, args)
    if args.augmenter:
        if state.get("augmentation_rng") is None:
            raise ValueError("missing augmentation RNG in checkpoint")
        args.augmenter.load_state_dict(state["augmentation_rng"])


def train(spec, root, directory, seed, device, no_wandb):
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    data, split_hash = load_training(spec, root, device)
    x, y, vx, vy, raw = data
    cfg = config(spec, seed, device, split_hash)
    model, opt, scaler, args = initialize(spec, seed, device)
    args.raw_train = raw
    m = spec["training"]
    checkpoint = directory / "last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
    history = state["history"] if state else []
    if state:
        restore(state, cfg, model, opt, scaler, args)
    run = None
    if not no_wandb:
        import wandb
        run = wandb.init(entity=cfg["entity"], project=cfg["project"],
                         group="winner-augmentation" if spec["architecture"] == "depth-winner" else "resnet18-cifar",
                         name=f"{spec['id']}, seed {seed}", id=state["wandb_id"] if state else None,
                         resume="must" if state else None,
                         config={**cfg, "params": sum(p.numel() for p in model.parameters()),
                                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
                         tags=["followup", "validation-selection", spec["augmentation"]])
        run.define_metric("epoch")
        for pattern in ("train/*", "eval/*", "performance/*", "optimizer/*"):
            run.define_metric(pattern, step_metric="epoch")
        if history and int(run.summary.get("last_epoch", 0)) < history[-1]["epoch"]:
            run.log(history[-1]); run.summary["last_epoch"] = history[-1]["epoch"]
    elif state and state["wandb_id"]:
        raise ValueError("cannot resume a W&B checkpoint with --no-wandb")
    loss_fn = nn.CrossEntropyLoss()
    steps = history[-1]["performance/optimizer_steps"] if history else 0
    elapsed = history[-1]["performance/elapsed_seconds"] if history else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(state["next_epoch"] if state else 1, m["epochs"] + 1):
            lr = base.learning_rate(epoch, m)
            for group in opt.param_groups:
                group["lr"] = lr
            batch = base.train_epoch(model, opt, scaler, loss_fn, x, y, args, device, check_gradients=True)
            clean = base.evaluate(model, loss_fn, x, y, args, device)
            val = base.evaluate(model, loss_fn, vx, vy, args, device)
            if not all(math.isfinite(r[k]) for r in (batch, clean, val) for k in ("loss", "accuracy")):
                raise FloatingPointError("non-finite training metrics")
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
            base.save_checkpoint(checkpoint, snapshot(model, opt, scaler, args, cfg, epoch, history, run.id if run else None))
            base.write_metrics(directory, history)
            if run:
                run.log(row); run.summary["last_epoch"] = epoch
            print(f"epoch {epoch}/{m['epochs']} lr={lr:.6f} train={clean['accuracy']:.2f}% "
                  f"val={val['accuracy']:.2f}% loss={batch['loss']:.4f}", flush=True)
        if len(history) != m["epochs"]:
            raise ValueError("checkpoint history differs from training budget")
        base.write_metrics(directory, history)
        score = statistics.mean(r["eval/validation_accuracy"] for r in history[-m["score_epochs"]:])
        summary = {"config": cfg, "epochs": m["epochs"], "score": score, "final": history[-1],
                   "params": sum(p.numel() for p in model.parameters()), "wandb_url": run.url if run else "",
                   "process_seconds": time.perf_counter() - started}
        if run:
            run.summary.update({"final/validation_score": score, "final/train_accuracy": history[-1]["train/eval_accuracy"]})
            run.finish()
        base.atomic_json(directory / "summary.json", summary)
    except BaseException:
        if run:
            run.finish(exit_code=1)
        raise


def smoke(spec, root, directory, seed, device):
    directory.mkdir(parents=True, exist_ok=True)
    data, split_hash = load_training(spec, root, device)
    x, y, vx, vy, raw = data
    cfg = config(spec, seed, device, split_hash)
    model, opt, scaler, args = initialize(spec, seed, device)
    args.raw_train = raw
    loss_fn = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base.train_epoch(model, opt, scaler, loss_fn, x, y, args, device, max_batches=3, check_gradients=True)
    measured = base.train_epoch(model, opt, scaler, loss_fn, x, y, args, device, max_batches=8, check_gradients=True)
    result = base.evaluate(model, loss_fn, vx[:2048], vy[:2048], args, device)
    if not all(math.isfinite(r[k]) for r in (measured, result) for k in ("loss", "accuracy")):
        raise FloatingPointError("non-finite preflight metrics")
    path = directory / "probe.pt"
    base.save_checkpoint(path, snapshot(model, opt, scaler, args, cfg, 1, [], None))
    args.raw_train = raw[:args.batch_size] if raw is not None else None
    base.train_epoch(model, opt, scaler, loss_fn, x[:args.batch_size], y[:args.batch_size], args, device, check_gradients=True)
    expected = copy.deepcopy(model.state_dict())
    resumed, ropt, rscaler, rargs = initialize(spec, seed, device)
    rargs.raw_train = args.raw_train
    state = torch.load(path, map_location="cpu", weights_only=False)
    restore(state, cfg, resumed, ropt, rscaler, rargs)
    base.train_epoch(resumed, ropt, rscaler, loss_fn, x[:args.batch_size], y[:args.batch_size], rargs, device, check_gradients=True)
    tolerance = {"rtol": 1e-3, "atol": 1e-4} if device.type == "cuda" else {"rtol": 0, "atol": 0}
    for k, v in expected.items():
        torch.testing.assert_close(v, resumed.state_dict()[k], **tolerance)
    for a, b in zip(opt.state.values(), ropt.state.values()):
        torch.testing.assert_close(a["momentum_buffer"], b["momentum_buffer"], **tolerance)
    torch.testing.assert_close(args.shuffle_generator.get_state(), rargs.shuffle_generator.get_state(), rtol=0, atol=0)
    if args.augmenter:
        for k, v in args.augmenter.state_dict().items():
            if v is not None:
                torch.testing.assert_close(v, rargs.augmenter.state_dict()[k], rtol=0, atol=0)
    path.unlink()
    row = {"config": cfg, "resume_verified": True, "train_images_per_second": measured["images_per_second"],
           "eval_images_per_second": min(2048, len(vx)) / result["seconds"],
           "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0}
    row["estimated_run_seconds"] = spec["training"]["epochs"] * (len(x) / row["train_images_per_second"] +
                                                                   (len(x) + len(vx)) / row["eval_images_per_second"])
    base.atomic_json(directory / "smoke.json", row)
    print(json.dumps(row), flush=True)


def assess(spec, root, directory, seed, device, no_wandb):
    split = json.loads((root / "split.json").read_text())
    cfg = config(spec, seed, device, split["hash"])
    state = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    if state["config"] != cfg or state["epoch"] != spec["training"]["epochs"]:
        raise ValueError("test assessment needs a completed compatible checkpoint")
    path = directory / "test-summary.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row["config"] != cfg:
            raise ValueError("cached test result differs")
    else:
        model, _, _, args = initialize(spec, seed, device)
        model.load_state_dict(state["model"])
        dataset = base.CIFAR10(str(HERE / "data"), train=False, download=False)
        x = base.normalize(dataset.data, device)
        y = torch.tensor(dataset.targets, device=device)
        result = base.evaluate(model, nn.CrossEntropyLoss(), x, y, args, device, diagnostics=True)
        row = {"config": cfg, "epoch": state["epoch"], "accuracy": result["accuracy"], "loss": result["loss"],
               "matrix": result["matrix"].tolist()}
        base.atomic_json(path, row)
    base.upload_assessment(directory, state, row, no_wandb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "smoke", "test"))
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    document = json.loads(a.specs.read_text())
    if document["implementation"] != implementation_hash() or a.seed not in document["seeds"]:
        raise ValueError("implementation or seed differs from frozen queue")
    spec = next(s for s in document["candidates"] if s["id"] == a.candidate)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if a.action == "smoke":
        smoke(spec, a.root, a.directory, a.seed, device)
    elif a.action == "train":
        train(spec, a.root, a.directory, a.seed, device, a.no_wandb)
    else:
        assess(spec, a.root, a.directory, a.seed, device, a.no_wandb)


if __name__ == "__main__":
    main()
