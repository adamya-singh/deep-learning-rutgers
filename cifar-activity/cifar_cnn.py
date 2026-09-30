"""CIFAR-10 CNN activity. Run python3 cifar_cnn.py --help."""
import argparse
import contextlib
import csv
import json
import math
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets

RUN_NAME = "2 conv CNN, 10 ep"
EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED = 10, 64, 0.01, 0
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def build_model():
    """The unchanged classroom model."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=RUN_NAME)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--precision", choices=("fp32", "tf32", "fp16", "bf16"), default="fp32")
    p.add_argument("--layout", choices=("contiguous", "channels_last"), default="contiguous")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--entity", default="7adamyasingh-rutgers-university")
    p.add_argument("--project", default="cifar-activity")
    p.add_argument("--group", choices=("baseline", "tuned", "smoke"), default="baseline")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-batches", type=int, default=96)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.eval_batch_size, a.benchmark_batches) <= 0 or a.lr <= 0:
        p.error("epochs, batch sizes, learning rate, and benchmark batches must be positive")
    if a.resume and a.benchmark:
        p.error("--resume and --benchmark cannot be combined")
    return a


def setup(seed, device, precision):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        torch.backends.cudnn.allow_tf32 = precision == "tf32"


def load_cifar10(data_dir, device, layout):
    train = datasets.CIFAR10(str(data_dir), train=True, download=True)
    test = datasets.CIFAR10(str(data_dir), train=False, download=True)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    def prepare(dataset):
        x = torch.from_numpy(dataset.data).to(device=device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).div_(255).sub_(mean).div_(std)
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        return x, torch.tensor(dataset.targets, dtype=torch.long, device=device)
    return (*prepare(train), *prepare(test))


def autocast(device, precision):
    if device.type == "cuda" and precision in ("fp16", "bf16"):
        return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
    return contextlib.nullcontext()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class GpuSampler:
    """Sample hardware counters during an epoch without synchronizing CUDA."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=2)
                self.samples.append([float(value.strip()) for value in output.splitlines()[0].split(",")])
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(0.5)

    def start(self):
        if self.enabled:
            self.thread.start()

    def finish(self):
        if not self.enabled:
            return {}
        self.stop_event.set()
        self.thread.join(timeout=3)
        if not self.samples:
            return {}
        columns = list(zip(*self.samples))
        return {"system/gpu_utilization_pct": sum(columns[0]) / len(columns[0]),
                "system/gpu_memory_used_mb": max(columns[1]),
                "system/gpu_power_w": sum(columns[2]) / len(columns[2]),
                "system/gpu_temperature_c": max(columns[3])}


def make_model(args, device):
    model = build_model().to(device)
    if args.layout == "channels_last":
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    forward = torch.compile(model) if args.compile else model
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    return model, forward, optimizer, scaler


def train_epoch(model, optimizer, scaler, loss_fn, x, y, args, device, max_batches=None, check_gradients=False):
    model.train()
    order = torch.randperm(len(x), device=device)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    batches = math.ceil(len(x) / args.batch_size)
    if max_batches is not None:
        batches = min(batches, max_batches)
    seen = 0
    sync(device)
    started = time.perf_counter()
    for i in range(batches):
        indices = order[i * args.batch_size:(i + 1) * args.batch_size]
        xb, yb = x[indices], y[indices]
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.precision):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        scaler.scale(loss).backward()
        if check_gradients and i == 0:
            scaler.unscale_(optimizer)
            if not bool(torch.isfinite(loss).item()) or not all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("non-finite loss or gradient")
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * len(indices)
        correct += (logits.detach().argmax(1) == yb).sum()
        seen += len(indices)
    sync(device)
    seconds = time.perf_counter() - started
    return {"loss": (loss_sum / seen).item(), "accuracy": (correct.float() / seen * 100).item(),
            "seconds": seconds, "images_per_second": seen / seconds, "steps": batches}


@torch.inference_mode()
def evaluate(model, loss_fn, x, y, args, device, diagnostics=False):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    matrix = torch.zeros(100, dtype=torch.long, device=device) if diagnostics else None
    mistakes = []
    sync(device)
    started = time.perf_counter()
    for pos in range(0, len(x), args.eval_batch_size):
        xb, yb = x[pos:pos + args.eval_batch_size], y[pos:pos + args.eval_batch_size]
        with autocast(device, args.precision):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        pred = logits.argmax(1)
        loss_sum += loss * len(xb)
        correct += (pred == yb).sum()
        if diagnostics:
            matrix += torch.bincount(yb * 10 + pred, minlength=100)
            if len(mistakes) < 16:
                for j in torch.nonzero(pred != yb).flatten()[:16-len(mistakes)].tolist():
                    mistakes.append((xb[j].cpu(), yb[j].item(), pred[j].item()))
    sync(device)
    return {"loss": (loss_sum / len(x)).item(), "accuracy": (correct.float() / len(x) * 100).item(),
            "seconds": time.perf_counter() - started,
            "matrix": matrix.reshape(10, 10).cpu() if diagnostics else None, "mistakes": mistakes}


def config(args, device):
    return {"run_name": args.run_name, "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
            "lr": args.lr, "seed": args.seed, "device": device.type, "precision": args.precision,
            "layout": args.layout, "compile": args.compile, "entity": args.entity,
            "project": args.project, "group": args.group}


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def benchmark(args, device, data):
    x, y, _, _ = data
    setup(args.seed, device, args.precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    reference_model, model, optimizer, scaler = make_model(args, device)
    loss_fn = nn.CrossEntropyLoss()
    train_epoch(model, optimizer, scaler, loss_fn, x, y, args, device, max_batches=6, check_gradients=True)
    evaluate(model, loss_fn, x[:8192], y[:8192], args, device)
    setup_seconds = time.perf_counter() - start
    sample = x[:min(256, args.batch_size)]
    reference_model.eval()
    model.eval()
    with torch.inference_mode():
        fp32_logits = reference_model(sample).float()
        with autocast(device, args.precision):
            candidate_logits = model(sample).float()
    numerical_difference = (candidate_logits - fp32_logits).abs().mean().item()
    prediction_agreement = (candidate_logits.argmax(1) == fp32_logits.argmax(1)).float().mean().item()
    if not math.isfinite(numerical_difference):
        raise FloatingPointError("non-finite benchmark output")
    trials = [train_epoch(model, optimizer, scaler, loss_fn, x, y, args, device,
                          max_batches=args.benchmark_batches) for _ in range(3)]
    evaluation = evaluate(model, loss_fn, x[:8192], y[:8192], args, device)
    speed = sum(t["images_per_second"] for t in trials) / len(trials)
    eval_speed = 8192 / evaluation["seconds"]
    result = {**config(args, device), "benchmark_batches": args.benchmark_batches,
              "setup_seconds": setup_seconds, "train_images_per_second": speed,
              "eval_images_per_second": eval_speed,
              "estimated_10_epoch_seconds": setup_seconds + 10 * (50000 / speed + 60000 / eval_speed),
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0,
              "fp32_mean_abs_logit_difference": numerical_difference,
              "fp32_prediction_agreement": prediction_agreement,
              "loss_first": trials[0]["loss"], "loss_last": trials[-1]["loss"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "benchmarks.jsonl").open("a") as file:
        file.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def record_result(path, row):
    fields = ("run_id", "run", "group", "batch_size", "epochs", "lr", "train_acc", "test_acc", "seconds", "wandb_url")
    rows = []
    if path.exists():
        with path.open(newline="") as file:
            rows = [r for r in csv.DictReader(file) if r["run_id"] != row["run_id"]]
    rows.append(row)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arguments()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu":
        if args.device == "cpu" and (args.precision != "fp32" or args.layout != "contiguous" or args.compile):
            raise ValueError("CPU supports fp32, contiguous, eager execution")
        args.precision, args.layout, args.compile = "fp32", "contiguous", False
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported on this GPU")
    setup(args.seed, device, args.precision)
    path = args.output_dir / "last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False) if args.resume else None
    current_config = config(args, device)
    if checkpoint:
        if current_config != checkpoint["config"]:
            raise ValueError(f"resume settings differ: {current_config} != {checkpoint['config']}")
        if args.epochs < checkpoint["epoch"]:
            raise ValueError("target epochs are before checkpoint epoch")
    data = load_cifar10(args.data_dir, device, args.layout)
    if args.benchmark:
        return benchmark(args, device, data)
    x_train, y_train, x_test, y_test = data
    model, forward, optimizer, scaler = make_model(args, device)
    loss_fn = nn.CrossEntropyLoss()
    run = None
    if not args.no_wandb:
        import wandb
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        run = wandb.init(entity=args.entity, project=args.project, group=args.group,
                         name=args.run_name, id=checkpoint["wandb_id"] if checkpoint else None,
                         resume="must" if checkpoint else None, tags=[args.group, "cifar10"],
                         config={**current_config, "epochs": args.epochs,
                                 "params": sum(p.numel() for p in model.parameters()),
                                 "torch_version": torch.__version__, "git_commit": git_commit,
                                 "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"})
        run.define_metric("epoch")
        for pattern in ("train/*", "eval/*", "performance/*", "system/*"):
            run.define_metric(pattern, step_metric="epoch")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first_epoch, steps, elapsed = checkpoint["epoch"] + 1, checkpoint["steps"], checkpoint["elapsed"]
        if run and checkpoint.get("pending_metrics") and int(run.summary.get("last_epoch", 0)) < checkpoint["epoch"]:
            run.log(checkpoint["pending_metrics"])
            run.summary["last_epoch"] = checkpoint["epoch"]
        print(f"Resuming at epoch {first_epoch} from {path}", flush=True)
    else:
        first_epoch, steps, elapsed = 1, 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    final_train = final_test = None
    if first_epoch > args.epochs:
        # The previous process may have stopped after saving the final epoch but
        # before uploading final diagnostics or its model artifact.
        final_train = evaluate(forward, loss_fn, x_train, y_train, args, device)
        final_test = evaluate(forward, loss_fn, x_test, y_test, args, device, diagnostics=True)
    for epoch in range(first_epoch, args.epochs + 1):
        sampler = GpuSampler(device.type == "cuda")
        sampler.start()
        train = train_epoch(forward, optimizer, scaler, loss_fn, x_train, y_train, args, device, check_gradients=epoch == first_epoch)
        steps += train["steps"]
        train_eval = evaluate(forward, loss_fn, x_train, y_train, args, device)
        test_eval = evaluate(forward, loss_fn, x_test, y_test, args, device, diagnostics=epoch == args.epochs)
        system_metrics = sampler.finish()
        elapsed += train["seconds"] + train_eval["seconds"] + test_eval["seconds"]
        metrics = {"epoch": epoch, "train/loss": train["loss"], "train/batch_accuracy": train["accuracy"],
                   "train/eval_loss": train_eval["loss"], "train/eval_accuracy": train_eval["accuracy"],
                   "eval/test_loss": test_eval["loss"], "eval/test_accuracy": test_eval["accuracy"],
                   "performance/train_images_per_second": train["images_per_second"],
                   "performance/train_seconds": train["seconds"],
                   "performance/eval_seconds": train_eval["seconds"] + test_eval["seconds"],
                   "performance/elapsed_seconds": elapsed, "performance/optimizer_steps": steps,
                   "system/peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0,
                   **system_metrics}
        state = {"epoch": epoch, "steps": steps, "elapsed": elapsed, "config": current_config,
                 "wandb_id": run.id if run else None, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "pending_metrics": metrics}
        save_started = time.perf_counter()
        save_checkpoint(path, state)
        metrics["performance/checkpoint_seconds"] = time.perf_counter() - save_started
        if run:
            run.log(metrics)
            run.summary["last_epoch"] = epoch
        print(f"epoch {epoch:2d}/{args.epochs} loss={train['loss']:.4f} train={train_eval['accuracy']:.2f}% "
              f"test={test_eval['accuracy']:.2f}% throughput={train['images_per_second']:.0f} img/s", flush=True)
        final_train, final_test = train_eval, test_eval
    if final_test and run:
        import wandb
        matrix = final_test["matrix"].numpy().tolist()
        truth = [i for i, row in enumerate(matrix) for j, count in enumerate(row) for _ in range(count)]
        predictions = [j for row in matrix for j, count in enumerate(row) for _ in range(count)]
        run.log({"diagnostics/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=truth, preds=predictions, class_names=list(CLASSES))})
        for i, name in enumerate(CLASSES):
            run.summary[f"diagnostics/class_accuracy/{name}"] = 100 * matrix[i][i] / sum(matrix[i])
        mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
        images = [wandb.Image(((image * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy(),
                              caption=f"true={CLASSES[truth]}, predicted={CLASSES[pred]}")
                  for image, truth, pred in final_test["mistakes"]]
        run.log({"diagnostics/misclassified": images})
        run.summary["final/test_accuracy"] = final_test["accuracy"]
        run.summary["final/train_accuracy"] = final_train["accuracy"]
        run.summary["final/elapsed_seconds"] = elapsed
        run.summary["final/optimizer_steps"] = steps
        artifact = wandb.Artifact(f"cifar-cnn-{run.id}", type="model")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
    if final_test:
        record_result(args.output_dir.parent / "results.csv", {
            "run_id": run.id if run else args.output_dir.name, "run": args.run_name,
            "group": args.group, "batch_size": args.batch_size, "epochs": args.epochs,
            "lr": args.lr, "train_acc": f"{final_train['accuracy']:.2f}",
            "test_acc": f"{final_test['accuracy']:.2f}", "seconds": f"{elapsed:.1f}",
            "wandb_url": run.url if run else ""})
    if run:
        print(f"W&B: {run.url}", flush=True)
        run.finish()


if __name__ == "__main__":
    main()
