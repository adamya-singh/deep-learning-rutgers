# CIFAR-10 CNN activity

The baseline keeps the template's two-convolution model, CIFAR-10 preprocessing,
SGD learning rate 0.01, batch size 64, seed 0, and 10 epochs. The script uses the
GPU when available, reports throughput, saves an epoch checkpoint, and logs to
[Weights & Biases](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity).

## Setup

Use Python 3.12 and a compatible CUDA driver. For the tested RTX 3090
environment:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock
```

Log in with `wandb login` if this machine has not been authenticated. The dataset
downloads into `data/` on the first run. Generated data, checkpoints, and logs
are ignored by Git.

## Run and resume

```bash
.venv/bin/python cifar_cnn.py --device cuda --precision fp32 --output-dir runs/baseline
.venv/bin/python cifar_cnn.py --device cuda --precision fp32 --output-dir runs/baseline --resume
```

Use the exact original training settings unless deliberately comparing an
experiment. `--resume` loads `runs/baseline/last.pt` and resumes the same W&B run
at the next epoch. Pass `--epochs N` to extend the target beyond the checkpoint.
Execution settings and hyperparameters must otherwise match. A completed epoch
is saved atomically before its metrics are sent. A crash may repeat an unfinished
epoch. The checkpoint also contains RNG and optimizer state.

For a separate batch-size experiment:

```bash
.venv/bin/python cifar_cnn.py --device cuda --precision fp32 --batch-size 256 \
  --group tuned --run-name 'batch 256' --output-dir runs/tuned-b256
```

To measure a configuration without creating a training run:

```bash
.venv/bin/python cifar_cnn.py --benchmark --device cuda --precision fp32 \
  --layout contiguous --batch-size 64 --output-dir runs/benchmarks
```

Benchmarks append JSON records to `runs/benchmarks/benchmarks.jsonl`. The
estimated runtime includes setup, training, and evaluation; actual full-run
time additionally includes checkpointing, W&B logging, and diagnostics. Compare
the complete runs for the final speed result. The training script also supports
`--device cpu --no-wandb` for local checks.

W&B groups the baseline and tuned runs and charts learning metrics, performance,
system measurements, and final class diagnostics. The script records training
batch accuracy separately from accuracy computed across the entire training set.
The curated [workspace](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity?nw=sno1vf8ff7t)
can be updated with `.venv/bin/python organize_wandb.py`. The measured results,
tradeoff, run links, and exact commands are in [BENCHMARKS.md](BENCHMARKS.md).
