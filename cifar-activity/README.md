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

## Augmentation experiments

The fixed suite in `augmentation_experiments.json` contains 15 recipes × seeds
0, 1, and 2: **45 runs**, each at batch 512, 80 epochs, SGD learning rate 0.01,
eager FP32, and contiguous layout. Crop + flip runs first, followed by fresh
unaugmented controls, crop/flip ablations, and separate additions. The model and
all 50,000 training images remain the same. Report final-epoch test accuracy
for every predefined recipe; these comparisons are exploratory rather than a
validation-based search or checkpoint selection.

`crop` and `crop-flip` add **four black pixels on every side**, then sample a
32×32 crop from the padded 40×40 image. Horizontal flip probability is 0.5.
Recipes built on crop + flip inherit this padding. `resized-crop` replaces the
padded crop with a crop/resize of the original image and retains the flip.
The complete fixed transform settings are in `augmentations.py`.

```bash
# Correctness, labels, RNG/resume, and queue restart tests (CUDA included if available).
.venv/bin/python test_augmentation.py -v
# Short training smoke checks for every recipe, with throughput estimates.
.venv/bin/python run_augmentation_queue.py smoke
# Run sequentially; repeat this command to resume an interrupted or failed queue.
.venv/bin/python -u run_augmentation_queue.py run
# Inspect queue status or regenerate the report from completed runs.
.venv/bin/python run_augmentation_queue.py status
.venv/bin/python run_augmentation_queue.py report
```

Outputs live under `runs/augmentation/`: each `<recipe>/seed-<seed>/` contains
`last.pt`, `run.log`, and a final `summary.json`. `queue-state.json` tracks all
45 jobs; failures stop the queue. Restarting restores incomplete checkpoints
and skips completed runs only when their configuration, checkpoint, and summary
agree. A process lock prevents duplicate runners. Changing a queue's settings
requires a separate `--root`. `--no-wandb` and `--device cpu` support local checks.
`comparison.md` and `comparison.json` update after each completed job, reporting
mean ± sample standard deviation and differences between paired seeds.

Reproducible previews in `runs/augmentation/previews/` show two rows of original
images followed by two rows of augmented images; adjacent JSON files record
labels, mixed targets, and transform settings. Generate them separately with
`run_augmentation_queue.py preview`.

Augmentation happens afresh during training, before normalization (except
erasing and mixing, which follow normalization). The raw dataset, vectorized
crop/flip, and training stay on the GPU. Other per-image torchvision tensor
transforms use CPU kernels with one CPU thread to avoid excessive small CUDA
launches; the backend is recorded as `hybrid-v1`. Normalized clean images are
retained for train/test evaluation. MixUp/CutMix use soft targets, so their
batch metric is `train/mixed_target_accuracy`; clean accuracy remains comparable.
Independent shuffle and augmentation RNG streams are saved in checkpoints.
Historical checkpoints without augmentation metadata remain resumable as `none`.
RNG restoration reproduces the sampled views and ordering. The existing cuDNN
benchmark settings can still cause small numerical differences across CUDA
processes; CPU resume tests compare weights exactly.

For an individual experiment:

```bash
.venv/bin/python cifar_cnn.py --device cuda --batch-size 512 --epochs 80 \
  --lr 0.01 --seed 0 --augmentation crop-flip --group augmentation \
  --run-name 'Crop + flip, seed 0' --output-dir runs/manual-crop-flip
```

Use fresh output directories; existing checkpoints require `--resume`.
The queue's smoke runtime estimate excludes checkpointing, W&B, process startup,
and final diagnostics. Completed run times provide the actual comparison.

W&B groups the baseline and tuned runs and charts learning metrics, performance,
system measurements, and final class diagnostics. The script records training
batch accuracy separately from accuracy computed across the entire training set.
The curated [workspace](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity?nw=sno1vf8ff7t)
can be updated with `.venv/bin/python organize_wandb.py`. This updates both the
default personal workspace and the saved view. Six unsmoothed learning charts
are pinned at the top: train/test accuracy and loss by epoch, accuracy and test
loss by optimizer updates, test accuracy by elapsed time, and optimization loss.
Performance follows; system measurements and diagnostics are collapsed below.
All runs are included, including smoke runs, and the benchmark table has its own
section. New runs using the same metric names appear in the existing charts.
The measured results,
tradeoff, run links, and exact commands are in [BENCHMARKS.md](BENCHMARKS.md).

## Depth and residual experiments

The standalone adaptive queue in `run_depth_queue.py` waits for the augmentation
suite, tests increasing depth with a fixed normalized training recipe, then
trains matched residual counterparts. It uses validation-only stopping and
selection, three seeds, inherited GPU locks, and epoch checkpoint resume.

```bash
.venv/bin/python test_depth.py -v
.venv/bin/python -u run_depth_queue.py run
.venv/bin/python run_depth_queue.py status
```

See [DEPTH_EXPERIMENTS.md](DEPTH_EXPERIMENTS.md) for setup, exact training and
stopping rules, restart behavior, and report locations.

## W&B metric consistency

See [WANDB_METRICS.md](WANDB_METRICS.md) for the shared chart order, metric
availability, historical repairs, and the independent metric watcher. Apply
layouts with `.venv/bin/python organize_wandb.py`; audit every recorded epoch with
`.venv/bin/python wandb_metric_audit.py --repair --backfill --verify-history`.
