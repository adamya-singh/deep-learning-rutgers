# Adaptive CIFAR-10 depth and residual experiments

The standalone queue tests 2, 4, 8, 16, and 32 convolutions, then compares
residual counterparts at every visited depth of four or more. It preserves the
32/64-channel stages, pooling locations, and 4096 → 128 → 10 classifier.
Every convolution has batch normalization; residual channel transitions append
zero channels to the shortcut, without learned projections. Plain/residual
pairs have identical parameter counts and initial weights for a given seed.

The fixed recipe uses seeds 0/1/2, a stratified 45,000/5,000 training/validation
split (seed 42), no augmentation, batch 128, evaluation batch 512, and 160
epochs. SGD has momentum 0.9 and weight decay 0.0005. Learning rate warms from
0.01 at epoch 1 to 0.1 at epoch 5, remains 0.1 at epoch 6, and decays by cosine
to 0.0001 at epoch 160. Execution is eager FP32 with deterministic cuDNN
convolutions. The historical classroom and augmentation runs are reference
points, not matched controls for this new recipe and data split.

## Setup and commands

Use the existing virtual environment. Additional plot dependencies are isolated
in `requirements-depth.txt`:

```bash
uv pip install --python .venv/bin/python -r requirements-depth.txt
.venv/bin/python test_depth.py -v
.venv/bin/python -u run_depth_queue.py run
.venv/bin/python run_depth_queue.py status
.venv/bin/python run_depth_queue.py report
```

`run` automatically waits for all 45 jobs in the existing augmentation queue,
verifies their checkpoints/summaries, acquires that queue's lock, and runs all
GPU preflight checks before training. A failed or exited incomplete augmentation
queue stops the waiting runner with an actionable error. Finish/resume that
suite, then repeat the depth `run` command.

`smoke` performs only preflight, using the same GPU gate. Each candidate gets
shape/parameter checks, finite-loss/gradient checks, measured throughput and
peak memory, and a serialized checkpoint continuation comparison. CUDA weight
and momentum comparisons use rtol 0.001 / atol 0.0001; CPU comparisons are exact.
No GPU depth checks run before augmentation completes. `--device cpu` bypasses
the GPU gate for development; full CIFAR training on CPU is slow.

The runner supports `--manifest`, `--root`, `--augmentation-root`, and
`--no-wandb`. Defaults are `depth_experiments.json`, `runs/depth`, and
`runs/augmentation`. Use a new output root when changing configuration or
implementation. Restarting the same command resumes incomplete runs and skips
only verified completions. A global depth-GPU lock plus inherited queue locks
prevent competing depth runners and augmentation restarts; unrelated manual
GPU jobs must use another device or wait.

## Adaptive decisions

The score is mean validation accuracy over epochs 151–160, then averaged
across the three seeds. All seeds finish before a depth decision is made.
Compared with the previous depth, stop plain training when:

- Two consecutive increases gain less than 0.3 percentage points each; or
- Mean accuracy falls by at least 1.0 point and at least two seeds decline; or
- The 32-convolution candidate finishes.

The depth that triggers stopping is included in the residual sweep. All visited
depths ≥4 receive all three residual runs, regardless of earlier shortcut gains.
At most 27 full runs are scheduled. Numerical errors, out-of-memory conditions,
or incompatible checkpoints stop the queue and are not interpreted as plateau.

Selection uses the highest mean validation score, with exact ties resolved by
fewer convolutions, then plain before residual. Only that model's three final
checkpoints are evaluated on the test set. Recorded test results are cached;
restarting does not re-evaluate them. A failed W&B upload can be retried from
the cached result. No best-epoch checkpoint selection or full-data retraining
is performed.

## Outputs and monitoring

`runs/depth/queue.log` is the detached runner log; `runner.pid` records its PID.
`queue-state.json` stores phase, commands, adaptive decisions, failures,
selection, and the remaining runtime estimate. `manifest.json` freezes the
recipe and source hash; `training-manifest.json` is passed to child processes.
`split.json` preserves the exact split indices and their hash.

Runs live under `<variant>/conv-<depth>/seed-<seed>/`, with `last.pt`, `run.log`,
`metrics.jsonl`, and `summary.json`. The epoch checkpoint is authoritative:
after a crash, the metrics file is reconstructed from its history. Checkpoints
include BN buffers, SGD momentum, learning-rate progress, all RNG states, split
identity, and W&B identity. An unfinished epoch is repeated on restart.

`smoke-results.json` contains per-candidate estimates; they initially exclude
process startup, checkpointing, and network logging. Completed sibling runs
replace estimates with observed wall time. Until the plain sweep stops, the
remaining estimate includes all maximum-budget candidates; afterward it includes
only required candidates. There is no runtime estimate derived from the old
two-layer throughput study.

`comparison.md` / `comparison.json` report validation scores, seed spread,
paired shortcut gains, training accuracy, validation loss, parameters, compute
and wall times, stopping decisions, and the selected model's final test results.
Four SVG plots in `plots/` show accuracy versus depth, paired residual gains,
training/validation curves, and accuracy versus compute time.

W&B uses the existing `cifar-activity` project with `plain-depth` and
`residual-depth` groups. Validation metrics have distinct names from historical
test metrics. Optimizer updates, learning rate, throughput, and peak allocated
GPU memory are logged per epoch. Individual training logs contain progress even
while the overall queue log only reports phase/job transitions.
