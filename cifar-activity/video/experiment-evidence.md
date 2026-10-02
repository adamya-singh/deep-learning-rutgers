## SOURCE: BENCHMARKS.md

# RTX 3090 CIFAR-10 throughput study

Environment: RTX 3090 (24 GB), PyTorch 2.7.1+cu126, torchvision 0.22.1,
Python 3.12, seed 0. The original two-convolution CNN and preprocessing are
unchanged. Each candidate warmed up, then ran three 96-batch training trials;
evaluation used 8,192 images. Times exclude data download and W&B upload.

| Batch | Precision | Layout | Compile | Train images/s | Estimated 10 epochs |
| ---: | --- | --- | :---: | ---: | ---: |
| 64 | FP32 | contiguous | no | 28,357 | 22.8 s |
| 64 | TF32 | contiguous | no | 25,045 | 25.9 s |
| 64 | BF16 | contiguous | no | 19,409 | 29.4 s |
| 64 | BF16 | channels-last | no | 17,830 | 31.8 s |
| 64 | FP16 | channels-last | no | 17,472 | 32.1 s |
| 64 | FP32 | channels-last | no | 25,880 | 32.8 s |
| 64 | FP32 | contiguous | yes | 20,637 | 28.9 s |

The compiled estimate is from a warm compiler cache. A cold compilation took
longer and is less favorable for this short run. Its evaluation compilation was
initially timed inside the evaluation measurement; the corrected trial warmed
evaluation first. Numerically, the largest observed mean logit difference from
FP32 was 0.00031 and the sampled predictions agreed 100%.

For the first baseline, use eager FP32 and contiguous tensors at batch 64. This
preserves the template's precision and optimizer settings as well as producing
the lowest measured 10-epoch estimate. Short benchmark estimates are
directional; completed run times include checkpointing and W&B overhead.

## Batch-size study and completed runs

With eager FP32 and contiguous tensors fixed, three-trial benchmarks measured:

| Batch | Training images/s | Estimated 10 epochs | Peak allocated GPU memory |
| ---: | ---: | ---: | ---: |
| 64 | 22,576 | 24.3 s | 1,168 MB |
| 128 | 48,504 | 12.1 s | 1,168 MB |
| 256 | 96,157 | 7.1 s | 1,168 MB |
| 512 | 155,451 | 5.1 s | 1,168 MB |
| 1024 | 133,123 | 5.9 s | 1,473 MB |

At batch 512, compiled execution reached 194,068 training images/s, but its
7.3-second setup cost gave an estimated 10.8-second run. Eager execution won
for ten epochs. The batch-1024 trial was slower than batch 512.

| Run | Test accuracy | Train/eval time | W&B runtime | Optimizer updates | Mean GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Template baseline, batch 64](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/812pre34) | 63.68% | 22.39 s | 24 s | 7,820 | 41.6% |
| [Throughput tuned, batch 512](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/oqsh0d4a) | 48.58% | 5.21 s | 7 s | 980 | 89.4% |

The tuned run was 4.3 times faster in measured training and evaluation, and
3.4 times faster by W&B runtime. Its accuracy was 15.10 percentage points
lower because batch 512 performs one-eighth as many SGD updates at the same
learning rate and epoch count. This is a throughput comparison, not an
accuracy-matched model improvement. The [W&B workspace](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity?nw=sno1vf8ff7t)
contains the curves, system measurements, diagnostics, and
[benchmark table](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/rjjf0tny).

Reproduce from this directory:

```bash
.venv/bin/python cifar_cnn.py --device cuda --precision fp32 --batch-size 64 \
  --lr 0.01 --seed 0 --epochs 10 --group baseline \
  --run-name 'Template baseline: batch 64' --output-dir runs/baseline
.venv/bin/python cifar_cnn.py --device cuda --precision fp32 --batch-size 512 \
  --lr 0.01 --seed 0 --epochs 10 --group tuned \
  --run-name 'Throughput tuned: batch 512' --output-dir runs/tuned-b512
```

## Fairer comparison: train batch 512 longer

The initial 10-epoch batch-512 run made only 980 optimizer updates, versus
7,820 for the template baseline. To compare at nearly the same update count,
we trained [batch 512 for 80 epochs](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/5j92g6cx)
with the same seed and learning rate. Eighty epochs yield 7,840 updates, 20
more than the baseline. Each update uses eight times as many images, so this
comparison also processes eight times as many total image passes.

| Comparison point | Test accuracy | Train/eval time | Optimizer updates |
| --- | ---: | ---: | ---: |
| Baseline, batch 64 at epoch 10 | 63.68% | 22.39 s | 7,820 |
| Batch 512 at epoch 42, first crossing baseline accuracy | 64.41% | 19.91 s | 4,116 |
| Batch 512 at epoch 48, stays above baseline accuracy afterward | 64.67% | 22.58 s | 4,704 |
| Batch 512 at epoch 80, nearly matched updates | 67.67% | 36.85 s | 7,840 |

The first threshold crossing is 2.48 seconds faster than baseline, but the
test curve briefly dips below the threshold afterward. By the more stable
epoch-48 point, time to baseline-level accuracy is essentially tied. At nearly
matched updates, batch 512 gains 3.99 accuracy points but takes 1.65 times as
long. These are descriptive comparisons of one seeded run per setting; the
test set was not used to select a new learning rate or model.

```bash
.venv/bin/python cifar_cnn.py --device cuda --precision fp32 --batch-size 512 \
  --lr 0.01 --seed 0 --epochs 80 --group tuned \
  --run-name 'Batch 512: matched optimizer updates' --output-dir runs/step-matched-b512
```


## SOURCE: DEPTH_EXPERIMENTS.md

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


## SOURCE: FOLLOWUP_EXPERIMENTS.md

# Follow-up: winner augmentation and CIFAR ResNet-18

This separate queue waits for `runs/depth` to complete its plain/residual sweep
and final assessment, then independently verifies its validation-selected
winner and completed checkpoints. It does not change the code or manifest used
by the active depth queue.

The comparison has four candidates, with seeds 0, 1, and 2:

| Candidate | Epochs | Training runs |
| --- | ---: | ---: |
| Depth winner, no augmentation | 160 | Reuse three verified depth runs |
| Depth winner, horizontal flip | 160 | Three new runs |
| Depth winner, crop + flip | 160 | Three new runs |
| CIFAR-adapted ResNet-18, crop + flip | 200 | Three new runs |

Winner augmentation runs start from the same seed-specific initial weights and
shuffle streams as the controls, rather than fine-tuning the trained winner.
They keep its SGD momentum, weight decay, warmup/cosine schedule, batch sizes,
normalization, and 45,000/5,000 split unchanged. Only augmentation changes.
Fresh views are sampled per batch, before normalization, using the existing
GPU-vectorized augmentation implementation. Flip probability is 0.5; crop uses
four black padding pixels on every side and samples a 32×32 window.

ResNet-18 starts from random weights (`weights=None`). Its input convolution is
3×3 with stride 1 and padding 1, and its initial max pool is removed. The four
residual stages use 64/128/256/512 channels, followed by global average pooling
and a 512 → 10 classifier. The optimizer is the same; cosine decay ends at
epoch 200. This is a stronger recipe comparison: architecture and epoch budget
both differ from the winner augmentation ablation.

## Run and monitor

The existing environment and `requirements-depth.txt` provide dependencies.

```bash
.venv/bin/python test_followup.py -v
.venv/bin/python -u run_followup_queue.py run
.venv/bin/python run_followup_queue.py status
.venv/bin/python run_followup_queue.py report
tail -f runs/followup/queue.log
```

`run` and `smoke` wait for the source queue to finish and release the shared GPU
lock. The follow-up then holds that lock plus the source and augmentation queue
locks. Training runs sequentially; there is no automatic batch/precision change
on OOM. If the upstream queue fails, this runner stops with an actionable error.
Resolve/resume the source queue, then repeat the follow-up command.

Optional arguments are `--manifest`, `--source-root`, `--root`,
`--augmentation-root`, `--device`, and `--no-wandb`. CPU mode is for development
fixtures and requires a CPU source queue. Changed recipe/source implementation
requires a new root. Restarting the same command resumes interrupted epochs and
skips completed runs only after checking checkpoint/configuration/score agreement.

GPU preflight checks all three new recipes, including serialized continuation
of weights, BN buffers, SGD momentum, shuffle state, and augmentation RNG. It
measures peak memory and per-recipe throughput before full training. Initial
runtime estimates exclude checkpoint/startup/upload overhead; completed sibling
wall times refine the remaining estimate. Preflight never runs while the depth
queue owns the GPU.

## Results and selection

Outputs are under `runs/followup`. `runner.pid` and `queue.log` identify the
detached runner. `manifest.json` freezes the request and source identity;
`specs.json` is resolved after the source winner is known. `split.json` copies
the original exact split. `queue-state.json` records phase, jobs, failures,
preflight estimates, selection, and final assessment.

Each `<candidate>/seed-<seed>/` contains an atomic `last.pt`, `run.log`,
`metrics.jsonl`, and `summary.json`. Checkpoints preserve optimizer, BN, all RNG
streams including augmentation, learning-rate progress, split/source identity,
and W&B identity. The checkpoint's epoch history repairs metrics after a crash.

Selection uses mean validation accuracy over each run's last ten epochs, then
averages across all three seeds. Exact ties prefer the reused control, then
flip, crop + flip, and ResNet-18, in that order. All nine new runs finish before
selection. Test scores do not influence scheduling or selection. Evaluate only
the selected candidate's three final checkpoints; if the reused control wins,
reuse its previously recorded test assessments. Test results are cached before
network upload so restarts avoid repeat evaluation.

`comparison.md` / `comparison.json` include means and seed spread, paired gains
against the unaugmented control, epoch budgets, training accuracy, parameter
counts, and selected test results. Two SVG plots show validation comparisons
and clean training/validation learning curves.

W&B uses the existing `cifar-activity` project with `winner-augmentation` and
`resnet18-cifar` groups. Validation names remain distinct from historical test
metrics. W&B uploads include recipe/source configuration, learning rate,
optimizer updates, throughput, elapsed time, and peak allocated GPU memory.


## SOURCE: research-log.md

![Starting network architecture: CIFAR-10 CNN](figures/cifar_cnn_architecture.png)

# CIFAR-10 research log

## Starting point — September 30, 2026

Our starting network is a small CNN with **545,098 trainable parameters**.
Each RGB image is scaled to 0–1 and normalized using CIFAR-10's channel
statistics before entering the network.

The first convolution block maps 3 input channels to 32 feature maps using a
3×3 convolution, ReLU, and 2×2 max pooling. The second block repeats those
operations with 64 feature maps. We flatten its 64×8×8 output into 4,096
features, then use a 128-unit hidden layer with ReLU and a final linear layer
that produces 10 class logits.

The template baseline uses SGD with learning rate **0.01**, batch size **64**,
seed **0**, and **10 epochs**. Cross-entropy loss consumes the logits directly.
This architecture is our reference for future experiments.

Figure exports: [SVG](figures/cifar_cnn_architecture.svg) ·
[PDF](figures/cifar_cnn_architecture.pdf).
The figure's visual style follows Figure 1 in
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762).

Existing training and throughput measurements are recorded in
[BENCHMARKS.md](BENCHMARKS.md).

## Experiment 1 — data augmentation

Question: does changing the training views improve final clean test accuracy
with the existing two-convolution CNN and batch-512, 80-epoch recipe?
The historical unaugmented seed-0 reference reached 67.67% test accuracy;
this suite includes fresh paired controls rather than assuming that single run
represents the recipe's mean.

The fixed queue contains 15 recipes and three seeds (0, 1, 2), for 45 runs.
Start with 32×32 random crops after four pixels of black padding on all sides
and horizontal flips with probability 0.5. Include an unaugmented control and
separate crop/flip ablations. Add color jitter, rotation, erasing, MixUp, CutMix,
RandAugment, affine transforms, grayscale, blur, and CIFAR-10 AutoAugment
separately to crop + flip. Also test a resized-crop replacement with flip.

Keep SGD learning rate 0.01, batch 512, 80 epochs, eager FP32, the architecture,
normalization, and all 50,000 training examples fixed. Paired seeds share model
initialization and training order. Evaluate clean images; report epoch 80 for
every predefined recipe without selecting checkpoints or tuning strengths on
the test set. Results are exploratory comparisons using the existing test set.

Design: [experiment manifest](augmentation_experiments.json). The resumable
runner writes previews, per-run checkpoints and summaries, and an aggregate
report to `runs/augmentation/`. The report includes mean and sample standard
deviation, paired accuracy differences, clean train–test gaps, throughput, and
runtime. Final results will be recorded after the queue completes.

### Interim results — October 1, 2026

The first 14 completed runs have [individual research-log entries](notes/research-log/README.md).
Flip-only improved all three paired seeds, averaging 68.77% test accuracy
versus 67.96% without augmentation. Crop-only averaged 66.72%, and crop + flip
averaged 66.38%. Their smaller train–test gaps came with lower test accuracy.
The first two color-jitter additions changed crop + flip accuracy by +0.14 and
−0.04 points, while taking roughly 48 and 52 minutes of training/evaluation.
The third color-jitter seed was unfinished at this snapshot.

### Log maintenance — October 1, 2026

The [experiment index](notes/research-log/README.md) now covers all 45 completed
augmentation runs, the three historical full-training comparisons, and all
completed depth runs. Each entry includes a Transformer-paper-style architecture
diagram with orange callouts showing the experimental change, plus SVG/PDF
exports. The 14-run section above is an archived snapshot, not the live status.

A five-minute heartbeat checks for newly completed runs and writes each entry
with `summarize-experiment-as-adamya` and its voice corpus. The training queues
remain unchanged. Depth and follow-up entries use validation scores; historical
and augmentation entries use test accuracy. The current index links each note
and keeps those metrics separate.


## SOURCE: runs/augmentation/comparison.md

# CIFAR-10 augmentation comparison

Fixed recipes; batch 512, 80 epochs, SGD lr 0.01. All 50,000 training images; clean test evaluation at the final epoch. Comparisons are exploratory.

Completed summaries: 45/45. Spread is sample standard deviation across seeds.

| Recipe | Seeds | Test accuracy, % | Δ vs none, pp | Δ vs crop+flip, pp | Clean train−test gap, pp | Train images/s | Train/eval time, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| crop-flip | 3 | 66.38 ± 1.01 | -1.58 ± 0.27 | 0.00 ± 0.00 | 0.74 | 138988 | 38.76 ± 0.27 |
| none | 3 | 67.96 ± 1.00 | 0.00 ± 0.00 | 1.58 ± 0.27 | 7.33 | 154771 | 35.77 ± 0.38 |
| flip | 3 | 68.77 ± 1.20 | 0.81 ± 0.63 | 2.39 ± 0.90 | 3.56 | 141699 | 38.22 ± 0.60 |
| crop | 3 | 66.72 ± 0.54 | -1.24 ± 0.47 | 0.34 ± 0.48 | 1.26 | 137658 | 38.94 ± 0.44 |
| color-jitter | 3 | 66.44 ± 1.06 | -1.52 ± 0.35 | 0.06 ± 0.09 | 0.41 | 1326 | 3049.96 ± 151.28 |
| rotation | 3 | 61.69 ± 1.43 | -6.27 ± 1.07 | -4.70 ± 0.80 | 0.65 | 3576 | 1146.59 ± 18.48 |
| erasing | 3 | 65.34 ± 0.97 | -2.62 ± 0.89 | -1.05 ± 0.65 | 0.72 | 11569 | 366.72 ± 9.16 |
| mixup | 3 | 64.46 ± 1.01 | -3.50 ± 0.69 | -1.92 ± 0.43 | 0.51 | 111123 | 46.62 ± 0.92 |
| cutmix | 3 | 58.94 ± 1.32 | -9.02 ± 1.98 | -7.44 ± 1.81 | 0.11 | 106535 | 48.10 ± 0.50 |
| randaugment | 3 | 64.01 ± 1.03 | -3.95 ± 0.79 | -2.37 ± 0.53 | 0.86 | 2375 | 1711.99 ± 13.45 |
| resized-crop | 3 | 66.38 ± 0.55 | -1.58 ± 0.79 | -0.00 ± 0.66 | 2.70 | 5977 | 694.79 ± 15.09 |
| affine | 3 | 58.32 ± 3.05 | -9.64 ± 2.51 | -8.06 ± 2.28 | 0.71 | 3007 | 1356.62 ± 7.65 |
| grayscale | 3 | 66.39 ± 0.80 | -1.57 ± 0.37 | 0.00 ± 0.22 | 0.73 | 27399 | 160.71 ± 2.44 |
| blur | 3 | 65.29 ± 1.60 | -2.67 ± 1.16 | -1.10 ± 0.90 | 0.81 | 12323 | 345.06 ± 7.93 |
| autoaugment | 3 | 63.79 ± 0.86 | -4.17 ± 0.61 | -2.59 ± 0.39 | 0.70 | 3718 | 1103.18 ± 21.52 |

Historical reference: the previous unaugmented seed-0 run reached 67.67% test accuracy. Fresh paired controls determine differences in this suite; no checkpoint or augmentation strength was selected using test results.


## SOURCE: runs/depth/comparison.md

# Depth and residual comparison

Selection uses validation only; score is mean accuracy over the final 10 epochs. Spread is sample standard deviation across seeds. Partial groups are marked.

Plain sweep stopping reason: decline; visited depths: [2, 4, 8, 16, 32].

| Variant | Conv layers | Seeds | Validation score, % | Final train, % | Final val loss | Params | Compute seconds | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain | 2 | 3/3 | 78.27 ± 0.41 | 100.00 | 0.8119 | 545194 | 186.7 | 199.5 |
| plain | 4 | 3/3 | 84.53 ± 0.29 | 100.00 | 0.5437 | 591466 | 276.9 | 290.3 |
| plain | 8 | 3/3 | 86.90 ± 0.44 | 100.00 | 0.4712 | 684010 | 502.3 | 516.4 |
| plain | 16 | 3/3 | 87.17 ± 0.26 | 100.00 | 0.4877 | 869098 | 996.6 | 1012.4 |
| plain | 32 | 3/3 | 83.08 ± 0.61 | 100.00 | 0.6175 | 1239274 | 1990.0 | 2026.2 |
| residual | 4 | 3/3 | 83.77 ± 0.63 | 100.00 | 0.5819 | 591466 | 314.8 | 328.6 |
| residual | 8 | 3/3 | 86.67 ± 0.30 | 100.00 | 0.4826 | 684010 | 534.9 | 550.0 |
| residual | 16 | 3/3 | 86.85 ± 0.45 | 100.00 | 0.5114 | 869098 | 1059.7 | 1075.6 |
| residual | 32 | 3/3 | 87.20 ± 0.57 | 100.00 | 0.5140 | 1239274 | 2123.8 | 2143.2 |

Selected by validation: {'variant': 'residual', 'depth': 32, 'score': 87.19799728393555, 'seeds': [0, 1, 2]}.

Selected model final test accuracy (3/3 seeds): 86.43 ± 0.49%.

## Plots

![Validation accuracy by depth](plots/accuracy-vs-depth.svg)
![Paired residual gains](plots/residual-gains.svg)
![Learning curves](plots/learning-curves.svg)
![Accuracy versus runtime](plots/accuracy-vs-runtime.svg)

## Plain depth decisions

```json
{
  "visited": [
    2,
    4,
    8,
    16,
    32
  ],
  "comparisons": [
    {
      "from": 2,
      "to": 4,
      "paired_differences_pp": [
        6.39600067138673,
        6.3659980773925895,
        6.037998962402355
      ],
      "mean_gain_pp": 6.266665903727225,
      "plateau_streak": 0
    },
    {
      "from": 4,
      "to": 8,
      "paired_differences_pp": [
        2.5819953918456946,
        1.7839988708496008,
        2.7399978637695312
      ],
      "mean_gain_pp": 2.368664042154942,
      "plateau_streak": 0
    },
    {
      "from": 8,
      "to": 16,
      "paired_differences_pp": [
        0.3440002441406307,
        0.404000854492196,
        0.06600036621092897
      ],
      "mean_gain_pp": 0.27133382161458525,
      "plateau_streak": 1
    },
    {
      "from": 16,
      "to": 32,
      "paired_differences_pp": [
        -3.4799972534179773,
        -3.781996917724612,
        -5.027999114990237
      ],
      "mean_gain_pp": -4.096664428710942,
      "plateau_streak": 2
    }
  ],
  "stop": "decline",
  "next_depth": null
}
```


## SOURCE: runs/followup/comparison.md

# Winner augmentation and CIFAR ResNet-18 comparison

Same 45,000/5,000 split and paired seeds. Winner augmentation uses the upstream recipe unchanged; ResNet-18 uses 200 epochs by default. Scores average the final ten validation epochs. These are recipe comparisons, not an isolated architecture comparison.

Upstream architecture: residual, 32 convolutions.

| Candidate | Seeds | Epochs | Validation score, % | Paired gain vs none, pp | Final train, % | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| winner-none | 3/3 | 160 | 87.20 ± 0.57 | 0.00 ± 0.00 | 100.00 | 1239274 |
| winner-flip | 3/3 | 160 | 90.31 ± 0.03 | 3.12 ± 0.54 | 99.99 | 1239274 |
| winner-crop-flip | 3/3 | 160 | 93.74 ± 0.27 | 6.54 ± 0.50 | 99.99 | 1239274 |
| resnet18-crop-flip | 1/3 | 200 | 95.86 (one seed) | 8.17 (one seed) | 100.00 | 11173962 |

The winner-none control reuses verified completed depth runs; no duplicate training is scheduled.

![Validation comparison](plots/validation-comparison.svg)
![Learning curves](plots/learning-curves.svg)
