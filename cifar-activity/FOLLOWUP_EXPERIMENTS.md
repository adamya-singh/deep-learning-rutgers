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
