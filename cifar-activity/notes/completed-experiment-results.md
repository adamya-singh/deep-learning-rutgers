# Completed depth and follow-up results

Both queues completed. All numbers below come from saved completed runs. Validation score is the mean accuracy over the final ten epochs, averaged across seeds 0, 1, and 2. Training metrics and validation loss refer to each run's final epoch. Runtime includes training and clean evaluation, excluding startup and upload overhead.

| Candidate | Mean validation score (%) | Clean train accuracy (%) | Clean train loss | Validation loss | Minutes / seed |
|---|---:|---:|---:|---:|---:|
| plain 2, no augmentation | 78.27 | 100.000 | 0.00566 | 0.812 | 3.1 |
| plain 4, no augmentation | 84.53 | 100.000 | 0.00315 | 0.544 | 4.6 |
| plain 8, no augmentation | 86.90 | 100.000 | 0.00227 | 0.471 | 8.4 |
| plain 16, no augmentation | 87.17 | 100.000 | 0.00148 | 0.488 | 16.6 |
| plain 32, no augmentation | 83.08 | 100.000 | 0.00260 | 0.617 | 33.2 |
| residual 4, no augmentation | 83.77 | 100.000 | 0.00325 | 0.582 | 5.2 |
| residual 8, no augmentation | 86.67 | 100.000 | 0.00181 | 0.483 | 8.9 |
| residual 16, no augmentation | 86.85 | 100.000 | 0.00119 | 0.511 | 17.7 |
| residual 32, no augmentation | 87.20 | 99.999 | 0.00101 | 0.514 | 35.4 |
| winner-flip | 90.31 | 99.993 | 0.00168 | 0.403 | 34.9 |
| winner-crop-flip | 93.74 | 99.987 | 0.00112 | 0.309 | 34.7 |
| resnet18-crop-flip | 95.77 | 100.000 | 0.00096 | 0.167 | 49.4 |

The plain sweep stopped for decline at 32 convolutions: the paired change from 16 to 32 averaged −4.10 percentage points, with all three seeds declining. The previous increase from 8 to 16 averaged +0.27 points.

| Convolutions | Residual minus plain, paired mean (percentage points) |
|---|---:|
| 4 | -0.77 |
| 8 | -0.24 |
| 16 | -0.33 |
| 32 | +4.12 |

Residual 32 won depth selection at 87.20%, only about 0.02 points above plain 16 at 87.17%. Recovering the deep plain model's lost accuracy does not imply a large advantage over the best shallower model.

The fixed depth winner improved to 90.31% with flips and 93.74% with crops plus flips. CIFAR ResNet-18 with crops plus flips won the follow-up selection at 95.77%. Its 11,173,962 parameters and 200 epochs differ from the custom model's 1,239,274 parameters and 160 epochs; this is a combined recipe comparison, not an isolated architecture experiment.

Test images were evaluated only for each queue's validation-selected architecture. These are saved final checkpoint evaluations, not per-epoch test curves. Test results did not select the candidates.

| Selected model | Seed | Test accuracy (%) | Test loss |
|---|---:|---:|---:|
| Depth: residual 32 | 0 | 86.78 | 0.525 |
| Depth: residual 32 | 1 | 85.87 | 0.559 |
| Depth: residual 32 | 2 | 86.63 | 0.547 |
| Depth: residual 32 | Mean | 86.43 | 0.543 |
| Follow-up: ResNet-18 crop + flip | 0 | 95.19 | 0.181 |
| Follow-up: ResNet-18 crop + flip | 1 | 95.21 | 0.185 |
| Follow-up: ResNet-18 crop + flip | 2 | 95.22 | 0.183 |
| Follow-up: ResNet-18 crop + flip | Mean | 95.21 | 0.183 |

See the [per-run log and architecture figures](research-log/README.md). Sources: `runs/depth/queue-state.json`, `runs/followup/queue-state.json`, and each candidate's `summary.json` and selected checkpoint's `test-summary.json`. No additional evaluation was run for this update.
