# CIFAR experiment metrics and dashboards

`organize_wandb.py` applies the same explicit panel order to the personal project
workspace, the saved comparison workspace, and the project's shared and personal
run workspace defaults. Automatic panel generation and alphabetical sorting are
disabled. The first two rows always show training, validation, and test accuracy,
then training, validation, and test loss. Final-result scalar panels also show
one-time evaluations that have no epoch curve.

The remaining sections show optimization, performance, optional system telemetry,
diagnostics, and throughput benchmarks. Both historical GPU-memory namespaces are
supported. Optimization batch accuracy and clean training accuracy remain distinct;
MixUp/CutMix mixed-target accuracy remains separately named.

## Metric contract

| Quantity | W&B key | Availability |
| --- | --- | --- |
| Optimization loss | `train/loss` | Every training epoch |
| Training batch accuracy | `train/batch_accuracy` or `train/mixed_target_accuracy` | Every training epoch; mixed-target recipes use the latter |
| Clean training loss / accuracy | `train/eval_loss`, `train/eval_accuracy` | Every training epoch |
| Validation loss / accuracy | `eval/validation_loss`, `eval/validation_accuracy` | Every depth/follow-up epoch; older runs had no validation split |
| Test loss / accuracy | `eval/test_loss`, `eval/test_accuracy` | Every legacy epoch; final selected checkpoints only in depth/follow-up |
| Learning rate | `optimizer/lr` | Every depth/follow-up epoch; historical constant LR reconstructed from saved configuration |
| Optimizer updates | `performance/optimizer_steps` | Every training epoch |
| Throughput | `performance/train_images_per_second` | Every training epoch |
| Train / evaluation / elapsed time | `performance/train_seconds`, `performance/eval_seconds`, `performance/elapsed_seconds` | Every training epoch |
| Peak allocated memory | `system/peak_gpu_memory_mb` or `performance/peak_gpu_memory_mb` | Every training epoch; historical names share a chart |
| Final loss / accuracy | `final/{train,validation,test}_{loss,accuracy}` | Completed runs, for datasets actually evaluated |
| Selection score | `final/validation_score` | Last 10 validation epochs, depth/follow-up |
| Final timing / updates | `final/elapsed_seconds`, `final/optimizer_steps` | Completed training runs |

GPU utilization, sampled memory, power, temperature, checkpoint timing, confusion
matrices, and misclassified images are also charted when originally collected.
Absent measurements remain absent. Benchmark-only runs have no classification
accuracy or loss; smoke runs remain identifiable by their group.

## Reconciliation and monitoring

Run `.venv/bin/python wandb_metric_audit.py --repair --backfill --verify-history`
to verify actual epoch coverage and reconcile completed runs. The report is saved
to `runs/wandb-metric-audit.json`; provenance for recovered curves is stored under
`runs/wandb-metric-backfills/`.

The separate CPU-only watcher uses the same command with `--watch --interval 120`.
It also restores the shared chart order if a saved workspace drifts back to
automatic panels. Layout updates write the manual-mode settings atomically.
It checks completed runs and final assessments throughout both queues, then exits
when both queues complete. Its lock prevents duplicate watchers. It never opens a
second SDK writer on a running run, accesses datasets, evaluates models, or edits
frozen training sources, manifests, checkpoints, optimizer state, or selection
scores. Failures are retained in its persistent log and retried.

Historical learning-rate recovery appends only the missing LR values and their
original epoch/update/time context. Existing loss and accuracy measurements are
not replayed. Interrupted recovery continues from missing epochs. Existing final
results take precedence over older history values.

Final test assessments previously logged only summary values. Once these results
exist and their W&B run is finished, the watcher also logs one test point at the
final epoch, labeled `logging/test_curve_kind = Final checkpoint only; no
historical test curve`. This uses an existing evaluation; it does not perform
another evaluation.

The current depth/follow-up protocol holds out the test set until validation-based
selection finishes. Empty test panels on unselected depth/follow-up runs reflect
that rule, not failed logging. Do not fabricate test curves, relabel validation as
test, or infer historical test values from final checkpoints. Changing this policy
requires a deliberate protocol change; no such change is made by these scripts.
