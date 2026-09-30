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
