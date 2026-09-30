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
