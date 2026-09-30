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
the lowest measured 10-epoch estimate. Larger batches will be benchmarked after
the baseline. Short benchmark estimates are directional; completed run times
will include checkpointing and W&B overhead.
