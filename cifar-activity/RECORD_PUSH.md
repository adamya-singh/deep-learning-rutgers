# CIFAR-10 accuracy push

Updated 2026-10-02 05:42 AM Eastern. Work window ends October 2, 2026 at 9am Eastern.

These runs use outside pretraining. Frozen-feature experiments keep the backbone fixed; fine-tuning experiments adapt its last four blocks and classifier. Classifier features, regularization, ensemble weights, and fine-tuning duration are selected on the existing 45,000/5,000 training/validation split. The selected recipe is then fitted on all 50,000 training images and assessed on the official 10,000-image test set. Test labels never fit a classifier or choose ensemble weights.

Best verified test accuracy: **99.72%**, with **28 errors out of 10,000**. The first target of 99% has been reached; work continues to maximize the result within the time window.

| Saved experiment | Validation (%) | Test (%) | Test errors | Verification |
|---|---:|---:|---:|---|
| [dinov2-dinov3-7b-multiresolution](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/ensembles/dinov2-dinov3-7b-multiresolution/test-summary.json) | 99.78 | 99.72 | 28 | CPU reconstruction |
| [dinov2-dinov3-projected7b](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/ensembles/dinov2-dinov3-projected7b/test-summary.json) | 99.78 | 99.71 | 29 | CPU reconstruction |
| [dinov2-dinov3-siglip2](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/ensembles/dinov2-dinov3-siglip2/test-summary.json) | 99.80 | 99.70 | 30 | CPU reconstruction |
| [dinov3-7b-gaussian](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/kernels/dinov3-7b-gaussian/test-summary.json) | 99.78 | 99.69 | 31 | CPU reconstruction |
| [vit_7b_patch16_dinov3.lvd1689m-224-original](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/vit_7b_patch16_dinov3.lvd1689m-224-original/test-summary.json) | 99.78 | 99.69 | 31 | CPU reconstruction |
| [dinov2-dinov3-224](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/ensembles/dinov2-dinov3-224/test-summary.json) | 99.72 | 99.68 | 32 | CPU reconstruction |
| [dinov2-dinov3-multiresolution](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/ensembles/dinov2-dinov3-multiresolution/test-summary.json) | 99.74 | 99.67 | 33 | CPU reconstruction |
| [vit_7b_patch16_dinov3.lvd1689m-224-projected](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/vit_7b_patch16_dinov3.lvd1689m-224-projected/test-summary.json) | 99.78 | 99.64 | 36 | CPU reconstruction |
| [dinov2-dinov3-views-multiresolution](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/ensembles/dinov2-dinov3-views-multiresolution/test-summary.json) | 99.76 | 99.61 | 39 | CPU reconstruction |
| [vit_huge_plus_patch16_dinov3.lvd1689m-224-average](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/vit_huge_plus_patch16_dinov3.lvd1689m-224-average/test-summary.json) | 99.72 | 99.61 | 39 | CPU reconstruction |
| [dinov2_vitg14-224-original](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/dinov2_vitg14-224-original/test-summary.json) | 99.70 | 99.60 | 40 | CPU reconstruction |
| [vit_huge_plus_patch16_dinov3.lvd1689m-224-original](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/vit_huge_plus_patch16_dinov3.lvd1689m-224-original/test-summary.json) | 99.68 | 99.57 | 43 | CPU reconstruction |
| [dinov2_vitg14-224-average](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/dinov2_vitg14-224-average/test-summary.json) | 99.70 | 99.55 | 45 | CPU reconstruction |
| [vit_huge_plus_patch16_dinov3.lvd1689m-384-original](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/vit_huge_plus_patch16_dinov3.lvd1689m-384-original/test-summary.json) | 99.64 | 99.55 | 45 | CPU reconstruction |
| [dinov2-average-gaussian](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/kernels/dinov2-average-gaussian/test-summary.json) | 99.70 | 99.55 | 45 | CPU reconstruction |
| [vit_giantopt_patch16_siglip_384.v2_webli-384-original](/home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity/runs/record-push/vit_giantopt_patch16_siglip_384.v2_webli-384-original/test-summary.json) | 99.18 | 99.10 | 90 | CPU reconstruction |

Previous best from-scratch result: 95.22% test accuracy with CIFAR ResNet-18. The pretrained results above belong to a different training track.

Published comparison: EfficientNet-L2 + SAM reports 99.70% CIFAR-10 accuracy (0.30% error, five-run mean with a 0.01 percentage-point 95% confidence interval) using outside pretraining, in [SAM Table 3](https://arxiv.org/html/2010.01412v3). DINOv3 7B reports 99.6% with linear probing in [DINOv3 Table 22](https://arxiv.org/html/2508.10104v1). These references do not establish an exhaustive current world record.

Each experiment directory preserves cached features, the validation-selected head, the refitted final head, test logits, confusion matrix, and checkpoint fingerprints. Independent verification recomputes frozen-feature predictions on CPU and checks official labels, counts, and fingerprints. Fine-tuned models are reloaded before test inference and assessed again with a different batch size; this does not constitute an independent CPU reconstruction of the backbone.

Recompute an individual result:

```bash
cd /home/win10ubuntu/dev/rutgers/fall-26/deep-learning-rutgers/cifar-activity
.venv/bin/python record_verify.py dinov2_vitg14-224-original
```

The run manifest and live queue log are under `runs/record-push`. Scores shown here are completed assessments, with one deterministic head fit per selected recipe; no multi-seed uncertainty estimate has been measured for these pretrained runs.
