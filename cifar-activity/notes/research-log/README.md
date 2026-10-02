# CIFAR-10 experiment log

One skill-written entry per completed training run. Each entry includes a bottom-to-top architecture diagram with orange callouts identifying its experimental change. Entries follow summarize-experiment-as-adamya and its voice corpus.

Augmentation and historical runs report final test accuracy. Depth and follow-up runs report validation-only scores over the last ten epochs; these scores are not interchangeable. Smoke/benchmark runs are excluded.

| Entry | Experiment | Metric | Score |
| --- | --- | --- | ---: |
| [01](01-crop-flip-seed-0.md) | augmentation/crop-flip/seed-0 | test accuracy | 66.25% |
| [02](02-crop-flip-seed-1.md) | augmentation/crop-flip/seed-1 | test accuracy | 65.45% |
| [03](03-crop-flip-seed-2.md) | augmentation/crop-flip/seed-2 | test accuracy | 67.45% |
| [04](04-none-seed-0.md) | augmentation/none/seed-0 | test accuracy | 67.53% |
| [05](05-none-seed-1.md) | augmentation/none/seed-1 | test accuracy | 67.25% |
| [06](06-none-seed-2.md) | augmentation/none/seed-2 | test accuracy | 69.10% |
| [07](07-flip-seed-0.md) | augmentation/flip/seed-0 | test accuracy | 67.66% |
| [08](08-flip-seed-1.md) | augmentation/flip/seed-1 | test accuracy | 68.62% |
| [09](09-flip-seed-2.md) | augmentation/flip/seed-2 | test accuracy | 70.04% |
| [10](10-crop-seed-0.md) | augmentation/crop/seed-0 | test accuracy | 66.56% |
| [11](11-crop-seed-1.md) | augmentation/crop/seed-1 | test accuracy | 66.28% |
| [12](12-crop-seed-2.md) | augmentation/crop/seed-2 | test accuracy | 67.32% |
| [13](13-color-jitter-seed-0.md) | augmentation/color-jitter/seed-0 | test accuracy | 66.39% |
| [14](14-color-jitter-seed-1.md) | augmentation/color-jitter/seed-1 | test accuracy | 65.41% |
| [15](15-augmentation-color-jitter-seed-2.md) | augmentation/color-jitter/seed-2 | test accuracy | 67.52% |
| [16](16-augmentation-rotation-seed-0.md) | augmentation/rotation/seed-0 | test accuracy | 62.42% |
| [17](17-augmentation-rotation-seed-1.md) | augmentation/rotation/seed-1 | test accuracy | 60.04% |
| [18](18-augmentation-rotation-seed-2.md) | augmentation/rotation/seed-2 | test accuracy | 62.60% |
| [19](19-augmentation-erasing-seed-0.md) | augmentation/erasing/seed-0 | test accuracy | 65.93% |
| [20](20-augmentation-erasing-seed-1.md) | augmentation/erasing/seed-1 | test accuracy | 64.22% |
| [21](21-augmentation-erasing-seed-2.md) | augmentation/erasing/seed-2 | test accuracy | 65.86% |
| [22](22-augmentation-mixup-seed-0.md) | augmentation/mixup/seed-0 | test accuracy | 64.82% |
| [23](23-augmentation-mixup-seed-1.md) | augmentation/mixup/seed-1 | test accuracy | 63.32% |
| [24](24-augmentation-mixup-seed-2.md) | augmentation/mixup/seed-2 | test accuracy | 65.24% |
| [25](25-augmentation-cutmix-seed-0.md) | augmentation/cutmix/seed-0 | test accuracy | 60.46% |
| [26](26-augmentation-cutmix-seed-1.md) | augmentation/cutmix/seed-1 | test accuracy | 58.29% |
| [27](27-augmentation-cutmix-seed-2.md) | augmentation/cutmix/seed-2 | test accuracy | 58.08% |
| [28](28-augmentation-randaugment-seed-0.md) | augmentation/randaugment/seed-0 | test accuracy | 64.49% |
| [29](29-augmentation-randaugment-seed-1.md) | augmentation/randaugment/seed-1 | test accuracy | 62.83% |
| [30](30-augmentation-randaugment-seed-2.md) | augmentation/randaugment/seed-2 | test accuracy | 64.72% |
| [31](31-augmentation-resized-crop-seed-0.md) | augmentation/resized-crop/seed-0 | test accuracy | 66.70% |
| [32](32-augmentation-resized-crop-seed-1.md) | augmentation/resized-crop/seed-1 | test accuracy | 65.75% |
| [33](33-augmentation-resized-crop-seed-2.md) | augmentation/resized-crop/seed-2 | test accuracy | 66.69% |
| [34](34-augmentation-affine-seed-0.md) | augmentation/affine/seed-0 | test accuracy | 59.94% |
| [35](35-augmentation-affine-seed-1.md) | augmentation/affine/seed-1 | test accuracy | 54.81% |
| [36](36-augmentation-affine-seed-2.md) | augmentation/affine/seed-2 | test accuracy | 60.22% |
| [37](37-augmentation-grayscale-seed-0.md) | augmentation/grayscale/seed-0 | test accuracy | 66.36% |
| [38](38-augmentation-grayscale-seed-1.md) | augmentation/grayscale/seed-1 | test accuracy | 65.60% |
| [39](39-augmentation-grayscale-seed-2.md) | augmentation/grayscale/seed-2 | test accuracy | 67.20% |
| [40](40-augmentation-blur-seed-0.md) | augmentation/blur/seed-0 | test accuracy | 66.04% |
| [41](41-augmentation-blur-seed-1.md) | augmentation/blur/seed-1 | test accuracy | 63.45% |
| [42](42-augmentation-blur-seed-2.md) | augmentation/blur/seed-2 | test accuracy | 66.37% |
| [43](43-augmentation-autoaugment-seed-0.md) | augmentation/autoaugment/seed-0 | test accuracy | 64.06% |
| [44](44-augmentation-autoaugment-seed-1.md) | augmentation/autoaugment/seed-1 | test accuracy | 62.83% |
| [45](45-augmentation-autoaugment-seed-2.md) | augmentation/autoaugment/seed-2 | test accuracy | 64.48% |
| [46](46-historical-baseline.md) | historical/baseline | test accuracy | 63.68% |
| [47](47-historical-tuned-b512.md) | historical/tuned-b512 | test accuracy | 48.58% |
| [48](48-historical-step-matched-b512.md) | historical/step-matched-b512 | test accuracy | 67.67% |
| [49](49-depth-plain-conv-2-seed-0.md) | depth/plain/conv-2/seed-0 | validation score (last ten epochs) | 77.81% |
| [50](50-depth-plain-conv-2-seed-1.md) | depth/plain/conv-2/seed-1 | validation score (last ten epochs) | 78.38% |
| [51](51-depth-plain-conv-2-seed-2.md) | depth/plain/conv-2/seed-2 | validation score (last ten epochs) | 78.61% |
| [52](52-depth-plain-conv-4-seed-0.md) | depth/plain/conv-4/seed-0 | validation score (last ten epochs) | 84.20% |
| [53](53-depth-plain-conv-4-seed-1.md) | depth/plain/conv-4/seed-1 | validation score (last ten epochs) | 84.75% |
| [54](54-depth-plain-conv-4-seed-2.md) | depth/plain/conv-4/seed-2 | validation score (last ten epochs) | 84.65% |
| [55](55-depth-plain-conv-8-seed-0.md) | depth/plain/conv-8/seed-0 | validation score (last ten epochs) | 86.79% |
| [56](56-depth-plain-conv-8-seed-1.md) | depth/plain/conv-8/seed-1 | validation score (last ten epochs) | 86.53% |
| [57](57-depth-plain-conv-8-seed-2.md) | depth/plain/conv-8/seed-2 | validation score (last ten epochs) | 87.39% |
| [58](58-depth-plain-conv-16-seed-0.md) | depth/plain/conv-16/seed-0 | validation score (last ten epochs) | 87.13% |
| [59](59-depth-plain-conv-16-seed-1.md) | depth/plain/conv-16/seed-1 | validation score (last ten epochs) | 86.93% |
| [60](60-depth-plain-conv-16-seed-2.md) | depth/plain/conv-16/seed-2 | validation score (last ten epochs) | 87.46% |
| [61](61-depth-plain-conv-32-seed-0.md) | depth/plain/conv-32/seed-0 | validation score (last ten epochs) | 83.65% |
| [62](62-depth-plain-conv-32-seed-1.md) | depth/plain/conv-32/seed-1 | validation score (last ten epochs) | 83.15% |
| [63](63-depth-plain-conv-32-seed-2.md) | depth/plain/conv-32/seed-2 | validation score (last ten epochs) | 82.43% |
| [64](64-depth-residual-conv-4-seed-0.md) | depth/residual/conv-4/seed-0 | validation score (last ten epochs) | 84.06% |
| [65](65-depth-residual-conv-4-seed-1.md) | depth/residual/conv-4/seed-1 | validation score (last ten epochs) | 83.04% |
| [66](66-depth-residual-conv-4-seed-2.md) | depth/residual/conv-4/seed-2 | validation score (last ten epochs) | 84.20% |
| [67](67-depth-residual-conv-8-seed-0.md) | depth/residual/conv-8/seed-0 | validation score (last ten epochs) | 86.45% |
| [68](68-depth-residual-conv-8-seed-1.md) | depth/residual/conv-8/seed-1 | validation score (last ten epochs) | 87.02% |
| [69](69-depth-residual-conv-8-seed-2.md) | depth/residual/conv-8/seed-2 | validation score (last ten epochs) | 86.53% |
| [70](70-depth-residual-conv-16-seed-0.md) | depth/residual/conv-16/seed-0 | validation score (last ten epochs) | 86.40% |
| [71](71-depth-residual-conv-16-seed-1.md) | depth/residual/conv-16/seed-1 | validation score (last ten epochs) | 86.85% |
| [72](72-depth-residual-conv-16-seed-2.md) | depth/residual/conv-16/seed-2 | validation score (last ten epochs) | 87.29% |
| [73](73-depth-residual-conv-32-seed-0.md) | depth/residual/conv-32/seed-0 | validation score (last ten epochs) | 87.69% |
| [74](74-depth-residual-conv-32-seed-1.md) | depth/residual/conv-32/seed-1 | validation score (last ten epochs) | 86.57% |
| [75](75-depth-residual-conv-32-seed-2.md) | depth/residual/conv-32/seed-2 | validation score (last ten epochs) | 87.33% |
| [76](76-followup-winner-flip-seed-0.md) | followup/winner-flip/seed-0 | validation score (last ten epochs) | 90.33% |
| [77](77-followup-winner-flip-seed-1.md) | followup/winner-flip/seed-1 | validation score (last ten epochs) | 90.28% |
| [78](78-followup-winner-flip-seed-2.md) | followup/winner-flip/seed-2 | validation score (last ten epochs) | 90.33% |
| [79](79-followup-winner-crop-flip-seed-0.md) | followup/winner-crop-flip/seed-0 | validation score (last ten epochs) | 93.67% |
| [80](80-followup-winner-crop-flip-seed-1.md) | followup/winner-crop-flip/seed-1 | validation score (last ten epochs) | 93.51% |
| [81](81-followup-winner-crop-flip-seed-2.md) | followup/winner-crop-flip/seed-2 | validation score (last ten epochs) | 94.04% |
| [82](82-followup-resnet18-crop-flip-seed-0.md) | followup/resnet18-crop-flip/seed-0 | validation score (last ten epochs) | 95.86% |
| [83](83-followup-resnet18-crop-flip-seed-1.md) | followup/resnet18-crop-flip/seed-1 | validation score (last ten epochs) | 95.64% |
| [84](84-followup-resnet18-crop-flip-seed-2.md) | followup/resnet18-crop-flip/seed-2 | validation score (last ten epochs) | 95.80% |

Completed training runs discovered: 84. Entries with registered diagrams: 84.

New completed runs are checked by the chat's experiment-log heartbeat. Each is drafted with the skill, then its figure is generated and the index refreshed. Pending work is listed by `experiment_log.py pending`.
