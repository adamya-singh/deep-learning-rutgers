# 84. Does the ResNet-18 recipe improve seed 2? (Oct 1, 2026)

<!-- experiment-key: followup/resnet18-crop-flip/seed-2 -->

**TL;DR:** I reached 95.80% validation with CIFAR ResNet-18 on seed 2, 1.76 points above the crop-and-flip depth winner.

**Problem.** Crops and flips helped our custom architecture substantially. I want to see whether a standard, wider residual network can improve the result further within the queued training budget.

*How we know:* the same-seed crop-and-flip depth model scored 94.04%. The no-augmentation depth control scored 87.33%; it is a reference, not an architecture-only comparison.

**Experiment.** I trained CIFAR-adapted ResNet-18 from scratch for 200 epochs with crops and flips, using a 3×3 input convolution and no initial max-pooling. Seed 2, the 45,000/5,000 split, batch size 128, and SGD recipe stayed fixed, but architecture, parameter count, and training duration changed together.

*What we hope to learn:* a gain would justify this combined model-and-training recipe. No gain would suggest the larger network and extra epochs did not earn their cost here.

**Outcome.** The final-ten-epoch validation score was 95.80%; final-epoch validation accuracy was 95.82%:

- Four stages widen from 64 to 512 channels, with learned shortcuts at transitions, like adding wider workbenches and fitted connecting paths. Global average pooling summarizes each feature map before classification, replacing our dense hidden head.
- Clean training accuracy reached 100%; final validation loss was 0.160 versus 0.295. Like acing practice and making fewer costly mistakes on fresh questions, both final validation measures improved.

Training plus clean evaluation took 49.5 minutes versus 34.6 for the augmented depth model. This model has 11,173,962 parameters versus 1,239,274. The group wins selection at 95.77% mean validation; afterward this checkpoint scored 95.22% test accuracy with 0.183 test loss.

**Takeaway:** The complete ResNet-18 recipe wins this queue. I cannot attribute the improvement solely to architecture because capacity and training duration changed too.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/bj0140h3), `runs/followup/resnet18-crop-flip/seed-2/summary.json`; control: `runs/depth/residual/conv-32/seed-2/summary.json`. Augmentation-matched comparator: `runs/followup/winner-crop-flip/seed-2/summary.json`. Final test: `runs/followup/resnet18-crop-flip/seed-2/test-summary.json`.


<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/followup-resnet18-crop-flip-seed-2.png)

Figure exports: [SVG](../../figures/experiments/followup-resnet18-crop-flip-seed-2.svg) · [PDF](../../figures/experiments/followup-resnet18-crop-flip-seed-2.pdf). Orange marks what changed; the diagram shows the trained architecture.
