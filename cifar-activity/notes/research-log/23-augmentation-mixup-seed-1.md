# 23. Do MixUp improve seed 1? (Oct 1, 2026)

**TL;DR:** Mixup reached 63.32% test accuracy, -2.13 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether MixUp would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 65.45% test accuracy and 66.41% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added MixUp with alpha 0.2 after crop + flip. Basically, I blended two images and their class targets instead of demanding one hard label. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 64.27% clean training accuracy and 63.32% test accuracy:

- MixUp blends images and labels, like overlapping two transparent pictures. Its training targets are mixtures, so I compare clean training accuracy rather than mixed-target batch accuracy.
- The clean train–test gap is 0.95 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -2.13-point paired change and 47.48 seconds of training/evaluation (1.22× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/k70c68bq) · [saved metrics](../../runs/augmentation/mixup/seed-1/summary.json).


<!-- experiment-key: augmentation/mixup/seed-1 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-mixup-seed-1.png)

Figure exports: [SVG](../../figures/experiments/augmentation-mixup-seed-1.svg) · [PDF](../../figures/experiments/augmentation-mixup-seed-1.pdf). Orange marks what changed; the diagram shows the trained architecture.
