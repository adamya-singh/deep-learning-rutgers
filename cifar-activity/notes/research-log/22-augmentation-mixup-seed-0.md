# 22. Do MixUp improve seed 0? (Oct 1, 2026)

**TL;DR:** Mixup reached 64.82% test accuracy, -1.43 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether MixUp would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 66.25% test accuracy and 66.75% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added MixUp with alpha 0.2 after crop + flip. Basically, I blended two images and their class targets instead of demanding one hard label. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 65.03% clean training accuracy and 64.82% test accuracy:

- MixUp blends images and labels, like overlapping two transparent pictures. Its training targets are mixtures, so I compare clean training accuracy rather than mixed-target batch accuracy.
- The clean train–test gap is 0.21 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -1.43-point paired change and 46.75 seconds of training/evaluation (1.20× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/i2lv1fzz) · [saved metrics](../../runs/augmentation/mixup/seed-0/summary.json).


<!-- experiment-key: augmentation/mixup/seed-0 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-mixup-seed-0.png)

Figure exports: [SVG](../../figures/experiments/augmentation-mixup-seed-0.svg) · [PDF](../../figures/experiments/augmentation-mixup-seed-0.pdf). Orange marks what changed; the diagram shows the trained architecture.
