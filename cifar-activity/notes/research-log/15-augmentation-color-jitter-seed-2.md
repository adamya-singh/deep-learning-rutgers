# 15. Do color jitter improve seed 2? (Oct 1, 2026)

**TL;DR:** Color jitter reached 67.52% test accuracy, +0.07 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether color jitter would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 67.45% test accuracy and 68.21% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added brightness, contrast, and saturation changes of ±0.2 and hue changes of ±0.05 to crop + flip. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 67.94% clean training accuracy and 67.52% test accuracy:

- Color jitter changes appearance while retaining the label, like practicing under different lighting. Accuracy alone cannot tell me which color change helped or hurt.
- The clean train–test gap is 0.42 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the +0.07-point paired change and 3159.01 seconds of training/evaluation (82.16× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/pve9cbb3) · [saved metrics](../../runs/augmentation/color-jitter/seed-2/summary.json).


<!-- experiment-key: augmentation/color-jitter/seed-2 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-color-jitter-seed-2.png)

Figure exports: [SVG](../../figures/experiments/augmentation-color-jitter-seed-2.svg) · [PDF](../../figures/experiments/augmentation-color-jitter-seed-2.pdf). Orange marks what changed; the diagram shows the trained architecture.
