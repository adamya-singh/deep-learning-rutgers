# 43. Do CIFAR-10 AutoAugment improve seed 0? (Oct 1, 2026)

**TL;DR:** Cifar-10 autoaugment reached 64.06% test accuracy, -2.19 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether CIFAR-10 AutoAugment would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 66.25% test accuracy and 66.75% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added the torchvision CIFAR-10 AutoAugment policy to crop + flip, using bilinear interpolation and black fill. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 64.74% clean training accuracy and 64.06% test accuracy:

- AutoAugment samples a predefined transform policy, like choosing a practice drill from a prepared list. This run tests that policy with this small model and training budget.
- The clean train–test gap is 0.68 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -2.19-point paired change and 1093.18 seconds of training/evaluation (28.08× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/0o9ghixj) · [saved metrics](../../runs/augmentation/autoaugment/seed-0/summary.json).


<!-- experiment-key: augmentation/autoaugment/seed-0 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-autoaugment-seed-0.png)

Figure exports: [SVG](../../figures/experiments/augmentation-autoaugment-seed-0.svg) · [PDF](../../figures/experiments/augmentation-autoaugment-seed-0.pdf). Orange marks what changed; the diagram shows the trained architecture.
