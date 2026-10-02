# 41. Do occasional blur improve seed 1? (Oct 1, 2026)

**TL;DR:** Occasional blur reached 63.45% test accuracy, -2.00 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether occasional blur would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 65.45% test accuracy and 66.41% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added a 3×3 Gaussian blur with probability 0.2 and sigma between 0.1 and 1.0 to crop + flip. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 64.32% clean training accuracy and 63.45% test accuracy:

- Blur softens fine detail, like looking through a slightly unfocused lens. That changes the evidence available during training, while evaluation images remain clean.
- The clean train–test gap is 0.87 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -2.00-point paired change and 343.92 seconds of training/evaluation (8.84× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/2mgd83i9) · [saved metrics](../../runs/augmentation/blur/seed-1/summary.json).


<!-- experiment-key: augmentation/blur/seed-1 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-blur-seed-1.png)

Figure exports: [SVG](../../figures/experiments/augmentation-blur-seed-1.svg) · [PDF](../../figures/experiments/augmentation-blur-seed-1.pdf). Orange marks what changed; the diagram shows the trained architecture.
