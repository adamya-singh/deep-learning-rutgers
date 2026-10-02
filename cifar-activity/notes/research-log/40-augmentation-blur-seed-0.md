# 40. Do occasional blur improve seed 0? (Oct 1, 2026)

**TL;DR:** Occasional blur reached 66.04% test accuracy, -0.21 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether occasional blur would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 66.25% test accuracy and 66.75% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added a 3×3 Gaussian blur with probability 0.2 and sigma between 0.1 and 1.0 to crop + flip. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 66.71% clean training accuracy and 66.04% test accuracy:

- Blur softens fine detail, like looking through a slightly unfocused lens. That changes the evidence available during training, while evaluation images remain clean.
- The clean train–test gap is 0.67 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -0.21-point paired change and 353.50 seconds of training/evaluation (9.08× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/c3ryftoq) · [saved metrics](../../runs/augmentation/blur/seed-0/summary.json).


<!-- experiment-key: augmentation/blur/seed-0 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-blur-seed-0.png)

Figure exports: [SVG](../../figures/experiments/augmentation-blur-seed-0.svg) · [PDF](../../figures/experiments/augmentation-blur-seed-0.pdf). Orange marks what changed; the diagram shows the trained architecture.
