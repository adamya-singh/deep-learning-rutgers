# 16. Do small rotations improve seed 0? (Oct 1, 2026)

**TL;DR:** Small rotations reached 62.42% test accuracy, -3.83 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether small rotations would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 66.25% test accuracy and 66.75% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added random rotations between −15° and +15° to crop + flip, using bilinear interpolation and black fill. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 63.00% clean training accuracy and 62.42% test accuracy:

- Rotation changes orientation and introduces interpolation and empty corners, like tilting a photograph. The measured effect belongs to that entire transform.
- The clean train–test gap is 0.58 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -3.83-point paired change and 1136.16 seconds of training/evaluation (29.18× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/vs3bctgg) · [saved metrics](../../runs/augmentation/rotation/seed-0/summary.json).


<!-- experiment-key: augmentation/rotation/seed-0 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-rotation-seed-0.png)

Figure exports: [SVG](../../figures/experiments/augmentation-rotation-seed-0.svg) · [PDF](../../figures/experiments/augmentation-rotation-seed-0.pdf). Orange marks what changed; the diagram shows the trained architecture.
