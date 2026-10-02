# 18. Do small rotations improve seed 2? (Oct 1, 2026)

**TL;DR:** Small rotations reached 62.60% test accuracy, -4.85 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether small rotations would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 67.45% test accuracy and 68.21% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added random rotations between −15° and +15° to crop + flip, using bilinear interpolation and black fill. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 63.06% clean training accuracy and 62.60% test accuracy:

- Rotation changes orientation and introduces interpolation and empty corners, like tilting a photograph. The measured effect belongs to that entire transform.
- The clean train–test gap is 0.46 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -4.85-point paired change and 1135.67 seconds of training/evaluation (29.54× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/rw6jsm1k) · [saved metrics](../../runs/augmentation/rotation/seed-2/summary.json).


<!-- experiment-key: augmentation/rotation/seed-2 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-rotation-seed-2.png)

Figure exports: [SVG](../../figures/experiments/augmentation-rotation-seed-2.svg) · [PDF](../../figures/experiments/augmentation-rotation-seed-2.pdf). Orange marks what changed; the diagram shows the trained architecture.
