# 34. Do affine transforms improve seed 0? (Oct 1, 2026)

**TL;DR:** Affine transforms reached 59.94% test accuracy, -6.31 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether affine transforms would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 66.25% test accuracy and 66.75% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added rotations up to 10°, translations up to 10%, scale 0.9–1.1, and shear between −5° and +5° to crop + flip. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 60.74% clean training accuracy and 59.94% test accuracy:

- Affine transforms shift, stretch, and shear the image, like changing the angle of a projected slide. This combined treatment cannot tell me which individual operation mattered.
- The clean train–test gap is 0.80 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -6.31-point paired change and 1354.87 seconds of training/evaluation (34.80× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/dirq4fqq) · [saved metrics](../../runs/augmentation/affine/seed-0/summary.json).


<!-- experiment-key: augmentation/affine/seed-0 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-affine-seed-0.png)

Figure exports: [SVG](../../figures/experiments/augmentation-affine-seed-0.svg) · [PDF](../../figures/experiments/augmentation-affine-seed-0.pdf). Orange marks what changed; the diagram shows the trained architecture.
