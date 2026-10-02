# 20. Do random erasing improve seed 1? (Oct 1, 2026)

**TL;DR:** Random erasing reached 64.22% test accuracy, -1.23 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether random erasing would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 65.45% test accuracy and 66.41% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** After crop + flip and normalization, I erased a random region with probability 0.2. Its area was 2–20% of the image. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 65.03% clean training accuracy and 64.22% test accuracy:

- Erasing hides part of the object, like putting a sticky note over a photo. It makes the training task harder without changing the clean evaluation images.
- The clean train–test gap is 0.81 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the -1.23-point paired change and 357.66 seconds of training/evaluation (9.20× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/wpwpol5a) · [saved metrics](../../runs/augmentation/erasing/seed-1/summary.json).


<!-- experiment-key: augmentation/erasing/seed-1 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-erasing-seed-1.png)

Figure exports: [SVG](../../figures/experiments/augmentation-erasing-seed-1.svg) · [PDF](../../figures/experiments/augmentation-erasing-seed-1.pdf). Orange marks what changed; the diagram shows the trained architecture.
