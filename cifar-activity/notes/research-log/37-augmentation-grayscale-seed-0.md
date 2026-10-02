# 37. Do occasional grayscale improve seed 0? (Oct 1, 2026)

**TL;DR:** Occasional grayscale reached 66.36% test accuracy, +0.11 percentage points versus crop + flip for the same seed.

**Problem.** I wanted to see whether occasional grayscale would help this small CNN recognize new images. The useful comparison is the matching seed's crop + flip recipe, rather than a different architecture or training budget.

*How we know:* That reference reached 66.25% test accuracy and 66.75% clean training accuracy. I did not know whether the changed views would improve recognition or just make fitting harder.

**Experiment.** I added grayscale conversion with probability 0.1 to crop + flip, while keeping three input channels. Batch 512, SGD learning rate 0.01, 80 epochs, and the architecture stayed fixed.

*What we hope to learn:* If clean test accuracy rises, this recipe is useful under these conditions. If training accuracy falls without a test gain, making the task harder did not pay off at this budget.

**Outcome.** The model reached 66.72% clean training accuracy and 66.36% test accuracy:

- Grayscale removes color cues, like practicing with a black-and-white photograph. It may encourage shape recognition, but this experiment only measures the combined recipe.
- The clean train–test gap is 0.36 points. A smaller gap alone is not success, like two lower exam scores becoming closer without either improving.

I recorded the +0.11-point paired change and 160.37 seconds of training/evaluation (4.12× the reference). The test comparisons are exploratory; I did not select checkpoints with them.

**Takeaway:** I carry the measured accuracy and implementation cost forward separately. This seed gives one result; the three-seed comparison is the evidence for whether the recipe repeats.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/h1dwtaj6) · [saved metrics](../../runs/augmentation/grayscale/seed-0/summary.json).


<!-- experiment-key: augmentation/grayscale/seed-0 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/augmentation-grayscale-seed-0.png)

Figure exports: [SVG](../../figures/experiments/augmentation-grayscale-seed-0.svg) · [PDF](../../figures/experiments/augmentation-grayscale-seed-0.pdf). Orange marks what changed; the diagram shows the trained architecture.
