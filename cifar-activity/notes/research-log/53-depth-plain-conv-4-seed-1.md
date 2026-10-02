# 53. Do 4 convolutions help seed 1? (Oct 1, 2026)

**TL;DR:** Increasing depth from 2 to 4 convolutions changed validation score by +6.37 points, reaching 84.75%.

**Problem.** I wanted to test whether increasing depth from 2 to 4 convolutions improves the features learned from these small images. A larger or easier-to-optimize model still has to help on held-out examples.

*How we know:* The matched reference reached 78.38% validation score in seed 1. I kept the same split and training recipe so the architecture change is the comparison.

**Experiment.** I used 2 convolutions in each of the 32- and 64-channel stages. Pooling locations and the 4096 → 128 → 10 classifier stayed fixed. Every run finishes 160 epochs before the queue makes a depth decision.

*What we hope to learn:* If validation improves, the extension helps under this recipe. If only training improves, extra fitting is not enough. Worse training would point me toward an optimization problem to investigate.

**Outcome.** Validation score reached 84.75% and final clean training accuracy reached 100.00%:

- Extra convolutions add processing before pooling shrinks the image, like combining small visual clues before summarizing them. This tests more depth at fixed channel widths.
- Final validation accuracy was 84.70% and validation loss was 0.5317. Training can improve independently, like getting better at familiar exercises without matching progress on new ones.

I retain the +6.37-point paired change. The queue waits for all three seeds before judging a plateau or decline; this entry covers one completed seed.

**Takeaway:** I judge the architecture by matched validation results, not parameter count or training accuracy alone. The shared recipe makes the direction of the comparison interpretable.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/njggmh89) · [saved metrics](../../runs/depth/plain/conv-4/seed-1/summary.json).


<!-- experiment-key: depth/plain/conv-4/seed-1 -->

<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/depth-plain-conv-4-seed-1.png)

Figure exports: [SVG](../../figures/experiments/depth-plain-conv-4-seed-1.svg) · [PDF](../../figures/experiments/depth-plain-conv-4-seed-1.pdf). Orange marks what changed; the diagram shows the trained architecture.
