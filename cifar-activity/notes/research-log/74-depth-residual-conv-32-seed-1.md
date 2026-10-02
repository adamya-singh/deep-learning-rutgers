# 74. Do thirty-two-layer shortcuts recover accuracy on seed 1? (Oct 1, 2026)

<!-- experiment-key: depth/residual/conv-32/seed-1 -->

**TL;DR:** I gained 3.42 validation percentage points with thirty-two residual convolutions on seed 1, reaching 86.57%.

**Problem.** Plain accuracy fell sharply at thirty-two layers. I want to check whether shortcuts recover that loss across random starts.

*How we know:* the matched plain seed-1 control scored 83.15% over its last ten epochs. I needed this completed pair to judge whether the shortcut benefit repeated.

**Experiment.** I trained thirty-two residual convolutions for 160 epochs, using eight two-convolution blocks per stage with shortcuts that add each block's input before the final ReLU, which clips negative values to zero. I kept initial weights, shuffle sequences, seed 1, widths, pooling, classifier, recipe, and the 45,000/5,000 split fixed, without augmentation.

*What we hope to learn:* a gain would support shortcuts at the depth where plain performance declined. A flat or lower score would show that the benefit did not repeat for this seed.

**Outcome.** The final-ten-epoch validation score was 86.57%; final-epoch validation accuracy was 86.60%:

- Shortcuts give inputs a direct route around two convolutions, like a side road alongside two processing stops. Zero padding adds no learned shortcut weights.
- Clean training accuracy was 99.99778% versus 100%; validation loss fell from 0.609 to 0.542. Like doing better on fresh questions after nearly acing practice, validation improved; these endpoints do not identify why.

Training plus clean evaluation took 35.4 minutes versus 33.2 for the control. The completed three-seed residual group gains 4.12 points over plain at this depth. Its 87.20% mean narrowly beats plain sixteen at 87.17%, winning validation selection. After validation selection, this checkpoint scored 85.87% test accuracy with 0.559 test loss.

**Takeaway:** Shortcuts recovered the deepest plain model's lost validation accuracy. That recovery is much larger than their tiny advantage over the best shallower plain architecture.

Source: [completed run](https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity/runs/47ogod0p), `runs/depth/residual/conv-32/seed-1/summary.json`; control: `runs/depth/plain/conv-32/seed-1/summary.json`. Final test: `runs/depth/residual/conv-32/seed-1/test-summary.json`.


<!-- experiment-architecture -->
![Architecture and experimental change](../../figures/experiments/depth-residual-conv-32-seed-1.png)

Figure exports: [SVG](../../figures/experiments/depth-residual-conv-32-seed-1.svg) · [PDF](../../figures/experiments/depth-residual-conv-32-seed-1.pdf). Orange marks what changed; the diagram shows the trained architecture.
