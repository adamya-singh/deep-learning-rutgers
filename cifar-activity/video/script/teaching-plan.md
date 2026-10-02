# Teaching plan: the CIFAR-10 film

Content plan for the narration rewrite. This is what the film has to teach and
what the evidence allows us to say. It is not narration. Every number below was
checked against `public/data/results.json`, `public/data/static.json` and the
frozen copies in `data/frozen/` (snapshot 2026-10-02T01:27:30Z).

## Audience and goal

The viewer is a student who has heard of neural networks but has never trained
one. By the end they should be able to explain:

1. What the input is (a 32×32×3 grid of numbers) and what the output is (ten
   scores, one per class).
2. How the starting CNN turns one into the other: normalization, convolution
   with shared weights, ReLU, max pooling, flatten, dense layers, logits.
3. How it learns: softmax probabilities, cross-entropy loss, backprop, gradient
   descent steps, batches and epochs.
4. Why each later change was tried, what happened, and what the numbers do and
   do not support.
5. The difference between test and validation numbers, between one seed and
   three seeds, and between a comparison and a causal claim.

The film should feel like one person walking a friend through their own
experiments in order, explaining what they expected and what actually happened.
No biography of anyone appears in it.

## Global facts and distinctions to keep straight

- **Dataset:** CIFAR-10, 60,000 32×32 RGB images, ten classes, 50,000 official
  training images and 10,000 official test images. 32 × 32 × 3 = 3,072 numbers.
- **Test vs validation.** Classroom, batch and first augmentation runs report
  TEST accuracy. From the stricter recipe onward, decisions use VALIDATION (5,000
  images carved out of the 50,000 training set). The test set is used only for the
  selected model of each later queue, but the first augmentation suite had
  already looked at test many times, so it is not a pristine holdout overall.
- **Seeds.** Classroom and batch runs are one seed (seed 0). Augmentation round
  one, depth, residual and follow-up runs are three seeds (0, 1, 2); ± is the
  sample SD across seeds.
- **Final-10 score.** In the later queues a run's score is the mean validation
  accuracy over its last ten epochs, then averaged over the three seeds. Never the
  best single epoch.
- **Measured vs illustrative.** Real: images, pixel values, conv1 filters,
  activations, logits/probabilities for the frog, the multiply-sum (0.6256, matches
  the model), the 25×25 loss slice, all curves and scores. Illustrative or
  schematic: descent path on the loss slice, ReLU/pool toy numbers, the backprop
  network drawing, receptive-field cone, residual vector picture, gradient-route
  animation, the "confident and wrong p = 0.02" example point.
- **Strongest here, not best anywhere.** The CIFAR ResNet-18 is the strongest
  model measured in this project. Wider and deeper residual networks and other
  recipes report higher CIFAR-10 accuracy (e.g. arXiv:1605.07146). No state of the
  art claim.

## Chapter plan

Segment IDs and counts stay as they are (the scenes index segments by
position). For each chapter: what the viewer must understand, the causal story,
what the data shows and cannot show, and visual constraints.

### 1. task (4 segments): the input

- Must understand: an image is just numbers to the network. 60k images, 10
  classes, 50k/10k official split. One pixel = R, G, B values 0–255; whole image =
  3,072 numbers. The question the film answers: what changes made the network
  better at mapping those numbers to the right label?
- Visual cues (existing): frog test image #4 centered; gallery of 8 per class;
  "60,000" and the 50,000/10,000 split bar are keyed to those spoken tokens; zoom
  to a real 4×4 pixel block; R/G/B slabs split on the first spoken "32"; "3,072"
  appears on its spoken token. Spoken text must contain `60,000`, `50,000`,
  `10,000` (segment 2), `32` and `3,072` (segment 3). "CIFAR" must be spoken as
  "SIFAR" (keep the `say` override).

### 2. cnn (8 segments): the starting network

- Must understand:
  - 545,098 trainable parameters; most (524,416, 96%) are in the 4,096→128 dense
    layer. Shown on screen already.
  - Normalization: divide by 255, subtract each channel's dataset mean, divide by
    its standard deviation, so all three channels sit on a similar scale around
    zero. Why: the weights' updates behave more evenly when inputs aren't all large
    positive numbers. (Stated as the usual reason, not something we measured.)
  - Convolution: one filter = 3×3 window over 3 channels = 27 weights + 1 bias. At
    one position, multiply 27 pairs and add (real example at row 14, col 15, filter
    13: sum + bias = 0.6256, which matches the model). Slide it to all 1,024
    positions (padding keeps 32×32) to get one output map.
  - Weight sharing: the same 28 numbers are used everywhere, so a pattern learned
    in one spot is detected anywhere. That's why conv1 has only 896 params.
  - 32 filters → 32 maps. The filters shown are learned from the data (real
    checkpoint, epoch 10), not designed by hand.
  - ReLU sets negatives to 0; 2×2 max pool keeps the largest of each block, 32→16.
  - Block 2: 64 filters, each 3×3×32 (288 weights + bias), so they combine the
    first block's maps. Pool to 8×8. "Combining simple patterns into parts" is the
    usual intuition; keep it framed as intuition.
  - Flatten 64×8×8 = 4,096 → 128 hidden (ReLU) → 10 scores.
- Visual cues (existing): `545,098` (seg 0), `Slide` triggers the sweep (seg 2),
  `Nobody` triggers the "not hand-designed" caption (seg 4), `max` (seg 5),
  `Pooling` (seg 6). These words may change; scene cues will be updated to match.

### 3. learn (9 segments): how it learns

- Must understand:
  - The ten outputs are logits: unbounded, any sign. For the frog: frog logit
    5.98 is the largest, cat 4.34 second.
  - Softmax: exponentiate, divide by the sum → positive, sum to 1. Frog gets
    73.6%, cat 14.3%. So the model is mostly right but not certain.
  - Cross-entropy = −log(probability on the true class). Frog: −log 0.736 = 0.31.
    A confident wrong answer (example p = 0.02) costs 3.91. The loss punishes
    confident mistakes hard.
  - PyTorch detail: the model outputs logits; `nn.CrossEntropyLoss` applies
    log-softmax internally. Softmax in the film is for reading the outputs, not a
    layer in the model.
  - Backprop: the chain rule gives, for every parameter, how the loss would change
    if that parameter moved a little (the gradient).
  - Gradient descent: move each weight a small step (learning rate 0.01) against
    its gradient. The surface shown is a real 2-D slice of the loss around the
    trained baseline (25×25 grid, two random filter-normalized directions, 1,000
    training images, height = log loss). The dashed path is illustrative, not
    recorded steps. The real space has 545,098 dimensions.
  - A batch of 64 images per step; 50,000 / 64 → 782 steps per epoch (last batch
    has 16).
  - 10 epochs at lr 0.01, batch 64, seed 0 → 63.68% TEST (train 67.76%). Chance is
    10%. Wrong on about 36% of test images.
- Visual cues: `Exponentiate` (seg 1), `Confident` twice in seg 3 (first = frog
  point, second = wrong example), `PyTorch's` and `Softmax` (seg 4), `path`
  (seg 6, illustrative tag), `percent` (seg 8, after the 63.68).

### 4. batch (4 segments): bigger batches

- Why tried: GPU utilization at batch 64 was low (41.6%); batch 512 should run
  faster. It did: 10 epochs in 5.21 s vs 22.39 s.
- What happened: test accuracy dropped to 48.58%. Cause: with the epoch count and
  learning rate fixed, batch 512 makes one-eighth as many weight updates (98 per
  epoch, 980 total vs 7,820). Each update is an average over more images, so it
  is less noisy, but at the same learning rate the model simply took far fewer
  steps.
- Follow-up: train batch 512 for 80 epochs → 7,840 updates (20 more than the
  baseline). Reached 67.67% TEST, 3.99 points above the baseline. It processed 8×
  as many image passes (4.0M vs 0.5M) and took 36.85 s vs 22.39 s.
- Can say: step count, not epoch count, explained most of the 10-epoch gap here.
  Bigger batches traded more work per step for fewer steps. Can't say: that 512 is
  better in general; these are single seeded runs, and the learning rate was not
  retuned for the larger batch.
- Visual cues: `Turns` (seg 1) shows "8× fewer steps" (word will change),
  `reached` (seg 2) labels the 67.67 point.

### 5. aug1 (5 segments): augmentation on the small network

- What augmentation is: train on randomly changed copies each time an image is
  used. Flip left-right with probability 0.5; pad 4 black pixels on each side and
  crop a random 32×32 window (the image shifts by up to 4 pixels).
- Why tried: the label doesn't change under these edits, so the model should learn
  to ignore them (invariance), and it never sees exactly the same image twice, so
  it is harder to memorize individual images (a regularizer). At 80 epochs the
  unaugmented model had a 7.33-point train–test gap, which suggests some
  memorization.
- Setup: the small CNN on the batch-512, 80-epoch, lr 0.01 recipe; 15 recipes ×
  3 seeds = 45 runs; TEST accuracy at epoch 80.
- Results: none 67.96 ± 1.00; flip 68.77 ± 1.20 (improved every paired seed,
  +0.81 mean); crop + flip 66.38 ± 1.01 (−1.58). Every recipe that added more on
  top of crop + flip scored lower than none (rotation 61.69, CutMix 58.94, affine
  58.32 at the bottom).
- Gap: none 7.33 points, crop + flip 0.74. The gap shrank, but test accuracy also
  fell, so the model fit training data less well without generalizing better. A
  smaller gap can mean useful regularization or just a model that learned less;
  here it was the second kind of outcome. Plausible reason (not tested): a small
  network on a short recipe doesn't have the capacity/training budget to fit the
  harder, more varied training set.
- Caveat: this suite looked at the test set repeatedly, so it's exploratory.
- Visual cues: `Flip`, `pad`, `crop` (seg 0); `invariance`, `regularization`
  (seg 1); `No`, `Flip`, `Crop` capitalised as sentence starts for bar
  highlights (seg 2); `useful`, `alone` (seg 3).

### 6. recipe (4 segments): stricter evaluation and training

- Why: repeated test use means choices could fit the test set by accident. So hold
  out validation for every decision.
- Split: 5,000 validation images, 500 per class (stratified, split seed 42),
  45,000 for training. Test reserved for the final check of the selected model.
  Not pristine (used in round one).
- New training recipe (also what made deeper networks trainable here): batch 128
  (352 steps/epoch), SGD with momentum 0.9, weight decay 0.0005, learning rate
  warms up from 0.01 to 0.1 over 5 epochs, then cosine decays to 0.0001 at epoch
  160, batch norm after every convolution, no augmentation.
  - Momentum: keeps a running direction so steps don't zig-zag.
  - Weight decay: pulls weights slightly toward zero every step.
  - Warmup + cosine: start gently, take big steps when it's safe, finish with tiny
    steps to settle.
  - Batch norm: normalizes each layer's outputs over the batch, which makes deeper
    stacks easier to train.
  - These are standard reasons, not measured individually here.
- Scoring: three seeds; mean of last ten validation epochs; ± = seed SD.
- Same 2-conv design + BN: 78.27 ± 0.41 VALIDATION vs 63.68 TEST before. Can't say
  how much of the ~15 points is due to which knob, or compare the two numbers
  strictly, because metric, split, epochs, optimizer and BN all changed.
- Visual cues: `stratified`, `reserved`, `pristine`, `decision` (seg 0);
  `warmup`, `cosine` (seg 1); `last` (seg 2); `scores`, `split` (seg 3).

### 7. depth (5 segments): more convolutions

- Why: more 3×3 layers let each unit see a wider part of the image (3×3 → 5×5 →
  7×7 after two, three layers) and compute more complex functions. Keep the same
  two stages (32 then 64 channels), pooling and 4096→128→10 head.
- How the sweep ran: 2, 4, 8, 16, 32 convolutions; three seeds each; continue
  until a stopping rule fires (two small gains in a row, or a clear decline).
- Results (VALIDATION): 78.27, 84.53, 86.90, 87.17, then 83.08 at 32. Paired gains
  +6.27, +2.37, +0.27, then −4.10 (all three seeds declined).
- Every depth reached 100.0% final training accuracy, so the 32-layer network did
  fit the training set; it generalized worse. We can't tell from these runs whether
  the cause is optimization dynamics, generalization, or both; no gradient
  measurements were taken.
- Contrast with He et al. 2015: there, deeper plain networks had higher TRAINING
  error (a degradation/optimization problem). Ours did not. Residual connections
  were still a reasonable thing to try because they're the standard fix for deep
  plain stacks, but this is a test, not a diagnosis.
- Visual cues: `5`, `7` (seg 1); `Two`, `Four`, `Eight`, `Sixteen`, `drops`
  (seg 2 — spoken number words reveal the chart points); `fitting` (seg 3); `ours`,
  `Residuals` (seg 4).

### 8. residual (6 segments): shortcuts

- Block structure: conv → BN → ReLU → conv → BN gives F(x); output is
  ReLU(F(x) + shortcut(x)). Shortcut is x itself, or x with zero channels appended
  when the channel count doubles (32→64). No learned weights, so each residual
  network has exactly the same parameter count and the same initial weights as
  its plain twin (1,239,274 at 32).
- Why it might help: to leave the input almost unchanged, a plain block has to
  learn an identity mapping through two convolutions; a residual block only needs
  F ≈ 0. And during backprop, gradients have a direct path through the additions.
  Schematic only, not measured.
- Results (VALIDATION): residual 4/8/16/32 = 83.77, 86.67, 86.85, 87.20. At 32:
  83.08 → 87.20 (+4.12). At 4, 8, 16 residual did not beat plain (83.77 vs 84.53,
  86.67 vs 86.90, 86.85 vs 87.17). Residual 32 (87.20 ± 0.57) vs plain 16 (87.17 ±
  0.26) is 0.03 points, inside seed spread. Selected by validation anyway (highest
  mean). TEST for the selected model: 86.43 ± 0.49.
- Can say: shortcuts removed the drop at 32 here. Can't say: shortcuts made the
  network better in general, or that residual 32 is meaningfully better than plain
  16.
- Visual cues: `shortcut` (seg 0); `learned` (seg 2); `zero`, `backprop` (seg 3);
  `83.77`, `Eight`, `Sixteen`, `Thirty` (seg 4); `matters`, `didnt`, `0.03`,
  `test` (seg 5).

### 9. aug2 (3 segments): augmentation again, on residual 32

- Why: augmentation hurt the small model, but that model ran a weak, short recipe.
  Re-test on residual 32 with the stricter recipe. Same seeds, same initial weights,
  same everything except augmentation.
- Results (VALIDATION): none 87.20 ± 0.57; flip 90.31 ± 0.03; crop + flip 93.74 ±
  0.27. Crop + flip paired gains +5.98, +6.93, +6.71 → +6.54, every seed.
- Contrast: small CNN crop + flip −1.58 (TEST, exploratory) vs residual 32 +6.54
  (VALIDATION). Same augmentation, opposite sign. Plausible reason: a bigger model
  with a longer, stronger recipe has the capacity to fit varied data and gains from
  it. Not proven: model, recipe, split and metric all differ between the two
  comparisons.
- Visual cues: `No`, `Flip`, `Crop` (seg 1, curve draws), `held` (seg 1, paired
  seed dots); `Plausibly`, `prove` (seg 2).

### 10. resnet (7 segments): CIFAR ResNet-18

- Why: a well-known residual architecture with much more width, to see how far the
  recipe could go. Random initialization, no ImageNet pre-training.
- Stem: torchvision's ImageNet stem (7×7 stride 2 conv + max pool stride 2) would
  cut 32×32 to 8×8 before any block. Replaced with a 3×3 stride-1 conv (3→64) and
  removed the max pool.
- Body: 8 basic residual blocks, two per stage, 64/128/256/512 channels;
  resolution 32, 16, 8, 4. Where shape changes (3 places), the shortcut is a learned
  1×1 stride-2 conv + BN.
- Head: global average pooling (512×4×4 → 512), one linear layer 512 → 10. No
  128-unit hidden layer.
- Why "18": stem conv + 16 block convs + classifier; projection convs not counted
  by convention.
- How it differs from residual 32: keeps resolution longer, many more channels,
  11,173,962 params (~9.0× residual 32's 1,239,274), global pooling head, learned
  projections, AND 200 epochs instead of 160. So it's a bundled recipe comparison,
  not a clean architecture test.
- Result: VALIDATION 95.77 ± 0.11 (final-10 mean, three seeds), vs 93.74 for
  residual 32 + crop + flip. Seed 0's best single epoch and last epoch shown on
  screen but not used for scoring.
- Visual cues: `ImageNet`, `swapped` (seg 1); `64`, `128`, `256`, `512`, `32`,
  `16`, `8`, `4`, `learned` (seg 2, in that order: channels said before
  resolutions); `linear` (seg 3); `stem`, `16`, `classifier`, `convention` (seg 4);
  comparison table rows keyed to `layer`, `longer`, `channels`, `globally`,
  `parameters`, `200`, `recipe` (seg 5).

### 11. end (5 segments): what we can claim

- Queue finished; selected by validation: ResNet-18 + crop + flip. TEST
  95.19 / 95.21 / 95.22 → 95.21 ± 0.02. This is the first time the test set was
  used for this model.
- Path: 63.68 TEST (classroom) → 78.27 VAL (stricter recipe, 2 conv) → 87.20 VAL
  (residual 32) → 93.74 VAL (+ crop and flip) → 95.77 VAL (ResNet-18). Mixed
  metrics and changing setups; a sequence, not a leaderboard and not a set of
  additive contributions.
- Scope: strongest measured here, not the best ResNet. Wider/deeper variants and
  other recipes do better (arXiv:1605.07146). Not state of the art.
- Practical takeaways the evidence supports (phrased as what we'd do next time,
  without absolute claims): compare by optimizer steps, not epochs; make decisions
  on validation across seeds; don't read a smaller gap as improvement on its own;
  re-test a technique when the model or recipe changes.
- Final beat: the improvement came from many changes together (training recipe,
  depth with shortcuts, augmentation, a bigger architecture), and the experiments
  don't separate their individual contributions.
- Visual cues: `story` (seg 1, mixed-metrics tag — rename cue if wording changes);
  numbers `63.68`, `78.27`, `87.20`, `93.74`, `95.77` spoken in order in seg 1;
  `Wider` (seg 2); lesson cues `Count`, `Decide`, `smaller`, `technique` (seg 3)
  — will be re-keyed to the new wording.

## Narration constraints

- Spoken numbers must come from `results.json` via `tools/build_script.py`
  (f-strings), never hand-typed where a field exists.
- Every metric mention says test or validation where the number first appears in
  a chapter.
- No claims beyond the "can say" lines above. Plausible explanations are marked as
  plausible.
- No biography, no talk about voice or style inside the script.
- Target ~10 minutes; complete explanations win over brevity.
