"""Build the chaptered narration script from frozen results.

    python3 tools/build_script.py

Reads public/data/results.json (written by extract_results.py) so the numbers
spoken in the film always come from the frozen evidence. Writes
script/narration.json (consumed by tts.py) and script/script.md (the readable
script + storyboard with source paths).
"""
import json
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
R = json.loads((VIDEO / "public" / "data" / "results.json").read_text())


def f2(x):
    return f"{x:.2f}"


dep = {(c["variant"], c["depth"]): c for c in R["depth"]["candidates"]}
fu = {c["candidate"]: c for c in R["followup"]["candidates"]}
aug = {a["recipe"]: a for a in R["augmentation"]["recipes"]}
rn = fu["resnet18-crop-flip"]
rn_done = len(rn["seeds"])
rn0 = next(s for s in rn["per_seed"] if s["seed"] == 0)
cl = R["classroom"]

# ---------------------------------------------------------------- follow-up status
WORDS = {1: "one", 2: "two", 3: "three"}
if rn_done == 3:
    rn_result = (f"Across three seeds, it averaged {f2(rn['mean'])} percent validation accuracy over the last ten epochs, "
                 f"plus or minus {f2(rn['sd'])}. We score the final ten epochs, never the single best one.")
    rn_short = f2(rn["mean"])
elif rn_done == 2:
    rn_result = (f"Two of three seeds have finished, averaging {f2(rn['mean'])} percent validation over their last ten epochs, "
                 f"plus or minus {f2(rn['sd'])}. We score the final ten epochs, never the single best one.")
    rn_short = f2(rn["mean"])
else:
    rn_result = (f"Seed 0 finished at {f2(rn0['final10'])} percent validation accuracy, averaged over its last ten epochs. "
                 f"Its best epoch hit {f2(rn0['peak'])}, but we score the final ten.")
    rn_short = f2(rn0["final10"])

sel = R["followup"].get("selection")
if R["followup"].get("test_results_available"):
    tm, ts = R["followup"]["test_mean"], R["followup"].get("test_sd")
    name = sel["candidate"] if isinstance(sel, dict) and "candidate" in sel else str(sel)
    pretty = {"resnet18-crop-flip": "ResNet-18 with crop plus flip", "winner-crop-flip": "residual 32 with crop plus flip",
              "winner-flip": "residual 32 with flip", "winner-none": "residual 32 without augmentation"}.get(name, name)
    status = (f"The queue has now finished and selected {pretty} by validation. Its final test accuracy is {f2(tm)} percent"
              + (f", plus or minus {f2(ts)}." if ts else "."))
elif rn_done == 3:
    status = ("All three ResNet-18 seeds have finished, but no selection or final test result was recorded at this snapshot. "
              "So there's no ResNet-18 test score yet.")
else:
    status = (f"At this snapshot, only {WORDS[rn_done]} of three ResNet-18 seeds {'has' if rn_done == 1 else 'have'} finished. "
              "No official winner yet, and no ResNet-18 test score.")

C = []


def chapter(cid, title, segs):
    C.append({"id": cid, "title": title, "segments": [
        {"id": f"{cid}-{i + 1}", "text": s[0], "say": s[1] if len(s) > 1 and s[1] else None,
         "pause": s[2] if len(s) > 2 else 0.45, "visual": s[3] if len(s) > 3 else ""}
        for i, s in enumerate(segs)]})


chapter("task", "Ten classes, 3,072 numbers", [
    ("This is a frog. Our network has to figure that out from nothing but numbers.", None, 0.4,
     "Real CIFAR-10 test images fill the frame; the frog is centered."),
    ("CIFAR-10 is 60,000 tiny photos in ten classes, from airplanes to trucks. 50,000 for training, and 10,000 held out for testing.",
     "SIFAR ten is 60,000 tiny photos in ten classes, from airplanes to trucks. 50,000 for training, and 10,000 held out for testing.", 0.4,
     "10x8 gallery of real test images with class labels; split bar 50,000 / 10,000."),
    ("Zoom in and the frog disappears. Each pixel is three numbers, red, green, and blue, from 0 to 255. 32 by 32 by 3. 3,072 numbers.",
     None, 0.5, "Camera pushes into the frog: real pixel values, then R, G, B slabs separate."),
    ("So, what actually makes a network better at turning those numbers into the right answer?",
     None, 0.7, "Question resolves over the slabs."),
])

chapter("cnn", "The starting network", [
    ("We started with a small convolutional network. 545,098 trainable parameters.", None, 0.4,
     "Full architecture strip: 3@32² → 32@32² → pool → 64@16² → pool → 4096 → 128 → 10."),
    ("First, normalization. Each color channel has its dataset mean subtracted, then gets divided by its spread.",
     None, 0.9, "Equation x̂ = (x − μ)/σ with the real CIFAR channel statistics."),
    ("Then, convolution. A filter is a 3 by 3 window through all three channels. 27 weights, plus a bias. At each position, it multiplies and sums. Slide it everywhere, and you get a new map.",
     None, 0.5, "Window sweeps the frog; real multiply-sum numbers; output map fills in behind it."),
    ("The same weights are reused at every position, so a pattern only has to be learned once, wherever it shows up.",
     None, 0.4, "Several windows linked to one shared kernel."),
    ("With 32 filters, we get 32 maps. These are the real filters our baseline learned, and their real maps for this frog. Nobody hardcoded edge detectors. Training found these.",
     None, 0.5, "Real conv1 filters and activations from ../runs/baseline/last.pt."),
    ("ReLU zeroes the negatives, and 2 by 2 max pooling keeps each block's strongest value, 32 by 32 down to 16 by 16.",
     None, 0.4, "Number grid: negatives → 0, then 2x2 max."),
    ("The second block repeats this with 64 filters reading all 32 maps, combining simple patterns into parts. Pooling takes it to 8 by 8.",
     None, 0.4, "Real conv2 activations; slab shrinks to 8x8."),
    ("That's 4,096 numbers. Flatten them, 128 hidden units with ReLU, and out come ten scores, one per class.",
     None, 0.6, "Slab dissolves into a 4096 column, particles flow to 128 then 10."),
])

chapter("learn", "How it learns", [
    ("Those ten scores are logits. Any size, positive or negative. Not probabilities yet.", None, 0.4,
     "Real logits for the frog as signed bars."),
    ("Softmax turns them into probabilities. Exponentiate each one, then divide by the total. Now they're positive, and they sum to one.", None, 1.0,
     "Softmax equation; bars morph into real probabilities."),
    ("For this frog, the baseline bets mostly on frog, but not all in.", None, 0.4,
     "Frog probability highlighted."),
    ("Cross-entropy is the negative log of the probability on the right answer. Confident and right costs almost nothing. Confident and wrong costs a lot.",
     None, 1.0, "L = −log p_y curve with the real frog point."),
    ("One detail. The network itself only outputs logits. PyTorch's cross-entropy applies log-softmax internally. Softmax here is for reading outputs, not an extra layer.",
     None, 0.5, "nn.CrossEntropyLoss(z, y) = −log softmax(z)_y; no softmax layer in build_model."),
    ("Backpropagation uses the chain rule to send that loss backwards, telling every parameter which way would lower it.",
     None, 0.9, "Forward particles, then gradient pulses flow backward; chain-rule equation."),
    ("This is a real slice of the loss around our trained baseline, along two random directions in weight space. Gradient descent steps downhill. The path drawn here is illustrative, not recorded steps.",
     None, 0.9, "Measured 25x25 loss slice (tools/extract_static.py); path labelled illustrative; update rule w ← w − η∇L."),
    ("Each step uses a batch of 64 images. One pass over the training set, an epoch, is 782 steps.",
     None, 0.4, "782 batch tiles fill up one epoch."),
    (f"Ten epochs at learning rate 0.01 gave our first baseline. {f2(cl['baseline_b64'][-1]['test'])} percent on test. Wrong more than a third of the time.",
     None, 0.7, "Real per-epoch test curve from ../runs/baseline.log."),
])

chapter("batch", "Faster isn't better", [
    ("First, we tried going faster. At batch 512, ten epochs took 5 seconds instead of 22.",
     None, 0.4, "Two lanes: batch 64 vs batch 512; timers 22.39 s vs 5.21 s."),
    (f"But accuracy fell to {f2(cl['tuned_b512_10ep'][-1]['test'])} percent. Turns out bigger batches mean fewer steps. 980 updates instead of 7,820.",
     None, 0.5, "Step dots: 7,820 vs 980; illustrative step paths (labelled) on the measured loss slice."),
    (f"With 80 epochs, 7,840 updates, almost exactly matched, it reached {f2(cl['b512_80ep'][-1]['test'])} percent.",
     None, 0.4, "Real test curves plotted against optimizer updates."),
    ("But each step used eight times as many images. Bigger batches aren't magic. They trade more work per step for fewer steps.",
     None, 0.7, "Image-pass bars: 0.5M vs 4M."),
])

chapter("aug1", "Augmentation, round one", [
    ("Next, augmentation. Show a variation of each photo instead of the exact same one. Flip it, or pad the border and crop a shifted window.",
     None, 0.4, "Real horse: flip, pad with 4 black pixels, crop windows."),
    ("A horse facing left is still a horse. This teaches invariance, and makes memorizing single images harder, a form of regularization.",
     None, 0.4, "All views map to one label."),
    (f"We tested 15 recipes, three seeds each, on the small network. No augmentation, {f2(aug['none']['mean'])} percent. Flip helped a little, {f2(aug['flip']['mean'])}. Crop plus flip hurt, {f2(aug['crop-flip']['mean'])}.",
     None, 0.5, "Real bar chart of all 15 recipes with seed error bars."),
    ("The subtle part. Crop plus flip shrank the train-test gap from about 7 points to under 1. But a smaller gap can mean useful regularization, or a model that just learned less. Test accuracy fell, so the gap alone isn't evidence of improvement.",
     None, 0.5, "Real train/test curves; gap 7.33 vs 0.74 points."),
    ("This suite also checked the test set repeatedly, so treat it as exploratory.", None, 0.7,
     "Caveat tag."),
])

chapter("recipe", "A stricter recipe", [
    ("Then we got stricter. We held out a stratified validation set of 5,000 images, trained on 45,000, and made every decision on validation. Test was reserved for final assessment, but the earlier suite had used it, so it's not a pristine holdout.",
     None, 0.5, "50k bar splits into 45k train / 5k validation; test set reserved, tagged 'previously used in exploration'."),
    ("And a new recipe. Batch 128, SGD with momentum and weight decay, a warmup then cosine learning rate over 160 epochs, and batch norm after every convolution.",
     None, 0.5, "Real learning-rate schedule; recipe card with exact values."),
    ("Every setup runs three seeds. Scores average the last ten validation epochs, not the luckiest one, and plus or minus is the seed spread.",
     None, 0.5, "Three real seed curves; final-10 window."),
    (f"The same two-convolution design, plus batch norm, now scores {f2(dep[('plain', 2)]['mean'])} on validation, versus {f2(cl['baseline_b64'][-1]['test'])} on test before. But the split, the metric, and several knobs all changed, so that's not an isolated ablation.",
     None, 0.7, "63.68 (test, classroom) vs 78.27 (validation, new recipe)."),
])

chapter("depth", "Going deeper", [
    ("Now, what if we stack more convolutions? Same stages, pooling, and head. Just more layers.",
     None, 0.4, "Architecture grows 2 → 4 → 8 → 16 → 32 convolutions."),
    ("Each extra 3 by 3 layer widens what a unit can see, 5 by 5, then 7 by 7, and lets it compose more complex functions.",
     None, 0.4, "Receptive field grows on the frog (schematic)."),
    (f"Two convolutions, {f2(dep[('plain', 2)]['mean'])}. Four, {f2(dep[('plain', 4)]['mean'])}. Eight, {f2(dep[('plain', 8)]['mean'])}. Sixteen, {f2(dep[('plain', 16)]['mean'])}. Smaller gains, but still going up. Then at 32, it drops to {f2(dep[('plain', 32)]['mean'])}.",
     None, 0.6, "Real validation accuracy vs depth with seed spread; the 32-layer point falls."),
    ("Every network fits its training labels, almost 100 percent. But fitting isn't generalizing. These results alone can't say whether optimization, generalization, or both caused the drop.",
     None, 0.5, "Real final train accuracy ≈100% for all depths; real curves 16 vs 32."),
    ("And unlike the original ResNet paper, where deeper plain nets had higher training error, ours didn't. Residuals are a plausible fix to test, not a diagnosis.",
     None, 0.6, "Paper (training error) vs ours (validation); transition to residual block."),
])

chapter("residual", "Residual connections", [
    ("A residual block keeps the same two convolutions, but adds a shortcut that skips around them and gets added back at the end.",
     None, 0.4, "Plain block and residual block side by side; shortcut draws."),
    ("So the block computes F of x. Conv, batch norm, ReLU, conv, batch norm. And the output is ReLU of F of x, plus the shortcut.", None, 1.0,
     "y = ReLU(F(x) + shortcut(x))."),
    ("When channels grow, our shortcut is padded with zeros. No learned weights, so each residual network matches its plain twin's parameters exactly.",
     None, 0.4, "32 channels + 32 zero channels → 64."),
    ("Why might that help? A block that should do nearly nothing just pushes F toward zero, instead of learning to copy its input. And in backprop, the shortcuts give gradients a direct route. Schematic, not measured.",
     None, 0.5, "Vector diagram x + F(x) = y, then gradient pulses on both paths (schematic)."),
    (f"Matched seeds, identical starting weights. Residual 4, {f2(dep[('residual', 4)]['mean'])}. Eight, {f2(dep[('residual', 8)]['mean'])}. Sixteen, {f2(dep[('residual', 16)]['mean'])}. Thirty-two, {f2(dep[('residual', 32)]['mean'])}.",
     None, 0.4, "Real residual line joins the plain line."),
    (f"At 32 layers, the shortcut matters, {f2(dep[('plain', 32)]['mean'])} up to {f2(dep[('residual', 32)]['mean'])}. But it didn't beat plain at 4, 8, or 16, and it's only 0.03 points above plain 16, inside the seed spread. On test, it scored {f2(R['depth']['test_mean'])} percent, plus or minus {f2(R['depth']['test_sd'])}.",
     None, 0.7, "Gap at 32 highlighted; spread band; test card."),
])

chapter("aug2", "Augmentation, round two", [
    ("Now the twist. Crop plus flip, which hurt the tiny network, on residual 32.", None, 0.4,
     "Augmented horse views return beside the residual-32 stack."),
    (f"No augmentation, {f2(fu['winner-none']['mean'])}. Flip, {f2(fu['winner-flip']['mean'])}. Crop plus flip, {f2(fu['winner-crop-flip']['mean'])}. About six and a half points, and it held for all three seeds.",
     None, 0.5, "Real validation curves and paired seed dots."),
    ("Same augmentation, opposite result. Plausibly, a bigger model with a stronger recipe can use that variety. We didn't prove that, and model, recipe, and split all changed together. A technique's value depends on what it's paired with.",
     None, 0.7, "Small CNN −1.58 vs residual-32 +6.54; caveat."),
])

chapter("resnet", "Building ResNet-18", [
    ("Finally, ResNet-18, adapted for CIFAR and trained from random weights. No ImageNet pre-training.", None, 0.4,
     "Empty pipeline; 'weights=None'."),
    ("The stem is one 3 by 3 convolution, 3 to 64 channels, stride 1. ImageNet's 7 by 7 stride 2 stem and max pool would crush a 32 by 32 image to 8 by 8, so we swapped one and dropped the other.",
     None, 0.4, "Stem slab; ghosted ImageNet stem crossed out."),
    ("Then eight basic residual blocks, two per stage, at 64, 128, 256, and 512 channels. Each stage halves resolution. 32, 16, 8, 4. Where shapes change, the shortcut is a learned 1 by 1 convolution with batch norm.",
     None, 0.4, "Blocks appear stage by stage; amber projection shortcuts."),
    ("Global average pooling turns each 4 by 4 map into one number, and one linear layer maps 512 features to ten classes. No hidden 128 layer.",
     None, 0.4, "512x4x4 → 512 → 10."),
    ("Why 18? The stem plus 16 block convolutions is 17, and the classifier makes 18. By convention, shortcut projections don't count.",
     None, 0.5, "Counter 1 + 16 + 1 = 18."),
    ("So it's not just layer count. It keeps resolution longer, has far more channels, about 11.2 million parameters, nine times residual 32, and pools globally. And it trained 200 epochs, not 160. A recipe comparison, not a clean architecture test.",
     None, 0.5, "Side-by-side comparison card."),
    (rn_result, None, 0.7, "Real ResNet-18 validation curve vs residual-32 crop+flip; final-10 window."),
])

chapter("end", "What we can actually claim", [
    (status, None, 0.5, "Queue status from runs/followup/queue-state.json at snapshot."),
    (f"The whole path. {f2(cl['baseline_b64'][-1]['test'])} for the classroom baseline. {f2(dep[('plain', 2)]['mean'])} with a stricter recipe. {f2(dep[('residual', 32)]['mean'])} with depth and shortcuts. {f2(fu['winner-crop-flip']['mean'])} with augmentation. {rn_short} with ResNet-18. Test and validation are mixed, so it's a story, not a leaderboard.",
     None, 0.5, "Staircase of real results with metric tags."),
    ("It's the strongest we've measured, not the best ResNet that exists. Wider and deeper variants and other recipes do better. This isn't state of the art.",
     None, 0.5, "Scope note."),
    ("The lessons generalize. Count optimizer steps, not just epochs. Decide on validation, across seeds. A smaller gap isn't better performance. And a technique's value depends on everything around it.",
     None, 0.5, "Four lessons appear one at a time."),
    ("No single trick did this. It was the design and the recipe, working together.", None, 1.8,
     "Final frame."),
])

out = {"snapshot_utc": R["snapshot_utc"], "followup_status": R["followup"]["jobs"], "chapters": C}
(VIDEO / "script").mkdir(exist_ok=True)
(VIDEO / "script" / "narration.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

SOURCES = {
    "task": "../data/cifar-10-batches-py; https://www.cs.toronto.edu/~kriz/cifar.html",
    "cnn": "../cifar_cnn.py (build_model, MEAN/STD); ../runs/baseline/last.pt via tools/extract_static.py",
    "learn": "../runs/baseline.log; ../runs/baseline/last.pt (logits, loss slice); ../research-log.md",
    "batch": "../BENCHMARKS.md; ../runs/tuned-b512.log; ../runs/step-matched-b512.log",
    "aug1": "../runs/augmentation/comparison.json + comparison.md; ../augmentations.py",
    "recipe": "../DEPTH_EXPERIMENTS.md; ../depth_cnn.py (learning_rate, split_indices); ../runs/depth/*/metrics.jsonl",
    "depth": "../runs/depth/comparison.json; ../depth_cnn.py (build_model)",
    "residual": "../depth_cnn.py (Block.shortcut zero-pad); ../runs/depth/comparison.json; https://arxiv.org/abs/1512.03385",
    "aug2": "../runs/followup/comparison.json; ../runs/followup/winner-*/seed-*/metrics.jsonl",
    "resnet": "../followup_cnn.py (build_model); torchvision 0.22.1 resnet18; ../runs/followup/resnet18-crop-flip/seed-*/metrics.jsonl",
    "end": "../runs/followup/queue-state.json; https://arxiv.org/abs/1605.07146",
}
md = ["# The CIFAR-10 journey: narration script and storyboard", "",
      f"Generated by `tools/build_script.py` from `public/data/results.json` (snapshot {R['snapshot_utc']}).",
      "Validation and test numbers are labelled as such; ± is sample SD across seeds 0, 1, 2.", ""]
for i, ch in enumerate(C, 1):
    md += [f"## {i}. {ch['title']}", "", f"*Sources:* {SOURCES[ch['id']]}", ""]
    for s in ch["segments"]:
        md += [f"**{s['id']}** — {s['text']}", "", f"> Visual: {s['visual']}", ""]
(VIDEO / "script" / "script.md").write_text("\n".join(md))
words = sum(len(s["text"].split()) for ch in C for s in ch["segments"])
print("chapters", len(C), "segments", sum(len(c["segments"]) for c in C), "words", words)
