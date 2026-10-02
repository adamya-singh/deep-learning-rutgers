"""Build the chaptered narration script from frozen results.

    python3 tools/build_script.py

This file is the single source of the narration. Reads public/data/results.json
and public/data/static.json (written by extract_results.py / extract_static.py)
so every number spoken in the film comes from the frozen evidence. Writes
script/narration.json (consumed by tts.py) and script/script.md (the readable
script + storyboard with source paths). Content follows script/teaching-plan.md.

Edit narration here, not in script/*.md or narration.json: those are outputs and
are overwritten on every run.
"""
import json
import math
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
R = json.loads((VIDEO / "public" / "data" / "results.json").read_text())
S = json.loads((VIDEO / "public" / "data" / "static.json").read_text())


def f2(x):
    return f"{x:.2f}"


dep = {(c["variant"], c["depth"]): c for c in R["depth"]["candidates"]}
fu = {c["candidate"]: c for c in R["followup"]["candidates"]}
aug = {a["recipe"]: a for a in R["augmentation"]["recipes"]}
rn = fu["resnet18-crop-flip"]
rn_done = len(rn["seeds"])
rn0 = next(s for s in rn["per_seed"] if s["seed"] == 0)
cl = R["classroom"]
base = cl["baseline_b64"][-1]["test"]
frog = S["featured"]["frog"]
p_frog, p_cat = frog["probs"][6], frog["probs"][3]
small_cf = -sum(aug["crop-flip"]["delta_vs_none"]) / 3          # points lost by crop+flip on the small CNN
big_cf = sum(fu["winner-crop-flip"]["paired_gain"]) / 3         # points gained on residual 32

# ---------------------------------------------------------------- follow-up status
WORDS = {1: "one", 2: "two", 3: "three"}
if rn_done == 3:
    rn_result = (f"Across three seeds, it averaged {f2(rn['mean'])} percent validation accuracy over the last ten epochs, "
                 f"plus or minus {f2(rn['sd'])}, compared to {f2(fu['winner-crop-flip']['mean'])} for residual 32 with the same augmentation.")
    rn_short = f2(rn["mean"])
elif rn_done == 2:
    rn_result = (f"Two of the three seeds have finished so far, and they average {f2(rn['mean'])} percent validation accuracy "
                 f"over their last ten epochs, plus or minus {f2(rn['sd'])}.")
    rn_short = f2(rn["mean"])
else:
    rn_result = (f"Only seed 0 has finished so far, at {f2(rn0['final10'])} percent validation accuracy averaged over its last ten epochs. "
                 f"Its best single epoch was {f2(rn0['peak'])}, but we only score the last ten.")
    rn_short = f2(rn0["final10"])

sel = R["followup"].get("selection")
if R["followup"].get("test_results_available"):
    tm, ts = R["followup"]["test_mean"], R["followup"].get("test_sd")
    name = sel["candidate"] if isinstance(sel, dict) and "candidate" in sel else str(sel)
    pretty = {"resnet18-crop-flip": "ResNet-18 with crop plus flip", "winner-crop-flip": "residual 32 with crop plus flip",
              "winner-flip": "residual 32 with flip", "winner-none": "residual 32 without augmentation"}.get(name, name)
    status = (f"When the queue finished, it picked {pretty} based on validation, and only then did we run that model on the test set. "
              f"It got {f2(tm)} percent test accuracy" + (f", plus or minus {f2(ts)} across the three seeds." if ts else "."))
elif rn_done == 3:
    status = ("All three ResNet-18 seeds have finished, but no selection or final test result had been recorded when these numbers "
              "were frozen, so there's no ResNet-18 test score yet.")
else:
    status = (f"When these numbers were frozen, only {WORDS[rn_done]} of the three ResNet-18 seeds "
              f"{'had' if rn_done == 1 else 'had'} finished, so nothing has been selected yet and there's no ResNet-18 test score.")

C = []


def chapter(cid, title, segs):
    C.append({"id": cid, "title": title, "segments": [
        {"id": f"{cid}-{i + 1}", "text": s[0], "say": s[1] if len(s) > 1 and s[1] else None,
         "pause": s[2] if len(s) > 2 else 0.45, "visual": s[3] if len(s) > 3 else ""}
        for i, s in enumerate(segs)]})


chapter("task", "The input: 3,072 numbers", [
    ("This is a frog from the CIFAR-10 test set. We wanted a neural network that could look at images like this and say "
     "what's in them, even though it never really sees a picture, it only sees numbers.",
     "This is a frog from the SIFAR ten test set. We wanted a neural network that could look at images like this and say "
     "what's in them, even though it never really sees a picture, it only sees numbers.", 0.4,
     "Real CIFAR-10 test images fill the frame; the frog is centered."),
    ("CIFAR-10 has 60,000 tiny color photos split across ten classes, things like airplanes, cats, frogs, and trucks. "
     "The official split gives us 50,000 images to train on and keeps 10,000 aside for testing.",
     "SIFAR ten has 60,000 tiny color photos split across ten classes, things like airplanes, cats, frogs, and trucks. "
     "The official split gives us 50,000 images to train on and keeps 10,000 aside for testing.", 0.4,
     "10x8 gallery of real test images with class labels; split bar 50,000 / 10,000."),
    ("If you zoom in on the frog, it's really just a 32 by 32 grid of pixels, and each pixel is three numbers between 0 and 255 "
     "for how much red, green, and blue it has. So every image is 32 times 32 times 3, which is 3,072 numbers.",
     None, 0.5, "Camera pushes into the frog: real pixel values, then R, G, B slabs separate on the third spoken 32."),
    ("The rest of this goes through the experiments we ran to figure out what makes a network better at turning those numbers "
     "into the right label.",
     None, 0.7, "Question resolves over the slabs."),
])

chapter("cnn", "The starting network", [
    ("The network we started with is a small convolutional network from class, with 545,098 trainable parameters, which are the numbers "
     "it adjusts while it learns.", None, 0.4,
     "Full architecture strip: 3@32² → 32@32² → pool → 64@16² → pool → 4096 → 128 → 10; parameter breakdown."),
    ("Before anything else, every pixel value gets divided by 255, and then each color channel has its dataset average subtracted "
     "and gets divided by its standard deviation, which puts the channels on a similar scale around zero and usually makes "
     "training better behaved.",
     None, 0.9, "Equation x̂ = (x − μ)/σ with the real CIFAR channel statistics and one real frog pixel worked through."),
    ("The first real layer is a convolution. A filter is a 3 by 3 window over all three color channels, "
     "so it has 27 weights plus a bias. At each position it multiplies the values under it by its weights and adds everything up, "
     "and this panel is that exact calculation from our trained model. Slide the same window across every position and you get "
     "a new 32 by 32 map.",
     None, 0.5, "Window moves to row 14, col 15; real multiply-sum (0.6256, matches the model); 'Slide' starts the sweep and the real output map fills in."),
    ("The filter uses the same weights at every position, so a pattern it learns in one spot can be "
     "picked up anywhere else in the image. That's also why all 32 filters in this layer only need 896 parameters.",
     None, 0.4, "Several windows linked to one shared kernel; 32 × 28 = 896."),
    ("These are the actual filters our baseline learned, and the maps they produce for this frog. Nobody designed these weights "
     "by hand, training found all of them.",
     None, 0.5, "Real conv1 filters and activations from ../runs/baseline/last.pt."),
    ("After the convolution, ReLU replaces every negative number with zero, and then 2 by 2 max pooling keeps only the biggest "
     "value in each 2 by 2 block, which cuts the maps from 32 by 32 down to 16 by 16.",
     None, 0.4, "Example number grid (illustrative): negatives → 0, then 2x2 max; real map before/after pooling."),
    ("The second block does the same thing with 64 filters, but now each filter reads all 32 maps from the first block, so it "
     "can combine simpler patterns into more complicated ones. Pooling then takes everything down to 8 by 8.",
     None, 0.4, "Real conv2 activations; slab shrinks to 8x8 on 'Pooling'."),
    ("That leaves 64 maps of 8 by 8, which is 4,096 numbers. We flatten them into one long list, pass it through a hidden layer "
     "of 128 units, and the last layer gives us ten scores, one for each class.",
     None, 0.6, "Slab dissolves into a 4096 column, particles flow to 128 then 10; dense-layer parameter count."),
])

chapter("learn", "How it learns", [
    ("Those ten scores are called logits. They can be any size, positive or negative, so on their own they aren't probabilities "
     "yet.", None, 0.4,
     "Real logits for the frog as signed bars."),
    ("Softmax is what turns them into probabilities. You exponentiate each score, which makes everything positive, and then "
     "divide by the total so they all add up to one.", None, 1.0,
     "Softmax equation; bars morph into real probabilities on 'exponentiate'."),
    (f"Here the model puts about {round(p_frog * 100)} percent on frog and about {round(p_cat * 100)} percent on cat, so it's "
     "mostly right but not very sure.", None, 0.4,
     "Frog probability highlighted."),
    ("To train it, we need one number for how wrong it was, and that's the cross-entropy loss, the negative log of the "
     f"probability it gave the correct class. Confident and right costs almost nothing, this frog only costs about "
     f"{-math.log(p_frog):.2f}. Confident and wrong costs a lot, so giving the right answer just 2 percent would cost about "
     f"{-math.log(0.02):.1f}.",
     None, 1.0, "L = −log p_y curve with the real frog point, then the labelled example point p = 0.02."),
    ("One detail about PyTorch here. The model itself only outputs logits, and PyTorch's cross-entropy loss applies log-softmax "
     "internally, so softmax is just how we read the outputs, it isn't an extra layer in the model.",
     None, 0.5, "nn.CrossEntropyLoss(z, y) = −log softmax(z)_y; no softmax layer in build_model."),
    ("Backpropagation then uses the chain rule to work backwards from the loss through every layer, and gives each of the "
     "545,098 parameters a gradient, which says how the loss would change if that parameter moved a tiny bit.",
     None, 0.9, "Forward particles, then gradient pulses flow backward; chain-rule equation; schematic tag."),
    ("Gradient descent moves every weight a small step against its gradient, with the learning rate setting the step size. "
     "This surface is a real slice of the loss around our trained baseline, along two random directions in weight space, "
     "and the dashed path is just an illustration of stepping downhill, not steps we recorded.",
     None, 0.9, "Measured 25x25 loss slice (tools/extract_static.py); dashed path labelled illustrative on 'path'; update rule w ← w − η∇L."),
    ("Each step uses a batch of 64 images, so one full pass through the 50,000 training images, which is called an epoch, "
     "takes 782 steps.",
     None, 0.4, "782 batch tiles fill up one epoch."),
    (f"We trained for ten epochs with a learning rate of 0.01, and this baseline reached {f2(base)} percent test accuracy. "
     "That's a lot better than the 10 percent you'd get from guessing, but it's still wrong on more than a third of the test images.",
     None, 0.7, "Real per-epoch test curve from ../runs/baseline.log; chance line at 10%."),
])

chapter("batch", "Bigger batches", [
    ("The first thing we tried was making training faster. At batch size 64 the GPU was only busy about 42 percent of the time, "
     "so we tried batch size 512, and ten epochs finished in 5 seconds instead of 22.",
     None, 0.4, "Two lanes: batch 64 vs batch 512; timers 22.39 s vs 5.21 s; mean GPU utilization 41.6% vs 89.4%."),
    (f"The catch was that test accuracy dropped to {f2(cl['tuned_b512_10ep'][-1]['test'])} percent. With the same number of epochs, "
     "a batch eight times bigger also means eight times fewer weight updates, "
     f"{R['classroom']['updates']['tuned_b512_10ep']:,} instead of {R['classroom']['updates']['baseline_b64']:,} at the same "
     "learning rate, so our guess was that the model just hadn't taken enough steps.",
     None, 0.5, "Step dots: 7,820 vs 980; '8× fewer steps' appears on 'eight'."),
    (f"To check that, we trained batch 512 for 80 epochs, which gives {R['classroom']['updates']['b512_80ep']:,} updates, almost "
     f"exactly the same as the baseline, and that run reached {f2(cl['b512_80ep'][-1]['test'])} percent. That fits the step count "
     "explanation, but it isn't a perfectly clean test, because this run also went through eight times as many images.",
     None, 0.4, "Real test curves plotted against optimizer updates; 67.67 labelled on 'reached'."),
    (f"So with matched steps, batch 512 ended up about {round(cl['b512_80ep'][-1]['test'] - base)} points higher, and it took "
     "37 seconds instead of 22. These are single runs with one seed each, so the exact gap isn't something to lean on too hard.",
     None, 0.7, "Image-pass bars: 0.5M vs 4M; 22.39 s vs 36.85 s."),
])

chapter("aug1", "Augmentation on the small network", [
    ("Next we tried data augmentation, which means every time the network sees a training image, it gets a randomly changed "
     "version of it. Flip it left to right half the time, or pad the border with 4 black pixels and crop a random 32 by 32 "
     "window, which shifts the image around a little.",
     None, 0.4, "Real horse: flip, pad with 4 black pixels, crop windows."),
    ("A horse facing left is still a horse, so the label stays the same, and the hope is that the network learns to ignore "
     "things like direction and small shifts, which is called invariance. Seeing varied views of each image also makes it "
     "harder to just memorize the exact training images, so it works as a kind of regularization.",
     None, 0.4, "All views map to one label."),
    (f"We tested 15 different recipes on the small network with three seeds each, measuring test accuracy. No augmentation got {f2(aug['none']['mean'])} percent. Flip on its own helped a little, "
     f"{f2(aug['flip']['mean'])}, and it improved every seed. Crop plus flip actually hurt, down to {f2(aug['crop-flip']['mean'])}, "
     "and everything we added on top of it did worse than no augmentation.",
     None, 0.5, "Real bar chart of all 15 recipes with seed error bars; none / flip / crop+flip light up when spoken."),
    (f"Then there's the gap between training and test accuracy, which went from about "
     f"{round(R['augmentation']['train_test_gap_pp']['none'])} points without augmentation to under 1 with crop plus flip. "
     "A smaller gap can mean useful regularization, or it can just mean the model learned less, and since test accuracy went down, "
     "the gap alone doesn't show an improvement. Our best guess is that this small network on a short recipe couldn't fit the "
     "harder data, but we didn't test that.",
     None, 0.5, "Real train/test curves; gap 7.33 vs 0.74 points."),
    ("This suite also looked at the test set over and over, so these results are exploratory.", None, 0.7,
     "Caveat tag."),
])

chapter("recipe", "A stricter setup", [
    ("Because of that, we got stricter about evaluation. We took 5,000 images out of the training set as a stratified validation "
     "set, 500 from each class, trained on the other 45,000, and made every decision on validation from then on. The test set "
     "was reserved for a final check of whichever model got selected, although it isn't a pristine holdout anymore, since the "
     "earlier suite already used it.",
     None, 0.5, "50k bar splits into 45k train / 5k validation; test set reserved, tagged 'previously used in exploration'."),
    ("We also switched to a stronger training recipe: batch 128, SGD with momentum and weight decay, batch norm after every "
     "convolution, and a learning rate that warms up for 5 epochs, then follows a cosine curve down over 160 epochs. "
     "Momentum smooths out the steps, weight decay keeps the weights small, and batch norm rescales each layer's outputs so "
     "deeper networks train more easily.",
     None, 0.5, "Real learning-rate schedule (draws on 'warms'); recipe card with exact values."),
    ("Every setup also ran with three different random seeds. A run's score is its average validation accuracy over the last "
     "ten epochs instead of its single best epoch, and the plus or minus is how much that score varies across seeds.",
     None, 0.5, "Three real seed curves; final-10 window."),
    (f"With this setup, the same two-convolution design plus batch norm scores {f2(dep[('plain', 2)]['mean'])} percent on validation, "
     f"compared to {f2(base)} percent on test before. But the split, the metric, the epochs, and the optimizer all changed at once, "
     "so we can't say which change caused that jump.",
     None, 0.7, "63.68 (test, classroom) vs 78.27 (validation, new recipe); 'not an isolated ablation'."),
])

chapter("depth", "Adding depth", [
    ("Next we wanted to know if more convolutions would help. We kept the same two stages, the same pooling, and the same "
     "classifier, and only stacked more convolutions inside each stage, from 2 up to 32.",
     None, 0.4, "Architecture grows 2 → 4 → 8 → 16 → 32 convolutions."),
    ("Each extra 3 by 3 layer lets a unit see a slightly bigger part of the image, 5 by 5 after two layers, then 7 by 7 after "
     "three, and it gives the network more steps to build up complicated features.",
     None, 0.4, "Receptive field grows on the frog (schematic)."),
    (f"On validation, two convolutions got {f2(dep[('plain', 2)]['mean'])}, four got {f2(dep[('plain', 4)]['mean'])}, "
     f"eight got {f2(dep[('plain', 8)]['mean'])}, and sixteen got {f2(dep[('plain', 16)]['mean'])}, so the gains were shrinking "
     f"but still positive. Then at 32 convolutions it dropped to {f2(dep[('plain', 32)]['mean'])}, and all three seeds got worse.",
     None, 0.6, "Real validation accuracy vs depth with seed spread; points appear on the spoken depths; the 32-layer point falls."),
    ("Every depth reached 100 percent training accuracy, so the 32 layer network could still fit the training set, it just did "
     "worse on images it hadn't seen, and these runs alone can't tell us whether that came from optimization, generalization, or both.",
     None, 0.5, "Real final train accuracy 100% for all depths; real curves 16 vs 32."),
    ("That's different from the original ResNet paper, where deeper plain networks had higher training error. Ours trained fine "
     "and fell on validation. Residual connections are the standard fix for deep plain networks, so they were a natural next "
     "thing to try, even though we didn't know the cause.",
     None, 0.6, "Paper (training error) vs ours (validation); transition to residual block."),
])

chapter("residual", "Residual connections", [
    ("A residual block keeps the same two convolutions, but adds a shortcut that carries the block's input around them and adds "
     "it back at the end.",
     None, 0.4, "Plain block and residual block side by side; shortcut draws."),
    ("So the main path computes F of x, which is conv, batch norm, ReLU, conv, batch norm, and the block outputs ReLU of F of x "
     "plus the shortcut.", None, 1.0,
     "y = ReLU(F(x) + shortcut(x))."),
    ("When the number of channels doubles from 32 to 64, our shortcut just pads the input with 32 channels of zeros. That adds no "
     "learned weights, so every residual network has exactly the same number of parameters as its plain version, and the two "
     "start from the same initial weights.",
     None, 0.4, "32 channels + 32 zero channels → 64; 1,239,274 parameters at 32."),
    ("The usual explanation is that if a block should barely change its input, a plain block has to learn to copy it through "
     "two convolutions, while a residual block just pushes F toward zero. During backprop, the shortcuts also give the gradient "
     "a direct path back. These pictures are schematics, not measurements.",
     None, 0.5, "Vector diagram x + F(x) = y, then gradient pulses on both paths (schematic)."),
    (f"Residual 4 got {f2(dep[('residual', 4)]['mean'])} on validation, "
     f"eight got {f2(dep[('residual', 8)]['mean'])}, sixteen got {f2(dep[('residual', 16)]['mean'])}, and thirty-two got "
     f"{f2(dep[('residual', 32)]['mean'])}.",
     None, 0.4, "Real residual line joins the plain line point by point."),
    (f"So at 32 layers the shortcuts clearly mattered, about {round(dep[('residual', 32)]['mean'] - dep[('plain', 32)]['mean'])} points over plain 32. "
     f"But residual didn't beat plain at 4, 8, or 16, and residual 32 is only "
     f"{dep[('residual', 32)]['mean'] - dep[('plain', 16)]['mean']:.2f} points above plain 16, well inside the variation between "
     f"seeds. It still had the highest average, so it was selected, and on the test set it scored "
     f"{f2(R['depth']['test_mean'])} percent, plus or minus {f2(R['depth']['test_sd'])}.",
     None, 0.7, "Gap at 32 highlighted; spread band; test card."),
])

chapter("aug2", "Augmentation on residual 32", [
    ("Since crop plus flip hurt the small network, we tried it again on residual 32, keeping the recipe, the seeds, and the "
     "starting weights the same, and only changing the augmentation.", None, 0.4,
     "Augmented horse views return beside the residual-32 stack."),
    (f"On validation, no augmentation got {f2(fu['winner-none']['mean'])}, flip got {f2(fu['winner-flip']['mean'])}, and crop plus "
     f"flip got {f2(fu['winner-crop-flip']['mean'])}. Crop plus flip improved all three "
     f"seeds, by roughly {math.floor(min(fu['winner-crop-flip']['paired_gain']) + 0.5)} to "
     f"{math.floor(max(fu['winner-crop-flip']['paired_gain']) + 0.5)} points each.",
     None, 0.5, "Real validation curves and paired seed dots (dots appear on 'improved')."),
    (f"So the same augmentation that cost the small network about {small_cf:.1f} points gave residual 32 about {big_cf:.1f}. "
     "Our best guess is that a bigger network with a longer, stronger recipe has room to learn from the extra variety. We didn't "
     "prove that though, since the model, the recipe, and the split all changed between the two experiments.",
     None, 0.7, "Small CNN −1.58 vs residual-32 +6.54; best-guess caption on 'guess'; caveat on 'prove'."),
])

chapter("resnet", "CIFAR ResNet-18", [
    ("Last, we tried ResNet-18, which is a much bigger residual network, adapted for CIFAR and trained from random weights, "
     "so there's no pre-training on ImageNet.",
     "Last, we tried ResNet-18, which is a much bigger residual network, adapted for SIFAR and trained from random weights, "
     "so there's no pre-training on ImageNet.", 0.4,
     "Code card: resnet18(weights=None), 3×3 stride-1 stem, maxpool = Identity."),
    ("The standard version starts with a 7 by 7 stride 2 convolution and a max pool, which is built for big ImageNet photos and "
     "would shrink our images to 8 by 8 before the blocks even start, so we swapped in one 3 by 3 stride 1 convolution and removed "
     "the max pool.",
     None, 0.4, "Stem slab; ghosted ImageNet stem crossed out on 'swapped'."),
    ("After that come eight basic residual blocks, two per stage, with 64, 128, 256, and 512 channels. The first stage stays at "
     "32 by 32, and each of the next three halves the resolution, so the maps go from 32 to 16 to 8 to 4. Where the shape changes, "
     "the shortcut is a learned 1 by 1 convolution with batch norm instead of zero padding.",
     None, 0.4, "Blocks appear stage by stage on the spoken channel counts; resolutions on the spoken sizes; amber projection shortcuts."),
    ("At the end, global average pooling turns each 4 by 4 map into a single number, and one linear layer maps those 512 numbers "
     "to the ten classes, with no 128 unit hidden layer like before.",
     None, 0.4, "512x4x4 → 512 → 10."),
    ("The 18 comes from counting the layers with weights along the main path. The stem is one, the blocks have 16 convolutions, "
     "and the classifier makes 18. The projection shortcuts aren't counted, by convention.",
     None, 0.5, "Counter 1 + 16 + 1 = 18, synced to 'stem', '16', 'classifier'."),
    ("So compared with residual 32, a lot more changed than the layer count. ResNet-18 keeps higher resolution for longer, has many "
     "more channels, learned shortcuts, a global pooling head, and about 11.2 million parameters, roughly nine times as many. "
     "It also trained for 200 epochs instead of 160, so this compares whole recipes, not just architectures.",
     None, 0.5, "Side-by-side comparison card; rows appear as each difference is spoken."),
    (rn_result, None, 0.7, "Real ResNet-18 validation curve vs residual-32 crop+flip; final-10 window."),
])

chapter("end", "What the results support", [
    (status, None, 0.5, "Queue status from runs/followup/queue-state.json at snapshot; selected model and its test score."),
    (f"Putting the whole path together, the classroom baseline got {f2(base)} on test, and then on validation the stricter recipe got "
     f"{f2(dep[('plain', 2)]['mean'])}, depth and shortcuts got {f2(dep[('residual', 32)]['mean'])}, crop and flip got "
     f"{f2(fu['winner-crop-flip']['mean'])}, and ResNet-18 got {rn_short}. The metrics and setups changed along the way, so these "
     "numbers show the order things happened in, not a leaderboard, and they shouldn't be added up.",
     None, 0.5, "Staircase of real results with TEST/VALIDATION tags; mixed-metrics note on 'leaderboard'."),
    ("This is the strongest result we measured in this project, and it's not the best ResNet out there. Wider and deeper "
     "residual networks and other training recipes get higher accuracy on CIFAR-10, so this isn't state of the art.",
     "This is the strongest result we measured in this project, and it's not the best ResNet out there. Wider and deeper "
     "residual networks and other training recipes get higher accuracy on SIFAR ten, so this isn't state of the art.",
     0.5, "Scope note."),
    ("If we did this again, we'd keep comparing runs by optimizer steps and not just epochs, deciding on validation across seeds, "
     "not treating a smaller train-test gap as an improvement by itself, and retesting a technique when the model or recipe "
     "changes, since crop plus flip went from hurting to helping.",
     None, 0.5, "Four takeaways appear one at a time as they are spoken."),
    (f"Going from {f2(base)} to {f2(R['followup']['test_mean']) if R['followup'].get('test_results_available') else rn_short} "
     f"percent on {'test' if R['followup'].get('test_results_available') else 'validation'} took a lot of changes stacked together, "
     "and these experiments don't split up exactly how much each one contributed.", None, 1.8,
     "Final frame: start and end numbers, the four kinds of change, credits."),
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
      f"Generated by `tools/build_script.py` from `public/data/results.json` and `public/data/static.json` "
      f"(snapshot {R['snapshot_utc']}). Content plan: `script/teaching-plan.md`. Edit the generator, not this file.",
      "Validation and test numbers are labelled as such; ± is sample SD across seeds 0, 1, 2.", ""]
for i, ch in enumerate(C, 1):
    md += [f"## {i}. {ch['title']}", "", f"*Sources:* {SOURCES[ch['id']]}", ""]
    for s in ch["segments"]:
        md += [f"**{s['id']}** — {s['text']}", "", f"> Visual: {s['visual']}", ""]
(VIDEO / "script" / "script.md").write_text("\n".join(md))
words = sum(len(s["text"].split()) for ch in C for s in ch["segments"])
print("chapters", len(C), "segments", sum(len(c["segments"]) for c in C), "words", words)
