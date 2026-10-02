# The CIFAR-10 journey: a narrated Remotion film

A ~10-minute, 1920×1080, 30 fps explanatory film about this project's CIFAR-10
experiments. It runs from the 545,098-parameter classroom CNN to the
validation-selected CIFAR ResNet-18, using real images, real learned filters and
activations, a measured loss slice, and real learning curves from `../runs`.
Written and built by Claude Code (Claude Opus 5.5).

## Deliverables

| File | What it is |
| --- | --- |
| `out/cifar-journey.mp4` | The full film (H.264 + AAC, 1080p30) |
| `out/cifar-journey-trailer.mp4` | Short narrated trailer cut from the same composition |
| `out/qa/*.mp4` | Motion-QA clips (sliding conv, backprop + loss slice, residual routes, ResNet build) |
| `out/cifar-journey.srt`, `.vtt` | Captions (also burned into the video) |
| `script/script.md` | Chaptered narration script + storyboard with source paths |
| `script/narration.json` | Machine-readable script (input to TTS) |
| `out/mp4-contact-sheet.jpg` | One frame per chapter decoded from the final MP4 |
| `out/contact-sheet-*.jpg`, `out/frames/` | Inspected preview stills (segment stills from the pre-refresh timeline; ResNet/End rechecked after refresh) |
| `public/data/results.json`, `public/data/static.json` | Frozen evidence the film renders from |
| `data/frozen/` | Verbatim copies of the source comparison/queue JSON at freeze time |
| `sources.md` | References |
| `build-progress.md` | Build log |

## Commands

Node: `export PATH=/home/win10ubuntu/.nvm/versions/node/v22.23.1/bin:$PATH`

```bash
npm ci                                  # project-local dependencies (package-lock.json)
npm run studio                          # Remotion Studio preview
npx tsc --noEmit                        # typecheck
npm run render                          # full 1080p MP4 → out/cifar-journey.mp4 (CPU, concurrency 4, swiftshader)
node tools/stills.mjs                   # one still per narration segment → out/frames/
node tools/stills.mjs 1663 4556         # specific frames
tools/render_clips.sh name:1557-1797    # short motion clip → out/qa/name.mp4
.venv-tools/bin/python tools/make_trailer.py   # out/qa/trailer-*.mp4 → out/cifar-journey-trailer.mp4
.venv-tools/bin/python tools/verify_render.py  # checks the final MP4 against the timeline
.venv-tools/bin/python tools/contact_sheet.py
```

Rendering never uses the GPU (the training queue owns it): `--gl=swiftshader`,
concurrency 4. If CPU time is tight, `npm run render:720` makes a labelled 720p
version. The 1080p command above is the reference.

## Refreshing results

All spoken and plotted numbers come from frozen JSON. To refresh after
`../runs` changes (read-only; no report scripts are run, nothing in `../runs` is written):

```bash
python3 tools/extract_results.py                 # ../runs → public/data/results.json + data/frozen/
python3 tools/build_script.py                    # results → script/narration.json + script.md
.venv-tools/bin/python tools/tts.py              # re-synthesizes only changed lines → timeline + captions
npx tsc --noEmit && node tools/stills.mjs && npm run render
```

`tools/extract_static.py` (images, filters, activations, loss slice) needs the
parent `.venv` (torch, used read-only on CPU with 2 threads):
`OMP_NUM_THREADS=2 ../.venv/bin/python tools/extract_static.py`.
`tools/derive_images.py` makes the R/G/B channel views.

Tools environment: `.venv-tools` (uv, Python 3.12) with edge-tts, imageio-ffmpeg,
numpy and pillow. It is local to this folder; the parent `.venv` is never modified.

## Narration

The narration uses Microsoft Edge neural TTS (`edge-tts`, voice `en-US-AndrewNeural`, rate +15%).
It is a stock synthetic voice, not a clone of anyone. A local writing reference
was used only as tone evidence (direct, first-person, plain, honest about
limits); that personal reference is excluded from the repository, and no biography from it appears in the film. Each line's audio length
drives its timing. Visual cues are keyed to word-level timestamps, so numbers
appear on screen when they are spoken.

## What is real and what is schematic

Tags in the film mark every element:

- **MEASURED:** CIFAR images and pixel values; conv1 filters, conv1/conv2 activations, logits and probabilities from `../runs/baseline/last.pt` (epoch 10); the multiply-sum example (matches the model's own output, 0.6256); the 25×25 loss slice (filter-normalized random directions, 1,000 training images); every learning curve, LR schedule and score.
- **ILLUSTRATIVE / SCHEMATIC:** the descent path drawn on the loss slice, the ReLU/max-pool toy numbers, the backprop network, receptive-field cone, residual vector picture and gradient-route animation.
- **TEST vs VALIDATION:** labelled everywhere. The early augmentation suite used the test set repeatedly (exploratory). Later queues select on a 45k/5k stratified split and reserve test for final assessment. That test set is still not a pristine holdout across the whole project.

## Results snapshot (frozen 2026-10-02 01:27:30 UTC)

The film was rendered after `../runs/followup/queue-state.json` reported
`phase: completed`. Every follow-up run is complete for all 3 seeds:

| Stage | Metric | Result |
| --- | --- | --- |
| Classroom CNN, b64, 10 ep | test, seed 0 | 63.68% |
| Batch 512, 10 ep / 80 ep | test, seed 0 | 48.58% / 67.67% |
| Early augmentation (none / flip / crop+flip) | test, 3 seeds, exploratory | 67.96 ± 1.00 / 68.77 ± 1.20 / 66.38 ± 1.01 |
| Plain 2/4/8/16/32 conv | val, final-10 mean, 3 seeds | 78.27 / 84.53 / 86.90 / 87.17 / 83.08 |
| Residual 4/8/16/32 conv | val | 83.77 / 86.67 / 86.85 / 87.20 |
| Residual 32 (depth-queue winner) | test, 3 seeds | 86.43 ± 0.49 |
| Residual 32 + none / flip / crop+flip | val | 87.20 ± 0.57 / 90.31 ± 0.03 / 93.74 ± 0.27 |
| **CIFAR ResNet-18 + crop+flip (selected by validation)** | val, final-10 mean, 3 seeds | **95.77 ± 0.11** |
| **same, final assessment** | test, 3 seeds (95.19, 95.21, 95.22) | **95.21 ± 0.02** |

± is the sample SD across seeds 0, 1, 2. Test values were cross-checked against
`../runs/followup/resnet18-crop-flip/seed-*/test-summary.json`. Copies are in `data/frozen/`.

## Scientific scope

The film describes measured results under this project's recipes. The CIFAR
ResNet-18 result is the strongest measured here. It is not a claim about the best
ResNet in existence: wider/deeper variants and other recipes do better on CIFAR
(e.g. arXiv:1605.07146). ResNet-18 also trained 200 epochs vs 160, so the final
comparison is a bundled recipe comparison, not an isolated architecture ablation.

## Verification of the delivered MP4

`tools/verify_render.py out/cifar-journey.mp4` → all PASS: H.264 1920×1080 at
30 fps; 598.55 s vs timeline 598.53 s; AAC narration at mean −23.5 dB (peak
−5.0 dB); no silence ≥ 4 s; a frame decodes from every chapter. Rendered on
CPU (swiftshader, concurrency 4) in 13 minutes. The GPU training queue was only
read, never modified.

Independent final check: the complete delivered MP4, including both video and audio, decoded through FFmpeg with exit code 0 and no errors.
