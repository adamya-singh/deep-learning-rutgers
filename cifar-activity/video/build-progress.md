# Build progress: CIFAR journey film

Author/implementer: Claude Code (Claude Opus 5.5). All work stays inside `video/`.
GPU queue is untouched (read-only checks of `../runs/followup/*` only). Video tooling is CPU-only, 2–4 threads.

## Milestones
- [x] 2026-10-02 00:08 UTC: Read brief, full voice reference, evidence, teaching scaffold, sources, source code and run artifacts.
- [x] Frozen film data: `tools/extract_static.py` (parent .venv, CPU, 2 threads, 316 s) → real CIFAR samples, real conv1 filters + conv1/conv2 activations from `../runs/baseline/last.pt`, real multiply-sum (0.6256, matches model), real 25×25 loss slice. `tools/extract_results.py` → `public/data/results.json` + verbatim copies in `data/frozen/`.
- [x] Narration script `tools/build_script.py` → `script/narration.json`, `script/script.md` (storyboard + source paths).
- [x] Applied every correction in REVIEW_NOTES.md (overfitting/degradation wording + ResNet-paper distinction; gap can reflect underfitting; logits vs softmax in PyTorch CE; illustrative loss-slice path; recipe comparison not an isolated ablation; test set not pristine).
- [x] Trimmed 2,080 → ~1,446 words. edge-tts en-US-AndrewNeural (+15%), clip completeness checks (one truncated stream caught and re-synthesized), balanced caption chunks → `src/data/timeline.json`, `out/cifar-journey.srt/.vtt`. ≈9.9 min.
- [x] Remotion project (11 scenes, word-synced cues), `npx tsc --noEmit` clean.
- [x] Preview stills: all 60 segment frames + 21 mid-animation frames rendered and inspected; fixed narrow multiply-sum cells, wrapped labels, chart overflow (clip paths), label/line collisions, lesson wrapping, edge-clipped tags, End overlap. Contact sheets in `out/contact-sheet-*.jpg`.
- [x] FINAL_RENDER_CHECKLIST.md read; schematic tags on residual vector + gradient pictures; ReLU/pool numbers tagged illustrative; normalization shows raw/255 first.
- [x] Motion QA clips `out/qa/qa-{conv,backprop-loss,residual,residual-grad,resnet-build}.mp4` (CPU, concurrency 4, ~12 fps). Sampled-frame tiles checked: window travels to (14,15), then sweeps while the real map fills row by row; forward particles then backward pulses; loss slice draws in; residual vector + gradient routes tagged schematic; ResNet stages, resolution labels and projection shortcuts land on their spoken words.
- [x] Narrated trailer `out/cifar-journey-trailer.mp4` (84.7 s, 6 segment-aligned excerpts, 1080p30 H.264/AAC, mean −23.6 dB). Built by `tools/make_trailer.py`.
- [x] 2026-10-02 01:27 UTC: follow-up queue `phase: completed` (read-only watcher, 60 s polls). Selected by validation: `resnet18-crop-flip`, 3-seed final-10 validation 95.77 ± 0.11; final test 95.19 / 95.21 / 95.22 → 95.21 ± 0.02, cross-checked against each `test-summary.json` (frozen in `data/frozen/`). Fixed extractor for the runner's `test_results` format; End cue made independent of status wording.
- [x] Regenerated only changed narration (resnet-7, end-1, end-2). Film is 598.5 s (9.98 min). Typecheck clean. Re-rendered and inspected all ResNet/End stills: 3/3 seeds, 95.77 ± 0.11 VALIDATION, selected winner + TEST 95.21 ± 0.02 labelled.
- [x] Full render `out/cifar-journey.mp4`: 1080p30 H.264 + AAC, 62.4 MB, CPU (swiftshader, concurrency 4), 13 min wall time. `tools/verify_render.py` passes all checks: codec, 1920×1080, 30 fps, duration 598.55 s vs timeline 598.53 s, narration mean −23.5 dB / peak −5.0 dB, no silence ≥ 4 s, every chapter decodes (`out/mp4-contact-sheet.jpg`). Captions contain the final numbers.
- [x] README with deliverables, commands, refresh steps, frozen results table, real-vs-schematic notes and scope. sources.md lists the local evidence.
- [x] Parent repo status unchanged (identical to session start); training queue never touched.

Note: `out/contact-sheet-*.jpg` were made from the pre-refresh timeline (only ResNet/End differed; those were re-rendered and rechecked). `out/mp4-contact-sheet.jpg` is decoded from the final MP4. The trailer's six excerpts contain no numbers that changed in the refresh.
