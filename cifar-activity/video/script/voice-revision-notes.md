# Narration revision notes

Revision of the film's narration, 2026-10-02, by Claude Code (Claude Opus 5.5).

## Workflow, in order

1. **Content plan first.** Before any narration was drafted, I read `sources.md`,
   `public/data/results.json`, `public/data/static.json`, the previous
   `script/script.md`, all eleven scene files, `REVIEW_NOTES.md`,
   `experiment-evidence.md` and the relevant parts of `../depth_cnn.py`,
   `../cifar_cnn.py` and `../augmentations.py`. The plan was saved as
   `script/teaching-plan.md` (file time 02:29 EDT). It covers what a viewer has to
   understand, the reason behind each experiment, what the data shows and can't
   show, and the cue words each scene depends on.
2. **Skill read after the plan.** Next I read
   `~/.agents/skills/write-as-adamya/SKILL.md`, then the skill's complete
   `references/style-prompt.txt`. It's a single 27,190-character line, the Read
   returned all of it, and I confirmed nothing was cut off by checking its first
   and last bytes against the file on disk. Drafting was done directly with the
   current model, as the skill says. No ChatGPT Web, no substitute reference, no
   summary. None of the corpus is copied into the repository, and no biography
   appears in the film.
3. **Drafted from the plan.** The narration is written straight into
   `tools/build_script.py`. Every number comes from the frozen JSON through
   f-strings.
4. **Length.** The first natural draft ran 15.3 minutes. To bring it down I cut
   repetition (numbers spoken twice, restated lines) rather than squeezing it back
   into clipped sentences. The final narration is 2,530 words and runs 14.6
   minutes.
5. **Review before returning.** I checked the draft against the skill's review
   list (answers every part of the task, natural first person, no added facts, no
   meta-commentary) and against the brief's banned patterns. Two lines failed and
   were revised once:
   - end-3 said "anywhere near state of the art", which goes beyond the sources.
     It now says "isn't state of the art".
   - depth-5 called residuals "the obvious next thing to try". It now says "a
     natural next thing to try".
6. **Factual corrections from review:**
   - aug1-2 no longer claims the network never sees the same image twice.
   - batch-2 presents fewer updates as a hypothesis, which batch-3 then checks,
     noting that the matched run also saw 8× as many images.
   - resnet-3 now says stage 1 stays at 32×32 and the three later stages each
     halve the resolution.

## Files updated

- `tools/build_script.py`: new narration (single source), new chapter titles,
  storyboard notes, and partial-results branches rewritten in the same voice.
  Regenerating writes the revised prose into `script/narration.json` and
  `script/script.md`.
- `script/teaching-plan.md` (new) and this file.
- Scenes, with cue words re-keyed to the new wording:
  - `Task` (third spoken "32"), `Batch` ("eight", GPU-utilization label),
    `Recipe` ("warms"), `Depth` ("dropped", "fit", "Residual"),
    `Residual` ("mattered").
  - `Aug2` ("improved", "guess", "Best guess" caption).
  - `ResNet` (table rows "shortcuts", "global").
  - `End` ("leaderboard", reworded takeaways, a plain scope line, and a final card
    showing 63.68% → 95.21% test with "contributions not separated").
- `tools/tts.py`: pronunciation overrides (CIFAR → "SIFAR ten") are now timed by
  aligning the displayed tokens with the real spoken word boundaries, instead of
  guessing proportionally.
- `tools/make_trailer.py`: excerpts are defined by segment IDs, and `--plan` prints
  the frame ranges from the current timeline.

## Before → after examples

| Segment | Before | After |
| --- | --- | --- |
| task-3 | "Zoom in and the frog disappears. … 32 by 32 by 3. 3,072 numbers." | "If you zoom in on the frog, it's really just a 32 by 32 grid of pixels, and each pixel is three numbers between 0 and 255 for how much red, green, and blue it has. So every image is 32 times 32 times 3, which is 3,072 numbers." |
| batch-2 | "But accuracy fell to 48.58 percent. Turns out bigger batches mean fewer steps." | "The catch was that test accuracy dropped to 48.58 percent. With the same number of epochs, a batch eight times bigger also means eight times fewer weight updates, 980 instead of 7,820 at the same learning rate, so our guess was that the model just hadn't taken enough steps." |
| aug1-4 | "The subtle part. Crop plus flip shrank the train-test gap …" | "Then there's the gap between training and test accuracy, which went from about 7 points without augmentation to under 1 with crop plus flip. A smaller gap can mean useful regularization, or it can just mean the model learned less …" |
| aug2-1 | "Now the twist. Crop plus flip, which hurt the tiny network, on residual 32." | "Since crop plus flip hurt the small network, we tried it again on residual 32, keeping the recipe, the seeds, and the starting weights the same, and only changing the augmentation." |
| end-4 | "The lessons generalize. Count optimizer steps, not just epochs. …" | "If we did this again, we'd keep comparing runs by optimizer steps and not just epochs, deciding on validation across seeds, …" |
| end-5 | "No single trick did this. It was the design and the recipe, working together." | "Going from 63.68 to 95.21 percent on test took a lot of changes stacked together, and these experiments don't split up exactly how much each one contributed." |

Chapter titles that read as slogans were renamed. For example, "Faster isn't
better" became "Bigger batches" and "What we can actually claim" became "What the
results support".

## Delivered

The full film `out/cifar-journey.mp4` (874 s) was rendered with this narration.
It passed `tools/verify_render.py` and a full FFmpeg decode before it replaced the
previous movie. Its audio was checked against the revised narration clips
(envelope correlation 0.97–0.99, compared with 0.06–0.29 for the old clips). The
captions in `out/cifar-journey.srt` and `.vtt` and the trailer were regenerated
from the same timeline.
