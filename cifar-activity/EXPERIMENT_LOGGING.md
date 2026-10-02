# Experiment log maintenance

Use `summarize-experiment-as-adamya` for every completed training run, following
its complete voice corpus and approved entry format. This applies to each seed,
including controls. Exclude smoke/benchmark runs and incomplete training jobs.

The existing training code is frozen while queues run. The user cancelled recurring maintenance. Update remaining notes in one batch
after both queues complete, or when explicitly requested. No scheduled logging
task remains.

```bash
.venv/bin/python experiment_log.py pending
.venv/bin/python experiment_log.py facts --key depth/plain/conv-8/seed-2
.venv/bin/python experiment_log.py record --key depth/plain/conv-8/seed-2 \
  --entry notes/research-log/57-depth-plain-conv-8-seed-2.md
.venv/bin/python experiment_log.py index
```

`pending` returns only completed artifacts that lack a current note/figure
registration, including stale facts or missing exports. `facts` provides the
verified summary and matching control. For new entries, use the next available
number and the key with slashes replaced by hyphens, such as
`58-depth-plain-conv-16-seed-0.md`. Write the note with the skill; this helper
does not generate its prose. Put `<!-- experiment-key: KEY -->` in the note so
an unfinished registration can be recovered safely.

`record` checks the required skill sections, generates PNG/SVG/PDF diagrams,
adds the figure and download links, and records the source-fact fingerprint in
`notes/research-log/entries.json`. It protects registry writes with a lock.
The Markdown index refreshes after registration.

Figures use the existing bottom-to-top, serif, pastel rounded-block layout:

- Augmentation highlights the training-view pipeline; the network stays fixed.
- Depth highlights repeated convolution blocks while keeping widths/pooling.
- Residual runs highlight shortcut sums and zero-padded channel transitions.
- ResNet-18 highlights its wider four-stage architecture, projection shortcuts,
  and global-average-pooling classifier.
- Historical throughput runs highlight batch size or training duration and
  explicitly show that the architecture stayed unchanged.

The first 14 reviewed notes retain their prose and now include figures. New
entries cite their exact source run and metrics. Test accuracy from the old
suite is not directly comparable to validation score in the depth/follow-up
suites. Validation scores average the final ten epochs; training accuracy and
final validation accuracy refer to the final epoch and must be labeled separately.

Logging never changes training settings, commits, or publishes drafts.
Final test results are recorded from saved test-summary.json files only after
validation selection. Do not infer missing test curves or evaluate more models.
