"""Render completed, saved CIFAR-10 record-push results without new evaluation."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from record_push import HERE, ROOT


def report():
    rows = []
    for path in ROOT.rglob('test-summary.json'):
        result = json.loads(path.read_text())
        audit_path = path.parent / 'verification.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        rows.append((path, result, audit.get('passed', False)))
    rows.sort(key=lambda row: (-row[1]['accuracy'], str(row[0])))
    verified = [row for row in rows if row[2]]
    local_now = datetime.now(ZoneInfo('America/New_York'))
    finished = local_now >= datetime(2026, 10, 2, 9, tzinfo=ZoneInfo('America/New_York'))
    now = local_now.strftime('%Y-%m-%d %I:%M %p Eastern')
    lines = ['# CIFAR-10 accuracy push', '',
             f"Updated {now}. Work window {'ended' if finished else 'ends'} October 2, 2026 at 9am Eastern.", '',
             'These runs use outside pretraining. Frozen-feature experiments keep the backbone fixed; '
             'fine-tuning experiments adapt its last four blocks and classifier. Classifier features, '
             'regularization, ensemble weights, and fine-tuning duration are selected on the existing '
             '45,000/5,000 training/validation split. The selected recipe is then fitted on all 50,000 '
             'training images and assessed on the official '
             '10,000-image test set. Test labels never fit a classifier or choose ensemble weights.', '']
    if verified:
        best = verified[0][1]
        lines += [f"Best verified test accuracy: **{best['accuracy']:.2f}%**, "
                  f"with **{best['errors']} errors out of {best['total']:,}**. "
                  + ('The 99% target was reached; the experiment window is complete.' if finished else
                     'The first target of 99% has been reached; work continues to maximize the result within the time window.'), '']
    lines += ['| Saved experiment | Validation (%) | Test (%) | Test errors | Verification |',
              '|---|---:|---:|---:|---|']
    for path, result, passed in rows:
        method = ('GPU reload + repeat' if path.parent.parent.name == 'finetunes' else 'CPU reconstruction')
        lines.append(f"| [{path.parent.name}]({path}) | {result['selection']['validation_accuracy']:.2f} | "
                     f"{result['accuracy']:.2f} | {result['errors']} | {method if passed else 'Pending'} |")
    lines += ['', 'Previous best from-scratch result: 95.22% test accuracy with CIFAR ResNet-18. '
              'The pretrained results above belong to a different training track.', '',
              'Published comparison: EfficientNet-L2 + SAM reports 99.70% CIFAR-10 accuracy '
              '(0.30% error, five-run mean with a 0.01 percentage-point 95% confidence interval) '
              'using outside pretraining, in [SAM Table 3](https://arxiv.org/html/2010.01412v3). '
              'DINOv3 7B reports 99.6% with linear probing in '
              '[DINOv3 Table 22](https://arxiv.org/html/2508.10104v1). '
              'These references do not establish an exhaustive current world record.', '',
              'Each experiment directory preserves cached features, the validation-selected head, the refitted '
              'final head, test logits, confusion matrix, and checkpoint fingerprints. Independent verification '
              'recomputes frozen-feature predictions on CPU and checks official labels, counts, and fingerprints. '
              'Fine-tuned models are reloaded before test inference and assessed again with a different batch size; '
              'this does not constitute an independent CPU reconstruction of the backbone.', '',
              'Recompute an individual result:', '', '```bash',
              f'cd {HERE}', '.venv/bin/python record_verify.py dinov2_vitg14-224-original', '```', '',
              'The run manifest and live queue log are under `runs/record-push`. '
              'Scores shown here are completed assessments, with one deterministic head fit per selected recipe; '
              'no multi-seed uncertainty estimate has been measured for these pretrained runs.', '']
    (HERE / 'RECORD_PUSH.md').write_text('\n'.join(lines))
    print(f'Reported {len(rows)} completed assessments; {len(verified)} independently verified')


if __name__ == '__main__':
    report()
