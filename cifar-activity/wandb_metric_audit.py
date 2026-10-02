"""Audit shared W&B metrics and repair completed-run summaries without touching training.

No GPU, dataset access, checkpoint migration, or second writer on active SDK runs.
The watcher also repairs summaries after future jobs and final assessments finish.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import wandb

HERE = Path(__file__).resolve().parent
PROJECT = '7adamyasingh-rutgers-university/cifar-activity'
VERSION = 1
CORE = ('train/loss', 'train/eval_loss', 'train/eval_accuracy',
        'performance/optimizer_steps', 'performance/train_images_per_second',
        'performance/train_seconds', 'performance/eval_seconds',
        'performance/elapsed_seconds')
FINAL = {'train/eval_accuracy': 'final/train_accuracy',
         'train/eval_loss': 'final/train_loss',
         'eval/validation_accuracy': 'final/validation_accuracy',
         'eval/validation_loss': 'final/validation_loss',
         'eval/test_accuracy': 'final/test_accuracy',
         'eval/test_loss': 'final/test_loss',
         'performance/elapsed_seconds': 'final/elapsed_seconds',
         'performance/optimizer_steps': 'final/optimizer_steps'}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def final_values(summary):
    # Existing final evaluations take precedence over the last training-history value.
    return {dest: summary[source] for source, dest in FINAL.items()
            if numeric(summary.get(source)) and not numeric(summary.get(dest))}


def classify(run):
    if run.group == 'benchmark':
        return 'benchmark'
    return 'validation-selection' if run.group in (
        'plain-depth', 'residual-depth', 'winner-augmentation', 'resnet18-cifar') else 'legacy-test-monitoring'


def backfill_rows(history, config, summary, kind):
    """Recover known constant LR; expose one-time test results at the final epoch.

    Never infer validation/test values, scheduled LR, or hardware telemetry.
    Replay only missing metrics with their original epoch and timing context.
    """
    epochs = {}
    for row in history:
        if numeric(row.get('epoch')):
            epochs.setdefault(int(row['epoch']), {}).update(row)
    epochs = {epoch: row for epoch, row in epochs.items() if numeric(row.get('train/eval_loss'))}
    output = {}
    if kind == 'legacy-test-monitoring' and numeric(config.get('lr')):
        for epoch, row in sorted(epochs.items()):
            if not numeric(row.get('optimizer/lr')):
                output[epoch] = {k: row[k] for k in ('epoch', 'performance/optimizer_steps',
                                                   'performance/elapsed_seconds') if k in row}
                output[epoch]['optimizer/lr'] = config['lr']
    if (kind == 'validation-selection' and epochs and
            numeric(summary.get('final/test_accuracy')) and numeric(summary.get('final/test_loss')) and
            not any(numeric(row.get('eval/test_accuracy')) for row in history)):
        epoch = max(epochs)
        row = epochs[epoch]
        output.setdefault(epoch, {k: row[k] for k in ('epoch', 'performance/optimizer_steps',
                                'performance/elapsed_seconds') if k in row})
        output[epoch].update({'eval/test_accuracy': summary['final/test_accuracy'],
                              'eval/test_loss': summary['final/test_loss']})
    return [output[epoch] for epoch in sorted(output)]


def backfill_run(public_run, kind):
    summary = dict(public_run.summary)
    missing_lr = kind == 'legacy-test-monitoring' and (
        not numeric(summary.get('optimizer/lr')) or summary.get('logging/lr_backfill_in_progress'))
    missing_test_point = (kind == 'validation-selection' and
                          numeric(summary.get('final/test_accuracy')) and
                          not numeric(summary.get('eval/test_accuracy')))
    if not (missing_lr or missing_test_point):
        return
    history = list(public_run.scan_history())
    rows = backfill_rows(history, public_run.config, summary, kind)
    if not rows:
        return
    source_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    with wandb.init(entity=public_run.entity, project=public_run.project, id=public_run.id,
                    resume='must', settings=wandb.Settings(quiet=True)) as run:
        run.define_metric('epoch')
        run.define_metric('optimizer/*', step_metric='epoch')
        run.define_metric('eval/*', step_metric='epoch')
        if missing_lr:
            run.summary['logging/lr_backfill_in_progress'] = True
        for row in rows:
            run.log(row)
        if missing_lr:
            run.summary['logging/lr_curve_source'] = 'Saved constant optimizer LR and original epoch history'
            run.summary['logging/lr_backfill_in_progress'] = False
        if missing_test_point:
            run.summary['logging/test_curve_kind'] = 'Final checkpoint only; no historical test curve'
        run.summary['logging/backfill_source_sha256'] = source_hash
    # Re-read from the server on the next sweep; don't trust this stale Run object.
    atomic_json(HERE / 'runs/wandb-metric-backfills' / f'{public_run.id}.json',
                {'rows': rows, 'source_sha256': source_hash, 'kind': kind})
    print(f'Backfilled {public_run.id}: {len(rows)} metric rows.', flush=True)
    return True


def inspect_run(run, repair=False):
    summary = dict(run.summary)
    kind = classify(run)
    keys = [*CORE, 'optimizer/lr']
    if kind == 'validation-selection':
        keys += ['eval/validation_accuracy', 'eval/validation_loss']
    elif kind != 'benchmark':
        keys += ['eval/test_accuracy', 'eval/test_loss']
    missing = [k for k in keys if not numeric(summary.get(k))] if kind != 'benchmark' else []
    if kind != 'benchmark' and not any(numeric(summary.get(k)) for k in
            ('system/peak_gpu_memory_mb', 'performance/peak_gpu_memory_mb')):
        missing.append('peak_gpu_memory_mb (either namespace)')
    if kind != 'benchmark' and not any(numeric(summary.get(k)) for k in
            ('train/batch_accuracy', 'train/mixed_target_accuracy')):
        missing.append('training batch accuracy (ordinary or mixed target)')
    has_test = numeric(summary.get('eval/test_accuracy')) or numeric(summary.get('final/test_accuracy'))
    row = {'id': run.id, 'name': run.name, 'state': run.state, 'group': run.group,
           'kind': kind, 'epoch': summary.get('last_epoch', summary.get('epoch')),
           'missing_required_metrics': missing,
           'test_status': ('not-applicable' if kind == 'benchmark' else
                           'evaluated' if has_test else 'held-out-until-final-selection'),
           'validation_status': 'measured' if numeric(summary.get('eval/validation_accuracy')) else 'not-collected',
           'gpu_telemetry_status': 'measured' if numeric(summary.get('system/gpu_utilization_pct')) else 'not-collected',
           'checkpoint_timing_status': 'measured' if numeric(summary.get('performance/checkpoint_seconds')) else 'not-collected',
           'url': run.url}
    if repair and run.state == 'finished' and kind != 'benchmark':
        changes = final_values(summary)
        metadata = {'logging/metric_schema_version': VERSION,
                    'logging/test_status': row['test_status'],
                    'logging/validation_status': row['validation_status']}
        changes.update({k: v for k, v in metadata.items() if summary.get(k) != v})
        if changes:
            # Public API updates summary values only; it does not resume a run,
            # append duplicate epochs, or change its final model/selection score.
            run.summary.update(changes)
            row['summary_fields_repaired'] = list(changes)
    return row


def verify_history(run, row):
    if row['kind'] == 'benchmark':
        return
    history = list(run.scan_history())
    epochs = {}
    for point in history:
        if numeric(point.get('epoch')):
            epochs.setdefault(int(point['epoch']), {}).update(point)
    epochs = {epoch: point for epoch, point in epochs.items() if numeric(point.get('train/loss'))}
    required = [*CORE, 'optimizer/lr']
    if row['kind'] == 'validation-selection':
        required += ['eval/validation_accuracy', 'eval/validation_loss']
    else:
        required += ['eval/test_accuracy', 'eval/test_loss']
    holes = {str(epoch): [key for key in required if not numeric(point.get(key))]
             for epoch, point in sorted(epochs.items())}
    holes = {epoch: missing for epoch, missing in holes.items() if missing}
    budget = run.config.get('epochs', run.config.get('manifest', {}).get('epochs'))
    missing_epochs = sorted(set(range(1, budget + 1)) - set(epochs)) if (
        row['state'] == 'finished' and isinstance(budget, int)) else []
    row.update(history_epoch_count=len(epochs), history_missing_metrics=holes,
               history_missing_epochs=missing_epochs)
    if holes or missing_epochs:
        row['missing_required_metrics'].append('epoch history incomplete; see history_missing_*')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sweep(repair=False, backfill=False, history=False):
    api = wandb.Api(timeout=60)
    rows = []
    for run in api.runs(PROJECT):
        if backfill and run.state == 'finished' and backfill_run(run, classify(run)):
            run = wandb.Api(timeout=60).run(f'{PROJECT}/{run.id}')
        row = inspect_run(run, repair)
        if history:
            verify_history(run, row)
        rows.append(row)
    if repair:
        from organize_wandb import ensure_layout
        ensure_layout()
    report = {'schema_version': VERSION, 'checked_at_unix': time.time(), 'runs': rows,
              'run_count': len(rows), 'missing_metrics': [r['id'] for r in rows if r['missing_required_metrics']]}
    atomic_json(HERE / 'runs/wandb-metric-audit.json', report)
    repaired = sum(bool(r.get('summary_fields_repaired')) for r in rows)
    print(f"Audited {len(rows)} runs; repaired {repaired} summaries; "
          f"{len(report['missing_metrics'])} runs missing required metrics.", flush=True)
    return report


def queues_done():
    paths = [HERE / 'runs' / root / 'queue-state.json' for root in ('depth', 'followup')]
    return all(p.exists() and json.loads(p.read_text()).get('phase') == 'completed' for p in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--backfill', action='store_true', help='Recover missing LR curves and final test points')
    parser.add_argument('--verify-history', action='store_true', help='Check required metrics at every recorded epoch')
    parser.add_argument('--interval', type=int, default=120)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('interval must be at least 10 seconds')
    root = HERE / 'runs'
    root.mkdir(exist_ok=True)
    with (root / 'wandb-metric-audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = sweep(args.repair, args.backfill, args.verify_history)
                if not args.watch or queues_done():
                    if result['missing_metrics']:
                        raise RuntimeError('Required metrics missing; inspect runs/wandb-metric-audit.json')
                    return
            except Exception as error:
                if not args.watch:
                    raise
                print(f'Audit failed; will retry: {type(error).__name__}: {error}', flush=True)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
