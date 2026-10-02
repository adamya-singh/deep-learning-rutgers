"""Select a pretrained-model ensemble on validation, then assess its frozen test predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from record_push import ROOT, save_json


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplex(count, units=20):
    if count == 1:
        yield (units,)
    else:
        for first in range(units + 1):
            for tail in simplex(count - 1, units - first):
                yield (first,) + tail


def centered(logits):
    return logits - logits.mean(1, keepdims=True)


def transform(logits, scale, kind):
    z = centered(logits.astype(np.float64)) / scale
    if kind == 'probabilities':
        z -= z.max(1, keepdims=True)
        z = np.exp(z)
        z /= z.sum(1, keepdims=True)
    return z


def run(name, member_names):
    paths = [ROOT / member for member in member_names]
    assert len(set(paths)) == len(paths) and 2 <= len(paths) <= 4
    out = ROOT / 'ensembles' / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'test-summary.json').exists():
        summary = json.loads((out / 'test-summary.json').read_text())
        assert summary['members'] == member_names
        assert [fingerprint(p / 'final-head.pt') for p in paths] == summary['final_head_sha256']
        print(summary, flush=True)
        return
    labels = [np.load(path / 'validation-labels.npy') for path in paths]
    indices = [np.load(path / 'validation-indices.npy') for path in paths]
    assert all(np.array_equal(labels[0], y) for y in labels[1:])
    assert all(np.array_equal(indices[0], x) for x in indices[1:])
    assert len(labels[0]) == 5000 and len(np.unique(indices[0])) == 5000
    validation = [np.load(path / 'validation-logits.npy') for path in paths]
    assert all(z.shape == (5000, 10) and np.isfinite(z).all() for z in validation)
    scales = [float(np.sqrt(np.mean(centered(z.astype(np.float64)) ** 2))) for z in validation]
    assert all(scale > 0 for scale in scales)
    best_key, best = None, None
    grid = list(simplex(len(paths)))
    for kind in ('centered-logits', 'probabilities'):
        candidates = [transform(z, scale, kind) for z, scale in zip(validation, scales)]
        for integers in grid:
            weights = np.asarray(integers, dtype=np.float64) / 20
            blended = sum(w * z for w, z in zip(weights, candidates))
            correct = int(np.sum(blended.argmax(1) == labels[0]))
            # Prefer an even mixture among equally scoring candidates, then logits.
            key = (correct, -float(np.sum((weights - 1 / len(paths)) ** 2)), kind == 'centered-logits')
            if best_key is None or key > best_key:
                best_key = key
                best = {'kind': kind, 'weights': weights.tolist(), 'validation_correct': correct,
                        'validation_total': 5000, 'validation_accuracy': correct / 50}
    config = {'members': member_names, 'selection': best, 'scales': scales,
              'validation_indices_sha256': fingerprint(paths[0] / 'validation-indices.npy'),
              'validation_head_sha256': [fingerprint(p / 'head.pt') for p in paths],
              'final_head_sha256': [fingerprint(p / 'final-head.pt') for p in paths],
              'selection_policy': 'Validation only, 0.05 weight grid, ties prefer even mixtures then centered logits',
              'track': 'outside pretraining / frozen backbones / validation-selected ensemble'}
    save_json(out / 'selection.json', config)
    print('ENSEMBLE SELECTION', config, flush=True)
    # Only after selection is frozen do we read test predictions or test labels.
    test_labels = [np.load(p / 'test-labels.npy') for p in paths]
    assert all(np.array_equal(test_labels[0], y) for y in test_labels[1:])
    assert len(test_labels[0]) == 10000
    tests = [np.load(p / 'test-logits.npy') for p in paths]
    assert all(z.shape == (10000, 10) and np.isfinite(z).all() for z in tests)
    logits = sum(w * transform(z, scale, best['kind']) for w, z, scale in zip(best['weights'], tests, scales))
    pred = logits.argmax(1)
    correct = int(np.sum(pred == test_labels[0]))
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (test_labels[0], pred), 1)
    np.save(out / 'test-logits.npy', logits)
    np.save(out / 'test-labels.npy', test_labels[0])
    summary = {**config, 'correct': correct, 'total': 10000, 'errors': 10000 - correct,
               'accuracy': correct / 100, 'matrix': matrix.tolist(), 'selection_sha256': fingerprint(out / 'selection.json')}
    save_json(out / 'test-summary.json', summary)
    print('ENSEMBLE TEST', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--members', nargs='+', required=True)
    args = parser.parse_args()
    run(args.name, args.members)
