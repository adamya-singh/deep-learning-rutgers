"""Recompute record-push predictions on CPU against official CIFAR-10 labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR10

from record_ensemble import transform
from record_push import HERE, ROOT, save_json


def predict(path):
    summary = json.loads((path / 'test-summary.json').read_text())
    if summary.get('kernel'):
        head_path = path / 'final-head.pt'
        assert hashlib.sha256(head_path.read_bytes()).hexdigest() == summary['head_sha256']
        head = torch.load(head_path, map_location='cpu', weights_only=True)
        config = json.loads((path / 'selection.json').read_text())
        assert head['config'] == config and head['train_images'] == 50000
        source = ROOT / config['source']
        assert hashlib.sha256((source / 'final-head.pt').read_bytes()).hexdigest() == config['source_head_sha256']
        base = predict(source)
        base = (base - base.mean(1, keepdims=True)) / config['selection']['base_scale']
        features = np.load(source / 'test.npy', mmap_mode='r')
        chosen = config['selection']
        chunks = []
        for i in range(0, 10000, 256):
            x = (np.array(features[i:i + 256, 3], dtype=np.float32) - head['mean'].numpy()) / head['scale'].numpy()
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            phi = np.exp(-chosen['gamma'] * np.maximum(2 - 2 * x @ head['centers'].numpy().T, 0))
            phi = np.concatenate([phi, np.ones((len(phi), 1), dtype=np.float32)], 1)
            z = phi @ head['w'].numpy()
            z = (z - z.mean(1, keepdims=True)) / chosen['kernel_scale']
            chunks.append((1 - chosen['blend']) * base[i:i + len(x)] + chosen['blend'] * z)
        logits = np.concatenate(chunks)
    elif 'members' in summary:
        selection = json.loads((path / 'selection.json').read_text())
        assert hashlib.sha256((path / 'selection.json').read_bytes()).hexdigest() == summary['selection_sha256']
        assert selection['selection'] == summary['selection']
        members = []
        for name, digest in zip(selection['members'], selection['final_head_sha256']):
            member = ROOT / name
            assert hashlib.sha256((member / 'final-head.pt').read_bytes()).hexdigest() == digest
            members.append(predict(member))
        chosen = selection['selection']
        logits = sum(w * transform(z, scale, chosen['kind']) for w, z, scale in
                     zip(chosen['weights'], members, selection['scales']))
    else:
        head_path = path / 'final-head.pt'
        assert hashlib.sha256(head_path.read_bytes()).hexdigest() == summary['head_sha256']
        head = torch.load(head_path, map_location='cpu', weights_only=True)
        assert head['selection'] == summary['selection'] and head['train_images'] == 50000
        features = np.load(path / 'test.npy', mmap_mode='r')
        chunks = []
        for i in range(0, 10000, 512):
            x = np.array(features[i:i + 512, head['indices']], dtype=np.float32).reshape(-1, len(head['mean']))
            x = (x - head['mean'].numpy()) / head['scale'].numpy()
            x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], 1)
            chunks.append(x @ head['w'].numpy())
        logits = np.concatenate(chunks)
    stored = np.load(path / 'test-logits.npy')
    assert logits.shape == stored.shape == (10000, 10)
    assert np.isfinite(logits).all()
    assert np.array_equal(logits.argmax(1), stored.argmax(1)), 'CPU and saved predictions differ'
    official = np.array(CIFAR10(str(HERE / 'data'), train=False, download=False).targets)
    assert np.array_equal(official, np.load(path / 'test-labels.npy'))
    pred = logits.argmax(1)
    correct = int(np.sum(pred == official))
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (official, pred), 1)
    assert correct == summary['correct'] and summary['total'] == 10000
    assert matrix.tolist() == summary['matrix']
    save_json(path / 'verification.json', {
        'passed': True, 'independent_cpu_predictions_agree': True,
        'official_test_labels_agree': True, 'checkpoint_fingerprints_verified': True,
        'correct': correct, 'total': 10000, 'accuracy': correct / 100, 'errors': 10000 - correct,
        'max_logit_difference': float(np.max(np.abs(logits - stored))),
    })
    print(f'VERIFIED {path.name}: {correct / 100:.2f}% ({10000 - correct} errors)', flush=True)
    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        predict(ROOT / args.directory)
