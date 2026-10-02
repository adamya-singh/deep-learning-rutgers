"""Validation-selected Gaussian landmark classifier on frozen CIFAR embeddings."""
import argparse
import fcntl
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from record_push import HERE, ROOT, save_json


def normalize(x, mean, scale):
    return F.normalize((x - mean) / scale, dim=1)


def run(name, member):
    torch.set_num_threads(4)
    source = ROOT / member
    out = ROOT / 'kernels' / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'test-summary.json').exists():
        print((out / 'test-summary.json').read_text(), flush=True)
        return
    features = np.load(source / 'train.npy', mmap_mode='r')
    labels = np.load(source / 'train-labels.npy')
    split = json.loads((HERE / 'runs/followup/split.json').read_text())['indices']
    train, valid = split['train'], split['validation']
    assert len(train) == 45000 and len(valid) == 5000 and not set(train) & set(valid)
    # Use the final class token, rather than concatenating increasingly correlated layers.
    x = torch.from_numpy(np.array(features[:, 3], dtype=np.float32)).cuda()
    mean, scale = x[train].mean(0), x[train].std(0).clamp_min(.01)
    x = normalize(x, mean, scale)
    rng = np.random.default_rng(2026)
    landmark_ids = np.concatenate([rng.choice(np.asarray(train)[labels[train] == cls], 205, replace=False)
                                   for cls in range(10)])
    assert set(landmark_ids).issubset(train)
    centers = x[landmark_ids]
    distance = (2 - 2 * x @ centers.T).clamp_min(0)
    base = torch.from_numpy(np.load(source / 'validation-logits.npy')).cuda()
    base = base - base.mean(1, keepdim=True)
    base_scale = base.square().mean().sqrt()
    base /= base_scale
    target = F.one_hot(torch.from_numpy(labels[train]).cuda(), 10).double()
    valid_labels = torch.from_numpy(labels[valid]).cuda()
    best_key, chosen = None, None
    rows = []
    for gamma in (1., 5., 10., 20., 50.):
        phi = torch.exp(-gamma * distance)
        phi = torch.cat([phi, torch.ones(len(phi), 1, device='cuda')], 1)
        z = phi[train].double()
        gram, rhs = z.T @ z, z.T @ target
        del z
        for alpha in (.1, 1., 10., 100., 1000.):
            w = torch.linalg.solve(gram + alpha * torch.eye(gram.shape[0], device='cuda', dtype=torch.float64), rhs).float()
            scores = phi[valid] @ w
            scores -= scores.mean(1, keepdim=True)
            kernel_scale = scores.square().mean().sqrt()
            scores /= kernel_scale
            for blend in (0., .1, .25, .5, .75, 1.):
                correct = int((((1 - blend) * base + blend * scores).argmax(1) == valid_labels).sum())
                row = {'gamma': gamma, 'alpha': alpha, 'blend': blend,
                       'validation_correct': correct, 'validation_accuracy': correct / 50,
                       'base_scale': float(base_scale), 'kernel_scale': float(kernel_scale)}
                rows.append(row)
                # Prefer less reliance on the new kernel when validation is tied.
                key = (correct, -blend, alpha, -gamma)
                if best_key is None or key > best_key:
                    best_key, chosen = key, row
        print('KERNEL VALIDATION', name, 'gamma', gamma, 'best', chosen, flush=True)
        save_json(out / 'validation-results.json', rows)
    config = {'source': member, 'selection': chosen, 'landmark_ids': landmark_ids.tolist(),
              'landmark_selection': '205 training-only examples per class, fixed seed 2026',
              'feature': 'normalized last class token, training-only feature standardization',
              'source_head_sha256': hashlib.sha256((source / 'final-head.pt').read_bytes()).hexdigest(),
              'track': 'outside pretraining / frozen backbone / validation-selected kernel blend'}
    save_json(out / 'selection.json', config)
    # Refit only the frozen selected kernel configuration on all 50k labels.
    phi = torch.exp(-chosen['gamma'] * distance)
    phi = torch.cat([phi, torch.ones(len(phi), 1, device='cuda')], 1)
    z = phi.double()
    gram = z.T @ z
    rhs = z.T @ F.one_hot(torch.from_numpy(labels).cuda(), 10).double()
    w = torch.linalg.solve(gram + chosen['alpha'] * torch.eye(gram.shape[0], device='cuda', dtype=torch.float64), rhs).float()
    head_path = out / 'final-head.pt'
    torch.save({'mean': mean.cpu(), 'scale': scale.cpu(), 'centers': centers.cpu(), 'w': w.cpu(),
                'config': config, 'train_images': 50000}, head_path)
    del phi, z, gram, rhs, distance, x
    test_features = np.load(source / 'test.npy', mmap_mode='r')
    test_x = torch.from_numpy(np.array(test_features[:, 3], dtype=np.float32)).cuda()
    test_x = normalize(test_x, mean, scale)
    phi = torch.exp(-chosen['gamma'] * (2 - 2 * test_x @ centers.T).clamp_min(0))
    phi = torch.cat([phi, torch.ones(len(phi), 1, device='cuda')], 1)
    kernel = phi @ w
    kernel -= kernel.mean(1, keepdim=True)
    kernel /= chosen['kernel_scale']
    base_test = torch.from_numpy(np.load(source / 'test-logits.npy')).cuda()
    base_test -= base_test.mean(1, keepdim=True)
    base_test /= chosen['base_scale']
    logits = ((1 - chosen['blend']) * base_test + chosen['blend'] * kernel).cpu().numpy()
    test_labels = np.load(source / 'test-labels.npy')
    correct = int(np.sum(logits.argmax(1) == test_labels))
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (test_labels, logits.argmax(1)), 1)
    np.save(out / 'test-logits.npy', logits)
    np.save(out / 'test-labels.npy', test_labels)
    result = {**config, 'kernel': True, 'accuracy': correct / 100, 'correct': correct, 'total': 10000,
              'errors': 10000 - correct, 'matrix': matrix.tolist(),
              'head_sha256': hashlib.sha256(head_path.read_bytes()).hexdigest()}
    save_json(out / 'test-summary.json', result)
    print('KERNEL TEST', name, result['accuracy'], result['errors'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--member', required=True)
    args = parser.parse_args()
    with open(HERE / 'runs/depth-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.name, args.member)
