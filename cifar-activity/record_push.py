"""Resumable pretrained CIFAR-10 feature extraction and validation-only head selection."""
import argparse
import fcntl
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/record-push'


def load_timm_streamed(name, checkpoint, device='cuda'):
    """Load converted timm weights without allocating the whole FP32 model in RAM."""
    import timm
    options = {'global_pool': 'token'} if 'dinov3' in name else {}
    with torch.device('meta'):
        model = timm.create_model(name, pretrained=False, num_classes=0,
                                  dynamic_img_size=True, dtype=torch.bfloat16, **options)
    model.to_empty(device=device)
    targets = model.state_dict()
    # safetensors' Torch reader maps the entire file with writable private pages;
    # Linux may reject a 27 GB mapping on this 16 GB host. Read each tensor's
    # documented safetensors byte interval instead, keeping host use bounded.
    dtypes = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16,
              'BF16': torch.bfloat16, 'I64': torch.int64, 'I32': torch.int32,
              'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool}
    with open(checkpoint, 'rb') as source:
        header_size = struct.unpack('<Q', source.read(8))[0]
        assert header_size < 100_000_000
        tensors = json.loads(source.read(header_size))
        tensors.pop('__metadata__', None)
        data_start = 8 + header_size
        file_size = Path(checkpoint).stat().st_size
        extra = set(tensors) - set(targets)
        # num_classes=0 deliberately omits the pretrained classification projection.
        assert extra.issubset({'head.weight', 'head.bias'})
        assert not set(targets) - set(tensors), sorted(set(targets) - set(tensors))
        with torch.no_grad():
            for key, target in targets.items():
                info = tensors[key]
                start, end = info['data_offsets']
                assert 0 <= start <= end <= file_size - data_start
                source.seek(data_start + start)
                raw = bytearray(end - start)
                assert source.readinto(raw) == len(raw), key
                value = torch.frombuffer(raw, dtype=dtypes[info['dtype']]).reshape(info['shape'])
                assert value.shape == target.shape, (key, value.shape, target.shape)
                target.copy_(value)
                del value, raw
    # RoPE periods are generated buffers, absent from safetensors. to_empty()
    # deliberately discards their meta values; regenerate them before inference.
    if getattr(model, 'rope', None) is not None:
        model.rope.reset_parameters()
    for key, buffer in model.named_buffers():
        if key not in targets and key.endswith('.k_bias'):
            buffer.zero_()
    assert not any(p.is_meta for p in model.parameters())
    assert all(torch.isfinite(b).all() for b in model.buffers())
    return model.eval()


def intermediate_features(model, x, name):
    if name.startswith('dinov2_'):
        layers = model.get_intermediate_layers(x, n=4, return_class_token=True)
        return torch.stack([v[1] for v in layers] + [layers[-1][0].mean(1)], dim=1)
    if 'dinov3' in name:
        layers = model.forward_intermediates(x, indices=4, return_prefix_tokens=True,
                                            norm=True, output_fmt='NLC', intermediates_only=True)
        return torch.stack([prefix[:, 0] for patches, prefix in layers] + [layers[-1][0].mean(1)], dim=1)
    final, layers = model.forward_intermediates(x, indices=4, return_prefix_tokens=False,
                                              norm=True, output_fmt='NLC', intermediates_only=False)
    return torch.stack([patches.mean(1) for patches in layers] +
                       [model.forward_head(final, pre_logits=True)], dim=1)


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def extract(a):
    if a.view in {'average', 'projected'}:
        raise ValueError('Use the average/project actions to derive features from completed caches')
    out = ROOT / f'{a.model}-{a.size}-{a.view}'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.model.startswith('dinov2_'):
        model = torch.hub.load('facebookresearch/dinov2', a.model, trust_repo=True)
        pretraining = 'DINOv2 LVD-142M'
    else:
        import timm
        if 'dinov3' in a.model:
            info_path = ROOT / ('dinov3-7b-download.json' if a.model.startswith('vit_7b') else 'dinov3-download.json')
        else:
            info_path = ROOT / 'backbones' / f'{a.model}.json'
        if info_path.exists() and json.loads(info_path.read_text())['repo'] == 'timm/' + a.model:
            model = load_timm_streamed(a.model, json.loads(info_path.read_text())['path'])
        else:
            options = {'global_pool': 'token'} if 'dinov3' in a.model else {}
            model = timm.create_model(a.model, pretrained=True, num_classes=0,
                                      dynamic_img_size=True, **options)
        pretraining = 'DINOv3 LVD-1689M' if 'dinov3' in a.model else model.pretrained_cfg.get('hf_hub_id', a.model)
    model = model.eval().to(device='cuda', dtype=torch.bfloat16)
    model.requires_grad_(False)
    dim = model.embed_dim
    native_cfg = getattr(model, 'pretrained_cfg', {})
    mean_values = native_cfg.get('mean', [.485, .456, .406])
    std_values = native_cfg.get('std', [.229, .224, .225])
    mean = torch.tensor(mean_values, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(std_values, device='cuda').view(1, 3, 1, 1)
    for split in (['train', 'test'] if a.test else ['train']):
        ds = CIFAR10(str(HERE / 'data'), train=split == 'train', download=False)
        n = len(ds)
        path = out / f'{split}.npy'
        state_path = out / f'{split}-state.json'
        cfg = {'model': a.model, 'size': a.size, 'view': a.view, 'n': n,
               'features': 'last four class tokens plus final mean patch token', 'dim': dim,
               'pretraining': pretraining, 'dtype': 'bfloat16 inference / float16 saved'}
        if 'siglip' in a.model:
            cfg['features'] = 'last four mean patch tokens plus final attention-pooled vector'
            cfg['normalization'] = {'mean': list(mean_values), 'std': list(std_values)}
        start = 0
        if state_path.exists():
            state = json.loads(state_path.read_text())
            assert state['config'] == cfg
            start = state['done']
        arr = (np.lib.format.open_memmap(path, mode='r+') if path.exists() else
               np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(n, 5, dim)))
        data = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
        t = time.monotonic()
        for i in range(start, n, a.batch):
            x = data[i:i + a.batch].cuda().float().div_(255)
            if a.view == 'flip':
                x = x.flip(-1)
            x = F.interpolate(x, size=(a.size, a.size), mode='bicubic', align_corners=False, antialias=True)
            x = ((x - mean) / std).to(torch.bfloat16)
            with torch.inference_mode():
                features = intermediate_features(model, x, a.model)
            assert torch.isfinite(features).all()
            arr[i:i + len(x)] = features.float().cpu().numpy()
            done = i + len(x)
            if done % (a.batch * 20) == 0 or done == n:
                arr.flush()
                elapsed = time.monotonic() - t
                rate = (done - start) / elapsed
                save_json(state_path, {'config': cfg, 'done': done, 'elapsed_this_session': elapsed,
                                       'images_per_second': rate})
                print(f'{out.name} {split} {done}/{n} {rate:.1f} img/s ETA {(n-done)/rate/60:.1f} min', flush=True)
                if a.smoke:
                    return
        np.save(out / f'{split}-labels.npy', np.array(ds.targets))
        print(f'COMPLETE {out.name} {split}', flush=True)


def average(a):
    """Average independently extracted original/flip features without test selection."""
    sources = [ROOT / f'{a.model}-{a.size}-{v}' for v in ('original', 'flip')]
    out = ROOT / f'{a.model}-{a.size}-average'
    out.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        states = [json.loads((p / f'{split}-state.json').read_text()) for p in sources]
        n = 50000 if split == 'train' else 10000
        assert all(s['done'] == n for s in states)
        inputs = [np.load(p / f'{split}.npy', mmap_mode='r') for p in sources]
        assert inputs[0].shape == inputs[1].shape and len(inputs[0]) == n
        labels = [np.load(p / f'{split}-labels.npy') for p in sources]
        assert np.array_equal(labels[0], labels[1])
        arr = np.lib.format.open_memmap(out / f'{split}.npy', mode='w+',
                                       dtype=np.float16, shape=inputs[0].shape)
        for i in range(0, n, 1024):
            arr[i:i + 1024] = (inputs[0][i:i + 1024].astype(np.float32) +
                              inputs[1][i:i + 1024].astype(np.float32)) * .5
        arr.flush()
        np.save(out / f'{split}-labels.npy', labels[0])
        save_json(out / f'{split}-state.json', {'done': n, 'config': {
            'model': a.model, 'size': a.size, 'view': 'average', 'weights': [.5, .5],
            'source_configs': [s['config'] for s in states],
        }})
        print(f'AVERAGED {out.name} {split} {n}', flush=True)


def project(a):
    """Retain every feature group with label-free fixed Gaussian projections."""
    torch.set_num_threads(4)
    source = ROOT / f'{a.model}-{a.size}-original'
    out = ROOT / f'{a.model}-{a.size}-projected'
    out.mkdir(parents=True, exist_ok=True)
    dimension = np.load(source / 'train.npy', mmap_mode='r').shape[-1]
    assert dimension > 1024
    projection_path = out / 'projection.pt'
    if projection_path.exists():
        matrices = torch.load(projection_path, map_location='cpu', weights_only=True)
    else:
        generator = torch.Generator().manual_seed(2026)
        matrices = torch.randn(5, dimension, 1024, generator=generator) / 1024 ** .5
        torch.save(matrices, projection_path)
    assert matrices.shape == (5, dimension, 1024)
    fingerprint = hashlib.sha256(projection_path.read_bytes()).hexdigest()
    matrices = matrices.cuda()
    for split in ('train', 'test'):
        state = json.loads((source / f'{split}-state.json').read_text())
        n = 50000 if split == 'train' else 10000
        assert state['done'] == n
        cfg = {'model': a.model, 'size': a.size, 'view': 'projected', 'seed': 2026,
               'projection': 'independent Gaussian matrix per group, scaled by sqrt(1024)',
               'projection_sha256': fingerprint, 'dim': 1024, 'source_config': state['config']}
        state_path = out / f'{split}-state.json'
        if state_path.exists():
            cached = json.loads(state_path.read_text())
            assert cached['config'] == cfg
            if cached['done'] == n:
                continue
        features = np.load(source / f'{split}.npy', mmap_mode='r')
        projected = np.lib.format.open_memmap(out / f'{split}.npy', mode='w+',
                      dtype=np.float16, shape=(n, 5, 1024))
        started = time.monotonic()
        for start in range(0, n, 256):
            x = torch.from_numpy(np.array(features[start:start + 256], dtype=np.float32)).cuda()
            z = torch.bmm(x.transpose(0, 1), matrices).transpose(0, 1)
            assert torch.isfinite(z).all()
            projected[start:start + len(x)] = z.cpu().numpy()
        projected.flush()
        np.save(out / f'{split}-labels.npy', np.load(source / f'{split}-labels.npy'))
        save_json(state_path, {'done': n, 'config': cfg, 'elapsed': time.monotonic() - started})
        print('PROJECTED', out.name, split, n, 'elapsed', time.monotonic() - started, flush=True)


def fit(a):
    torch.set_num_threads(8)
    directory = ROOT / f'{a.model}-{a.size}-{a.view}'
    state = json.loads((directory / 'train-state.json').read_text())
    assert state['done'] == 50000
    features = np.load(directory / 'train.npy', mmap_mode='r')
    labels = np.load(directory / 'train-labels.npy')
    split_path = HERE / 'runs/followup/split.json'
    split = json.loads(split_path.read_text())
    train = split['indices']['train']
    valid = split['indices'].get('validation', split['indices'].get('val'))
    assert valid is not None and len(set(train) & set(valid)) == 0
    results = []
    modes = (['last-mean', 'last-mean-pooled', 'four-means', 'four-means-pooled'] if 'siglip' in a.model else
             ['last-cls', 'last-cls-mean', 'four-cls', 'four-cls-mean'])
    for mode, indices in zip(modes, ([3], [3, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4])):
        if len(indices) * features.shape[-1] > 8192:
            print(f'SKIP {mode}: feature dimension exceeds 8192-dimensional head budget', flush=True)
            continue
        x = torch.from_numpy(np.array(features[:, indices], dtype=np.float32).reshape(50000, -1)).cuda()
        mu = x[train].mean(0)
        scale = x[train].std(0).clamp_min(.01)
        x = (x - mu) / scale
        x = torch.cat([x, torch.ones(len(x), 1, device='cuda')], 1)
        xt, xv = x[train], x[valid]
        yt = torch.from_numpy(labels[train]).cuda()
        y = F.one_hot(yt, 10).float()
        # The concatenated class tokens are correlated. Accumulate/solve ridge in
        # float64 so small regularizers do not disappear in float32 roundoff.
        xt64 = xt.double()
        gram = xt64.T @ xt64
        rhs = xt64.T @ y.double()
        del xt64
        for alpha in [.01, .1, 1., 10., 100., 1000., 10000.]:
            w = torch.linalg.solve(gram + alpha * torch.eye(x.shape[1], device='cuda', dtype=torch.float64), rhs).float()
            assert torch.isfinite(w).all()
            pred = (xv @ w).argmax(1).cpu().numpy()
            acc = float(np.mean(pred == labels[valid]) * 100)
            row = {'mode': mode, 'method': 'ridge', 'alpha': alpha, 'validation_accuracy': acc}
            results.append(row)
            print(row, flush=True)
            if acc == max(r['validation_accuracy'] for r in results):
                torch.save({'w': w.cpu(), 'mean': mu.cpu(), 'scale': scale.cpu(),
                            'indices': indices, 'selection': row, 'split_hash': split['hash']}, directory / 'head.pt')
                np.save(directory / 'validation-logits.npy', (xv @ w).detach().cpu().numpy())
        del gram, rhs
        for regularization in [.00001, .0001, .001, .01]:
            w = torch.zeros(x.shape[1], 10, device='cuda', requires_grad=True)
            opt = torch.optim.LBFGS([w], lr=1, max_iter=100, history_size=10,
                                    line_search_fn='strong_wolfe', tolerance_grad=1e-7)
            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(xt @ w, yt) + regularization * w[:-1].square().sum() / 2
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                pred = (xv @ w).argmax(1).cpu().numpy()
                acc = float(np.mean(pred == labels[valid]) * 100)
            row = {'mode': mode, 'method': 'logistic', 'regularization': regularization,
                   'validation_accuracy': acc}
            results.append(row)
            print(row, flush=True)
            if acc == max(r['validation_accuracy'] for r in results):
                torch.save({'w': w.detach().cpu(), 'mean': mu.cpu(), 'scale': scale.cpu(),
                            'indices': indices, 'selection': row, 'split_hash': split['hash']}, directory / 'head.pt')
                np.save(directory / 'validation-logits.npy', (xv @ w).detach().cpu().numpy())
        save_json(directory / 'validation-results.json', results)
    # Equal validation scores deliberately keep the last enumerated candidate.
    # Record the exact persisted head, rather than a different tied row.
    selected = torch.load(directory / 'head.pt', weights_only=True)['selection']
    save_json(directory / 'selection.json', selected)
    np.save(directory / 'validation-labels.npy', labels[valid])
    np.save(directory / 'validation-indices.npy', np.asarray(valid))


def refit(a):
    """Freeze validation selection, then fit the same head on all 50k training images."""
    torch.set_num_threads(8)
    directory = ROOT / f'{a.model}-{a.size}-{a.view}'
    selection = json.loads((directory / 'selection.json').read_text())
    chosen = torch.load(directory / 'head.pt', weights_only=True)
    assert chosen['selection']['validation_accuracy'] == selection['validation_accuracy']
    selection = chosen['selection']
    save_json(directory / 'selection.json', selection)
    final_path = directory / 'final-head.pt'
    if final_path.exists():
        cached = torch.load(final_path, map_location='cpu', weights_only=True)
        assert cached['selection'] == selection and cached['train_images'] == 50000
        assert (cached['backbone'], cached['size'], cached['view']) == (a.model, a.size, a.view)
        print('REUSE COMPLETED REFIT', selection, flush=True)
        return
    features = np.load(directory / 'train.npy', mmap_mode='r')
    labels = np.load(directory / 'train-labels.npy')
    x = torch.from_numpy(np.array(features[:, chosen['indices']], dtype=np.float32).reshape(len(labels), -1)).cuda()
    mu, scale = x.mean(0), x.std(0).clamp_min(.01)
    x = (x - mu) / scale
    x = torch.cat([x, torch.ones(len(x), 1, device='cuda')], 1)
    y = torch.from_numpy(labels).cuda()
    if selection['method'] == 'ridge':
        x64 = x.double()
        gram = x64.T @ x64
        w = torch.linalg.solve(gram + selection['alpha'] * torch.eye(x.shape[1], device='cuda', dtype=torch.float64),
                               x64.T @ F.one_hot(y, 10).double()).float()
    else:
        w = torch.zeros(x.shape[1], 10, device='cuda', requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=100, history_size=10,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-7)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(x @ w, y) + selection['regularization'] * w[:-1].square().sum() / 2
            loss.backward()
            return loss
        opt.step(closure)
    assert torch.isfinite(w).all()
    torch.save({'w': w.detach().cpu(), 'mean': mu.cpu(), 'scale': scale.cpu(),
                'indices': chosen['indices'], 'selection': selection, 'train_images': len(labels),
                'backbone': a.model, 'size': a.size, 'view': a.view,
                'track': 'outside pretraining / frozen backbone'}, directory / 'final-head.pt')
    print('REFIT COMPLETE', selection, flush=True)


def evaluate(a):
    """Evaluate a frozen validation-selected, train-only-refitted classifier once."""
    torch.set_num_threads(8)
    directory = ROOT / f'{a.model}-{a.size}-{a.view}'
    result_path = directory / 'test-summary.json'
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        assert hashlib.sha256((directory / 'final-head.pt').read_bytes()).hexdigest() == cached['head_sha256'], (
            'Cached test result belongs to a different head; use a fresh experiment directory')
        print(result_path.read_text(), flush=True)
        return
    state = json.loads((directory / 'test-state.json').read_text())
    assert state['done'] == 10000
    head_path = directory / 'final-head.pt'
    head = torch.load(head_path, weights_only=True)
    assert head['train_images'] == 50000
    assert head['selection'] == json.loads((directory / 'selection.json').read_text())
    features = np.load(directory / 'test.npy', mmap_mode='r')
    labels = np.load(directory / 'test-labels.npy')
    x = torch.from_numpy(np.array(features[:, head['indices']], dtype=np.float32).reshape(len(labels), -1)).cuda()
    x = (x - head['mean'].cuda()) / head['scale'].cuda()
    x = torch.cat([x, torch.ones(len(x), 1, device='cuda')], 1)
    logits = (x @ head['w'].cuda()).cpu().numpy()
    assert np.isfinite(logits).all()
    pred = logits.argmax(1)
    correct = int(np.sum(pred == labels))
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, pred), 1)
    np.save(directory / 'test-logits.npy', logits)
    result = {'accuracy': correct / len(labels) * 100, 'correct': correct, 'total': len(labels),
              'errors': len(labels) - correct, 'matrix': matrix.tolist(), 'selection': head['selection'],
              'backbone': head['backbone'], 'size': head['size'], 'view': head['view'],
              'track': head['track'], 'train_images': head['train_images'],
              'head_sha256': hashlib.sha256(head_path.read_bytes()).hexdigest(),
              'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    save_json(result_path, result)
    print('TEST RESULT', result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['extract', 'fit', 'refit', 'evaluate', 'average', 'project'])
    p.add_argument('--model', default='dinov2_vitg14')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--view', choices=['original', 'flip', 'average', 'projected'], default='original')
    p.add_argument('--test', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(HERE / 'runs/depth-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'extract': extract, 'fit': fit, 'refit': refit, 'evaluate': evaluate,
         'average': average, 'project': project}[a.action](a)
