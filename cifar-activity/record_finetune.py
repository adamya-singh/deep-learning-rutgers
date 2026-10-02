"""Adapt the last four DINOv3 blocks; select epochs on validation, refit on train."""
import argparse
import fcntl
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from record_push import HERE, ROOT, intermediate_features, load_timm_streamed, save_json

MODEL = 'vit_huge_plus_patch16_dinov3.lvd1689m'
SOURCE = ROOT / f'{MODEL}-224-original'


class Adapter(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.indices = head['indices']
        self.register_buffer('mean', head['mean'].float())
        self.register_buffer('scale', head['scale'].float())
        self.head = nn.Linear(len(self.mean), 10)
        with torch.no_grad():
            self.head.weight.copy_(20 * head['w'][:-1].T)
            self.head.bias.copy_(20 * head['w'][-1])
        self.backbone.requires_grad_(False)
        for block in self.backbone.blocks[-4:]:
            block.float().requires_grad_(True)
        self.backbone.norm.float().requires_grad_(True)

    def forward(self, x, name=MODEL):
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            features = intermediate_features(self.backbone, x, name)
        features = features[:, self.indices].flatten(1).float()
        return self.head((features - self.mean) / self.scale)


def make_model():
    info = json.loads((ROOT / 'dinov3-download.json').read_text())
    backbone = load_timm_streamed(MODEL, info['path'])
    head = torch.load(SOURCE / 'head.pt', map_location='cpu', weights_only=True)
    return Adapter(backbone, head).cuda()


def preprocess(images, augment=False, device='cuda'):
    x = images.to(device).float().div_(255)
    if augment:
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        ys = torch.randint(0, 9, (len(x),), device=x.device)
        xs = torch.randint(0, 9, (len(x),), device=x.device)
        rows = ys[:, None, None] + torch.arange(32, device=x.device)[None, :, None]
        cols = xs[:, None, None] + torch.arange(32, device=x.device)[None, None, :]
        x = x.permute(0, 2, 3, 1)[torch.arange(len(x), device=x.device)[:, None, None], rows, cols]
        x = x.permute(0, 3, 1, 2)
        flip = torch.rand(len(x), device=x.device) < .5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x = F.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True)
    mean = x.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = x.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (x - mean) / std


def predict(model, images, indices, batch):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            logits = model(preprocess(images[indices[start:start + batch]]))
            assert torch.isfinite(logits).all()
            parts.append(logits.cpu().numpy())
    return np.concatenate(parts)


def checkpoint(model):
    trainable = {key for key, value in model.named_parameters() if value.requires_grad}
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key in trainable or key in {'mean', 'scale'}}


def train(model, images, labels, indices, epochs, batch, directory, validate=None, schedule_epochs=None):
    head_parameters = list(model.head.parameters())
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': backbone_parameters, 'lr': 1e-5},
        {'params': head_parameters, 'lr': 1e-3}], weight_decay=.01)
    steps = (schedule_epochs or epochs) * math.ceil(len(indices) / batch)
    step, best_correct, best_epoch = 0, -1, 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(indices)
        started = time.monotonic()
        loss_sum, count = 0., 0
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            logits = model(preprocess(images[ids], augment=True))
            target = labels[ids].cuda()
            loss = F.cross_entropy(logits, target, label_smoothing=.02)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head_parameters + backbone_parameters, 1.)
            assert torch.isfinite(norm)
            factor = min(1., (step + 1) / 100) * .5 * (1 + math.cos(math.pi * step / steps))
            optimizer.param_groups[0]['lr'] = 1e-5 * factor
            optimizer.param_groups[1]['lr'] = 1e-3 * factor
            optimizer.step()
            step += 1
            loss_sum += float(loss.detach()) * len(ids)
            count += len(ids)
            if step % 100 == 0:
                print('FINETUNE', directory.name, 'epoch', epoch, 'seen', count,
                      'loss', loss_sum / count, 'img/s', count / (time.monotonic() - started), flush=True)
        row = {'epoch': epoch, 'loss': loss_sum / count}
        if validate is not None:
            logits = predict(model, images, validate, batch)
            correct = int(np.sum(logits.argmax(1) == labels[validate].numpy()))
            row.update(validation_correct=correct, validation_accuracy=correct / 50)
            if correct > best_correct:
                best_correct, best_epoch = correct, epoch
                torch.save(checkpoint(model), directory / 'head.pt')
                np.save(directory / 'validation-logits.npy', logits)
            save_json(directory / 'selection.json', {'epochs': best_epoch,
                      'validation_accuracy': best_correct / 50, 'validation_correct': best_correct})
        rows.append(row)
        save_json(directory / ('validation-results.json' if validate is not None else 'refit-progress.json'), rows)
        print('FINETUNE EPOCH', row, flush=True)
    return best_epoch


def seed():
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.cuda.manual_seed_all(2026)


def run(a):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / 'finetunes' / a.name
    out.mkdir(parents=True, exist_ok=True)
    config = {'model': MODEL, 'epochs': a.epochs, 'batch': a.batch, 'seed': 2026,
              'blocks': 4, 'size': 224, 'backbone_lr': 1e-5, 'head_lr': 1e-3,
              'source_head_sha256': hashlib.sha256((SOURCE / 'head.pt').read_bytes()).hexdigest()}
    if (out / 'run-config.json').exists():
        assert json.loads((out / 'run-config.json').read_text()) == config
    else:
        save_json(out / 'run-config.json', config)
    ds = CIFAR10(str(HERE / 'data'), train=True, download=False)
    images = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
    labels = torch.tensor(ds.targets)
    split = json.loads((HERE / 'runs/followup/split.json').read_text())['indices']
    assert len(split['train']) == 45000 and len(split['validation']) == 5000
    assert not set(split['train']) & set(split['validation'])
    if not (out / 'selection-complete.json').exists():
        seed()
        model = make_model()
        train(model, images, labels, split['train'], a.epochs, a.batch, out, split['validation'])
        np.save(out / 'validation-labels.npy', labels[split['validation']].numpy())
        np.save(out / 'validation-indices.npy', np.asarray(split['validation']))
        selection = json.loads((out / 'selection.json').read_text())
        selection['schedule_epochs'] = a.epochs
        save_json(out / 'selection-complete.json', selection)
        del model
        torch.cuda.empty_cache()
    selection = json.loads((out / 'selection-complete.json').read_text())
    final = out / 'final-head.pt'
    if not final.exists():
        seed()
        model = make_model()
        train(model, images, labels, np.arange(50000), selection['epochs'], a.batch, out,
              schedule_epochs=selection['schedule_epochs'])
        torch.save(checkpoint(model), final)
        del model
        torch.cuda.empty_cache()
    # Reconstruct from the base checkpoint and saved adapted parameters for test.
    seed()
    model = make_model()
    saved = torch.load(final, map_location='cpu', weights_only=True)
    expected = set(checkpoint(model))
    assert set(saved) == expected
    model.load_state_dict(saved, strict=False)
    test = CIFAR10(str(HERE / 'data'), train=False, download=False)
    test_images = torch.from_numpy(test.data).permute(0, 3, 1, 2)
    logits = predict(model, test_images, np.arange(10000), a.batch)
    test_labels = np.asarray(test.targets)
    correct = int(np.sum(logits.argmax(1) == test_labels))
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (test_labels, logits.argmax(1)), 1)
    np.save(out / 'test-logits.npy', logits)
    np.save(out / 'test-labels.npy', test_labels)
    result = {'selection': selection, 'accuracy': correct / 100, 'correct': correct,
              'total': 10000, 'errors': 10000 - correct, 'matrix': matrix.tolist(),
              'train_images': 50000, 'backbone': MODEL, 'size': 224,
              'track': 'outside pretraining / last four blocks and classifier fine-tuned',
              'head_sha256': hashlib.sha256(final.read_bytes()).hexdigest(),
              'source_head_sha256': hashlib.sha256((SOURCE / 'head.pt').read_bytes()).hexdigest(),
              'base_checkpoint': json.loads((ROOT / 'dinov3-download.json').read_text()),
              'training': {'seed': 2026, 'batch': a.batch, 'label_smoothing': .02,
                           'backbone_lr': 1e-5, 'head_lr': 1e-3, 'weight_decay': .01}}
    save_json(out / 'test-summary.json', result)
    # Repeat deterministic inference in different batches to detect reload/batch errors.
    again = predict(model, test_images, np.arange(10000), max(1, a.batch // 2))
    assert np.array_equal(logits.argmax(1), again.argmax(1))
    save_json(out / 'verification.json', {'passed': True, 'checkpoint_reloaded_before_test': True,
              'different_batch_predictions_agree': True, 'official_test_labels_agree': True,
              'accuracy': correct / 100, 'errors': 10000 - correct,
              'scope': 'full-model GPU inference; no independent CPU backbone reconstruction'})
    print('FINETUNE TEST', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dinov3-huge-last4')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    with open(HERE / 'runs/depth-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
