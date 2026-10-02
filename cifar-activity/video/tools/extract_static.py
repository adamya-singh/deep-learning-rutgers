"""Extract real visual evidence for the film (read-only, CPU, 2 threads).

Run with the parent project's interpreter (its torch/numpy/pillow are only used,
never modified):

    OMP_NUM_THREADS=2 ../.venv/bin/python tools/extract_static.py

Reads ../data/cifar-10-batches-py and ../runs/baseline/last.pt. Writes only to
video/public/. Nothing in the parent project is changed.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

torch.set_num_threads(2)
VIDEO = Path(__file__).resolve().parents[1]
PARENT = VIDEO.parent
OUT = VIDEO / "public"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)
MEAN = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32)
STD = np.array((0.2470, 0.2435, 0.2616), dtype=np.float32)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_batch(name):
    with open(PARENT / "data" / "cifar-10-batches-py" / name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # N,H,W,C uint8
    return x, np.array(d[b"labels"])


def build_model():
    # Identical to cifar_cnn.build_model (the unchanged classroom model).
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))


def to_tensor(x_uint8):
    x = (x_uint8.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def save(arr, name, scale=1):
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    im.save(IMG / name)


def fmap_png(m, name):
    """Save a feature map with a fixed navy→cyan→cream ramp (per-map scaling)."""
    m = m - m.min()
    m = m / (m.max() + 1e-8)
    stops = np.array([[11, 18, 38], [30, 110, 150], [96, 210, 235], [246, 236, 214]], np.float32)
    t = m * (len(stops) - 1)
    i = np.clip(t.astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    save(rgb.astype(np.uint8), name)


def main():
    t0 = time.time()
    test_x, test_y = load_batch("test_batch")
    train_x, train_y = load_batch("data_batch_1")
    meta = {"classes": CLASSES, "source": "../data/cifar-10-batches-py (official CIFAR-10 python batches)"}

    # Gallery: first 8 test images of each class.
    gallery = []
    for c in range(10):
        idx = np.where(test_y == c)[0][:8]
        for j, i in enumerate(idx):
            save(test_x[i], f"cifar_{c}_{j}.png")
        gallery.append([int(i) for i in idx])
    meta["gallery_test_indices"] = gallery

    state = torch.load(PARENT / "runs" / "baseline" / "last.pt", map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(state["model"])
    model.eval()
    meta["checkpoint"] = {"path": "../runs/baseline/last.pt", "epoch": state["epoch"],
                          "steps": state["steps"], "wandb_id": state["wandb_id"]}

    # Pick featured images: correctly classified with clear confidence, for frog/cat/ship/horse.
    with torch.no_grad():
        probs = torch.softmax(model(to_tensor(test_x[:2000])), 1).numpy()
    featured = {}
    for cname in ("frog", "ship", "horse", "cat", "automobile", "deer"):
        c = CLASSES.index(cname)
        cand = [i for i in range(2000) if test_y[i] == c and probs[i].argmax() == c and 0.55 < probs[i, c] < 0.92]
        featured[cname] = cand[0]
    # A misclassified example for honesty.
    wrong = [i for i in range(2000) if probs[i].argmax() != test_y[i] and probs[i].max() > 0.5]
    featured["wrong"] = wrong[0]
    meta["featured"] = {}
    for k, i in featured.items():
        save(test_x[i], f"feat_{k}.png")
        with torch.no_grad():
            logits = model(to_tensor(test_x[i:i + 1]))[0].numpy()
        meta["featured"][k] = {"test_index": int(i), "label": CLASSES[test_y[i]],
                               "logits": [round(float(v), 4) for v in logits],
                               "probs": [round(float(v), 5) for v in np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()]}

    # Raw pixels of the hero image (frog) for the pixel grid.
    hero = featured["frog"]
    meta["hero_pixels"] = test_x[hero].tolist()  # 32x32x3 uint8

    # Real conv1 filters (32 x 3 x 3 x 3) from the trained baseline.
    w1 = model[0].weight.detach().numpy()
    b1 = model[0].bias.detach().numpy()
    meta["conv1_weights"] = np.round(w1, 4).tolist()
    meta["conv1_bias"] = np.round(b1, 4).tolist()
    for k in range(32):
        f = w1[k].transpose(1, 2, 0)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        save((f * 255).astype(np.uint8), f"filter_{k}.png")

    # Real activations for the hero image.
    x = to_tensor(test_x[hero:hero + 1])
    with torch.no_grad():
        a1 = model[1](model[0](x))
        p1 = model[2](a1)
        a2 = model[4](model[3](p1))
        p2 = model[5](a2)
        h = model[8](model[7](model[6](p2)))
    for k in range(32):
        fmap_png(a1[0, k].numpy(), f"act1_{k}.png")
        fmap_png(p1[0, k].numpy(), f"pool1_{k}.png")
    for k in range(64):
        fmap_png(a2[0, k].numpy(), f"act2_{k}.png")
        fmap_png(p2[0, k].numpy(), f"pool2_{k}.png")
    meta["hidden128"] = np.round(h[0].numpy(), 3).tolist()
    # Mean activation per conv1 map (to pick vivid maps in the film).
    meta["act1_mean"] = np.round(a1[0].mean((1, 2)).numpy(), 4).tolist()
    meta["act2_mean"] = np.round(a2[0].mean((1, 2)).numpy(), 4).tolist()

    # A real multiply-sum: filter k at output pixel (r, c), using the normalized input.
    k = int(np.argmax(a1[0].mean((1, 2)).numpy()))
    r, c = 14, 15
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1))[0].numpy()
    patch = xp[:, r:r + 3, c:c + 3]
    value = float((patch * w1[k]).sum() + b1[k])
    meta["multiply_sum"] = {"filter": k, "row": r, "col": c,
                            "patch": np.round(patch, 3).tolist(), "weights": np.round(w1[k], 3).tolist(),
                            "bias": round(float(b1[k]), 4), "pre_relu": round(value, 4),
                            "check_from_model": round(float(model[0](x)[0, k, r, c]), 4)}

    # Augmentation views of real images: pad 4 black px, crop offsets, horizontal flip
    # (same operations as augmentations.py crop-flip, applied deterministically here).
    aug = {}
    for name in ("horse", "ship", "cat"):
        i = featured[name]
        img = test_x[i]
        padded = np.zeros((40, 40, 3), np.uint8)
        padded[4:36, 4:36] = img
        save(padded, f"aug_{name}_padded.png")
        views = []
        for j, (dy, dx, flip) in enumerate([(0, 0, False), (4, 4, True), (8, 2, False), (1, 7, True), (6, 8, True), (3, 0, False)]):
            v = padded[dy:dy + 32, dx:dx + 32]
            if flip:
                v = v[:, ::-1]
            save(np.ascontiguousarray(v), f"aug_{name}_{j}.png")
            views.append({"dy": dy, "dx": dx, "flip": flip})
        save(np.ascontiguousarray(img[:, ::-1]), f"aug_{name}_flip.png")
        aug[name] = views
    meta["augment_views"] = aug

    # Real 2-D loss slice around the trained baseline (Li et al. 2018 filter-normalized
    # random directions), cross-entropy on 1,000 training images.
    gen = torch.Generator().manual_seed(0)
    params = [p.detach().clone() for p in model.parameters()]
    def direction():
        d = []
        for p in params:
            r_ = torch.randn(p.shape, generator=gen)
            if p.dim() > 1:
                r_ = r_ * (p.flatten(1).norm(dim=1) / (r_.flatten(1).norm(dim=1) + 1e-10)).view(-1, *[1] * (p.dim() - 1))
            else:
                r_ = torch.zeros_like(p)  # biases fixed, as in Li et al.
            d.append(r_)
        return d
    d1, d2 = direction(), direction()
    xs = to_tensor(train_x[:1000])
    ys = torch.from_numpy(train_y[:1000]).long()
    grid = np.linspace(-1.0, 1.0, 25)
    loss = np.zeros((25, 25), np.float32)
    lossf = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, a in enumerate(grid):
            for j, b in enumerate(grid):
                for p, p0, u, v in zip(model.parameters(), params, d1, d2):
                    p.copy_(p0 + a * u + b * v)
                loss[i, j] = float(lossf(model(xs), ys))
        for p, p0 in zip(model.parameters(), params):
            p.copy_(p0)
    meta["loss_slice"] = {"alpha": grid.round(4).tolist(), "beta": grid.round(4).tolist(),
                          "loss": loss.round(4).tolist(),
                          "method": "baseline checkpoint; two filter-normalized Gaussian directions (seed 0); "
                                    "cross-entropy on first 1,000 images of data_batch_1; biases held fixed",
                          "center_loss": round(float(loss[12, 12]), 4)}

    meta["extracted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["seconds"] = round(time.time() - t0, 1)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "data" / "static.json").write_text(json.dumps(meta))
    print("done", meta["seconds"], "s; multiply-sum check", meta["multiply_sum"]["pre_relu"],
          meta["multiply_sum"]["check_from_model"], "featured", {k: v["test_index"] for k, v in meta["featured"].items()})


if __name__ == "__main__":
    main()
