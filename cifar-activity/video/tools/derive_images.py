"""Derive per-channel views of the hero frog (read-only input, writes public/img).

    .venv-tools/bin/python tools/derive_images.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

IMG = Path(__file__).resolve().parents[1] / "public" / "img"
x = np.asarray(Image.open(IMG / "feat_frog.png").convert("RGB"))
for i, name in enumerate("RGB"):
    out = np.zeros_like(x)
    out[..., i] = x[..., i]
    Image.fromarray(out).save(IMG / f"frog_{name}.png")
print("ok")
