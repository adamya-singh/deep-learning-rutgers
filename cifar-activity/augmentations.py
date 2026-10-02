"""Fixed CIFAR-10 recipes; image randomness is independent within a batch."""
import contextlib

import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode, v2


BASE = {"crop": {"size": 32, "padding": 4, "fill": 0}, "flip": {"p": 0.5}}
RECIPES = {
    "crop-flip": {**BASE},
    "none": {},
    "flip": {"flip": BASE["flip"]},
    "crop": {"crop": BASE["crop"]},
    "color-jitter": {**BASE, "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}},
    "rotation": {**BASE, "rotation": {"degrees": 15}},
    "erasing": {**BASE, "erasing": {"p": 0.5, "scale": [0.02, 0.20], "ratio": [0.3, 3.3], "value": 0}},
    "mixup": {**BASE, "mixup": {"alpha": 0.2, "num_classes": 10}},
    "cutmix": {**BASE, "cutmix": {"alpha": 1.0, "num_classes": 10}},
    "randaugment": {**BASE, "randaugment": {"num_ops": 2, "magnitude": 9, "num_magnitude_bins": 31}},
    "resized-crop": {"resized_crop": {"size": [32, 32], "scale": [0.8, 1.0], "ratio": [0.9, 1.1], "antialias": True}, "flip": BASE["flip"]},
    "affine": {**BASE, "affine": {"degrees": 10, "translate": [0.1, 0.1], "scale": [0.9, 1.1], "shear": [-5, 5]}},
    "grayscale": {**BASE, "grayscale": {"p": 0.1}},
    "blur": {**BASE, "blur": {"p": 0.2, "kernel_size": 3, "sigma": [0.1, 1.0]}},
    "autoaugment": {**BASE, "autoaugment": {"policy": "CIFAR10"}},
}


class Augmenter:
    """Own CPU/CUDA RNG streams; torchvision policies sample on CPU."""

    def __init__(self, recipe, seed, device, mean, std):
        self.recipe = recipe
        self.spec = RECIPES[recipe]
        self.device = torch.device(device)
        self.cpu_rng = torch.Generator().manual_seed(seed + 100_000).get_state()
        self.cuda_rng = (torch.Generator(device=self.device).manual_seed(seed + 100_000).get_state()
                         if self.device.type == "cuda" else None)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        bilinear = {"interpolation": InterpolationMode.BILINEAR, "fill": 0}
        self.image_transform = None
        if "color_jitter" in self.spec:
            self.image_transform = v2.ColorJitter(**self.spec["color_jitter"])
        elif "rotation" in self.spec:
            self.image_transform = v2.RandomRotation(**self.spec["rotation"], **bilinear)
        elif "affine" in self.spec:
            self.image_transform = v2.RandomAffine(**self.spec["affine"], **bilinear)
        elif "grayscale" in self.spec:
            self.image_transform = v2.RandomGrayscale(**self.spec["grayscale"])
        elif "blur" in self.spec:
            spec = self.spec["blur"]
            self.image_transform = v2.RandomApply([v2.GaussianBlur(spec["kernel_size"], spec["sigma"])], p=spec["p"])
        elif "randaugment" in self.spec:
            self.image_transform = v2.RandAugment(**self.spec["randaugment"], **bilinear)
        elif "autoaugment" in self.spec:
            self.image_transform = v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10, **bilinear)
        self.resized_crop = (v2.RandomResizedCrop(**self.spec["resized_crop"], interpolation=InterpolationMode.BILINEAR)
                             if "resized_crop" in self.spec else None)
        self.erase = v2.RandomErasing(**self.spec["erasing"]) if "erasing" in self.spec else None
        self.mix = (v2.MixUp(**self.spec["mixup"]) if "mixup" in self.spec else
                    v2.CutMix(**self.spec["cutmix"]) if "cutmix" in self.spec else None)

    def state_dict(self):
        return {"cpu_rng": self.cpu_rng.clone(), "cuda_rng": self.cuda_rng.clone() if self.cuda_rng is not None else None}

    def load_state_dict(self, state):
        self.cpu_rng = state["cpu_rng"].cpu()
        self.cuda_rng = state["cuda_rng"].cpu() if state["cuda_rng"] is not None else None

    @contextlib.contextmanager
    def random_stream(self):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self.cuda_rng, self.device)
            try:
                yield
            finally:
                self.cpu_rng = torch.get_rng_state()
                if devices:
                    self.cuda_rng = torch.cuda.get_rng_state(self.device)

    def __call__(self, raw, labels):
        with self.random_stream():
            x = raw.clone()
            n = len(x)
            if "crop" in self.spec:
                # Independent offsets for every image, including the final partial batch.
                padded = F.pad(x, (4, 4, 4, 4), value=0)
                offsets = torch.randint(9, (n, 2), device=x.device)
                rows = offsets[:, 0, None, None] + torch.arange(32, device=x.device)[None, :, None]
                cols = offsets[:, 1, None, None] + torch.arange(32, device=x.device)[None, None, :]
                x = padded.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
            elif self.resized_crop:
                # Tiny per-image kernels are faster on CPU than launching hundreds
                # of CUDA operations. Keep the resident dataset and base crop/flip on GPU.
                x = torch.stack([self.resized_crop(image) for image in x.cpu()]).to(self.device)
            if "flip" in self.spec:
                mask = torch.rand(n, device=x.device) < self.spec["flip"]["p"]
                x = torch.where(mask[:, None, None, None], x.flip(-1), x)
            if self.image_transform:
                x = torch.stack([self.image_transform(image) for image in x.cpu()]).to(self.device)
            x = x.float().div_(255).sub_(self.mean).div_(self.std)
            if self.erase:
                x = torch.stack([self.erase(image) for image in x.cpu()]).to(self.device)
            if self.mix:
                x, labels = self.mix(x, labels)
            return x.contiguous(), labels
