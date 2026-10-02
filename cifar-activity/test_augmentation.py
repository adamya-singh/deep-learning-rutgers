"""Transform, RNG/resume, and queue invariants using synthetic fixtures."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.transforms.v2 import functional as VF

from augmentations import Augmenter, RECIPES
from cifar_cnn import MEAN, STD, compatible_config, train_epoch
import run_augmentation_queue as queue


class AugmentationTests(unittest.TestCase):
    def test_all_recipes_devices_shapes_and_rng(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            raw = torch.arange(8 * 3 * 32 * 32, device=device).remainder(256).byte().reshape(8, 3, 32, 32)
            labels = torch.arange(8, device=device)
            original = raw.clone()
            for recipe in RECIPES:
                with self.subTest(device=device, recipe=recipe):
                    augmenter = Augmenter(recipe, 0, device, MEAN, STD)
                    cpu_state = torch.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
                    saved = augmenter.state_dict()
                    x, y = augmenter(raw, labels)
                    self.assertEqual(x.shape, raw.shape)
                    self.assertEqual(x.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(x).all())
                    self.assertTrue(torch.equal(raw, original))
                    self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
                    if cuda_state is not None:
                        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_state))
                    self.assertEqual(tuple(y.shape), (8, 10) if recipe in ("mixup", "cutmix") else (8,))
                    if y.ndim == 2:
                        torch.testing.assert_close(y.sum(1), torch.ones(8, device=device))
                        self.assertTrue((y >= 0).all())
                    else:
                        self.assertTrue(torch.equal(y, labels))
                    augmenter.load_state_dict(saved)
                    repeated, repeated_y = augmenter(raw, labels)
                    torch.testing.assert_close(x, repeated, rtol=0, atol=0)
                    torch.testing.assert_close(y, repeated_y, rtol=0, atol=0)

    def test_crop_flip_padding_and_independence(self):
        image = torch.arange(3 * 32 * 32).remainder(255).add(1).byte().reshape(3, 32, 32)
        raw = image.repeat(16, 1, 1, 1)
        augmenter = Augmenter("crop-flip", 7, "cpu", MEAN, STD)
        with augmenter.random_stream():
            offsets = torch.randint(9, (16, 2))
            flips = torch.rand(16) < 0.5
        augmenter = Augmenter("crop-flip", 7, "cpu", MEAN, STD)
        x, _ = augmenter(raw, torch.zeros(16, dtype=torch.long))
        mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
        for i in range(16):
            expected = VF.crop(VF.pad(image, [4, 4, 4, 4], fill=0), int(offsets[i, 0]), int(offsets[i, 1]), 32, 32)
            if flips[i]:
                expected = VF.horizontal_flip(expected)
            torch.testing.assert_close(x[i], (expected.float() / 255 - mean) / std)
        self.assertGreater(len(torch.unique(offsets, dim=0)), 1)
        self.assertFalse(torch.equal(x, augmenter(raw, torch.zeros(16, dtype=torch.long))[0]))

    def test_cutmix_actual_area_and_singleton(self):
        raw = torch.stack([torch.full((3, 32, 32), value, dtype=torch.uint8) for value in (30, 80, 130, 180)])
        labels = torch.arange(4)
        augmenter = Augmenter("cutmix", 0, "cpu", MEAN, STD)
        # Isolate CutMix to inspect its actual clipped rectangle on constant images.
        augmenter.spec = {}
        x, y = augmenter(raw, labels)
        restored = x * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)
        for i in range(4):
            own_fraction = torch.isclose(restored[i, 0], torch.tensor(raw[i, 0, 0, 0].item() / 255)).float().mean()
            torch.testing.assert_close(y[i, i], own_fraction)
            torch.testing.assert_close(y[i, (i - 1) % 4], 1 - own_fraction)
        for recipe in ("mixup", "cutmix"):
            one, targets = Augmenter(recipe, 0, "cpu", MEAN, STD)(raw[:1], labels[:1])
            self.assertTrue(torch.isfinite(one).all())
            torch.testing.assert_close(targets, nn.functional.one_hot(labels[:1], 10).float())

    def test_clean_normalization_and_legacy_metadata(self):
        raw = torch.randint(256, (2, 3, 32, 32), dtype=torch.uint8)
        x, _ = Augmenter("none", 0, "cpu", MEAN, STD)(raw, torch.arange(2))
        expected = raw.float().div(255).sub(torch.tensor(MEAN).view(1, 3, 1, 1)).div(torch.tensor(STD).view(1, 3, 1, 1))
        torch.testing.assert_close(x, expected, rtol=0, atol=0)
        saved = {"seed": 0}
        self.assertEqual(compatible_config(saved), {"seed": 0, "augmentation": "none", "augmentation_params": {}, "augmentation_backend": "none", "rng_policy": "legacy"})
        self.assertEqual(saved, {"seed": 0})

    def test_training_resume_and_paired_shuffle(self):
        torch.manual_seed(42)
        raw = torch.randint(256, (7, 3, 32, 32), dtype=torch.uint8)
        labels = torch.arange(7)
        clean, _ = Augmenter("none", 0, "cpu", MEAN, STD)(raw, labels)
        initial = nn.Sequential(nn.Flatten(), nn.Linear(3072, 10))
        def make():
            model = copy.deepcopy(initial)
            args = SimpleNamespace(batch_size=3, precision="fp32", layout="contiguous", raw_train=raw,
                                   augmenter=Augmenter("mixup", 0, "cpu", MEAN, STD),
                                   shuffle_generator=torch.Generator().manual_seed(0))
            return model, torch.optim.SGD(model.parameters(), lr=0.01), args
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        device = torch.device("cpu")
        continuous, opt, args = make()
        train_epoch(continuous, opt, scaler, nn.CrossEntropyLoss(), clean, labels, args, device, check_gradients=True)
        saved_model = copy.deepcopy(continuous.state_dict())
        saved_opt = copy.deepcopy(opt.state_dict())
        shuffle = args.shuffle_generator.get_state()
        aug = args.augmenter.state_dict()
        train_epoch(continuous, opt, scaler, nn.CrossEntropyLoss(), clean, labels, args, device)
        resumed, resumed_opt, resumed_args = make()
        resumed.load_state_dict(saved_model)
        resumed_opt.load_state_dict(saved_opt)
        resumed_args.shuffle_generator.set_state(shuffle)
        resumed_args.augmenter.load_state_dict(aug)
        train_epoch(resumed, resumed_opt, scaler, nn.CrossEntropyLoss(), clean, labels, resumed_args, device)
        for a, b in zip(continuous.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for recipe in ("none", "crop-flip", "mixup"):
            generator = torch.Generator().manual_seed(10)
            expected = torch.randperm(7, generator=torch.Generator().manual_seed(10))
            Augmenter(recipe, 10, "cpu", MEAN, STD)(raw, labels)
            self.assertTrue(torch.equal(torch.randperm(7, generator=generator), expected))


class QueueTests(unittest.TestCase):
    def test_failure_resume_skip_and_report(self):
        manifest = {"version": 1, "epochs": 1, "batch_size": 512, "eval_batch_size": 1024,
                    "lr": 0.01, "precision": "fp32", "layout": "contiguous", "seeds": [0, 1], "recipes": ["crop-flip", "none"]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fail = [True]
            calls = []
            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                recipe = cmd[cmd.index("--augmentation") + 1]
                seed = int(cmd[cmd.index("--seed") + 1])
                directory = Path(cmd[cmd.index("--output-dir") + 1])
                expected = queue.expected_config(manifest, recipe, seed, "cpu")
                epoch = 0 if fail[0] else 1
                torch.save({"config": expected, "epoch": epoch}, directory / "last.pt")
                if fail[0]:
                    fail[0] = False
                    return SimpleNamespace(returncode=1)
                queue.atomic_json(directory / "summary.json", {"config": expected, "epochs": 1,
                    "test_accuracy": 60 + seed + (recipe != "none"), "train_accuracy": 70,
                    "elapsed_seconds": 1, "mean_train_images_per_second": 50000, "wandb_url": ""})
                return SimpleNamespace(returncode=0)
            with patch.object(queue, "previews"), patch.object(queue.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(RuntimeError):
                    queue.run_queue(manifest, root, "cpu", True)
                state = json.loads((root / "queue-state.json").read_text())
                self.assertEqual(state["jobs"]["crop-flip/seed-0"]["status"], "failed")
                queue.run_queue(manifest, root, "cpu", True)
                self.assertIn("--resume", calls[1])
                count = len(calls)
                queue.run_queue(manifest, root, "cpu", True)
                self.assertEqual(len(calls), count)
                report = json.loads((root / "comparison.json").read_text())
                self.assertEqual(report["completed"], 4)
                self.assertEqual(report["recipes"][0]["paired_differences"]["none"], [1, 1])
                with self.assertRaises(ValueError):
                    queue.run_queue({**manifest, "epochs": 2}, root, "cpu", True)
                summary_path = root / "none/seed-0/summary.json"
                value = json.loads(summary_path.read_text())
                value["config"]["augmentation"] = "flip"
                queue.atomic_json(summary_path, value)
                with self.assertRaises(ValueError):
                    queue.completed(summary_path.parent, queue.expected_config(manifest, "none", 0, "cpu"), 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
