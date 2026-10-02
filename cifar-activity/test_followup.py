"""CPU tests for follow-up pairing, augmentation resume, ResNet shape, and queue restart."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import depth_cnn as base
import followup_cnn as trainer
import run_followup_queue as queue

torch.set_num_threads(1)


def specs():
    request = queue.load_manifest(base.HERE / "followup_experiments.json")
    upstream = base.load_manifest(base.HERE / "depth_experiments.json")
    return queue.make_specs(request, upstream, {"variant": "residual", "depth": 4})


def fixture(spec):
    spec = copy.deepcopy(spec)
    spec["training"].update(epochs=4, warmup_epochs=1, score_epochs=2, batch_size=2,
                            lr_start=.0001, lr_peak=.001, lr_min=.00001)
    raw = torch.arange(4 * 3 * 32 * 32).remainder(256).byte().reshape(4, 3, 32, 32)
    x = (raw.float() / 255 - torch.tensor(base.MEAN).view(1, 3, 1, 1)) / torch.tensor(base.STD).view(1, 3, 1, 1)
    return spec, (x, torch.tensor([0, 1, 2, 3]), x[:2].clone(), torch.tensor([0, 1]), raw)


class FollowupTests(unittest.TestCase):
    def test_recipe_budgets_and_cifar_resnet_shape(self):
        candidates = specs()
        self.assertEqual([s["augmentation"] for s in candidates], ["flip", "crop-flip", "crop-flip"])
        self.assertEqual([s["training"]["epochs"] for s in candidates], [160, 160, 200])
        self.assertEqual(candidates[0]["training"]["augmentation"], "flip")
        self.assertEqual(candidates[-1]["training"]["augmentation"], "crop-flip")
        model = trainer.build_model(candidates[-1])
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, nn.Identity)
        self.assertIsInstance(model.avgpool, nn.AdaptiveAvgPool2d)
        self.assertEqual(model.fc.out_features, 10)
        self.assertEqual(model(torch.zeros(2, 3, 32, 32)).shape, (2, 10))

    def test_winner_weights_and_shuffle_match_upstream(self):
        upstream = base.load_manifest(base.HERE / "depth_experiments.json")
        original, _, _, original_args = base.initialize(upstream, "residual", 4, 2, torch.device("cpu"))
        for spec in specs()[:2]:
            model, _, _, args = trainer.initialize(spec, 2, torch.device("cpu"))
            for k, value in original.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[k], rtol=0, atol=0)
            generator = torch.Generator().set_state(original_args.shuffle_generator.get_state())
            torch.testing.assert_close(torch.randperm(40, generator=generator), torch.randperm(40, generator=args.shuffle_generator), rtol=0, atol=0)

    def test_serialized_augmented_resume_exact_weights_bn_momentum_and_rng(self):
        for original_spec in specs():
            spec, data = fixture(original_spec)
            x, y, _, _, raw = data
            model, opt, scaler, args = trainer.initialize(spec, 0, torch.device("cpu"))
            args.raw_train = raw
            cfg = trainer.config(spec, 0, "cpu", "fixture")
            base.train_epoch(model, opt, scaler, nn.CrossEntropyLoss(), x, y, args, torch.device("cpu"), check_gradients=True)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "last.pt"
                base.save_checkpoint(path, trainer.snapshot(model, opt, scaler, args, cfg, 1, [], None))
                saved = torch.load(path, weights_only=False)
            base.train_epoch(model, opt, scaler, nn.CrossEntropyLoss(), x, y, args, torch.device("cpu"), check_gradients=True)
            resumed, ropt, rscaler, rargs = trainer.initialize(spec, 0, torch.device("cpu"))
            rargs.raw_train = raw
            trainer.restore(saved, cfg, resumed, ropt, rscaler, rargs)
            base.train_epoch(resumed, ropt, rscaler, nn.CrossEntropyLoss(), x, y, rargs, torch.device("cpu"), check_gradients=True)
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[k], rtol=0, atol=0)
            for a, b in zip(opt.state.values(), ropt.state.values()):
                torch.testing.assert_close(a["momentum_buffer"], b["momentum_buffer"], rtol=0, atol=0)
            for k, value in args.augmenter.state_dict().items():
                if value is not None:
                    torch.testing.assert_close(value, rargs.augmenter.state_dict()[k], rtol=0, atol=0)
            torch.testing.assert_close(args.shuffle_generator.get_state(), rargs.shuffle_generator.get_state(), rtol=0, atol=0)

    def test_training_restart_and_cached_test_result(self):
        spec, data = fixture(specs()[0])
        indices = {"train": [0, 1, 2, 3], "validation": [4, 5]}
        split_hash = base.digest(indices)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / "winner-flip/seed-0"
            base.atomic_json(root / "split.json", {"hash": split_hash, "indices": indices})
            with patch.object(trainer, "load_training", return_value=(data, split_hash)), \
                    patch.object(base, "CIFAR10", side_effect=AssertionError("test accessed during training")):
                trainer.train(spec, root, directory, 0, torch.device("cpu"), True)
                trainer.train(spec, root, directory, 0, torch.device("cpu"), True)
            row = json.loads((directory / "summary.json").read_text())
            cfg = trainer.config(spec, 0, "cpu", split_hash)
            self.assertEqual(queue.depth_queue.completed(directory, cfg, spec["training"], False)["score"], row["score"])
            dataset = argparse.Namespace(data=np.zeros((2, 32, 32, 3), dtype=np.uint8), targets=[0, 1])
            with patch.object(base, "CIFAR10", return_value=dataset) as loader:
                trainer.assess(spec, root, directory, 0, torch.device("cpu"), True)
                trainer.assess(spec, root, directory, 0, torch.device("cpu"), True)
                self.assertEqual(loader.call_count, 1)
                self.assertFalse(loader.call_args.kwargs["train"])

    def test_gpu_smoke_logic_on_cpu_fixtures(self):
        # Exercise the serialized continuation path without touching the active GPU.
        for original_spec in specs():
            spec, data = fixture(original_spec)
            with tempfile.TemporaryDirectory() as tmp, patch.object(trainer, "load_training", return_value=(data, "fixture")):
                root = Path(tmp); directory = root / "smoke"
                trainer.smoke(spec, root, directory, 0, torch.device("cpu"))
                self.assertTrue(json.loads((directory / "smoke.json").read_text())["resume_verified"])


class QueueTests(unittest.TestCase):
    def test_verified_source_winner_split_copy_and_real_report(self):
        m = base.load_manifest(base.HERE / "depth_experiments.json")
        m.update(seeds=[0], depths=[2], declining_seeds=1, epochs=4, warmup_epochs=1,
                 score_epochs=2, batch_size=2, lr_start=.0001, lr_peak=.001, lr_min=.00001)
        request = queue.load_manifest(base.HERE / "followup_experiments.json")
        request.update(seeds=[0], resnet_epochs=4)
        original_spec = specs()[0]
        _, data = fixture(original_spec)
        x, y, vx, vy, raw = data
        indices = {"train": [0, 1, 2, 3], "validation": [4, 5]}
        split_hash = base.digest(indices)
        with tempfile.TemporaryDirectory() as tmp:
            source, root = Path(tmp) / "source", Path(tmp) / "followup"
            source.mkdir(); root.mkdir()
            base.atomic_json(source / "split.json", {"seed": 42, "hash": split_hash, "indices": indices})
            directory = source / queue.depth_queue.key("plain", 2, 0)
            with patch.object(base, "load_training", return_value=((x, y, vx, vy), split_hash)):
                base.train(m, source, directory, "plain", 2, 0, torch.device("cpu"), True)
            rows = queue.depth_queue.records_at(m, source, "cpu", True)
            selection = queue.depth_queue.select_winner(m, rows)
            state = {"phase": "completed", "selection": selection, "plain_decision": {"visited": [2]}}
            frozen = {"manifest": m, "device": "cpu", "no_wandb": True, "implementation": base.implementation_hash()}
            document = queue.prepare(request, source, root, state, frozen)
            self.assertEqual(document["selection"], selection)
            self.assertEqual(json.loads((root / "split.json").read_text()), json.loads((source / "split.json").read_text()))
            self.assertEqual(len(document["controls"]), 1)
            spec = document["candidates"][0]
            with patch.object(trainer, "load_training", return_value=(data, split_hash)):
                trainer.train(spec, root, root / queue.job_key(spec["id"], 0), 0, torch.device("cpu"), True)
            records = queue.report(document, root, "cpu", True, {"jobs": {}})
            self.assertEqual(set(records), {("winner-none", 0), ("winner-flip", 0)})
            self.assertEqual(len(list((root / "plots").glob("*.svg"))), 2)
            self.assertTrue((root / "comparison.md").exists())
            corrupt = {**state, "selection": {**selection, "score": selection["score"] + 1}}
            with self.assertRaises(ValueError): queue.prepare(request, source, root, corrupt, frozen)

    def test_wait_gate_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; source.mkdir()
            frozen = {"manifest": {}, "implementation": "fixture"}
            base.atomic_json(source / "manifest.json", frozen)
            base.atomic_json(source / "queue-state.json", {"phase": "failed", "error": "fixture failure"})
            with self.assertRaisesRegex(RuntimeError, "depth queue failed"):
                queue.wait_for_depth(source, root, "cpu", {}, root / "state.json", frozen, .001)
            base.atomic_json(source / "queue-state.json", {"phase": "completed", "selection": {}})
            upstream, locks = queue.wait_for_depth(source, root, "cpu", {}, root / "state.json", frozen, .001)
            self.assertEqual(upstream["phase"], "completed")
            for lock in locks: lock.close()
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                queue.wait_for_depth(source, root, "cpu", {}, root / "state.json", {"different": True}, .001)

    def test_nine_runs_control_reuse_failure_restart_and_test_selection(self):
        candidates = specs(); seeds = [0, 1, 2]
        controls = [{"seed": s, "summary": {"score": 70}, "directory": "unused"} for s in seeds]
        document = {"implementation": trainer.implementation_hash(), "seeds": seeds, "split_hash": "fixture",
                    "candidates": candidates, "controls": controls}
        stored, launches = {}, []
        fail = [True]
        def complete(directory, expected, m, require_wandb):
            return stored.get(str(directory))
        def report(document, root, device, no_wandb, state):
            rows = {("winner-none", s): {"score": 70} for s in seeds}
            for spec in candidates:
                for s in seeds:
                    row = stored.get(str(root / queue.job_key(spec["id"], s)))
                    if row: rows[spec["id"], s] = row
            return rows
        def run(cmd, **kwargs):
            action = cmd[3]; candidate = cmd[cmd.index("--candidate") + 1]
            seed = int(cmd[cmd.index("--seed") + 1]); directory = Path(cmd[cmd.index("--directory") + 1])
            launches.append((action, candidate, seed))
            if action == "smoke":
                base.atomic_json(directory / "smoke.json", {"peak_gpu_memory_mb": 1, "estimated_run_seconds": 1})
            elif action == "train":
                if candidate == "winner-crop-flip" and seed == 0 and fail[0]:
                    fail[0] = False; return argparse.Namespace(returncode=1)
                stored[str(directory)] = {"score": 80 if candidate == "resnet18-crop-flip" else 75}
            else:
                base.atomic_json(directory / "test-summary.json", {"accuracy": 79, "loss": .4})
            return argparse.Namespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state = {"jobs": {}}; state_path = root / "queue-state.json"
            args = argparse.Namespace(action="run", device="cpu", no_wandb=True)
            with patch.object(queue, "report", side_effect=report), patch.object(queue.depth_queue, "completed", side_effect=complete), \
                    patch.object(queue.subprocess, "run", side_effect=run):
                with self.assertRaises(RuntimeError):
                    queue.execute(args, document, root, state, state_path, [])
                self.assertEqual(state["jobs"]["winner-crop-flip/seed-0"]["status"], "failed")
                queue.execute(args, document, root, state, state_path, [])
            self.assertEqual(state["phase"], "completed")
            self.assertEqual(state["selection"]["candidate"], "resnet18-crop-flip")
            self.assertEqual(sum(a == "train" and c == "winner-flip" for a, c, s in launches), 3)
            self.assertFalse(any(c == "winner-none" for a, c, s in launches))
            self.assertEqual(sum(a == "test" for a, c, s in launches), 3)
            self.assertEqual(len([j for j in state["jobs"].values() if j["status"] == "completed"]), 9)

    def test_selection_ignores_test_metrics_and_needs_all_seeds(self):
        document = {"seeds": [0, 1, 2], "candidates": specs()}
        rows = {(candidate, seed): {"score": 70, "test_accuracy": 100 if candidate == "resnet18-crop-flip" else 0}
                for candidate in ["winner-none"] + [s["id"] for s in specs()] for seed in document["seeds"]}
        self.assertEqual(queue.select_winner(document, rows)["candidate"], "winner-none")
        del rows["winner-flip", 2]
        with self.assertRaises(ValueError): queue.select_winner(document, rows)


if __name__ == "__main__":
    unittest.main()
