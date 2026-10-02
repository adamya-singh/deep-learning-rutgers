"""CPU-only correctness and restart tests. GPU continuation checks run in queued preflight."""
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

import depth_cnn as trainer
import run_depth_queue as queue
from cifar_cnn import train_epoch

torch.set_num_threads(1)


def manifest():
    return trainer.load_manifest(trainer.HERE / "depth_experiments.json")


class ArchitectureTests(unittest.TestCase):
    def test_shapes_counts_and_exact_paired_initialization(self):
        for depth in manifest()["depths"]:
            torch.manual_seed(7)
            plain = trainer.build_model(depth)
            self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in plain.modules()), depth)
            self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in plain.modules()), depth)
            before = copy.deepcopy(plain.state_dict())
            if depth >= 4:
                torch.manual_seed(7)
                residual = trainer.build_model(depth, "residual")
                self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in residual.parameters()))
                for k, v in before.items():
                    torch.testing.assert_close(v, residual.state_dict()[k], rtol=0, atol=0)
                self.assertEqual(residual(torch.randn(2, 3, 32, 32)).shape, (2, 10))
            self.assertEqual(plain(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_zero_branch_and_channel_padding(self):
        for inputs, channels in ((3, 32), (32, 64), (64, 64)):
            block = trainer.Block(inputs, channels, True).eval()
            with torch.no_grad():
                block.conv1.weight.zero_(); block.conv2.weight.zero_()
            x = torch.randn(2, inputs, 8, 8)
            expected = torch.cat((x, x.new_zeros(2, channels - inputs, 8, 8)), 1).relu()
            torch.testing.assert_close(block(x), expected, rtol=0, atol=0)


class TrainerTests(unittest.TestCase):
    def test_stratified_split_hash_and_disjointness(self):
        targets = np.repeat(np.arange(10), 5000)
        a = trainer.split_indices(targets, 42, 500)
        b = trainer.split_indices(targets, 42, 500)
        self.assertEqual(a, b)
        self.assertEqual(len(a["train"]), 45000)
        self.assertEqual(len(a["validation"]), 5000)
        self.assertFalse(set(a["train"]) & set(a["validation"]))
        self.assertEqual(set(a["train"]) | set(a["validation"]), set(range(50000)))
        np.testing.assert_array_equal(np.bincount(targets[a["validation"]]), np.full(10, 500))
        self.assertEqual(trainer.digest(a), trainer.digest(b))

    def test_schedule_endpoints(self):
        m = manifest()
        self.assertAlmostEqual(trainer.learning_rate(1, m), .01)
        self.assertAlmostEqual(trainer.learning_rate(5, m), .1)
        self.assertAlmostEqual(trainer.learning_rate(6, m), .1)
        self.assertAlmostEqual(trainer.learning_rate(160, m), .0001)
        self.assertGreater(trainer.learning_rate(50, m), trainer.learning_rate(100, m))

    def test_exact_serialized_resume_bn_momentum_and_rng(self):
        m = manifest(); m.update(batch_size=2, epochs=8, warmup_epochs=2, score_epochs=2)
        device = torch.device("cpu")
        torch.manual_seed(99)
        x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
        for variant in ("plain", "residual"):
            model, opt, scaler, args = trainer.initialize(m, variant, 4, 1, device)
            for group in opt.param_groups:
                group["lr"] = trainer.learning_rate(1, m)
            train_epoch(model, opt, scaler, nn.CrossEntropyLoss(), x, y, args, device, check_gradients=True)
            cfg = trainer.config(m, variant, 4, 1, device, "test-split")
            with tempfile.TemporaryDirectory() as tmp:
                trainer.save_checkpoint(Path(tmp) / "last.pt", trainer.snapshot(model, opt, scaler, args, cfg, 1, [], None))
                saved = torch.load(Path(tmp) / "last.pt", weights_only=False)
            for group in opt.param_groups:
                group["lr"] = trainer.learning_rate(2, m)
            train_epoch(model, opt, scaler, nn.CrossEntropyLoss(), x, y, args, device, check_gradients=True)
            expected_shuffle = args.shuffle_generator.get_state().clone()
            resumed, ropt, rscaler, rargs = trainer.initialize(m, variant, 4, 1, device)
            trainer.restore(saved, cfg, resumed, ropt, rscaler, rargs)
            for group in ropt.param_groups:
                group["lr"] = trainer.learning_rate(saved["next_epoch"], m)
            train_epoch(resumed, ropt, rscaler, nn.CrossEntropyLoss(), x, y, rargs, device, check_gradients=True)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, resumed.state_dict()[k], rtol=0, atol=0)
            for a, b in zip(opt.state.values(), ropt.state.values()):
                torch.testing.assert_close(a["momentum_buffer"], b["momentum_buffer"], rtol=0, atol=0)
            torch.testing.assert_close(expected_shuffle, rargs.shuffle_generator.get_state(), rtol=0, atol=0)
            self.assertEqual(opt.param_groups[0]["lr"], ropt.param_groups[0]["lr"])
            with self.assertRaises(ValueError):
                trainer.restore(saved, {**cfg, "depth": 8}, resumed, ropt, rscaler, rargs)

    def test_shuffle_independent_of_model_depth_and_variant(self):
        orders = []
        for variant, depth in (("plain", 2), ("plain", 8), ("residual", 8)):
            _, _, _, args = trainer.initialize(manifest(), variant, depth, 2, torch.device("cpu"))
            orders.append(torch.randperm(20, generator=args.shuffle_generator))
        for order in orders[1:]:
            torch.testing.assert_close(order, orders[0], rtol=0, atol=0)

    def test_real_training_resume_history_and_test_isolation(self):
        m = manifest(); m.update(epochs=4, warmup_epochs=1, score_epochs=2, batch_size=2)
        data = (torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]),
                torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        with tempfile.TemporaryDirectory() as tmp, patch.object(trainer, "load_training", return_value=(data, "fixture")), \
                patch.object(trainer, "CIFAR10", side_effect=AssertionError("test data accessed during training")):
            root = Path(tmp)
            directory = root / "run"
            # Simulate a crash immediately after the first completed-epoch checkpoint is written.
            real_save = trainer.save_checkpoint
            calls = []
            def interrupted_save(path, state):
                real_save(path, state)
                calls.append(state["epoch"])
                raise RuntimeError("interrupted")
            with patch.object(trainer, "save_checkpoint", side_effect=interrupted_save), self.assertRaises(RuntimeError):
                trainer.train(m, root, directory, "plain", 2, 0, torch.device("cpu"), True)
            trainer.train(m, root, directory, "plain", 2, 0, torch.device("cpu"), True)
            summary = json.loads((directory / "summary.json").read_text())
            cfg = trainer.config(m, "plain", 2, 0, "cpu", "fixture")
            self.assertEqual(queue.completed(directory, cfg, m, False)["score"], summary["score"])
            history = [json.loads(r) for r in (directory / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in history], [1, 2, 3, 4])
            self.assertAlmostEqual(summary["score"], np.mean([r["eval/validation_accuracy"] for r in history[-2:]]))
            self.assertFalse(any("test" in k for r in history for k in r))
            # A completed training process is restartable without adding duplicate epochs.
            trainer.train(m, root, directory, "plain", 2, 0, torch.device("cpu"), True)
            self.assertEqual((directory / "metrics.jsonl").read_text().count("\n"), 4)

    def test_final_test_assessment_cached_and_report_plots(self):
        m = manifest(); m.update(epochs=4, warmup_epochs=1, score_epochs=2, batch_size=2)
        data = (torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]),
                torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        split = {"train": [0, 1, 2, 3], "validation": [4, 5]}
        split_hash = trainer.digest(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / queue.key("plain", 2, 0)
            trainer.atomic_json(root / "split.json", {"hash": split_hash, "indices": split})
            with patch.object(trainer, "load_training", return_value=(data, split_hash)):
                trainer.train(m, root, directory, "plain", 2, 0, torch.device("cpu"), True)
            dataset = argparse.Namespace(data=np.zeros((3, 32, 32, 3), dtype=np.uint8), targets=[0, 1, 2])
            with patch.object(trainer, "CIFAR10", return_value=dataset) as loader:
                trainer.assess(m, root, directory, "plain", 2, 0, torch.device("cpu"), True)
                trainer.assess(m, root, directory, "plain", 2, 0, torch.device("cpu"), True)
                self.assertEqual(loader.call_count, 1)
                self.assertFalse(loader.call_args.kwargs["train"])
            records, outcome = queue.report(m, root, "cpu", True, {"jobs": {}})
            self.assertIn(("plain", 2, 0), records)
            self.assertIsNone(outcome["stop"])
            self.assertEqual(len(list((root / "plots").glob("*.svg"))), 4)
            self.assertTrue((root / "comparison.md").exists())
            # Completion verification rejects summary tampering.
            path = directory / "summary.json"
            row = json.loads(path.read_text()); row["score"] += 1
            trainer.atomic_json(path, row)
            with self.assertRaises(ValueError):
                queue.records_at(m, root, "cpu", True)


class QueueTests(unittest.TestCase):
    def scores(self, values):
        return {d: {s: v + s * .01 for s in manifest()["seeds"]} for d, v in zip(manifest()["depths"], values)}

    def test_plateau_decline_cap_and_incomplete_group(self):
        m = manifest()
        plateau = queue.decision(m, self.scores([70, 72, 72.2, 72.3]))
        self.assertEqual(plateau["stop"], "plateau")
        self.assertEqual(plateau["visited"], [2, 4, 8, 16])
        self.assertEqual(queue.decision(m, self.scores([70, 68]))["stop"], "decline")
        self.assertEqual(queue.decision(m, self.scores([70, 72, 74, 76, 78]))["stop"], "cap")
        incomplete = self.scores([70, 72]); del incomplete[4][2]
        result = queue.decision(m, incomplete)
        self.assertIsNone(result["stop"])
        self.assertEqual(result["next_depth"], 4)
        self.assertEqual(result["visited"], [2])
        # One stalled increase is insufficient; later progress resets the streak.
        self.assertEqual(queue.decision(m, self.scores([70, 70.1, 72, 72.1]))["next_depth"], 32)

    def test_selection_ties_and_test_metrics_ignored(self):
        m = manifest()
        records = {}
        for variant, depth in (("plain", 2), ("plain", 4), ("residual", 4)):
            for seed in m["seeds"]:
                records[variant, depth, seed] = {"score": 75, "test_accuracy": 100 if variant == "residual" else 0}
        self.assertEqual(queue.select_winner(m, records)["depth"], 2)
        for seed in m["seeds"]:
            records["plain", 2, seed]["score"] = 74
        self.assertEqual(queue.select_winner(m, records)["variant"], "plain")

    def test_augmentation_gate_failure_and_incomplete_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state_path = root / "depth-state.json"
            trainer.atomic_json(root / "queue-state.json", {"jobs": {"example": {"status": "failed"}}})
            with self.assertRaisesRegex(RuntimeError, "augmentation suite failed"):
                queue.wait_for_augmentation(root, {}, state_path, .001)
            trainer.atomic_json(root / "queue-state.json", {"jobs": {}})
            with self.assertRaisesRegex(RuntimeError, "absent/incomplete"):
                queue.wait_for_augmentation(root, {}, state_path, .001)
            from run_augmentation_queue import load_manifest
            augmentation = load_manifest(trainer.HERE / "augmentation_experiments.json")
            trainer.atomic_json(root / "manifest.json", {"manifest": augmentation, "device": "cuda", "no_wandb": True})
            with patch("run_augmentation_queue.completed", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "incomplete job"):
                    queue.wait_for_augmentation(root, {}, state_path, .001)
            with patch("run_augmentation_queue.completed", return_value=True) as check:
                lock = queue.wait_for_augmentation(root, {}, state_path, .001)
                self.assertEqual(check.call_count, 45)
                lock.close()

    def test_adaptive_execution_failure_restart_and_residual_transition(self):
        m = manifest(); m["depths"] = [2, 4, 8, 16]
        stored, launches = {}, []
        fail = [True]
        def fake_completed(directory, expected, m, require_wandb):
            return stored.get(str(directory))
        def fake_report(m, root, device, no_wandb, state=None):
            records = {}
            for v, d in queue.candidates(m):
                for s in m["seeds"]:
                    row = stored.get(str(root / queue.key(v, d, s)))
                    if row: records[v, d, s] = row
            scores = {d: {s: records["plain", d, s]["score"] for s in m["seeds"] if ("plain", d, s) in records} for d in m["depths"]}
            return records, queue.decision(m, scores)
        def fake_subprocess(cmd, **kwargs):
            action = cmd[3]
            directory = Path(cmd[cmd.index("--directory") + 1])
            variant = cmd[cmd.index("--variant") + 1]
            depth = int(cmd[cmd.index("--depth") + 1]); seed = int(cmd[cmd.index("--seed") + 1])
            launches.append((action, variant, depth, seed))
            if action == "smoke":
                trainer.atomic_json(directory / "smoke.json", {"peak_gpu_memory_mb": 10, "estimated_run_seconds": 1})
                trainer.atomic_json(root / "split.json", {"hash": "fixture"})
            elif action == "train":
                if variant == "plain" and depth == 4 and seed == 1 and fail[0]:
                    fail[0] = False
                    return argparse.Namespace(returncode=1)
                stored[str(directory)] = {"score": {2: 70, 4: 70.1, 8: 70.2, 16: 90}[depth] + (1 if variant == "residual" else 0), "final": {}}
            return argparse.Namespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state_path = root / "queue-state.json"
            state = {"jobs": {}}
            args = argparse.Namespace(action="run", device="cpu", no_wandb=True)
            with patch.object(queue, "validate_architectures"), patch.object(queue, "report", side_effect=fake_report), \
                    patch.object(queue, "completed", side_effect=fake_completed), patch.object(queue.subprocess, "run", side_effect=fake_subprocess):
                with self.assertRaises(RuntimeError):
                    queue.execute(args, m, root, state, state_path, [])
                self.assertEqual(state["jobs"][queue.key("plain", 4, 1)]["status"], "failed")
                queue.execute(args, m, root, state, state_path, [])
                self.assertEqual(state["phase"], "completed")
                self.assertEqual(state["plain_decision"]["stop"], "plateau")
                self.assertEqual(state["selection"]["variant"], "residual")
                self.assertFalse(any(depth == 16 and action == "train" for action, v, depth, s in launches))
                self.assertEqual(sum(action == "train" and v == "plain" and d == 2 for action, v, d, s in launches), 3)
                self.assertEqual(sum(action == "train" and v == "residual" for action, v, d, s in launches), 6)
                self.assertEqual(sum(action == "test" for action, v, d, s in launches), 3)
                self.assertEqual(state["plain_decision"]["visited"], [2, 4, 8])


if __name__ == "__main__":
    unittest.main()
