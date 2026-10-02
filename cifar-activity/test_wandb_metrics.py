"""Tests for truthful recovery of historic metrics; no W&B writes or GPU use."""
import unittest
from wandb_metric_audit import backfill_rows, final_values


def epoch(n):
    return {'epoch': n, 'train/eval_loss': 0.5, 'train/eval_accuracy': 80.0,
            'performance/optimizer_steps': n * 10, 'performance/elapsed_seconds': n * 2}


class MetricsTest(unittest.TestCase):
    def test_final_test_is_not_overwritten(self):
        values = final_values({'eval/test_accuracy': 70.0, 'eval/test_loss': 1.0,
                               'final/test_accuracy': 72.0, 'final/test_loss': 0.9,
                               'train/eval_loss': 0.5})
        self.assertEqual(values, {'final/train_loss': 0.5})

    def test_constant_lr_preserves_epoch_context(self):
        rows = backfill_rows([epoch(1), epoch(2)], {'lr': 0.01}, {}, 'legacy-test-monitoring')
        self.assertEqual([r['optimizer/lr'] for r in rows], [0.01, 0.01])
        self.assertEqual(rows[1]['performance/optimizer_steps'], 20)
        self.assertNotIn('train/eval_loss', rows[1])

    def test_partial_backfill_continues_without_duplicate_points(self):
        rows = backfill_rows([epoch(1), epoch(2), {'epoch': 1, 'optimizer/lr': 0.01}],
                             {'lr': 0.01}, {}, 'legacy-test-monitoring')
        self.assertEqual([r['epoch'] for r in rows], [2])

    def test_no_test_values_inferred_from_validation(self):
        row = {**epoch(1), 'eval/validation_accuracy': 90.0}
        self.assertEqual(backfill_rows([row], {}, {'final/validation_accuracy': 90.0},
                                      'validation-selection'), [])

    def test_final_assessment_is_one_point(self):
        rows = backfill_rows([epoch(1), epoch(2)], {},
                             {'final/test_accuracy': 85.0, 'final/test_loss': 0.7},
                             'validation-selection')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['epoch'], 2)
        self.assertEqual(rows[0]['eval/test_accuracy'], 85.0)

    def test_no_scheduled_lr_guess(self):
        self.assertEqual(backfill_rows([epoch(1)], {'lr': 0.1}, {}, 'validation-selection'), [])

    def test_existing_final_point_is_not_replayed(self):
        history = [epoch(1), {**epoch(2), 'eval/test_accuracy': 85.0, 'eval/test_loss': 0.7}]
        self.assertEqual(backfill_rows(history, {}, {'final/test_accuracy': 85.0, 'final/test_loss': 0.7},
                                      'validation-selection'), [])

    def test_nonfinite_metrics_never_repaired(self):
        self.assertEqual(final_values({'train/eval_loss': float('nan')}), {})


if __name__ == '__main__':
    unittest.main()
