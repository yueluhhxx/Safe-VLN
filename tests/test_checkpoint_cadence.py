"""CPU-only checkpoint gating and atomic payload compatibility checks."""

import ast
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'vlnce_baselines/common/checkpoint_cadence.py'
SPEC = importlib.util.spec_from_file_location('checkpoint_cadence_test_helper', str(HELPER_PATH))
cadence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cadence)


class PureCadenceTest(unittest.TestCase):
    def test_requested_window_contains_exactly_41_checkpoints(self):
        observed = [iteration for iteration in range(200, 20001, 200)
                    if cadence.should_save_checkpoint(iteration, 12000, 20000, 200)]
        self.assertEqual(observed, list(range(12000, 20001, 200)))
        self.assertEqual(len(observed), 41)

    def test_inclusive_bounds_and_off_cadence_are_explicit(self):
        expected = {11999: False, 12000: True, 12001: False, 12199: False,
                    12200: True, 19999: False, 20000: True, 20001: False, 20200: False}
        for iteration, decision in expected.items():
            self.assertEqual(cadence.should_save_checkpoint(iteration, 12000, 20000, 200), decision)

    def test_default_legacy_keeps_every_existing_valid_save_call(self):
        for iteration in (0, 1, 200, 999, 12000, 20000, 20001):
            self.assertTrue(cadence.should_save_checkpoint(iteration))
        self.assertEqual(cadence.validate_checkpoint_cadence(), (0, 0, 0))

    def test_unbounded_end_and_zero_interval(self):
        self.assertTrue(cadence.should_save_checkpoint(100000, 12000, 0, 200))
        self.assertFalse(cadence.should_save_checkpoint(11999, 12000, 0, 0))
        self.assertTrue(cadence.should_save_checkpoint(12001, 12000, 20000, 0))
        self.assertTrue(cadence.should_save_checkpoint(20000, 12000, 20000, 0))
        self.assertFalse(cadence.should_save_checkpoint(20001, 12000, 20000, 0))

    def test_positive_cadence_is_anchored_at_start(self):
        self.assertTrue(cadence.should_save_checkpoint(125, 25, 0, 100))
        self.assertFalse(cadence.should_save_checkpoint(100, 25, 0, 100))
        self.assertTrue(cadence.should_save_checkpoint(25, 25, 25, 100))

    def test_invalid_boundaries_and_noninteger_types_are_rejected(self):
        for values in [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (12000, 10000, 200)]:
            with self.assertRaises(ValueError):
                cadence.validate_checkpoint_cadence(*values)
        for position in range(3):
            for invalid in (True, False, 200., '200', None):
                values = [0, 0, 0]; values[position] = invalid
                with self.assertRaises(ValueError):
                    cadence.validate_checkpoint_cadence(*values)
        for invalid in (-1, True, 200., '200'):
            with self.assertRaises(ValueError):
                cadence.should_save_checkpoint(invalid)

    def test_fresh_training_cannot_silently_miss_requested_save_calls(self):
        cadence.validate_checkpoint_save_calls(12000, 20000, 200, 200)
        cadence.validate_checkpoint_save_calls(12000, 20000, 400, 200)
        for values in [(12000, 20000, 200, 2000), (12050, 20000, 200, 200),
                       (12000, 20000, 200, 0), (12000, 20000, 200, True)]:
            with self.assertRaises(ValueError):
                cadence.validate_checkpoint_save_calls(*values)
        # A pure legacy gate does not redefine the parent's call schedule.
        cadence.validate_checkpoint_save_calls(12050, 0, 0, 2000)

    def test_new_config_defaults_preserve_legacy_behavior(self):
        source = ast.parse((ROOT / 'vlnce_baselines/config/default.py').read_text())
        defaults = {}
        for node in source.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Attribute) and target.attr.startswith('checkpoint_'):
                    defaults[target.attr] = ast.literal_eval(node.value)
        for name in ('checkpoint_start_iter', 'checkpoint_end_iter', 'checkpoint_interval'):
            self.assertEqual(defaults[name], 0)


class ManagedCheckpointGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FakeParent:
            def __init__(self, config):
                self.config = config
                self.parent_train_calls = 0
            def train(self):
                self.parent_train_calls += 1
                return 'parent training entered'
        source = ast.parse((ROOT / 'vlnce_baselines/managed_ss_trainer.py').read_text())
        function = copy.deepcopy(next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == '_fsync_directory'))
        trainer = copy.deepcopy(next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ManagedCheckpointRLTrainer'))
        trainer.decorator_list = []
        scope = {'RLTrainer': FakeParent, 'os': os, 'tempfile': tempfile, 'Path': Path, 'torch': torch,
                 'should_save_checkpoint': cadence.should_save_checkpoint,
                 'validate_checkpoint_cadence': cadence.validate_checkpoint_cadence,
                 'validate_checkpoint_save_calls': cadence.validate_checkpoint_save_calls}
        exec(compile(ast.Module(body=[function, trainer], type_ignores=[]), 'managed_checkpoint_gate', 'exec'), scope)
        cls.Trainer = scope['ManagedCheckpointRLTrainer']

    def trainer(self, destination, window=True, **overrides):
        options = {'log_every': 200, 'load_from_ckpt': False}
        if window:
            options.update(checkpoint_start_iter=12000, checkpoint_end_iter=20000, checkpoint_interval=200)
        options.update(overrides)
        trainer = self.Trainer(SimpleNamespace(IL=SimpleNamespace(**options), CHECKPOINT_FOLDER=str(destination)))
        trainer.policy = SimpleNamespace(state_dict=mock.Mock(return_value={'weight': torch.tensor([1.])}))
        trainer.optimizer = SimpleNamespace(state_dict=mock.Mock(return_value={'state': {0: {'step': 7}}}))
        trainer.scaler = SimpleNamespace(state_dict=mock.Mock(return_value={'scale': 8192., '_growth_tracker': 7}))
        return trainer

    def test_skipped_save_has_no_filesystem_or_state_extraction_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'must-not-be-created'
            trainer = self.trainer(destination)
            for iteration in (200, 11800, 12001, 20200):
                trainer.save_checkpoint(iteration)
            self.assertFalse(destination.exists())
            trainer.policy.state_dict.assert_not_called()
            trainer.optimizer.state_dict.assert_not_called()
            trainer.scaler.state_dict.assert_not_called()

    def test_actual_save_calls_publish_exactly_41_and_keep_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoints'
            trainer = self.trainer(destination)
            for iteration in range(200, 20001, 200):
                trainer.save_checkpoint(iteration)
            self.assertEqual(sorted(path.name for path in destination.iterdir()),
                             sorted('ckpt.iter{}.pth'.format(iteration) for iteration in range(12000, 20001, 200)))
            self.assertEqual(trainer.policy.state_dict.call_count, 41)
            checkpoint = torch.load(destination / 'ckpt.iter20000.pth', map_location='cpu')
            self.assertEqual(checkpoint['iteration'], 20000)
            self.assertEqual(checkpoint['checkpoint_format_version'], 1)
            self.assertEqual(checkpoint['optim_state'], {'state': {0: {'step': 7}}})
            self.assertEqual(checkpoint['amp_scaler_state'], {'scale': 8192., '_growth_tracker': 7})
            self.assertEqual(checkpoint['state_dict']['weight'].tolist(), [1.])
            self.assertEqual(checkpoint['config'].IL.checkpoint_start_iter, 12000)

    def test_missing_config_fields_keep_original_save_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoints'
            trainer = self.trainer(destination, window=False)
            for iteration in (0, 1, 200, 12345):
                trainer.save_checkpoint(iteration)
            self.assertEqual(len(list(destination.glob('ckpt.iter*.pth'))), 4)

    def test_invalid_window_fails_before_parent_training_or_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'not-created'
            for overrides in ({'checkpoint_start_iter': -1}, {'checkpoint_end_iter': 10000},
                              {'checkpoint_interval': True}, {'log_every': 2000}):
                trainer = self.trainer(destination, **overrides)
                with self.assertRaises(ValueError):
                    trainer.train()
                self.assertEqual(trainer.parent_train_calls, 0)
                self.assertFalse(destination.exists())

    def test_valid_training_delegates_without_changing_training_arguments(self):
        trainer = self.trainer(Path('/unused-checkpoint-path'))
        self.assertEqual(trainer.train(), 'parent training entered')
        self.assertEqual(trainer.parent_train_calls, 1)
        legacy = self.trainer(Path('/unused-checkpoint-path'), window=False)
        self.assertEqual(legacy.train(), 'parent training entered')

    def test_failed_atomic_write_does_not_replace_published_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoints'
            trainer = self.trainer(destination)
            trainer.save_checkpoint(12000)
            published = destination / 'ckpt.iter12000.pth'
            original = published.read_bytes()
            with mock.patch.object(torch, 'save', side_effect=RuntimeError('simulated write failure')):
                with self.assertRaises(RuntimeError):
                    trainer.save_checkpoint(12000)
            self.assertEqual(published.read_bytes(), original)
            self.assertEqual([path.name for path in destination.iterdir()], ['ckpt.iter12000.pth'])


if __name__ == '__main__':
    unittest.main()
