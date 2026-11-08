"""CPU-only tests: the release launcher must not import training dependencies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_launcher', ROOT / 'scripts/run_experiment.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def options(command):
    start = command.index('--exp-config') + 2
    return dict(zip(command[start::2], command[start+1::2]))


class ReleaseCommandsTest(unittest.TestCase):
    def test_train_contract(self):
        command = launcher.build_command('train', Path('/example/data'), Path('/example/output'), 29500)
        values = options(command)
        self.assertEqual(values['IL.back_algo'], 'teleport')
        self.assertEqual(values['IL.iters'], '20000')
        self.assertEqual(values['IL.log_every'], '200')
        self.assertEqual(values['IL.load_from_ckpt'], 'False')
        self.assertIn('retrain_window_12000_20000.yaml', command[command.index('--exp-config')+1])

    def test_eval_contract(self):
        values = options(launcher.build_command('eval', Path('/example/data'), Path('/example/output'),
                                                29500, Path('/example/ckpt.iter12800.pth')))
        self.assertEqual(values['GPU_NUMBERS'], '8')
        self.assertEqual(values['NUM_ENVIRONMENTS'], '2')
        self.assertEqual(values['IL.log_every'], '2000')
        self.assertEqual(values['IL.back_algo'], 'control')
        self.assertEqual(values['IL.controller_motion_mode'], 'etpnav_distance')
        self.assertEqual(values['IL.ghost_route_mode'], 'nearest_front')
        self.assertEqual(values['MODEL.SAFE_VLN.DELTA'], '1e-05')
        self.assertEqual(values['EVAL.EPISODE_COUNT'], '-1')
        self.assertEqual(values['EVAL.fast_eval'], 'False')
        self.assertEqual(values['EVAL.USE_CKPT_CONFIG'], 'False')
        self.assertEqual(values['TASK_CONFIG.SEED'], '0')

    def test_filename_index_matches_trainer(self):
        self.assertEqual(launcher.checkpoint_index('ckpt.iter12800.pth'), 5)
        self.assertEqual(launcher.checkpoint_index('ckpt.iter13800.pth'), 5)
        self.assertEqual(launcher.checkpoint_index('ckpt.iter20000.pth'), 9)
        self.assertEqual(launcher.checkpoint_index('best.pth'), 0)

    def test_all_41_outputs_are_distinct_and_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'never-created'
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                launcher.main(['eval-window', '--checkpoint-dir', '/not/downloaded/checkpoints',
                               '--output-dir', str(output), '--dry-run'])
            plan = json.loads(stream.getvalue())
            self.assertEqual(len(plan['commands']), 41)
            self.assertEqual(len({options(c)['RESULTS_DIR'] for c in plan['commands']}), 41)
            self.assertFalse(output.exists())

    def test_nonempty_output_refused_without_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            marker = path / 'keep'
            marker.write_text('keep')
            with self.assertRaises(ValueError):
                launcher.new_output(path)
            self.assertEqual(marker.read_text(), 'keep')

    def test_data_logs_are_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output = root / 'input', root / 'output'
            (data / 'logs').mkdir(parents=True)
            (data / 'wp_pred').mkdir()
            output.mkdir()
            work = launcher.create_work(output, data)
            self.assertFalse((work / 'data/logs').is_symlink())
            self.assertTrue((work / 'data/wp_pred').is_symlink())
            self.assertFalse(any((data / 'logs').iterdir()))

    def test_gpu_protocol_not_silently_changed(self):
        self.assertEqual(launcher.visible_gpus('0,1,2,3,4,5,6,7'), '0,1,2,3,4,5,6,7')
        with self.assertRaises(Exception):
            launcher.visible_gpus('0,1')

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            launcher.window_candidates(Path('/example'), 12000, 20000, 0)
        with self.assertRaises(ValueError):
            launcher.window_candidates(Path('/example'), 12000, 20000, 300)


if __name__ == '__main__':
    unittest.main()
