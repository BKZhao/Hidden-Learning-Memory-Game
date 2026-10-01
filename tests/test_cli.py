import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hidden_memory.checkpoint import load_checkpoint


class CliTests(unittest.TestCase):
    def test_compiled_benchmark_reloads_cache_in_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                output = Path(directory) / str(index)
                result = subprocess.run(
                    [sys.executable, '-m', 'hidden_memory', 'benchmark', '--output', str(output),
                     '--events', '200', '--nodes', '20', '--backend', 'numba'],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((output / 'benchmark.json').read_text())
                self.assertEqual(report['backend'], 'numba')
                self.assertEqual(report['micro_updates'], 200)

    def test_audit_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit'
            cmd = [sys.executable, '-m', 'hidden_memory', 'audit', '--output', str(output),
                   '--nodes', '12', '--training-mcs', '20', '--response-mcs', '10']
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / 'audit.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['purpose'], 'engineering_validation_only')
            self.assertEqual(report['frozen_theta'], 0.0)
            self.assertEqual(report['branch_count'], 6)
            state, _, _, metadata = load_checkpoint(output / 'checkpoint.npz')
            self.assertEqual(state.event, 240)
            self.assertEqual(report['checkpoint_id'], metadata['checkpoint_id'])
            with np.load(output / 'paired_trajectories.npz', allow_pickle=False) as data:
                self.assertEqual(data['lambda_0.5_z_1_q'].shape, (11,))
            self.assertTrue((output / 'events.jsonl').stat().st_size > 0)
            again = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
