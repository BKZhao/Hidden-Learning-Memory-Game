from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hidden_memory.experiment_specs import Job
from hidden_memory.training import train_job


class CheckpointSelectionTests(unittest.TestCase):
    def test_shared_graph_independent_histories_and_first_passing_checkpoint(self):
        job = Job('E2', 0, 0, 7301, 20, 20, .85, .1, .4, 5000,
                  window_mcs=100, chunk_mcs=100)
        with tempfile.TemporaryDirectory() as directory:
            summaries = []
            for history in (0, 1):
                output = Path(directory)/str(history)
                summary = train_job(replace(job, replicate=history), output,
                                    network_seed=82, selection_times=(1000, 2000, 5000))
                summaries.append(summary)
                selection = summary['selection']
                passed = [d['mcs'] for d in selection['checked'] if d['stability']['passed']]
                self.assertTrue(passed)
                self.assertEqual(selection['selected_mcs'], passed[0])
                self.assertLess(selection['completed_mcs'], 5000)
                data = np.load(output/'trajectory.npy')
                self.assertEqual(len(data), selection['completed_mcs']+1)
                self.assertEqual(data[-1, 0], selection['completed_mcs'])
                config = json.loads((output/'config.json').read_text())
                self.assertEqual(config['seeds']['network'], 82)
            self.assertEqual(summaries[0]['network_hash'], summaries[1]['network_hash'])
            with np.load(Path(directory)/'0/checkpoints/mcs-0.npz') as a, np.load(Path(directory)/'1/checkpoints/mcs-0.npz') as b:
                np.testing.assert_array_equal(a['edges'], b['edges'])
                self.assertFalse(np.array_equal(a['strategy'], b['strategy']))

    def test_no_passing_checkpoint_preserves_final_failure(self):
        job = Job('E2', 0, 0, 7302, 20, 20, .21, .8, .0, 2000,
                  window_mcs=100, chunk_mcs=100)
        with tempfile.TemporaryDirectory() as directory:
            summary = train_job(job, Path(directory)/'run', selection_times=(1000, 2000))
            self.assertIsNone(summary['selection']['selected_mcs'])
            self.assertEqual(summary['selection']['completed_mcs'], 2000)
            self.assertEqual(len(summary['selection']['checked']), 2)
