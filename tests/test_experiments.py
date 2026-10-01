from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hidden_memory.campaign import execute_jobs
from hidden_memory.experiment_specs import build_jobs, job_seeds
from hidden_memory.training import train_job
from hidden_memory.pilot import screen_pilot
from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.events import make_tape
from hidden_memory.runner import run_trajectory
from hidden_memory.state import clone


class ExperimentTests(unittest.TestCase):
    def test_protocol_matrix_and_independent_seeds(self):
        e0, e1 = build_jobs('E0', 1701), build_jobs('E1', 1701)
        self.assertEqual(len(e0), 60)
        self.assertEqual(len(e1), 108)
        self.assertEqual({j.edges for j in e0}, {1382})
        self.assertEqual({j.training_mcs for j in e0}, {500000})
        self.assertEqual({j.nodes for j in e1}, {300})
        self.assertEqual(len({(j.r, j.alpha, j.beta) for j in e1}), 36)
        all_seeds = [seed for job in e0 + e1 for seed in job_seeds(job).values()]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertEqual(job_seeds(e1[0]), job_seeds(build_jobs('E1', 1701)[0]))

    def tiny_job(self):
        return replace(build_jobs('E1', 1701)[0], nodes=20, edges=32,
                       training_mcs=20, window_mcs=5, continuation_mcs=10,
                       response_mcs=10, chunk_mcs=3)

    def test_training_stream_checkpoint_continuation(self):
        job = self.tiny_job()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / job.job_id
            summary = train_job(job, output)
            self.assertEqual(summary['training_micro_updates'], 400)
            end, params, streams, meta = load_checkpoint(output / 'checkpoints/mcs-20.npz')
            initial, _, initial_streams, _ = load_checkpoint(output / 'checkpoints/mcs-0.npz')
            rng = np.random.default_rng()
            rng.bit_generator.state = initial_streams['training']
            reference = clone(initial)
            expected = run_trajectory(reference, params, make_tape(rng, 400), backend='numba')
            np.testing.assert_array_equal(end.q, reference.q)
            columns = json.loads((output / 'trajectory_columns.json').read_text())
            data = np.load(output / 'trajectory.npy')
            np.testing.assert_array_equal(data[:, columns.index('q')], expected['q'])
            self.assertEqual(data.shape[0], 21)
            self.assertEqual(meta['diagnostics']['training_id'], job.job_id)
            self.assertEqual(streams['training'], rng.bit_generator.state)
            with self.assertRaises(FileExistsError):
                train_job(job, output)

    def test_campaign_preserves_all_results(self):
        jobs = [replace(self.tiny_job(), stage='E0'),
                replace(self.tiny_job(), stage='E0', replicate=1)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'campaign'
            status = execute_jobs(jobs, output, workers=1)
            self.assertEqual(status['completed'], 2)
            self.assertEqual(status['failed'], 0)
            records = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
            self.assertEqual({r['job_id'] for r in records}, {j.job_id for j in jobs})
            self.assertTrue((output / 'source.tar.gz').exists())

    def test_pilot_saves_sham_pairs_and_skips_failed_training(self):
        job = self.tiny_job()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / job.job_id
            summary = train_job(job, output)
            failed = {**summary, 'stability': {'passed': False}}
            self.assertFalse(screen_pilot(job, output, failed)['candidate'])
            # Exercise the response pipeline independently of the tiny fixture's stability.
            fixture = {**summary, 'stability': {'passed': True}}
            result = screen_pilot(job, output, fixture)
            self.assertEqual(len(result['shocks']), 4)
            self.assertFalse(result['work_point_locked'])
            self.assertEqual(result['baseline']['delta'], 0)
            self.assertEqual(result['baseline']['shock_nodes'], [])
            for row in result['shocks']:
                if row['status'] == 'completed':
                    self.assertEqual(len(row['shock_nodes']), int(np.floor(row['delta'] * job.nodes)))
                    self.assertEqual(row['future_event_seed'], result['baseline']['future_event_seed'])
            self.assertTrue((output / 'pilot/summary.json').exists())

    def test_failed_job_is_retained_and_campaign_fails(self):
        job = replace(self.tiny_job(), edges=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failure'
            status = execute_jobs([job], output, workers=1)
            self.assertEqual(status['state'], 'failed')
            self.assertEqual(status['failed'], 1)
            failure = json.loads((output / 'jobs' / job.job_id / 'failure.json').read_text())
            self.assertEqual(failure['error_type'], 'ValueError')
            self.assertIn('Too few edges', failure['message'])
