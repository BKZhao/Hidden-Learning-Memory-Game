import json
from pathlib import Path
import unittest

from scripts.run_dense_phase import dense_jobs
from scripts.run_dense_phase_response import response_seeds


ROOT = Path(__file__).resolve().parents[1]


class DensePhaseDesignTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/dense-phase-v1.0.json').read_text())

    def test_dense_grid_has_fixed_size_and_unique_jobs(self):
        jobs = dense_jobs(self.config)
        self.assertEqual(len(jobs), 378)
        self.assertEqual(len({job.job_id for job in jobs}), 378)
        self.assertEqual(len({(job.r, job.alpha, job.beta) for job in jobs}), 189)
        self.assertTrue(all(job.nodes == 1000 and job.edges == 1600 for job in jobs))
        self.assertTrue(all(job.training_mcs == 100000 for job in jobs))

    def test_response_streams_are_reproducible_and_distinct(self):
        seeds = [response_seeds(self.config['response_seed'], point, replicate)
                 for point in range(189) for replicate in range(2)]
        self.assertEqual(seeds, [response_seeds(self.config['response_seed'], point, replicate)
                                 for point in range(189) for replicate in range(2)])
        self.assertEqual(len({seed for pair in seeds for seed in pair}), 756)


if __name__ == '__main__':
    unittest.main()
