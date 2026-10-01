import unittest

import numpy as np

from hidden_memory.events import make_tape
from hidden_memory.accelerated import _pairwise_sum
from hidden_memory.hypergraph import sample_connected
from hidden_memory.parameters import Parameters
from hidden_memory.runner import run_paired, run_trajectory
from hidden_memory.state import clone, initialize


class AcceleratedTests(unittest.TestCase):
    def test_reward_reduction_matches_strided_numpy_mean(self):
        rng = np.random.default_rng(41)
        for degree in (*range(1, 35), 127, 128, 129, 255, 256, 257, 1000):
            values = rng.normal(size=(degree, 3))
            for action in range(3):
                self.assertEqual(_pairwise_sum(values[:, action].copy(), 0, degree) / degree,
                                 float(values[:, action].mean()))

    def test_same_events_match_reference(self):
        for seed in (5, 29, 103):
            for freeze in (False, True):
                with self.subTest(seed=seed, freeze=freeze):
                    graph = sample_connected(20, 5, 32, np.random.default_rng(seed))
                    reference = initialize(graph, seed)
                    fast = clone(reference)
                    tape = make_tape(np.random.default_rng(seed + 1), 4000)
                    a = run_trajectory(reference, Parameters(r=0.7), tape, freeze)
                    b = run_trajectory(fast, Parameters(r=0.7), tape, freeze, backend='numba')
                    for metric in a:
                        np.testing.assert_array_equal(a[metric], b[metric])
                    for field in ('strategy', 'counts', 'visit_count', 'last_update_event'):
                        np.testing.assert_array_equal(getattr(reference, field), getattr(fast, field))
                    np.testing.assert_allclose(reference.q, fast.q, atol=1e-12, rtol=1e-12)
                    self.assertEqual(reference.event, fast.event)

    def test_accelerated_frozen_pairing_and_chunking(self):
        graph = sample_connected(20, 5, 32, np.random.default_rng(17))
        state = initialize(graph, 17)
        tape = make_tape(np.random.default_rng(19), 4000)
        params = Parameters(r=0.7)
        run_trajectory(state, params, tape, backend='numba')
        shocked = np.flatnonzero(state.strategy != 1)[:2]
        result = run_paired(state, params, tape, shocked, freeze=True, backend='numba')
        self.assertEqual(result.theta, 0.0)
        a, b = clone(state), clone(state)
        run_trajectory(a, params, tape, backend='numba')
        run_trajectory(b, params, tape[:1200], backend='numba')
        run_trajectory(b, params, tape[1200:], backend='numba')
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.strategy, b.strategy)

    def test_unknown_backend_and_trace_mode_are_explicit(self):
        graph = sample_connected(20, 5, 32, np.random.default_rng(17))
        state = initialize(graph, 17)
        tape = make_tape(np.random.default_rng(19), 20)
        with self.assertRaises(ValueError):
            run_trajectory(state, Parameters(r=0.7), tape, backend='invalid')
        with self.assertRaises(ValueError):
            run_trajectory(state, Parameters(r=0.7), tape, traces=[], backend='numba')
