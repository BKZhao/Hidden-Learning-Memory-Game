import unittest

import numpy as np

from hidden_memory.events import make_tape
from hidden_memory.hypergraph import from_edges, sample_connected
from hidden_memory.learner import choose_action, policy, step
from hidden_memory.parameters import Parameters
from hidden_memory.payoff import group_payoffs, projected_reward, reward, reward_bounds
from hidden_memory.state import initialize, validate_state


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.p = Parameters(r=0.7, alpha=0.1, beta=0.8)
        self.g = from_edges(5, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    def test_T01_group_payoffs(self):
        for counts, action, expected in [
            ([5, 0, 0], 0, 2.5), ([0, 5, 0], 1, 0),
            ([0, 0, 5], 2, 2.5),
        ]:
            self.assertAlmostEqual(group_payoffs(np.array(counts), self.p)[action], expected)
        np.testing.assert_allclose(group_payoffs(np.array([1, 2, 2]), self.p), [1.1, 0.5, 0.9])

    def test_T02_welfare_exhaustive(self):
        for c in range(6):
            for d in range(6 - c):
                p = 5 - c - d
                counts = np.array([c, d, p])
                expected = (5 * self.p.r - 1) * (c + p) - 0.9 * p * d
                self.assertAlmostEqual(counts @ group_payoffs(counts, self.p), expected)

    def test_T03_new_action_in_reward(self):
        s = initialize(self.g, 7)
        s.strategy[:] = 1
        s.counts[:] = [0, 3, 0]
        before = s.q.copy()
        event = np.array([0.0, 0.0, 0.0, 0.2])
        trace = step(s, self.p, event)
        self.assertEqual(trace.action, 0)
        self.assertAlmostEqual(trace.reward, -0.3)
        self.assertEqual(np.count_nonzero(s.q != before), 1)
        validate_state(s)

    def test_T04_old_target_when_state_equals_action(self):
        s = initialize(self.g, 7)
        s.strategy[:] = 0
        s.counts[:] = [3, 0, 0]
        s.q[0, 0] = [2, 0, 0]
        before = s.q.copy()
        trace = step(s, self.p, np.array([0.0, 0.9, 0.1, 0.0]))
        expected_target = 1.1 + 0.8 * 2
        self.assertAlmostEqual(trace.target, expected_target)
        self.assertAlmostEqual(s.q[0, 0, 0], 2 + 0.8 * (expected_target - 2))
        before[0, 0, 0] = s.q[0, 0, 0]
        np.testing.assert_array_equal(s.q, before)
        self.assertEqual(s.visit_count.sum(), 1)
        self.assertEqual(s.last_update_event[0, 0, 0], 0)

    def test_T05_ties_and_exploration(self):
        for row in [np.zeros(3), np.array([2.0, 2.0, 0.0]), np.array([0.0, 2.0, 0.0])]:
            expected = policy(row, 0.2)
            tape = make_tape(np.random.default_rng(19), 30000)
            selected = [choose_action(row, 0.2, e)[0] for e in tape]
            actual = np.bincount(selected, minlength=3) / len(selected)
            np.testing.assert_allclose(actual, expected, atol=0.012, rtol=0)
        np.testing.assert_array_equal(policy(np.zeros((2, 3, 3)), 0.02), np.full((2, 3, 3), 1 / 3))

    def test_T06_projection_all_candidate_actions(self):
        s = initialize(self.g, 71)
        for _ in range(30):
            for node in range(5):
                for action in range(3):
                    self.assertAlmostEqual(reward(s, node, action, self.p), projected_reward(s, node, action, self.p), delta=1e-10)
            step(s, self.p, make_tape(np.random.default_rng(s.event), 1)[0])

    def test_graph_sampling_and_validation(self):
        g = sample_connected(20, 4, 30, np.random.default_rng(13))
        self.assertEqual(g.edges.shape, (30, 4))
        self.assertTrue(np.all(g.degree > 0))
        self.assertEqual(len({tuple(e) for e in g.edges}), 30)
        with self.assertRaises(ValueError):
            from_edges(4, [[0, 1], [2, 3]])
        with self.assertRaises(ValueError):
            from_edges(3, [[0, 1, 1]])
        with self.assertRaises(ValueError):
            sample_connected(5, 3, 11, np.random.default_rng(1))

    def test_reward_bounds_only_occupied_actions(self):
        lo, hi = reward_bounds(3, self.p)
        values = []
        for c in range(4):
            for d in range(4 - c):
                counts = np.array([c, d, 3 - c - d])
                values.extend(group_payoffs(counts, self.p)[counts > 0])
        self.assertEqual((lo, hi), (min(values), max(values)))

    def test_invalid_parameters(self):
        for kwargs in [dict(r=float('nan')), dict(r=0.7, gamma=1), dict(r=0.7, eta=-1), dict(r=0.7, alpha=-1)]:
            with self.assertRaises(ValueError):
                Parameters(**kwargs)
