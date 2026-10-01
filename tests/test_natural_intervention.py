import unittest

import numpy as np

from hidden_memory.events import make_tape
from hidden_memory.hypergraph import sample_connected
from hidden_memory.learner import policy
from hidden_memory.natural_intervention import transplant
from hidden_memory.parameters import Parameters
from hidden_memory.runner import run_trajectory
from hidden_memory.state import clone, initialize


class NaturalInterventionTests(unittest.TestCase):
    def setUp(self):
        self.params = Parameters(r=.55, alpha=.02, beta=.4)
        self.state = initialize(sample_connected(30, 5, 30, np.random.default_rng(15)), 27)
        self.state.q[:] = [3., 0., 2.]
        self.high = clone(self.state)
        self.high.q[..., 2] = 1.

    def test_self_mapping_policy_and_frozen_reference_acceleration(self):
        branches, mapping, report = transplant(self.state, self.state, self.high, self.params, 1)
        self.assertEqual(report['matched'], 90)
        np.testing.assert_array_equal(branches['low'].q, self.state.q)
        np.testing.assert_array_equal(mapping['high_gaps'], np.full(90, 2.))
        np.testing.assert_array_equal(policy(branches['high'].q, .02), policy(self.state.q, .02))
        tape = make_tape(np.random.default_rng(9), 300)
        reference = run_trajectory(clone(branches['low']), self.params, tape, freeze=True)
        for initial in branches.values():
            actual = run_trajectory(initial, self.params, tape, freeze=True, backend='numba')
            for key in actual: np.testing.assert_array_equal(actual[key], reference[key])

    def test_missing_support_and_joint_bound_exclusion(self):
        _, _, report = transplant(self.state, self.state, self.high, self.params, 100)
        self.assertEqual(report['matched'], 0)
        self.high.q[..., 2] = -100.
        branches, mapping, report = transplant(self.state, self.state, self.high, self.params, 1)
        self.assertEqual(report['matched'], 0)
        for branch in branches.values(): np.testing.assert_array_equal(branch.q, self.state.q)
        self.assertEqual(len(mapping['rows']), 0)
