import tempfile
import unittest
from pathlib import Path

import numpy as np

from hidden_memory.checkpoint import load_checkpoint, save_checkpoint
from hidden_memory.events import make_tape
from hidden_memory.hypergraph import from_edges
from hidden_memory.intervention import intervene
from hidden_memory.learner import policy, step
from hidden_memory.parameters import Parameters
from hidden_memory.runner import run_paired, run_trajectory
from hidden_memory.shock import apply_shock, sample_shock
from hidden_memory.state import clone, initialize, validate_state


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.p = Parameters(r=0.7, alpha=0.1, beta=0.8)
        self.s = initialize(from_edges(6, [[0, 1, 2], [2, 3, 4], [0, 4, 5]]), 4)
        self.s.strategy[:] = [0, 0, 2, 0, 2, 0]
        self.s.counts[:] = [[2, 0, 1], [1, 0, 2], [2, 0, 1]]
        self.s.q[:] = [0.4, 0.2, 0.1]
        self.tape = make_tape(np.random.default_rng(88), 600)

    def test_T07_all_state_policies_and_common_exclusion(self):
        self.s.q[0, 0] = [0.0, 0.0, 0.0]
        self.s.q[1, 1] = [0.0, 0.0, 0.5]
        self.s.q[2, 2] = [0.0, 0.0, -7.0]
        branches, eligible = intervene(self.s, self.p)
        self.assertFalse(eligible[0, 0])
        self.assertFalse(eligible[1, 1])
        self.assertFalse(eligible[2, 2])
        for scale, branch in branches.items():
            np.testing.assert_array_equal(policy(branch.q, self.p.epsilon), policy(self.s.q, self.p.epsilon))
            np.testing.assert_array_equal(branch.q[..., :2], self.s.q[..., :2])
            np.testing.assert_array_equal(branch.q[..., 2][~eligible], self.s.q[..., 2][~eligible])
            if scale == 1:
                np.testing.assert_array_equal(branch.q, self.s.q)

    def test_T08_frozen_every_event(self):
        branches, _ = intervene(self.s, self.p)
        for branch in branches.values():
            apply_shock(branch, np.array([0, 2]))
        before = {k: v.q.copy() for k, v in branches.items()}
        for event in self.tape:
            traces = [step(s, self.p, event, freeze=True) for s in branches.values()]
            self.assertEqual(len({(t.action, t.reward) for t in traces}), 1)
            for s in branches.values():
                np.testing.assert_array_equal(s.strategy, branches[1.0].strategy)
        for k, v in branches.items():
            np.testing.assert_array_equal(v.q, before[k])

    def test_T09_sham_clone_every_event(self):
        branches, _ = intervene(self.s, self.p)
        a, b = branches[1], clone(self.s)
        for event in self.tape:
            self.assertEqual(step(a, self.p, event), step(b, self.p, event))
            np.testing.assert_array_equal(a.q, b.q)
            np.testing.assert_array_equal(a.strategy, b.strategy)

    def test_T10_shock_only_strategy_and_counts(self):
        nodes = sample_shock(self.s, 0.5, np.random.default_rng(4))
        before = clone(self.s)
        apply_shock(self.s, nodes)
        self.assertEqual(np.count_nonzero(self.s.strategy == 1), 3)
        for name in ('q', 'visit_count', 'last_update_event'):
            np.testing.assert_array_equal(getattr(before, name), getattr(self.s, name))
        self.assertEqual(self.s.event, before.event)
        validate_state(self.s)
        with self.assertRaises(ValueError):
            sample_shock(self.s, 0.9, np.random.default_rng(4))

    def test_T12_checkpoint_reload_and_no_shared_memory(self):
        rng = np.random.default_rng(67)
        for event in make_tape(rng, 13):
            step(self.s, self.p, event)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            save_checkpoint(path, self.s, self.p, {'future': rng.bit_generator.state}, {'training_id': 'test', 'stable': False})
            restored, p, streams, metadata = load_checkpoint(path)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, self.s, self.p, {}, {})
            rng2 = np.random.default_rng()
            rng2.bit_generator.state = streams['future']
            a, b = make_tape(rng, 90), make_tape(rng2, 90)
            np.testing.assert_array_equal(a, b)
            for ea, eb in zip(a, b):
                self.assertEqual(step(self.s, self.p, ea), step(restored, p, eb))
            np.testing.assert_array_equal(self.s.q, restored.q)
            self.assertTrue(metadata['checksum'])
            for name in ('q', 'strategy', 'counts', 'visit_count', 'last_update_event'):
                self.assertFalse(np.shares_memory(getattr(self.s, name), getattr(restored, name)))
        copied = clone(self.s)
        self.assertFalse(np.shares_memory(self.s.graph.edges, copied.graph.edges))

    def test_T12_tape_chunk_invariance(self):
        a, b = np.random.default_rng(23), np.random.default_rng(23)
        np.testing.assert_array_equal(make_tape(a, 71), np.concatenate([make_tape(b, 13), make_tape(b, 58)]))

    def test_T12_checkpoint_detects_modified_array(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'original.npz'
            modified = Path(directory) / 'modified.npz'
            save_checkpoint(original, self.s, self.p, {}, {'training_id': 'corruption-test'})
            with np.load(original, allow_pickle=False) as data:
                payload = {key: data[key].copy() for key in data.files}
            payload['q'][0, 0, 0] += 1
            np.savez_compressed(modified, **payload)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch: q'):
                load_checkpoint(modified)

    def test_T07_intervened_arrays_are_independent(self):
        branches, _ = intervene(self.s, self.p)
        before = self.s.q.copy()
        branches[0.5].q[:] = 100
        np.testing.assert_array_equal(self.s.q, before)
        np.testing.assert_array_equal(branches[1.0].q, before)
        for name in ('q', 'strategy', 'counts', 'visit_count', 'last_update_event'):
            self.assertFalse(np.shares_memory(getattr(branches[0.5], name), getattr(branches[2.0], name)))

    def test_six_branches_mcs_and_frozen_loss(self):
        result = run_paired(self.s, self.p, self.tape, np.array([0]), freeze=True)
        self.assertEqual(len(result.trajectories), 6)
        self.assertEqual(result.theta, 0.0)
        for z in (0, 1):
            np.testing.assert_array_equal(result.trajectories[(0.5, z)]['q'], result.trajectories[(2.0, z)]['q'])
        self.assertEqual(len(result.trajectories[(1.0, 0)]['q']), 101)
        self.assertEqual(self.s.event, 0)
        with self.assertRaises(ValueError):
            run_trajectory(clone(self.s), self.p, self.tape[:7])
