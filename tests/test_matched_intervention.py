import unittest

import numpy as np

from hidden_memory.events import make_tape
from hidden_memory.hypergraph import sample_connected
from hidden_memory.learner import policy
from hidden_memory.matched_intervention import edit_subset,matched_rows,strata
from hidden_memory.parameters import Parameters
from hidden_memory.runner import run_trajectory
from hidden_memory.state import initialize


class MatchedInterventionTests(unittest.TestCase):
    def setUp(self):
        self.params=Parameters(r=.55,alpha=.02,beta=.4)
        self.state=initialize(sample_connected(20,5,20,np.random.default_rng(16)),17)
        self.state.q[:]=[3.,1.,1.]
        self.state.q[::2]=[1.,3.,1.]

    def test_matching_preserves_strata_count_and_dose(self):
        for action in (0,1):
            p,c,audit=matched_rows(self.state,self.params,action)
            self.assertEqual(len(p),30)
            self.assertEqual(len(c),30)
            np.testing.assert_array_equal(strata(self.state).ravel()[p],strata(self.state).ravel()[c])
            self.assertEqual(audit['l1_relative_difference'],0.)
            a=edit_subset(self.state,self.params,2,p,.5)
            b=edit_subset(self.state,self.params,action,c,.5)
            self.assertEqual(abs(a.q-self.state.q).sum(),abs(b.q-self.state.q).sum())
            np.testing.assert_array_equal(policy(a.q,.02),policy(b.q,.02))
            np.testing.assert_array_equal(edit_subset(self.state,self.params,2,p,1.).q,self.state.q)
            tape=make_tape(np.random.default_rng(29),400)
            x=run_trajectory(a,self.params,tape,freeze=True,backend='numba')
            y=run_trajectory(b,self.params,tape,freeze=True,backend='numba')
            for key in x:np.testing.assert_array_equal(x[key],y[key])

    def test_illegal_rows_rejected_and_no_gap_overlap_reported(self):
        with self.assertRaises(ValueError):edit_subset(self.state,self.params,0,[3],.5)
        self.state.q[:]=[3.,0.,2.9]
        p,c,audit=matched_rows(self.state,self.params,1)
        self.assertEqual(len(p),0)
        self.assertEqual(len(c),0)
        self.assertIsNone(audit['l1_relative_difference'])
