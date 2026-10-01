import unittest

import numpy as np

from hidden_memory.metrics import recovery_summary, signed_loss, stability
from hidden_memory.config import validate_confirmatory


class MetricsTests(unittest.TestCase):
    def test_T11_recovery_start_and_right_censoring(self):
        result = recovery_summary(np.r_[np.zeros(5), np.ones(8)], 1.0, smooth=2, sustain=3)
        self.assertEqual(result['recovery_time'], 7)
        self.assertFalse(result['right_censored'])
        result = recovery_summary(np.zeros(20), 1.0, smooth=2, sustain=3)
        self.assertIsNone(result['recovery_time'])
        self.assertTrue(result['right_censored'])
        self.assertTrue(result['collapse'])
        self.assertEqual(result['restricted_unrecovered_time'], 20)

    def test_T11_threshold_equality_and_incomplete_window(self):
        self.assertFalse(recovery_summary(np.full(10, 0.8), 1.0, smooth=1, sustain=3)['collapse'])
        self.assertEqual(recovery_summary(np.full(10, 0.95), 1.0, smooth=1, sustain=3)['recovery_time'], 1)
        self.assertTrue(recovery_summary(np.r_[np.zeros(8), np.ones(2)], 1, smooth=1, sustain=3)['right_censored'])
        self.assertFalse(recovery_summary(np.full(300, 0.8), 1.0)['collapse'])
        self.assertEqual(recovery_summary(np.full(300, 0.95), 1.0)['recovery_time'], 50)
        self.assertTrue(recovery_summary(np.ones(10), 1.0)['right_censored'])

    def test_signed_loss_not_clipped(self):
        self.assertAlmostEqual(signed_loss(np.array([0.2, 0.4]), np.array([0.3, 0.5])), -0.1)

    def test_stability_requires_two_windows(self):
        self.assertTrue(stability(np.full(10, 0.9), np.full(10, 0.3), window=5)['passed'])
        self.assertFalse(stability(np.r_[np.ones(5), np.full(5, 0.8)], np.full(10, 0.3), window=5)['passed'])
        with self.assertRaises(ValueError):
            stability(np.ones(9), np.zeros(9), window=5)

    def test_unlocked_confirmatory_rejected(self):
        with self.assertRaisesRegex(ValueError, 'locked'):
            validate_confirmatory({'status': 'awaiting_pilot_lock'})
