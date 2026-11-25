import unittest
import numpy as np

from core import PROTOCOL, Watermark, fft, ifft, strength
from metrics import evaluate


class PaperFormulaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(PROTOCOL["key_seed"])
        self.key = rng.standard_normal((4, 64, 64))
        self.initial = rng.standard_normal((4, 64, 64))
        self.feature = rng.standard_normal((4, 64, 64))
        self.core = Watermark(self.key)

    def test_strength_and_coordinates(self):
        actual = strength(np.zeros_like(self.feature))
        np.testing.assert_allclose(actual, 1.6 + 0.6/(1+np.exp(5)))
        self.assertAlmostEqual(self.core.mask[0, 0], np.exp(-4))
        self.assertAlmostEqual(self.core.window[0, 0], np.exp(-1/.7**2))
        self.assertTrue(self.core.band[32, 42])
        self.assertFalse(self.core.band[32, 43])
        self.assertFalse(np.allclose(self.core.template[0], self.core.template[1]))

    def test_candidate_and_modulation(self):
        fourier = fft(self.initial)
        fourier[:, self.core.band] = fft(self.key)[:, self.core.band]
        candidate = ifft(fourier) - self.initial
        full = self.core.modulation(self.initial, self.feature)
        np.testing.assert_allclose(full, candidate * self.core.mask * self.core.window * strength(self.feature))
        for name in ("no_mcr", "no_mcre"):
            self.assertTrue(np.all(self.core.modulation(self.initial, self.feature, name)[1:] == 0))
        matched = self.core.modulation(self.initial, self.feature, "fixed_equal_energy")
        self.assertAlmostEqual(np.linalg.norm(full), np.linalg.norm(matched))

    def test_detector_scores(self):
        actual = np.abs(fft(self.core.mask*self.initial))[:, self.core.band].ravel()
        target = np.abs(fft(self.core.mask*self.key))[:, self.core.band].ravel()
        self.assertAlmostEqual(self.core.score(self.initial), np.corrcoef(actual,target)[0,1])
        self.assertAlmostEqual(self.core.score(self.initial,"negative_l1"), -np.mean(np.abs(actual-target)))

    def test_independent_threshold_and_ties(self):
        cal = np.linspace(-10, -1, 100)
        results = evaluate(cal, np.array([-1., 1.]), np.array([-2., -2.]), 20, 20262026)
        self.assertEqual(results["threshold"], -1.)
        self.assertEqual(results["fixed_tpr"], .5)
        self.assertEqual(results["actual_fpr"], 0.)
        self.assertEqual(results["roc_tpr_at_fpr_le_1pct"], 1.)


if __name__ == "__main__":
    unittest.main()
