"""utils/calibration.py: ECE / NLL / Brier, reliability bins, mutual information and error coverage (synthetic)."""
import unittest

import numpy as np

from utils.calibration import (brier, calibration_report, coverage_area, coverage_curve, ece, mutual_information, nll,
                               reliability)


class TestCalibrationMetrics(unittest.TestCase):
    def test_calibrated_predictions_have_zero_ece(self):
        p, y = np.full(1000, 0.3), np.r_[np.ones(300), np.zeros(700)]
        self.assertAlmostEqual(ece(p, y), 0.0)
        self.assertAlmostEqual(ece(p, y, equal_mass=False), 0.0)

    def test_overconfident_predictions_have_ece_gap(self):
        p, y = np.full(100, 0.9), np.r_[np.ones(60), np.zeros(40)]
        self.assertAlmostEqual(ece(p, y), 0.3)
        self.assertAlmostEqual(ece(p, y, equal_mass=False), 0.3)

    def test_nll_and_brier_hand_values(self):
        p, y = np.array([0.8, 0.2]), np.array([1, 0])
        self.assertAlmostEqual(nll(p, y), -np.log(0.8))
        self.assertAlmostEqual(brier(p, y), 0.04)

    def test_equal_mass_bins_hold_equal_counts(self):
        bins = reliability(np.linspace(0, 1, 30), np.zeros(30), bins=15)
        self.assertEqual([b['n'] for b in bins], [2] * 15)
        self.assertLess(bins[0]['p_hi'], bins[1]['p_lo'])

    def test_tied_predictions_share_one_bin(self):
        p, y = np.r_[np.full(80, 0.999), np.linspace(0.1, 0.5, 20)], np.r_[np.ones(80), np.zeros(20)]
        bins = reliability(p, y, bins=10)
        self.assertEqual(sum(b['n'] for b in bins), 100)
        self.assertEqual(bins[-1]['n'], 80)
        self.assertEqual((bins[-1]['p_lo'], bins[-1]['p_hi']), (0.999, 0.999))

    def test_empty_population_gives_none(self):
        self.assertIsNone(ece([], []))
        self.assertEqual(reliability([], []), [])
        self.assertEqual(calibration_report(np.array([]), np.array([]))['n'], 0)


class TestMutualInformation(unittest.TestCase):
    def test_disagreement_is_high_agreement_is_low(self):
        mi = mutual_information(np.array([[0.95, 0.05], [0.6, 0.4], [0.9, 0.9]]))
        self.assertAlmostEqual(mi[0], 0.4946, places=3)          # confident members that disagree
        self.assertAlmostEqual(mi[1], 0.0201, places=3)          # both unsure: aleatoric, not epistemic
        self.assertAlmostEqual(mi[2], 0.0, places=6)


class TestCoverageCurve(unittest.TestCase):
    def test_perfect_ranking_finds_all_errors_first(self):
        error = np.r_[np.ones(2, bool), np.zeros(8, bool)]
        curve = coverage_curve(np.r_[[0.5, 0.4], np.zeros(8)], error)
        self.assertEqual(curve[20], 100.0)
        self.assertEqual(curve[10], 50.0)
        self.assertGreater(coverage_area(curve), 0.85)

    def test_uninformative_score_is_near_random(self):
        rng = np.random.default_rng(1)
        error = rng.random(20000) < 0.1
        self.assertLess(abs(coverage_area(coverage_curve(np.zeros(20000), error)) - 0.5), 0.03)

    def test_no_errors_gives_none(self):
        self.assertIsNone(coverage_curve(np.ones(5), np.zeros(5, bool)))
        self.assertIsNone(coverage_area(None))


if __name__ == '__main__':
    unittest.main()
