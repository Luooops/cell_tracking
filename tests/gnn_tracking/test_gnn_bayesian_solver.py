"""models/gnn_bayesian/solver.py: continuations on p_same, divisions on p_division (synthetic)."""
import unittest

import numpy as np

from models.gnn_bayesian.solver import assign, match_with_divisions


class TestMatchWithDivisions(unittest.TestCase):
    def test_per_row_threshold_matches_the_original_rule(self):
        # one row, one column: linked iff p > the row's threshold
        for p, thresh, linked in ((0.45, 0.4, True), (0.35, 0.4, False), (0.45, 0.5, False)):
            links = assign(np.array([-np.log(p)]), np.array([0]), np.array([0]), 1, 1, [thresh], 0.4)
            self.assertEqual(bool(links), linked)

    def test_division_on_two_likely_daughters(self):
        # cell 0: daughters 0 and 1 likely (p_div 0.8, 0.7), no good continuation; cell 1 continues to 2
        row, col = np.array([0, 0, 1]), np.array([0, 1, 2])
        p_same, p_div = np.array([0.2, 0.1, 0.9]), np.array([0.8, 0.7, 0.01])
        links = sorted(match_with_divisions(p_same, p_div, row, col, 2, 3, 0.4, 0.5))
        self.assertEqual(links, [(0, 0), (0, 1), (1, 2)])

    def test_no_division_when_continuation_is_more_likely_or_daughters_weak(self):
        row, col = np.array([0, 0]), np.array([0, 1])
        for p_same, p_div in ((np.array([0.9, 0.1]), np.array([0.8, 0.7])),     # 0.56 < 0.9: continue
                              (np.array([0.3, 0.1]), np.array([0.8, 0.4]))):    # second daughter below 0.5
            links = match_with_divisions(p_same, p_div, row, col, 1, 2, 0.4, 0.5)
            self.assertLessEqual(len(links), 1)

    def test_without_division_probability_it_is_the_plain_assignment(self):
        rng = np.random.default_rng(0)
        row, col = np.repeat(np.arange(5), 3), rng.integers(0, 6, 15)
        p = rng.random(15)
        from models.gnn_tracking.solver import match
        plain = sorted(match(p, row, col, 5, 6, 0.4))
        self.assertEqual(sorted(match_with_divisions(p, np.zeros(15), row, col, 5, 6, 0.4, 0.5)), plain)


if __name__ == '__main__':
    unittest.main()
