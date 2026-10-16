"""models/gnn_tracking/gating.py: the adaptive gating radius and the term that bounds it (synthetic positions)."""
import unittest

import numpy as np

from models.gnn_tracking.gating import R_FALLBACK, R_MIN, SAFETY, SPACING_CAP, gating_radius


def moving_grid(n_frames, step, spacing, side=4):
    """side x side cells on a grid of the given spacing, all moving `step` px along x per frame."""
    grid = np.array([(i * spacing, j * spacing) for i in range(side) for j in range(side)], dtype=float)
    return {t: grid + (t * step, 0.0) for t in range(n_frames)}


def moving_pairs(n_frames, step, gap, n_pairs=6):
    """Pairs of cells `gap` px apart along x, pairs 200 px apart, all moving `step` px along y per frame."""
    base = np.array([(k * 200.0 + dx, 0.0) for k in range(n_pairs) for dx in (0.0, gap)])
    return {t: base + (0.0, t * step) for t in range(n_frames)}


class TestGatingRadius(unittest.TestCase):
    def test_too_few_cells_falls_back(self):
        pos = {0: np.array([[0.0, 0.0], [100.0, 0.0]]), 1: np.array([[1.0, 0.0], [101.0, 0.0]])}
        r, details = gating_radius([0, 1], pos, 0)
        self.assertEqual((r, details['bound']), (R_FALLBACK, 'fallback'))

    def test_small_steps_hit_floor(self):
        pos = moving_grid(4, step=1.0, spacing=100.0)
        r, details = gating_radius(sorted(pos), pos, 1)
        self.assertEqual((r, details['bound']), (R_MIN, 'floor'))
        self.assertAlmostEqual(details['step_q99'], 1.0)

    def test_moderate_steps_use_q99(self):
        pos = moving_grid(4, step=20.0, spacing=100.0)
        r, details = gating_radius(sorted(pos), pos, 1)
        self.assertEqual(details['bound'], 'q99')
        self.assertAlmostEqual(r, SAFETY * 20.0)

    def test_dense_cells_hit_spacing_cap(self):
        pos = moving_pairs(4, step=8.0, gap=10.0)
        r, details = gating_radius(sorted(pos), pos, 1)
        self.assertEqual(details['bound'], 'cap')
        self.assertAlmostEqual(details['spacing'], 10.0)
        self.assertAlmostEqual(r, SPACING_CAP * 10.0)


if __name__ == '__main__':
    unittest.main()
