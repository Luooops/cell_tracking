"""models/gnn_tracking/fit_calibration.py: per-member maps, nested over folds (synthetic answer sheets)."""
import unittest

import numpy as np

from models.gnn_tracking.fit_calibration import fit_maps, mapped, nested, score
from utils.calibration import sigmoid
from utils.lineage import SAME_CELL, UNRELATED


def sheet(seed, n=20000):
    """Round-1 edges whose 2 members are overconfident and biased upwards: true p = sigmoid(0.5 z - 1)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 4, (n, 2))
    y = rng.random(n) < sigmoid(0.5 * z.mean(1) - 1)
    p = sigmoid(z).mean(1)
    e = dict(round=np.ones(n, int), logits=z, p=p, in_solver=np.ones(n, bool), chosen=p > 0.5)
    return dict(edges=e, relation=np.where(y, SAME_CELL, UNRELATED).astype(object))


class TestFitCalibration(unittest.TestCase):
    def test_identity_maps_reproduce_the_tracker_probability(self):
        part = sheet(0)
        q = mapped(part['edges'], fit_maps(part['edges'], part['relation'], 'none'))
        np.testing.assert_allclose(q.mean(1), part['edges']['p'])

    def test_nested_platt_scores_each_fold_with_other_folds_and_improves_nll(self):
        parts = {f'w{i}': sheet(i) for i in range(3)}
        folds = [['w0'], ['w1'], ['w2']]
        results = {}
        for method in ('none', 'platt'):
            e, rel, q, per_fold = nested(parts, folds, method)
            results[method] = score(e, rel, q)['calibration']['round1_all']
            self.assertEqual([f['wells'] for f in per_fold], folds)
        self.assertLess(results['platt']['nll'], results['none']['nll'])
        self.assertLess(results['platt']['ece'], results['none']['ece'])


if __name__ == '__main__':
    unittest.main()
