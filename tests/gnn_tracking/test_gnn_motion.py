"""models/gnn_tracking/motion.py: Kalman filter, maximum-likelihood fit and model selection (synthetic tracks)."""
import unittest

import numpy as np

from models.gnn_tracking.motion import features, fit, histories, predict, run_filter, state


def simulate(model, n=400, T=20, q=1.0, r=0.5, v0=4.0, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.empty((n, T, 2))
    pos = rng.uniform(0, 1000, (n, 2))
    vel = rng.normal(0, np.sqrt(v0), (n, 2)) if model == 'cv' else np.zeros((n, 2))
    for t in range(T):
        Z[:, t] = pos + rng.normal(0, np.sqrt(r), (n, 2))
        if model == 'cv':
            vel = vel + rng.normal(0, np.sqrt(q), (n, 2))
            pos = pos + vel
        else:
            pos = pos + rng.normal(0, np.sqrt(q), (n, 2))
    return Z


class TestMotion(unittest.TestCase):
    def test_model_selection_picks_the_generating_model(self):
        for model in ('cv', 'rw'):
            out = fit(simulate(model, n=150, T=15, seed=1))
            self.assertEqual(out['model'], model)

    def test_random_walk_noise_is_recovered(self):
        out = fit(simulate('rw', n=300, T=20, q=4.0, r=1.0, seed=2))
        rw = out['fits']['rw']
        self.assertAlmostEqual(rw['q'], 4.0, delta=0.6)
        self.assertAlmostEqual(rw['r'], 1.0, delta=0.6)

    def test_gaps_and_single_detections_widen_the_prediction(self):
        params = dict(model='cv', q=1.0, r=0.5, v0=4.0)
        full = [[(f, 10.0 * f, 0.0) for f in range(6)]]
        gap = [[(f, 10.0 * f, 0.0) for f in (0, 1, 2, 3, 5)]]
        once = [[(5, 50.0, 0.0)]]
        var = {name: predict(**{k: v for k, v in state(tr, 6, params).items() if k in ('mean', 'cov')}, g=1, params=params)[1][0]
               for name, tr in (('full', full), ('gap', gap), ('once', once))}
        self.assertLess(var['full'], var['gap'])
        self.assertLess(var['gap'], var['once'])
        pos, _ = predict(**{k: v for k, v in state(full, 6, params).items() if k in ('mean', 'cov')}, g=1, params=params)
        self.assertAlmostEqual(pos[0, 0], 60.0, delta=1.0)          # keeps moving 10 px per frame

    def test_rows_never_observed_stay_empty(self):
        Z = histories([[], [(0, 1.0, 2.0)]], 3)
        _, _, seen, _, _ = run_filter(Z, dict(model='rw', q=1.0, r=1.0))
        self.assertEqual(seen.tolist(), [0, 1])

    def test_features_scale_residuals_by_the_predicted_sd(self):
        params = dict(model='rw', q=1.0, r=1.0, v0=0.0)
        motion = state([[(0, 0.0, 0.0), (1, 0.0, 0.0)]], 2, params)
        edge, node = features(motion, 1, None, np.array([[3.0, 4.0]]), np.array([0]), np.array([0]), r=10.0)
        _, var = predict(motion['mean'], motion['cov'], 1, params)
        self.assertAlmostEqual(edge[0, 2], 5.0 / np.sqrt(var[0]), places=4)
        self.assertEqual((edge[0, 3], node[0, 2]), (1.0, 1.0))
        self.assertEqual(edge.shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
