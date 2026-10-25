"""models/gnn_bayesian model.py / track.py / data.py on a synthetic video.

Needs torch_geometric (skipped where it is not installed, e.g. on machines that do not run the GNN)."""
import importlib.util
import unittest

import numpy as np

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None


@unittest.skipUnless(HAS_PYG, 'torch_geometric is not installed')
class TestGnnBayesian(unittest.TestCase):
    def test_priors_are_smoothed_and_exchanged_per_dataset(self):
        from models.gnn_bayesian.model import _shift, class_priors
        pri = class_priors({('scripta', 1): [900, 100, 0], ('hela', 1): [700, 250, 50]})
        self.assertGreater(pri['datasets']['scripta'][1][2], 0)
        self.assertAlmostEqual(sum(pri['mix'][1]), 1.0)

        class M:
            priors = pri
        self.assertLess(_shift(M, 1, 'scripta')[2], -3)            # division much rarer on scripta than in the mix
        self.assertGreater(_shift(M, 1, 'hela')[2], 0)
        self.assertEqual(_shift(M, 1, 'unknown').tolist(), [0, 0, 0])

    def test_three_way_network_tracks_with_and_without_divisions(self):
        import torch
        from models.gnn_bayesian.model import TrackerGNN3, class_priors, edge_scores
        from models.gnn_bayesian.track import Recorder, track_video
        from models.gnn_tracking.graph import pair_graph
        from models.gnn_tracking.model import DEVICE
        from tests.gnn_tracking.test_gnn_track_recording import synthetic_video
        torch.manual_seed(0)
        video = synthetic_video()
        models = [TrackerGNN3().to(DEVICE).eval() for _ in range(2)]
        for m in models:
            m.priors = class_priors({('syn', 1): [80, 20, 1], ('syn', 2): [80, 20, 1]})
        gr = pair_graph(video, video.frames[0], 1, {})
        s = edge_scores(models, gr, 1, 'syn')
        self.assertTrue(np.all(s['p_same'] + s['p_div'] <= 1 + 1e-6))
        self.assertEqual(s['logits'].shape, (gr.candidate_edge_index.shape[1], 2))
        for divisions in (False, True):
            rec = Recorder(video)
            assign, parent = track_video(models, video, 'syn', divisions=divisions, recorder=rec)
            self.assertEqual(len(assign), sum(len(video.rows[t]) for t in video.frames))
            e = rec.arrays()
            self.assertEqual(len(e['p_div']), len(e['p']))
            if not divisions:
                self.assertEqual(parent, {})


if __name__ == '__main__':
    unittest.main()
