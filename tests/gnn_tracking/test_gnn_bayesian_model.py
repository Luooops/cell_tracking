"""models/gnn_bayesian model.py / track.py / data.py on a synthetic video.

Needs torch_geometric (skipped where it is not installed, e.g. on machines that do not run the GNN)."""
import importlib.util
import unittest

import numpy as np

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None


@unittest.skipUnless(HAS_PYG, 'torch_geometric is not installed')
class TestGnnBayesian(unittest.TestCase):
    def test_priors_are_smoothed_and_only_division_is_exchanged(self):
        from models.gnn_bayesian.model import _shift, class_priors
        pri = class_priors({('scripta', 1): [900, 100, 0], ('hela', 1): [700, 250, 50]})
        self.assertGreater(pri['datasets']['scripta'][1][2], 0)
        self.assertAlmostEqual(sum(pri['mix'][1]), 1.0)

        class M:
            priors, n_classes = pri, 3
        self.assertEqual(_shift(M, 1, 'scripta').tolist(), [0, 0, 0])                     # default: none
        only_div = _shift(M, 1, 'scripta', 'division')
        self.assertEqual(only_div[:2].tolist(), [0, 0])               # same : unrelated left to the network
        self.assertLess(only_div[2], -3)                              # division much rarer on scripta than in the mix
        self.assertGreater(_shift(M, 1, 'hela', 'division')[2], 0)
        self.assertEqual(_shift(M, 1, 'unknown', 'division').tolist(), [0, 0, 0])

    def test_active_bias_history_and_capped_weights(self):
        import torch
        from models.gnn_bayesian.train import AB_CAP, PredictionHistory, active_bias_weights
        from models.gnn_tracking.model import DEVICE
        rng = np.random.default_rng(0)
        qs = rng.dirichlet(np.ones(3), size=(6, 50))
        hist = PredictionHistory(50, 3)
        for q in qs:                                                        # one epoch, two shuffled batches
            order = rng.permutation(50)
            for part in (order[:20], order[20:]):
                hist.add(torch.as_tensor(part, device=DEVICE), torch.as_tensor(q[part], dtype=torch.float32, device=DEVICE))
        u = hist.spread(torch.arange(50, device=DEVICE)).cpu().numpy()
        np.testing.assert_allclose(u, np.sqrt(qs.var(0, ddof=1).sum(1)), atol=1e-5)
        spiky = torch.cat([torch.zeros(99, device=DEVICE), torch.tensor([100.0], device=DEVICE)])
        w = active_bias_weights(spiky, torch.tensor(0.01, device=DEVICE))
        self.assertLessEqual(float(w.max()), AB_CAP + 1e-6)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=3)

    def test_three_way_network_tracks_with_and_without_divisions(self):
        import torch
        from models.gnn_bayesian.model import TrackerGNN3, class_priors, edge_scores
        from models.gnn_bayesian.confidence import summarize
        from models.gnn_bayesian.track import Recorder, _order, track_video
        from models.gnn_tracking.graph import pair_graph
        from models.gnn_tracking.model import DEVICE
        from tests.gnn_tracking.test_gnn_track_recording import synthetic_video
        torch.manual_seed(0)
        video = synthetic_video()
        models = [TrackerGNN3().to(DEVICE).eval() for _ in range(2)]
        for m in models:
            m.priors = class_priors({('syn', 1): [80, 20, 1], ('syn', 2): [80, 20, 1]})
        gr = pair_graph(video, video.frames[0], 1, {})
        s = edge_scores(models, [(gr, None)], 1, 'syn')
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
        two = edge_scores(models, [(gr, None), (gr, _order(gr, gr))], 1, 'syn')      # a view identical to the graph
        np.testing.assert_allclose(two['p_same'], s['p_same'], atol=1e-6)
        self.assertEqual(two['logits'].shape[1], 2 * len(models))
        for kw in (dict(ablate='appearance'), dict(ablate='velocity'), dict(cues=True)):
            rec = Recorder(video)
            assign, parent = track_video(models, video, 'syn', divisions=True, recorder=rec, **kw)
            self.assertEqual(len(assign), sum(len(video.rows[t]) for t in video.frames))
        e = rec.arrays()
        self.assertEqual(len(e['p_noapp']), len(e['p']))
        import pandas as pd
        tracks = pd.DataFrame([dict(frame_index=t, instance_id=int(video.df.loc[r, 'track_id']), track_id=assign[(fi, i)] + 1)
                               for fi, t in enumerate(video.frames) for i, r in enumerate(video.rows[t])])
        links, table, summary = summarize(e, tracks)
        self.assertEqual(len(links), int(e['chosen'].sum()))
        self.assertTrue(((table.p_all_correct >= 0) & (table.p_all_correct <= 1)).all())
        self.assertAlmostEqual(summary['expected_false_links'], float((1 - links.p).sum()), places=1)
        binary = [TrackerGNN3(2).to(DEVICE).eval()]
        binary[0].priors = class_priors({('syn', 1): [80, 21], ('syn', 2): [80, 21]})
        s = edge_scores(binary, [(gr, None)], 1, 'syn', 'division')
        self.assertEqual(float(np.abs(s['p_div']).max()), 0.0)
        assign, parent = track_video(binary, video, 'syn', divisions=True, prior='none')
        self.assertEqual(len(assign), sum(len(video.rows[t]) for t in video.frames))


if __name__ == '__main__':
    unittest.main()
