"""models/gnn_tracking track.py / track_division.py with an EdgeRecorder: recording must not change any track.

Needs torch_geometric (skipped where it is not installed, e.g. on machines that do not run the GNN)."""
import importlib.util
import unittest

import numpy as np
import pandas as pd

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None


def synthetic_video():
    """6 frames of 9 cells on a 40 px grid drifting 3 px per frame; cell 4 is missing in frame 3 (a gap for round 2)."""
    from models.gnn_tracking.graph import Video
    rng = np.random.default_rng(0)
    rows = []
    for t in range(6):
        for k in range(9):
            if k == 4 and t == 3:
                continue
            rows.append(dict(video='syn', frame=t, track_id=k + 1, cell_id=k + 1, x=40.0 * (k % 3) + 3 * t + rng.normal(0, 0.5),
                             y=40.0 * (k // 3) + rng.normal(0, 0.5), area=100.0 + k, mean_intensity=1.0 + 0.1 * k))
    df = pd.DataFrame(rows)
    crops = rng.random((len(df), 64, 64)).astype(np.float16)
    return Video(df, crops)


@unittest.skipUnless(HAS_PYG, 'torch_geometric is not installed')
class TestTrackVideoRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from models.gnn_tracking.model import DEVICE, TrackerGNN
        torch.manual_seed(0)
        cls.models = [TrackerGNN().to(DEVICE).eval() for _ in range(2)]

    def test_recording_leaves_tracks_unchanged(self):
        from models.gnn_tracking.edge_record import EdgeRecorder
        from models.gnn_tracking.track import track_video
        video = synthetic_video()
        rec = EdgeRecorder(video)
        self.assertEqual(track_video(self.models, video), track_video(self.models, video, recorder=rec))
        edges = rec.arrays()
        self.assertEqual(edges['logits'].shape[1], 2)
        p = 1 / (1 + np.exp(-edges['logits'].astype(np.float64)))
        np.testing.assert_allclose(edges['p'], p.mean(1), atol=1e-5)
        self.assertTrue(set(edges['round'].tolist()) <= {1, 2})
        self.assertTrue(edges['in_solver'][edges['round'] == 1].all())
        self.assertFalse((edges['chosen'] & ~edges['in_solver']).any())

    def test_recorded_round1_links_match_assignment(self):
        from models.gnn_tracking.edge_record import EdgeRecorder
        from models.gnn_tracking.track import track_video
        video = synthetic_video()
        rec = EdgeRecorder(video)
        assign = track_video(self.models, video, gap_round=False, recorder=rec)
        edges = rec.arrays()
        label = {(fi, i): int(video.df.loc[video.rows[t][i], 'track_id'])
                 for fi, t in enumerate(video.frames) for i in range(len(video.rows[t]))}
        same_track = {(fi, label[(fi, i)], label[(fi + 1, j)])
                      for fi in range(len(video.frames) - 1) for i in range(len(video.rows[video.frames[fi]]))
                      for j in range(len(video.rows[video.frames[fi + 1]])) if assign[(fi, i)] == assign[(fi + 1, j)]}
        chosen = {(int(t), int(a), int(b)) for t, a, b, c in
                  zip(edges['src_frame'], edges['src_label'], edges['dst_label'], edges['chosen']) if c}
        self.assertEqual(chosen, same_track)

    def test_division_recording_leaves_tracks_unchanged(self):
        from models.gnn_tracking.edge_record import EdgeRecorder
        from models.gnn_tracking.track_division import track_video_division
        video = synthetic_video()
        self.assertEqual(track_video_division(self.models, video),
                         track_video_division(self.models, video, recorder=EdgeRecorder(video)))

    def test_prior_correction_subtracts_each_members_log_pos_weight(self):
        from models.gnn_tracking.edge_record import EdgeRecorder
        from models.gnn_tracking.track import track_video
        video = synthetic_video()
        raw, corrected = EdgeRecorder(video), EdgeRecorder(video)
        for m, w in zip(self.models, (4.0, 2.0)):
            m.pos_weight = w
        try:
            track_video(self.models, video, gap_round=False, recorder=raw)
            track_video(self.models, video, gap_round=False, recorder=corrected, prior_correction=True)
        finally:
            for m in self.models:
                m.pos_weight = None
        a, b = raw.arrays(), corrected.arrays()
        first = a['src_frame'] == video.frames[0]          # the first frame pair has no velocities: same graph
        np.testing.assert_allclose(b['logits'][first], a['logits'][first] - np.log([4.0, 2.0]), atol=1e-4)

    def test_prior_correction_needs_recorded_weights(self):
        from models.gnn_tracking.track import track_video
        for m in self.models:
            m.pos_weight = None
        with self.assertRaises(ValueError):
            track_video(self.models, synthetic_video(), prior_correction=True)

    def test_kalman_ensemble_builds_motion_features_and_tracks(self):
        import torch
        from models.gnn_tracking import motion as kalman
        from models.gnn_tracking.graph import pair_graph
        from models.gnn_tracking.model import DEVICE, TrackerGNN
        from models.gnn_tracking.track import track_video
        from models.gnn_tracking.train import gt_motion
        params = dict(model='cv', q=1.0, r=0.5, v0=4.0, fits={})
        video = synthetic_video()
        state = gt_motion(video, video.frames[2], np.random.default_rng(0), params)
        gr = pair_graph(video, video.frames[2], 1, {}, motion=state)
        self.assertEqual(gr.candidate_edge_attr.shape[1], 14)
        self.assertEqual(gr.message_edge_attr.shape[1], 14)
        self.assertEqual(int(state['seen'].max()), 3)                       # frames 0, 1, 2 of a GT cell
        torch.manual_seed(0)
        models = [TrackerGNN(motion=params).to(DEVICE).eval() for _ in range(2)]
        assign = track_video(models, video)
        self.assertEqual(len(assign), sum(len(video.rows[t]) for t in video.frames))
        with self.assertRaises(ValueError):                                 # members must share the motion setting
            track_video([models[0], self.models[0]], video)
        self.assertIsNotNone(kalman)

    def test_original_thresholds_are_the_defaults(self):
        from models.gnn_tracking.track import track_video
        video = synthetic_video()
        self.assertEqual(track_video(self.models, video), track_video(self.models, video, link_thresh=0.5, gap_thresh=0.9))


if __name__ == '__main__':
    unittest.main()
