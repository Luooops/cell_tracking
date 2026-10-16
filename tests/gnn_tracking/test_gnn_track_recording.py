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
            rows.append(dict(video='syn', frame=t, track_id=k + 1, x=40.0 * (k % 3) + 3 * t + rng.normal(0, 0.5),
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


if __name__ == '__main__':
    unittest.main()
