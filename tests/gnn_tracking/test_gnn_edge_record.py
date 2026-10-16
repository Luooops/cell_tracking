"""models/gnn_tracking/edge_record.py: rows, columns and the saved file of the edge recorder (fake video, no torch)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from models.gnn_tracking.edge_record import COLUMNS, EdgeRecorder, load_edges


class FakeVideo:
    """Two frames of two cells; labels 7, 8 in frame 0 and 7, 9 in frame 1; radius 10 px."""
    def __init__(self):
        self.df = pd.DataFrame(dict(frame=[0, 0, 1, 1], track_id=[7, 8, 7, 9]))
        self.rows = {0: np.array([0, 1]), 1: np.array([2, 3])}
        self.pos = {0: np.array([[0.0, 0.0], [20.0, 0.0]]), 1: np.array([[3.0, 4.0], [20.0, 5.0]])}

    def radius(self, t):
        return 10.0


class TestEdgeRecorder(unittest.TestCase):
    def record(self):
        rec = EdgeRecorder(FakeVideo())
        logits = np.array([[2.0, 1.0], [-1.0, -2.0], [0.0, 0.5]])
        rec(1, 0, 1, [0, 0, 1], [0, 1, 1], [0.8, 0.2, 0.55], logits, [True, True, True], [(0, 0), (1, 1)])
        return rec

    def test_rows_hold_labels_distances_and_choices(self):
        edges = self.record().arrays()
        self.assertEqual(edges['src_label'].tolist(), [7, 7, 8])
        self.assertEqual(edges['dst_label'].tolist(), [7, 9, 9])
        np.testing.assert_allclose(edges['distance'], [5.0, np.hypot(20, 5), 5.0], rtol=1e-6)
        self.assertEqual(edges['chosen'].tolist(), [True, False, True])
        self.assertEqual(edges['logits'].shape, (3, 2))
        np.testing.assert_allclose(edges['gate'], 10.0)

    def test_round2_gate_scales_with_sqrt_gap(self):
        video = FakeVideo()
        video.rows[2], video.pos[2] = video.rows[1], video.pos[1]
        rec = EdgeRecorder(video)
        rec(2, 0, 2, [0], [0], [0.9], [[2.2]], [False], [])
        edges = rec.arrays()
        self.assertAlmostEqual(float(edges['gate'][0]), 10.0 * np.sqrt(2), places=4)
        self.assertEqual((edges['in_solver'].tolist(), edges['chosen'].tolist()), ([False], [False]))

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'edges.npz'
            self.record().save(path)
            edges = load_edges(path)
            self.assertEqual(set(edges), set(COLUMNS) | {'logits'})
            self.assertEqual(len(edges['p']), 3)

    def test_empty_recorder_saves_empty_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'edges.npz'
            EdgeRecorder(FakeVideo()).save(path)
            self.assertEqual(len(load_edges(path)['p']), 0)


if __name__ == '__main__':
    unittest.main()
