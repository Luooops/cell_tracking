"""utils/runners.py export_gnn_well: files of one GNN-tracked well (fake video, no torch_geometric)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from models.gnn_tracking.edge_record import EdgeRecorder, load_edges
from utils.runners import export_gnn_well


class FakeVideo:
    """Two frames: labels 7, 8 in frame 0 and 7, 9 in frame 1 (9 is a new GT track); radius 10 px."""
    def __init__(self):
        self.df = pd.DataFrame(dict(video='v', frame=[0, 0, 1, 1], track_id=[7, 8, 7, 9], x=[0.0, 20, 3, 20],
                                    y=[0.0, 0, 4, 5], area=[10, 10, 10, 10], mean_intensity=[1.0, 1, 1, 1]))
        self.frames = [0, 1]
        self.rows = {0: np.array([0, 1]), 1: np.array([2, 3])}
        self.pos = {t: self.df.loc[self.rows[t], ['x', 'y']].to_numpy() for t in self.frames}

    def radius(self, t):
        return 10.0


class TestExportGnnWell(unittest.TestCase):
    def test_writes_tracks_shifted_by_one_and_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'well'
            assign = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 2}
            report = export_gnn_well(out, FakeVideo(), assign)
            df = pd.read_csv(out / 'instance_tracks.csv')
            self.assertEqual(df.track_id.tolist(), [1, 2, 1, 3])
            self.assertEqual(df.instance_id.tolist(), [7, 8, 7, 9])
            self.assertEqual((report['perfect'], report['id_switches']), (100.0, 0))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['instance_tracks.csv'])

    def test_lineage_and_edges_written_when_given(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'well'
            video = FakeVideo()
            recorder = EdgeRecorder(video)
            recorder(1, 0, 1, [0], [0], [0.9], [[2.2, 2.0]], [True], [(0, 0)])
            export_gnn_well(out, video, {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}, {2: 0, 3: 0}, recorder)
            self.assertEqual(json.loads((out / 'lineage.json').read_text(encoding='utf-8')), {'3': 1, '4': 1})
            self.assertEqual(load_edges(out / 'edges.npz')['logits'].shape, (1, 2))

    def test_existing_destination_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                export_gnn_well(Path(folder), FakeVideo(), {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 2})


if __name__ == '__main__':
    unittest.main()
