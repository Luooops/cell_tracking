import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import tifffile

from models.trackastra import evaluate
from models.trackastra.run_trackastra import csv_read, csv_write, track_sequence
from utils.legacy_evaluate import main as shared_evaluate


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seq = self.root / 'source'
        self.seq.mkdir()
        self.uuid = '12345678-1234-1234-1234-123456789abc'
        self.sid = self.uuid + '__r01c01/f01__p01__ch01'
        self.frames, self.rows = [], []
        for i in range(2):
            name = f'r01c01f01p01-ch01t{i + 1:02}.tiff'
            mask = np.zeros((8, 8), dtype=np.uint16)
            mask[2:5, 2:5] = 1
            tifffile.imwrite(self.seq / name, mask)
            self.frames.append(dict(sequence_id=self.sid, frame_index=i, time_index=i + 1,
                image_name=name, source_relative_path=name, mask_path=name, overlay_path='',
                height=8, width=8, status='complete'))
            self.rows.append(dict(image_name=name, instance_id=1, track_id=99, frame_index=i,
                frame=i, x=3, y=3, mask_path=name, track_length=2, passes_min_track_length=True))
        self.save()
        graph = nx.DiGraph()
        graph.add_node(0, time=0, label=1)
        graph.add_node(1, time=1, label=1)
        graph.add_edge(0, 1)
        self.model = Mock()
        self.model._pretrained_name = 'test'
        self.model.device = 'cpu'
        self.model.track.return_value = (graph, None)

    def save(self):
        csv_write(self.seq / 'frames.csv', self.frames, list(self.frames[0]))
        csv_write(self.seq / 'instance_tracks.csv', self.rows,
            list(self.rows[0]) if self.rows else ['image_name', 'instance_id', 'track_id', 'frame_index', 'mask_path'])
        (self.seq / 'summary.json').write_text(json.dumps(dict(status='complete',
            input_dir=str(self.seq), parameters=dict(min_track_length=2))))

    def run_tracking(self, copy_masks=False):
        with patch.dict('sys.modules', {'trackastra': SimpleNamespace(__version__='test'),
                                       'torch': SimpleNamespace(__version__='test')}):
            return track_sequence(self.seq, None, self.root / 'predictions', self.model,
                                  'greedy_nodiv', copy_masks, False)

    def test_output_and_shared_evaluation(self):
        out = self.run_tracking()
        rows = csv_read(out / 'instance_tracks.csv')
        self.assertEqual([r['track_id'] for r in rows], ['1', '1'])
        for frame, row in zip(csv_read(out / 'frames.csv'), rows):
            self.assertEqual(frame['mask_path'], row['mask_path'])
            self.assertTrue((out / row['mask_path']).is_file())
        gt = self.root / 'gt.xml'
        gt.write_text('<annotations>' + ''.join(
            f'<image id="{i}" name="{f["image_name"]}" width="8" height="8">'
            '<polygon label="cell" points="2,2;4,2;4,4;2,4">'
            '<attribute name="track_id">A</attribute></polygon></image>'
            for i, f in enumerate(self.frames)) + '</annotations>')
        args = ['--gt', str(gt), '--dataset-id', self.uuid, '--batch', '--visualize', 'all',
                '--predictions-root', str(self.root / 'predictions')]
        self.assertEqual(evaluate.main(args + ['--output-root', str(self.root / 'a')]), 0)
        self.assertEqual(shared_evaluate(args + ['--output-root', str(self.root / 'b')]), 0)
        a, b = [self.root / x / self.uuid / 'r01c01' for x in ('a', 'b')]
        self.assertEqual({p.relative_to(a) for p in a.rglob('*')},
                         {p.relative_to(b) for p in b.rglob('*')})
        for p in a.rglob('*'):
            # Plot SVGs include generation dates and random element IDs.
            if p.is_file() and p.suffix in {'.csv', '.json', '.png'} and p.name != 'visualizations.csv':
                self.assertEqual(p.read_bytes(), (b / p.relative_to(a)).read_bytes())

    def test_copied_masks(self):
        out = self.run_tracking(True)
        self.assertTrue(all((out / r['mask_path']).is_file() for r in csv_read(out / 'instance_tracks.csv')))

    def test_incomplete_and_time_gap_rejected(self):
        self.frames[0]['status'] = 'failed'
        self.save()
        with self.assertRaisesRegex(ValueError, 'not complete'):
            self.run_tracking()
        self.frames[0]['status'] = 'complete'
        self.frames[1]['time_index'] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, 'time points'):
            self.run_tracking()
        self.model.track.assert_not_called()

    def test_empty_detections(self):
        for f in self.frames:
            tifffile.imwrite(self.seq / f['mask_path'], np.zeros((8, 8), dtype=np.uint16))
        self.rows = []
        self.save()
        out = self.run_tracking()
        self.assertEqual(csv_read(out / 'instance_tracks.csv'), [])
        self.model.track.assert_not_called()


if __name__ == '__main__':
    unittest.main()
