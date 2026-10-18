"""models/gnn_tracking/evaluate_answers.py: a synthetic answer sheet with a division, scored end to end."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from models.gnn_tracking.evaluate_answers import main


def write_run(root):
    """GT: mother 1 (frames 0-1) divides into 2 and 3 (frame 2); track 4 runs through frames 0-2.
    The tracker linked 1 -> 1, 4 -> 4, and one daughter (1 -> 2) as a continuation; no lineage.json."""
    gt = root / 'data' / 's01_GT' / 'TRA'
    gt.mkdir(parents=True)
    (gt / 'man_track.txt').write_text('1 0 1 0\n2 2 2 1\n3 2 2 1\n4 0 2 0\n', encoding='utf-8')
    well = root / 'run' / 's01'
    well.mkdir(parents=True)
    pd.DataFrame(dict(frame_index=[0, 0, 1, 1, 2, 2, 2], instance_id=[1, 4, 1, 4, 2, 3, 4],
                      track_id=[1, 2, 1, 2, 1, 3, 2])).to_csv(well / 'instance_tracks.csv', index=False)
    #        t  src dst    p     chosen  relation
    rows = [(0, 1, 1, 0.95, True),      # same cell
            (0, 1, 4, 0.10, False),     # unrelated
            (0, 4, 4, 0.99, True),      # same cell
            (0, 4, 1, 0.05, False),     # unrelated
            (1, 1, 2, 0.80, True),      # mother -> daughter
            (1, 1, 3, 0.70, False),     # mother -> daughter
            (1, 4, 4, 0.97, True),      # same cell
            (1, 1, 4, 0.30, False)]     # unrelated
    t, a, b, p, chosen = map(np.array, zip(*rows))
    z = np.log(p / (1 - p))
    np.savez(well / 'edges.npz', round=np.ones(8, np.int8), src_frame=t, g=np.ones(8, np.int8), src_label=a,
             dst_label=b, distance=np.ones(8), gate=np.full(8, 10.0), p=p, in_solver=np.ones(8, bool), chosen=chosen,
             logits=np.stack([z, z + 0.1], 1))


class TestEvaluateAnswers(unittest.TestCase):
    def test_scores_synthetic_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_run(root)
            out = root / 'eval'
            self.assertEqual(main(['--data-root', str(root / 'data'), '--run', str(root / 'run'),
                                   '--output-root', str(out)]), 0)
            report = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
            o = report['overall']
            self.assertEqual(o['relations'], {'mother_daughter': 2, 'same_cell': 3, 'unrelated': 3})
            self.assertEqual((o['calibration']['round1_all']['n'], o['calibration']['round1_all']['positives']), (6, 3))
            self.assertEqual(len(o['calibration']['round1_all']['members']), 2)
            self.assertEqual(o['calibration']['round1_decision']['n'], 5)     # binary edges with 0.02 <= p <= 0.98
            self.assertEqual(o['mother_daughter']['round1']['share_chosen'], 0.5)
            self.assertEqual((o['division']['gt_divisions'], o['division']['predicted'], o['division']['recall']),
                             (1, 0, 0.0))
            self.assertEqual(o['coverage']['false_links']['errors'], 0)      # no chosen edge is unrelated
            for name in ('metrics.json', 'reliability.csv', 'coverage.csv', 'figures/answers_summary.png',
                         'figures/answers_summary_legend.txt'):
                self.assertTrue((out / name).is_file(), name)
            with self.assertRaises(SystemExit):
                main(['--data-root', str(root / 'data'), '--run', str(root / 'run'), '--output-root', str(out)])


if __name__ == '__main__':
    unittest.main()
