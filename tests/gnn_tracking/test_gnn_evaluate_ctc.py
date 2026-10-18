"""models/gnn_tracking/evaluate_ctc.py end to end with py-ctcmetrics: a perfect run scores 1, a broken link lowers LNK.

Needs py-ctcmetrics (skipped where it is not installed)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

HAS_CTC = importlib.util.find_spec('ctc_metrics') is not None


def write_dataset(root):
    """Sequence s01, 5 frames: cells 1 (x drifts) and 2 (still); 2 is missed in frame 2 (a gap in its label)."""
    (root / 'data' / 's01').mkdir(parents=True)
    tra = root / 'data' / 's01_GT' / 'TRA'
    tra.mkdir(parents=True)
    for t in range(5):
        mask = np.zeros((20, 80), np.uint16)
        mask[4:10, 5 + 2 * t:11 + 2 * t] = 1
        if t != 2:
            mask[4:10, 50:56] = 2
        tifffile.imwrite(root / 'data' / 's01' / f't{t:03d}.tif', np.ones_like(mask))
        tifffile.imwrite(tra / f'man_track{t:03d}.tif', mask)
    (tra / 'man_track.txt').write_text('1 0 4 0\n2 0 4 0\n', encoding='utf-8')


def write_run(root, name, tracks):
    folder = root / name / 's01'
    folder.mkdir(parents=True)
    rows = [(t, 1, tracks[0][t]) for t in range(5)] + [(t, 2, tracks[1][t]) for t in range(5) if t != 2]
    pd.DataFrame(rows, columns=['frame_index', 'instance_id', 'track_id']).to_csv(folder / 'instance_tracks.csv',
                                                                                  index=False)
    return root / name


@unittest.skipUnless(HAS_CTC, 'py-ctcmetrics is not installed')
class TestEvaluateCtc(unittest.TestCase):
    def run_eval(self, root, run, out):
        from models.gnn_tracking.evaluate_ctc import main
        self.assertEqual(main(['--data-root', str(root / 'data'), '--run', str(run), '--output-root', str(out),
                               '--threads', '1']), 0)
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))

    def test_perfect_run_scores_one_and_broken_link_lowers_lnk(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_dataset(root)
            perfect = self.run_eval(root, write_run(root, 'perfect', ({t: 1 for t in range(5)}, {t: 2 for t in range(5)})),
                                    root / 'eval_perfect')['per_well']['s01']
            self.assertEqual((perfect['gt'], perfect['DET'], perfect['TRA'], perfect['LNK']), ('canonical', 1.0, 1.0, 1.0))
            self.assertIsNone(perfect['BC(1)'])                              # no GT division: BC not requested
            broken = self.run_eval(root, write_run(root, 'broken', ({0: 1, 1: 1, 2: 3, 3: 3, 4: 3}, {t: 2 for t in range(5)})),
                                   root / 'eval_broken')['per_well']['s01']
            self.assertLess(broken['LNK'], 1.0)
            self.assertLess(broken['TRA'], 1.0)
            self.assertTrue((root / 'ctc_gt' / 'data' / 's01_GT' / 'canonical.json').is_file())   # built once, reused
            self.assertFalse((root / 'eval_broken' / 'res').exists())                              # RES removed


if __name__ == '__main__':
    unittest.main()
