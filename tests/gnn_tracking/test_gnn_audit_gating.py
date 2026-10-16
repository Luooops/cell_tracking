"""models/gnn_tracking/audit_gating.py: gate decisions and output files on a synthetic CTC sequence with a division."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from models.gnn_tracking.audit_gating import audit_well, main


def write_sequence(root, well='s01'):
    """5 frames, 64 x 160 px. Track 1 sits still at x = 20 and divides after frame 2 into tracks 2 (x = 22, close)
    and 3 (x = 140, far beyond any gate); 6 background tracks keep the radius estimate away from the fallback."""
    (root / well).mkdir(parents=True)
    (root / f'{well}_GT' / 'TRA').mkdir(parents=True)
    still = [(8 + 12 * k, 52) for k in range(6)]                           # labels 10..15, never move
    for t in range(5):
        mask = np.zeros((64, 160), np.uint16)
        cells = [(10 + k, x, y) for k, (x, y) in enumerate(still)]
        cells += [(1, 20, 20)] if t <= 2 else [(2, 22, 20), (3, 140, 20)]
        for label, x, y in cells:
            mask[y - 2:y + 3, x - 2:x + 3] = label
        tifffile.imwrite(root / well / f't{t:03d}.tif', np.ones_like(mask))
        tifffile.imwrite(root / f'{well}_GT' / 'TRA' / f'man_track{t:03d}.tif', mask)
    table = ['1 0 2 0', '2 3 4 1', '3 3 4 1'] + [f'{10 + k} 0 4 0' for k in range(6)]
    (root / f'{well}_GT' / 'TRA' / 'man_track.txt').write_text('\n'.join(table) + '\n', encoding='utf-8')
    return well


class TestAuditWell(unittest.TestCase):
    def test_near_daughter_inside_far_daughter_outside(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, radius_rows, anomalies, summary = audit_well(root, write_sequence(root))
            md = {r['dst_label']: r for r in rows if r['kind'] == 'mother_daughter'}
            self.assertEqual(sorted(md), [2, 3])
            self.assertTrue(md[2]['in_gate'])
            self.assertFalse(md[3]['in_gate'])
            self.assertAlmostEqual(md[2]['distance'], 2.0)
            self.assertTrue(all(r['in_gate'] for r in rows if r['kind'] == 'continuation'))
            self.assertEqual(len(radius_rows), 4)                          # every frame but the last
            self.assertEqual((anomalies, summary['divisions']), ([], 1))


class TestAuditMain(unittest.TestCase):
    def test_writes_outputs_and_refuses_existing_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_sequence(root / 'data')
            out = root / 'audit'
            self.assertEqual(main(['--data-root', str(root / 'data'), '--output-root', str(out)]), 0)
            for name in ('links.csv', 'radius.csv', 'anomalies.csv', 'summary.json'):
                self.assertTrue((out / name).is_file(), name)
            summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['gate_recall']['mother_daughter']['1']['in_gate'], 1)
            self.assertTrue((root / 'logs' / 'gnn_tracking' / 'audit_audit_gating_data.log').is_file())
            with self.assertRaises(SystemExit):
                main(['--data-root', str(root / 'data'), '--output-root', str(out)])


if __name__ == '__main__':
    unittest.main()
