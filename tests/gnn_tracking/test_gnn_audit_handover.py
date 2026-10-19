"""models/gnn_tracking/audit_handover.py: prepare (panels + review page) and apply (refined dataset) end to end."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from models.gnn_tracking.audit_handover import main
from utils.lineage import read_track_table


def write_dataset(root):
    """Sequence s01, 5 frames, 64 x 160 px: track 1 sits at x=40 in frames 0-2, the same nucleus continues as
    track 2 in frames 3-4 (a hand-over); 6 still background cells keep the radius estimate stable."""
    (root / 's01').mkdir(parents=True)
    tra = root / 's01_GT' / 'TRA'
    tra.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for t in range(5):
        mask = np.zeros((64, 160), np.uint16)
        mask[18:23, 38:43] = 1 if t <= 2 else 2
        for k in range(6):
            mask[50:55, 8 + 24 * k:13 + 24 * k] = 10 + k
        tifffile.imwrite(root / 's01' / f't{t:03d}.tif', (rng.random((64, 160)) * 100 + (mask > 0) * 500).astype(np.uint16))
        tifffile.imwrite(tra / f'man_track{t:03d}.tif', mask)
    rows = ['1 0 2 0', '2 3 4 0'] + [f'{10 + k} 0 4 0' for k in range(6)]
    (tra / 'man_track.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')


class TestAuditHandover(unittest.TestCase):
    def test_prepare_then_apply_builds_refined_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_dataset(root / 'data')
            review = root / 'review'
            self.assertEqual(main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(review)]), 0)
            cands = json.loads((review / 'candidates.json').read_text(encoding='utf-8'))
            self.assertEqual([(i['a_label'], [s['b_label'] for s in i['starts']]) for i in cands['items']], [(1, [2])])
            self.assertTrue((review / cands['items'][0]['panel']).is_file())
            page = (review / 'review.html').read_text(encoding='utf-8')
            self.assertIn(cands['review_id'], page)
            self.assertNotIn('/*__REVIEW_DATA__*/', page)

            decisions = root / 'decisions.json'
            decisions.write_text(json.dumps(dict(review_id=cands['review_id'], decisions={'s01:1': dict(choice=2)})))
            target = root / 'data_refine'
            self.assertEqual(main(['apply', '--data-root', str(root / 'data'), '--review', str(review),
                                   '--decisions', str(decisions), '--target', str(target)]), 0)
            self.assertEqual(read_track_table(target / 's01_GT' / 'TRA' / 'man_track.txt')[2], (3, 4, 1))
            self.assertEqual(read_track_table(root / 'data' / 's01_GT' / 'TRA' / 'man_track.txt')[2], (3, 4, 0))
            log = json.loads((target / 'refine_log.json').read_text(encoding='utf-8'))
            self.assertEqual((log['counts']['same'], log['merges']), (1, [dict(well='s01', end=1, start=2)]))
            np.testing.assert_array_equal(tifffile.imread(target / 's01_GT' / 'TRA' / 'man_track003.tif'),
                                          tifffile.imread(root / 'data' / 's01_GT' / 'TRA' / 'man_track003.tif'))

    def test_apply_refuses_other_review_and_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_dataset(root / 'data')
            review = root / 'review'
            main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(review)])
            review_id = json.loads((review / 'candidates.json').read_text(encoding='utf-8'))['review_id']
            for content in (dict(review_id='other', decisions={}), dict(review_id=review_id, decisions={})):
                path = root / 'd.json'
                path.write_text(json.dumps(content))
                with self.assertRaises(ValueError):
                    main(['apply', '--data-root', str(root / 'data'), '--review', str(review), '--decisions', str(path),
                          '--target', str(root / 'out')])
                self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
