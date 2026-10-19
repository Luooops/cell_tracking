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


def write_swap_dataset(root):
    """Like r14c03:15: track 1 (x=40) ends at frame 2; from frame 3 its nucleus carries the label of the existing
    track 3 (x=52 before), and 3's own nucleus continues as the new track 2. Background cells drift 8 px/frame so
    the gate is ~17 px."""
    (root / 's01').mkdir(parents=True)
    tra = root / 's01_GT' / 'TRA'
    tra.mkdir(parents=True)
    for t in range(5):
        mask = np.zeros((64, 200), np.uint16)
        cells = [(1, 40)] if t <= 2 else [(3, 41), (2, 53)]
        cells += [(3, 52)] if t <= 2 else []
        for label, x in cells:
            mask[18:23, x - 2:x + 3] = label
        for k in range(6):
            x = 8 + 24 * k + 8 * t
            mask[50:55, x:x + 5] = 10 + k
        tifffile.imwrite(root / 's01' / f't{t:03d}.tif', (mask > 0).astype(np.uint16) * 500 + 50)
        tifffile.imwrite(tra / f'man_track{t:03d}.tif', mask)
    rows = ['1 0 2 0', '2 3 4 0', '3 0 4 0'] + [f'{10 + k} 0 4 0' for k in range(6)]
    (tra / 'man_track.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')


class TestAuditHandover(unittest.TestCase):
    def test_swap_relabels_masks_and_translates(self):
        from utils.gt_refine import relabel_lookup, translate
        from utils.lineage import Lineage
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_swap_dataset(root / 'data')
            review = root / 'review'
            self.assertEqual(main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(review)]), 0)
            cands = json.loads((review / 'candidates.json').read_text(encoding='utf-8'))
            item = [i for i in cands['items'] if i['a_label'] == 1][0]
            self.assertIn(2, [s['b_label'] for s in item['starts']])
            self.assertIn((3, 3), [(n['label'], n['frame']) for n in item['neighbours']])
            self.assertEqual(item['frames_shown'][0], 0)                 # strip starts at A's first frame
            decisions = root / 'decisions.json'
            decisions.write_text(json.dumps(dict(review_id=cands['review_id'], decisions={
                item['id']: dict(choice='swap', swap_track=3, swap_frame=3, swap_rest=2)})))
            target = root / 'data_refine'
            self.assertEqual(main(['apply', '--data-root', str(root / 'data'), '--review', str(review),
                                   '--decisions', str(decisions), '--target', str(target), '--allow-incomplete']), 0)
            table = read_track_table(target / 's01_GT' / 'TRA' / 'man_track.txt')
            self.assertEqual((table[3], table[16], table[2]), ((0, 2, 0), (3, 4, 1), (3, 4, 3)))
            m3 = tifffile.imread(target / 's01_GT' / 'TRA' / 'man_track003.tif')
            self.assertEqual((int(m3[20, 41]), int(m3[20, 53])), (16, 2))
            self.assertEqual(int(tifffile.imread(root / 'data' / 's01_GT' / 'TRA' / 'man_track003.tif')[20, 41]), 3)
            entries = relabel_lookup(target)['s01']
            self.assertEqual((translate(entries, 2, 3), translate(entries, 3, 3)), (3, 16))
            obs = [(f, l) for f in range(5) for l in np.unique(tifffile.imread(target / 's01_GT' / 'TRA' /
                                                                           f'man_track{f:03d}.tif')) if l]
            self.assertEqual(Lineage(table, obs).relation((0, 1), (4, 16)), 'same_cell')

    def test_gap_item_and_manual_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_swap_dataset(root / 'data')
            # put a one-frame track 5 into a gap of track 2: 2's pixels in frame 4 become 5 (copy of the data)
            tra = root / 'data' / 's01_GT' / 'TRA'
            m = tifffile.imread(tra / 'man_track004.tif'); m[m == 2] = 5; tifffile.imwrite(tra / 'man_track004.tif', m)
            rows = ['1 0 2 0', '2 3 3 0', '3 0 4 0', '5 4 4 0'] + [f'{10 + k} 0 4 0' for k in range(6)]
            (tra / 'man_track.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            review = root / 'review'
            main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(review)])
            cands = json.loads((review / 'candidates.json').read_text(encoding='utf-8'))
            self.assertNotIn('s01:gap:5', [i['id'] for i in cands['items']])  # 2 is not seen after 5: no gap
            decisions = root / 'decisions.json'
            decisions.write_text(json.dumps(dict(review_id=cands['review_id'], decisions={}, manual=[
                dict(well='s01', label=5, from_frame=4, to_frame=4, target=2)])))
            target = root / 'data_refine'
            self.assertEqual(main(['apply', '--data-root', str(root / 'data'), '--review', str(review),
                                   '--decisions', str(decisions), '--target', str(target), '--allow-incomplete']), 0)
            table = read_track_table(target / 's01_GT' / 'TRA' / 'man_track.txt')
            self.assertEqual((5 in table, table[2]), (False, (3, 4, 0)))
            self.assertEqual(int(tifffile.imread(target / 's01_GT' / 'TRA' / 'man_track004.tif')[20, 53]), 2)
            log = json.loads((target / 'refine_log.json').read_text(encoding='utf-8'))
            self.assertEqual((log['counts']['manual'], log['relabels']),
                             (1, [dict(well='s01', label=5, from_frame=4, to_frame=4, new_label=2)]))

    def test_chain_through_unlabelled_detections(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_swap_dataset(root / 'data')
            # track 1 ends at frame 2; its nucleus continues unlabelled in frames 3 and 4 (the pixels of 3 there
            # become 50001 / 50002), and track 3 is cut back to frames 0-2
            tra = root / 'data' / 's01_GT' / 'TRA'
            for f, new in ((3, 50001), (4, 50002)):
                m = tifffile.imread(tra / f'man_track{f:03d}.tif'); m[m == 3] = new
                tifffile.imwrite(tra / f'man_track{f:03d}.tif', m)
            rows = ['1 0 2 0', '2 3 4 0', '3 0 2 0', '50001 3 3 0', '50002 4 4 0'] + [f'{10 + k} 0 4 0' for k in range(6)]
            (tra / 'man_track.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            review = root / 'review'
            main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(review)])
            cands = json.loads((review / 'candidates.json').read_text(encoding='utf-8'))
            item = [i for i in cands['items'] if i['id'] == 's01:1'][0]
            self.assertEqual([(c['label'], c['frame']) for c in item['chain']], [(50001, 3), (50002, 4)])
            decisions = root / 'decisions.json'
            decisions.write_text(json.dumps(dict(review_id=cands['review_id'], decisions={
                's01:1': dict(choice='chain', chain_until=4, chain_link=False)})))
            target = root / 'data_refine'
            self.assertEqual(main(['apply', '--data-root', str(root / 'data'), '--review', str(review),
                                   '--decisions', str(decisions), '--target', str(target), '--allow-incomplete']), 0)
            table = read_track_table(target / 's01_GT' / 'TRA' / 'man_track.txt')
            self.assertEqual((table[1], 50001 in table, 50002 in table), ((0, 4, 0), False, False))
            self.assertEqual(int(tifffile.imread(target / 's01_GT' / 'TRA' / 'man_track004.tif')[20, 41]), 1)

    def test_reuse_review_id_keeps_progress_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_dataset(root / 'data')
            main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(root / 'r1')])
            main(['prepare', '--data-root', str(root / 'data'), '--output-root', str(root / 'r2'),
                  '--reuse-review-id', str(root / 'r1' / 'candidates.json')])
            ids = [json.loads((root / r / 'candidates.json').read_text(encoding='utf-8'))['review_id'] for r in ('r1', 'r2')]
            self.assertEqual(ids[0], ids[1])

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
            self.assertEqual((log['counts']['same'], log['merges'], log['relabels']),
                             (1, [dict(well='s01', end=1, end_frame=2, start=2, start_frame=3)], []))
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
