"""utils/gt_refine.py: model-free candidates, decision checks, hand-overs, swaps, refined dataset, label translation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from utils.gt_refine import (build_refined_dataset, check_decisions, gap_candidates, handover_candidates, refine_well,
                             relabel_lookup, translate, validate_table, write_track_table)
from utils.lineage import Lineage, read_track_table


def objects_for(tracks, n_frames=6, background=6):
    """tracks: {label: {frame: (x, y)}} plus still background cells far away (labels 100+) so the gating radius
    is estimated (floor 10 px). -> (frames, objects)."""
    out = {}
    for f in range(n_frames):
        rows = [(l, xy) for l, fs in tracks.items() for g, xy in fs.items() if g == f]
        rows += [(100 + k, (300.0 + 40 * k, 300.0)) for k in range(background)]
        rows.sort()
        out[f] = (np.array([l for l, _ in rows]), np.array([xy for _, xy in rows], dtype=float))
    return list(range(n_frames)), out


class TestHandoverCandidates(unittest.TestCase):
    def setUp(self):
        # 1 ends at frame 2; 2 starts at 3 nearby (g=1), 3 starts at 4 nearby (g=2), 4 starts at 3 far away,
        # 5 starts at 5 nearby (g=3, beyond the rule), 50001 (unknown identity) starts at 3 nearby,
        # 6 is an existing track that is near 1's end at frame 3 (a possible swap target)
        self.tracks = {1: {f: (50.0, 50.0) for f in range(3)}, 2: {f: (53.0, 50.0) for f in range(3, 6)},
                       3: {f: (50.0, 55.0) for f in range(4, 6)}, 4: {f: (150.0, 50.0) for f in range(3, 6)},
                       5: {5: (51.0, 50.0)}, 50001: {f: (52.0, 52.0) for f in range(3, 6)},
                       6: {**{f: (90.0, 50.0) for f in range(3)}, **{f: (55.0, 52.0) for f in range(3, 6)}}}
        self.table = {1: (0, 2, 0), 2: (3, 5, 0), 3: (4, 5, 0), 4: (3, 5, 0), 5: (5, 5, 0), 6: (0, 5, 0),
                      50001: (3, 5, 0)}
        self.table.update({100 + k: (0, 5, 0) for k in range(6)})

    def test_near_starts_and_existing_neighbours(self):
        frames, objects = objects_for(self.tracks)
        items = handover_candidates('w', frames, objects, self.table)
        self.assertEqual([i['a_label'] for i in items], [1])
        item = items[0]
        self.assertEqual([(s['b_label'], s['g']) for s in item['starts']], [(2, 1), (3, 2)])
        self.assertEqual([(n['label'], n['frame']) for n in item['neighbours']], [(6, 3)])
        self.assertEqual((item['id'], item['a_first']), ('w:1', 0))

    def test_end_with_child_or_start_with_parent_is_skipped(self):
        frames, objects = objects_for(self.tracks)
        table = {**self.table, 2: (3, 5, 1)}                             # 2 already continues 1
        self.assertEqual(handover_candidates('w', frames, objects, table), [])


ITEMS = [dict(id='w:1', well='w', a_label=1, a_frame=2, starts=[dict(b_label=2, b_frame=3), dict(b_label=3, b_frame=4)],
              neighbours=[dict(label=6, frame=3)]),
         dict(id='w:7', well='w', a_label=7, a_frame=2, starts=[dict(b_label=2, b_frame=3)], neighbours=[]),
         dict(id='w:9', well='w', a_label=9, a_frame=2, starts=[dict(b_label=10, b_frame=3)], neighbours=[])]


class TestCheckDecisions(unittest.TestCase):
    def test_counts_merges_and_undecided(self):
        r = check_decisions(ITEMS, {'w:1': dict(choice=3), 'w:7': dict(choice='none')})
        self.assertEqual((r['merges'], r['undecided'], r['problems']), ([('w', 1, 2, 3, 4)], ['w:9'], []))
        self.assertEqual((r['counts']['same'], r['counts']['none']), (1, 1))

    def test_swap_recorded_with_its_continuation(self):
        r = check_decisions(ITEMS, {'w:1': dict(choice='swap', swap_track=6, swap_frame=3, swap_rest=2)})
        self.assertEqual((r['swaps'], r['problems'], r['counts']['swap']), ([('w', 1, 2, 6, 3, 2, 3)], [], 1))

    def test_swap_must_name_a_neighbour(self):
        r = check_decisions(ITEMS, {'w:1': dict(choice='swap', swap_track=4, swap_frame=3, swap_rest=None)})
        self.assertEqual(len(r['problems']), 1)

    def test_start_claimed_by_handover_and_swap_is_a_problem(self):
        r = check_decisions(ITEMS, {'w:1': dict(choice='swap', swap_track=6, swap_frame=3, swap_rest=2),
                                    'w:7': dict(choice=2)})
        self.assertEqual(len(r['problems']), 1)

    def test_invalid_choice_and_unknown_item_are_problems(self):
        r = check_decisions(ITEMS, {'w:1': dict(choice=99), 'x:5': dict(choice='none')})
        self.assertEqual(len(r['problems']), 2)

    def test_cleared_choice_counts_as_undecided(self):
        self.assertIn('w:1', check_decisions(ITEMS, {'w:1': dict(note='looked twice')})['undecided'])


class TestRefineWell(unittest.TestCase):
    table = {1: (0, 2, 0), 2: (3, 5, 0), 6: (0, 5, 0)}
    observed = {1: [0, 1, 2], 2: [3, 4, 5], 6: [0, 1, 2, 3, 4, 5]}

    def test_handover_becomes_single_child(self):
        table, relabels = refine_well(self.table, self.observed, [(1, 2, 2, 3)], [])
        self.assertEqual((table[2], relabels), ((3, 5, 1), []))
        lineage = Lineage(table, [(f, l) for l, fs in self.observed.items() for f in fs])
        self.assertEqual(lineage.relation((0, 1), (5, 2)), 'same_cell')

    def test_swap_splits_existing_track(self):
        table, relabels = refine_well(self.table, self.observed, [], [(1, 2, 6, 3, 2, 3)])
        self.assertEqual(relabels, [(6, 3, None, 7)])                   # 6 from frame 3 on becomes the new label 7
        self.assertEqual((table[6], table[7], table[2]), ((0, 2, 0), (3, 5, 1), (3, 5, 6)))
        obs = [(f, 1) for f in range(3)] + [(f, 6) for f in range(3)] + [(f, 7) for f in range(3, 6)] + \
              [(f, 2) for f in range(3, 6)]
        lineage = Lineage(table, obs)
        self.assertEqual((lineage.relation((0, 1), (4, 7)), lineage.relation((0, 6), (4, 2))), ('same_cell', 'same_cell'))

    def test_split_track_resolved_for_another_decision(self):
        # 6 is split at 3 by end 1; end 8 (ends at 4) continues as start 9 -- and 6's later part ends at 5,
        # so a hand-over from 6 at frame 5 must attach to the new label
        table = {1: (0, 2, 0), 6: (0, 5, 0), 9: (6, 7, 0)}
        observed = {1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 9: [6, 7]}
        out, relabels = refine_well(table, observed, [(6, 5, 9, 6)], [(1, 2, 6, 3, None, None)])
        self.assertEqual(out[9], (6, 7, relabels[0][3]))

    def test_impossible_parent_raises(self):
        with self.assertRaises(ValueError):
            refine_well({1: (0, 3, 0), 2: (3, 5, 0)}, {1: [0, 1, 2, 3], 2: [3, 4, 5]}, [(1, 3, 2, 3)], [])
        with self.assertRaises(ValueError):
            validate_table({1: (0, 2, 5)})


class TestBuildRefinedDataset(unittest.TestCase):
    def make_source(self, src):
        for seq in ('a', 'b'):
            (src / seq).mkdir(parents=True)
            (src / f'{seq}_GT' / 'TRA').mkdir(parents=True)
            for t in range(2):
                (src / seq / f't00{t}.tif').write_bytes(b'raw')
                mask = np.zeros((4, 8), np.uint16); mask[1:3, 1:3] = 6; mask[1:3, 5:7] = 1
                tifffile.imwrite(src / f'{seq}_GT' / 'TRA' / f'man_track00{t}.tif', mask)
            write_track_table({1: (0, 1, 0), 6: (0, 1, 0)}, src / f'{seq}_GT' / 'TRA' / 'man_track.txt')

    def test_tables_rewritten_relabelled_masks_rest_linked(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / 'src', Path(folder) / 'src_refine'
            self.make_source(src)
            original = tifffile.imread(src / 'a_GT' / 'TRA' / 'man_track001.tif')
            modes = build_refined_dataset(src, dst, ['a', 'b'], {'a': {1: (0, 1, 0), 6: (0, 0, 0), 7: (1, 1, 1)}},
                                          {'a': [(6, 1, None, 7)]})
            self.assertEqual(modes['rewritten'], 1)                        # only frame 1 of well a
            m1 = tifffile.imread(dst / 'a_GT' / 'TRA' / 'man_track001.tif')
            self.assertEqual(set(np.unique(m1)), {0, 1, 7})
            self.assertEqual(set(np.unique(tifffile.imread(dst / 'a_GT' / 'TRA' / 'man_track000.tif'))), {0, 1, 6})
            np.testing.assert_array_equal(tifffile.imread(src / 'a_GT' / 'TRA' / 'man_track001.tif'), original)
            self.assertEqual(read_track_table(dst / 'a_GT' / 'TRA' / 'man_track.txt')[7], (1, 1, 1))
            self.assertEqual(read_track_table(dst / 'b_GT' / 'TRA' / 'man_track.txt')[6], (0, 1, 0))
            with self.assertRaises(FileExistsError):
                build_refined_dataset(src, dst, ['a'], {})


class TestGapCandidates(unittest.TestCase):
    def test_track_inside_a_gap_is_a_candidate(self):
        # 353 is seen in frames 0-2 and 4-5 at x=50; 140 sits at x=52 only in frame 3; 7 is far away in frame 3
        tracks = {353: {f: (50.0, 50.0) for f in (0, 1, 2, 4, 5)}, 140: {3: (52.0, 50.0)}, 7: {3: (200.0, 50.0)}}
        frames, objects = objects_for(tracks)
        table = {353: (0, 5, 0), 140: (3, 3, 0), 7: (3, 3, 0), **{100 + k: (0, 5, 0) for k in range(6)}}
        items = gap_candidates('w', frames, objects, table)
        self.assertEqual([(i['id'], i['x_frames'], [t['label'] for t in i['targets']]) for i in items],
                         [('w:gap:140', [3], [353])])
        self.assertEqual((items[0]['targets'][0]['before_frame'], items[0]['targets'][0]['after_frame']), (2, 4))


class TestMoves(unittest.TestCase):
    table = {353: (0, 5, 0), 140: (3, 3, 0), 9: (0, 5, 0)}
    observed = {353: [0, 1, 2, 4, 5], 140: [3], 9: [0, 1, 2, 3, 4, 5]}

    def test_gap_fill_moves_track_into_target(self):
        r = check_decisions([dict(id='w:gap:140', kind='gap', well='w', x_label=140, x_frames=[3],
                                  targets=[dict(label=353)])], {'w:gap:140': dict(choice=353)})
        self.assertEqual((r['moves'], r['counts']['gap']), ([('w', 140, 3, 3, 353, 'w:gap:140')], 1))
        table, relabels = refine_well(self.table, self.observed, [], [], [m[1:] for m in r['moves']])
        self.assertEqual((140 in table, table[353], relabels), (False, (0, 5, 0), [(140, 3, 3, 353)]))

    def test_manual_edit_to_new_label(self):
        r = check_decisions([], {}, [dict(well='w', label=9, from_frame=4, to_frame=5, target='new')])
        table, relabels = refine_well(self.table, self.observed, [], [], [m[1:] for m in r['moves']])
        self.assertEqual((table[9], table[354], relabels), ((0, 3, 0), (4, 5, 0), [(9, 4, 5, 354)]))

    def test_target_already_present_or_track_decided_twice_raises(self):
        with self.assertRaises(ValueError):
            refine_well(self.table, self.observed, [], [], [(140, 3, 3, 9, 'manual 1')])
        with self.assertRaises(ValueError):
            refine_well(self.table, self.observed, [(140, 3, 353, 4)], [], [(140, 3, 3, 353, 'manual 1')])

    def test_invalid_manual_edit_is_a_problem(self):
        r = check_decisions([], {}, [dict(well='w', label=9, from_frame=5, to_frame=4, target=3)])
        self.assertEqual(len(r['problems']), 1)


class TestTranslate(unittest.TestCase):
    def test_lookup_and_translate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(relabel_lookup(root), {})
            (root / 'refine_log.json').write_text(json.dumps(dict(relabels=[
                dict(well='a', label=6, from_frame=3, new_label=7)])), encoding='utf-8')
            entries = relabel_lookup(root)['a']
            self.assertEqual([translate(entries, f, 6) for f in (2, 3, 9)], [6, 7, 7])
            self.assertEqual(translate(entries, 5, 7), 7)                   # refined labels stay as they are
            chained = [(140, 8, 8, 353), (353, 9, None, 595)]               # gap fill, then a swap of the target
            self.assertEqual([translate(chained, f, 140) for f in (7, 8)], [140, 353])
            self.assertEqual([translate(chained, f, 353) for f in (8, 9)], [353, 595])


if __name__ == '__main__':
    unittest.main()
