"""utils/gt_refine.py: model-free hand-over candidates, decision checks, merges and the refined dataset (synthetic)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.gt_refine import (apply_merges, build_refined_dataset, check_decisions, handover_candidates,
                             write_track_table)
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
        # 5 starts at 5 nearby (g=3, beyond the rule), 50001 (unknown identity) starts at 3 nearby
        self.tracks = {1: {f: (50.0, 50.0) for f in range(3)}, 2: {f: (53.0, 50.0) for f in range(3, 6)},
                       3: {f: (50.0, 55.0) for f in range(4, 6)}, 4: {f: (150.0, 50.0) for f in range(3, 6)},
                       5: {5: (51.0, 50.0)}, 50001: {f: (52.0, 52.0) for f in range(3, 6)}}
        self.table = {1: (0, 2, 0), 2: (3, 5, 0), 3: (4, 5, 0), 4: (3, 5, 0), 5: (5, 5, 0), 50001: (3, 5, 0)}
        self.table.update({100 + k: (0, 5, 0) for k in range(6)})

    def test_near_starts_within_two_frames_are_candidates(self):
        frames, objects = objects_for(self.tracks)
        items = handover_candidates('w', frames, objects, self.table)
        self.assertEqual([i['a_label'] for i in items], [1])
        self.assertEqual([(s['b_label'], s['g']) for s in items[0]['starts']], [(2, 1), (3, 2)])
        self.assertEqual(items[0]['id'], 'w:1')

    def test_end_with_child_or_start_with_parent_is_skipped(self):
        frames, objects = objects_for(self.tracks)
        table = {**self.table, 2: (3, 5, 1)}                             # 2 already continues 1
        self.assertEqual(handover_candidates('w', frames, objects, table), [])


class TestDecisions(unittest.TestCase):
    items = [dict(id='w:1', well='w', a_label=1, starts=[dict(b_label=2), dict(b_label=3)]),
             dict(id='w:7', well='w', a_label=7, starts=[dict(b_label=2)]),
             dict(id='w:9', well='w', a_label=9, starts=[dict(b_label=10)])]

    def test_counts_merges_and_undecided(self):
        r = check_decisions(self.items, {'w:1': dict(choice=3), 'w:7': dict(choice='none')})
        self.assertEqual((r['merges'], r['counts'], r['undecided'], r['problems']),
                         ([('w', 1, 3)], dict(same=1, none=1, unsure=0), ['w:9'], []))

    def test_one_start_claimed_twice_is_a_problem(self):
        r = check_decisions(self.items, {'w:1': dict(choice=2), 'w:7': dict(choice=2)})
        self.assertEqual(len(r['problems']), 1)

    def test_invalid_choice_and_unknown_item_are_problems(self):
        r = check_decisions(self.items, {'w:1': dict(choice=99), 'x:5': dict(choice='none')})
        self.assertEqual(len(r['problems']), 2)

    def test_cleared_choice_counts_as_undecided(self):
        r = check_decisions(self.items, {'w:1': dict(note='looked twice')})
        self.assertIn('w:1', r['undecided'])


class TestApplyMerges(unittest.TestCase):
    def test_start_becomes_single_child_of_end(self):
        table = apply_merges({1: (0, 2, 0), 2: (3, 5, 0), 3: (0, 5, 0)}, [(1, 2)])
        self.assertEqual(table[2], (3, 5, 1))
        lineage = Lineage(table, [(f, 1) for f in range(3)] + [(f, 2) for f in range(3, 6)])
        self.assertEqual(lineage.relation((0, 1), (5, 2)), 'same_cell')

    def test_overlapping_tracks_cannot_merge(self):
        with self.assertRaises(ValueError):
            apply_merges({1: (0, 3, 0), 2: (3, 5, 0)}, [(1, 2)])


class TestBuildRefinedDataset(unittest.TestCase):
    def test_tables_rewritten_files_linked_source_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / 'src', Path(folder) / 'src_refine'
            for seq in ('a', 'b'):
                (src / seq).mkdir(parents=True)
                (src / f'{seq}_GT' / 'TRA').mkdir(parents=True)
                (src / seq / 't000.tif').write_bytes(b'raw')
                (src / f'{seq}_GT' / 'TRA' / 'man_track000.tif').write_bytes(b'mask')
                write_track_table({1: (0, 2, 0), 2: (3, 5, 0)}, src / f'{seq}_GT' / 'TRA' / 'man_track.txt')
            original = (src / 'a_GT' / 'TRA' / 'man_track.txt').read_text()
            modes = build_refined_dataset(src, dst, ['a', 'b'], {'a': {1: (0, 2, 0), 2: (3, 5, 1)}})
            self.assertEqual(read_track_table(dst / 'a_GT' / 'TRA' / 'man_track.txt')[2], (3, 5, 1))
            self.assertEqual(read_track_table(dst / 'b_GT' / 'TRA' / 'man_track.txt')[2], (3, 5, 0))
            self.assertEqual((dst / 'a' / 't000.tif').read_bytes(), b'raw')
            self.assertEqual((src / 'a_GT' / 'TRA' / 'man_track.txt').read_text(), original)
            self.assertEqual(sum(modes.values()), 5)                       # 2 raw + 2 masks + b's man_track
            with self.assertRaises(FileExistsError):
                build_refined_dataset(src, dst, ['a'], {})


if __name__ == '__main__':
    unittest.main()
