"""utils/lineage.py: GT links and relations from man_track.txt + observed detections (synthetic tables only)."""
from pathlib import Path
import tempfile
import unittest

from utils.lineage import (CONTINUATION, MOTHER_DAUGHTER, SAME_CELL, UNKNOWN, UNRELATED, Lineage,
                           read_lineage, read_track_table)


def observe(**tracks):
    """observe(l1=[0, 1, 3]) -> [(0, 1), (1, 1), (3, 1)]; keyword l<label> = observed frames of that label."""
    return [(f, int(name[1:])) for name, frames in tracks.items() for f in frames]


def links_of(lineage):
    return [(k.src, k.dst, k.kind, k.via, k.g) for k in lineage.links]


class TestReadTrackTable(unittest.TestCase):
    def write(self, folder, text):
        path = Path(folder) / 'man_track.txt'
        path.write_text(text, encoding='utf-8')
        return path

    def test_valid_table_parsed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(read_track_table(self.write(folder, '1 0 2 0\n2 3 4 1\n\n')), {1: (0, 2, 0), 2: (3, 4, 1)})

    def test_wrong_column_count_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                read_track_table(self.write(folder, '1 0 2\n'))

    def test_duplicate_label_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                read_track_table(self.write(folder, '1 0 2 0\n1 3 4 0\n'))

    def test_end_before_begin_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                read_track_table(self.write(folder, '1 5 2 0\n'))

    def test_read_lineage_from_file(self):
        with tempfile.TemporaryDirectory() as folder:
            lineage = read_lineage(self.write(folder, '1 0 1 0\n'), observe(l1=[0, 1]))
            self.assertEqual(links_of(lineage), [((0, 1), (1, 1), CONTINUATION, 'same_label', 1)])


class TestContinuationLinks(unittest.TestCase):
    def test_same_label_gap_is_continuation_with_g2(self):
        lineage = Lineage({1: (0, 3, 0)}, observe(l1=[0, 1, 3]))
        self.assertEqual(links_of(lineage), [((0, 1), (1, 1), CONTINUATION, 'same_label', 1),
                                             ((1, 1), (3, 1), CONTINUATION, 'same_label', 2)])
        self.assertEqual(lineage.summary()['tracks_with_internal_gap'], 1)

    def test_single_child_is_continuation(self):
        lineage = Lineage({1: (0, 2, 0), 2: (5, 6, 1)}, observe(l1=[0, 1, 2], l2=[5, 6]))
        self.assertIn(((2, 1), (5, 2), CONTINUATION, 'single_child', 3), links_of(lineage))
        self.assertEqual(lineage.relation((0, 1), (6, 2)), SAME_CELL)
        self.assertEqual(lineage.summary()['single_child_continuations'], 1)
        self.assertEqual(lineage.anomalies, [])

    def test_single_child_chain_shares_one_cell(self):
        lineage = Lineage({1: (0, 0, 0), 2: (2, 2, 1), 3: (4, 4, 2)}, observe(l1=[0], l2=[2], l3=[4]))
        self.assertEqual({lineage.cell_of[l] for l in (1, 2, 3)}, {1})
        self.assertEqual([k.g for k in lineage.links], [2, 2])


class TestDivisionLinks(unittest.TestCase):
    def test_two_children_are_mother_daughter(self):
        lineage = Lineage({1: (0, 2, 0), 2: (3, 4, 1), 3: (3, 4, 1)}, observe(l1=[0, 1, 2], l2=[3, 4], l3=[3, 4]))
        md = [l for l in links_of(lineage) if l[2] == MOTHER_DAUGHTER]
        self.assertEqual(md, [((2, 1), (3, 2), MOTHER_DAUGHTER, 'two_children', 1),
                              ((2, 1), (3, 3), MOTHER_DAUGHTER, 'two_children', 1)])
        self.assertEqual(lineage.relation((2, 1), (3, 2)), MOTHER_DAUGHTER)
        self.assertEqual(lineage.relation((3, 2), (4, 3)), UNRELATED)          # sisters are different cells
        self.assertEqual(lineage.summary()['divisions'], 1)
        self.assertEqual(lineage.anomalies, [])

    def test_mother_missing_last_frame_gives_g2_link(self):
        lineage = Lineage({1: (0, 2, 0), 2: (3, 3, 1), 3: (3, 3, 1)}, observe(l1=[0, 1], l2=[3], l3=[3]))
        self.assertEqual({(l.src, l.g) for l in lineage.links if l.kind == MOTHER_DAUGHTER}, {((1, 1), 2)})
        self.assertEqual([a['kind'] for a in lineage.anomalies], ['unobserved_endpoint'])

    def test_three_children_are_unknown(self):
        table = {1: (0, 1, 0), 2: (2, 2, 1), 3: (2, 2, 1), 4: (2, 2, 1)}
        lineage = Lineage(table, observe(l1=[0, 1], l2=[2], l3=[2], l4=[2]))
        self.assertEqual({(l.kind, l.via) for l in lineage.links if l.src[1] == 1 and l.dst[1] != 1},
                         {(UNKNOWN, 'many_children')})
        self.assertEqual(lineage.relation((1, 1), (2, 2)), UNKNOWN)
        self.assertIn('many_children', lineage.summary()['anomalies'])

    def test_late_daughter_flagged(self):
        lineage = Lineage({1: (0, 1, 0), 2: (3, 3, 1), 3: (2, 2, 1)}, observe(l1=[0, 1], l2=[3], l3=[2]))
        self.assertEqual([(a['kind'], a['label']) for a in lineage.anomalies], [('child_begin_mismatch', 2)])


class TestUnknownIdentity(unittest.TestCase):
    def test_unknown_labels_take_part_in_no_link(self):
        lineage = Lineage({1: (0, 1, 0), 50001: (0, 1, 0)}, observe(l1=[0, 1], l50001=[0, 1]))
        self.assertEqual(links_of(lineage), [((0, 1), (1, 1), CONTINUATION, 'same_label', 1)])
        self.assertEqual(lineage.relation((0, 1), (1, 50001)), UNKNOWN)
        self.assertEqual(lineage.summary()['unknown_identity_tracks'], 1)


class TestAnomalies(unittest.TestCase):
    def test_missing_parent_and_unlisted_label(self):
        lineage = Lineage({2: (0, 1, 9)}, observe(l2=[0, 1], l7=[0]))
        self.assertEqual(sorted(a['kind'] for a in lineage.anomalies), ['label_not_in_table', 'missing_parent'])
        self.assertEqual(lineage.relation((0, 7), (1, 2)), UNKNOWN)

    def test_unobserved_track(self):
        lineage = Lineage({1: (0, 1, 0)}, [])
        self.assertEqual([a['kind'] for a in lineage.anomalies], ['unobserved_track'])
        self.assertEqual(lineage.links, [])

    def test_duplicate_observation_raises(self):
        with self.assertRaises(ValueError):
            Lineage({1: (0, 0, 0)}, [(0, 1), (0, 1)])


if __name__ == '__main__':
    unittest.main()
