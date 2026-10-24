"""models/gnn_tracking/audit_confident.py: picking and classing confident false links, stratified sampling."""
import unittest

import numpy as np

from models.gnn_tracking.audit_confident import confident_false_links, stratified_sample
from utils.lineage import SAME_CELL, UNRELATED


class TestAuditConfident(unittest.TestCase):
    def test_confident_false_links_are_picked_and_classed(self):
        # frame 0 -> 1. 1 continues as 1 (true, also a candidate), but the tracker linked 1 -> 2 (2 existed before);
        # 3 ends at 0 and the tracker linked it to 4, which starts at 1; one member of 5 -> 6 is not confident.
        table = {1: (0, 1, 0), 2: (0, 1, 0), 3: (0, 0, 0), 4: (1, 1, 0), 5: (0, 0, 0), 6: (0, 1, 0)}
        z = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 0.0]])
        edges = dict(round=np.ones(4, int), src_frame=np.zeros(4, int), src_label=np.array([1, 1, 3, 5]),
                     dst_label=np.array([1, 2, 4, 6]), logits=z, chosen=np.array([False, True, True, True]))
        relation = np.array([SAME_CELL, UNRELATED, UNRELATED, UNRELATED], dtype=object)
        found = confident_false_links(edges, relation, table)
        self.assertEqual([(f['a'], f['b'], f['cls']) for f in found], [(1, 2, 'mixup'), (3, 4, 'a_ends_b_new')])

    def test_sample_keeps_small_classes_and_its_size(self):
        items = [dict(cls='mixup')] * 50 + [dict(cls='a_ends_b_new')] * 3 + [dict(cls='other')] * 20
        sample = stratified_sample([dict(i, k=k) for k, i in enumerate(items)], 30, seed=0)
        classes = [s['cls'] for s in sample]
        self.assertEqual(classes.count('a_ends_b_new'), 3)
        self.assertAlmostEqual(len(sample), 30, delta=1)
        self.assertEqual(len({s['k'] for s in sample}), len(sample))


if __name__ == '__main__':
    unittest.main()
