import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.dataset import cell_ids


class TestCellIds(unittest.TestCase):
    def test_single_child_continuation_shares_the_cell_and_daughters_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = Path(tmp) / 'w_GT' / 'TRA'
            gt.mkdir(parents=True)
            # 1 -> 2 -> 6 interrupted track (single children); 3 divides into 4 and 5
            (gt / 'man_track.txt').write_text('1 0 3 0\n2 5 7 1\n6 8 9 2\n3 0 2 0\n4 3 5 3\n5 3 5 3\n', encoding='utf-8')
            ids = cell_ids('w', [1, 2, 6, 3, 4, 5, 50001], root=tmp)
        self.assertEqual(ids.tolist(), [1, 1, 1, 3, 4, 5, 50001])

    def test_well_without_track_table_keeps_mask_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = cell_ids('w', np.array([7, 8, 50002]), root=tmp)
        self.assertEqual(ids.tolist(), [7, 8, 50002])


if __name__ == '__main__':
    unittest.main()
