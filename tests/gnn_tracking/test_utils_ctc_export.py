"""utils/ctc_export.py: CTC RES export and canonical GT on synthetic masks (no py-ctcmetrics)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from utils.ctc_export import _segments, canonical_gt, export_res


def blob(mask, label, x):
    mask[2:6, x:x + 4] = label


def write_masks(folder, cells_per_frame, n_frames=4, name='man_track'):
    """cells_per_frame(t) -> [(label, x)]; writes 3-digit mask files and returns [(frame, path)]."""
    folder.mkdir(parents=True)
    frames = []
    for t in range(n_frames):
        mask = np.zeros((8, 60), np.uint16)
        for label, x in cells_per_frame(t):
            blob(mask, label, x)
        path = folder / f'{name}{t:03d}.tif'
        tifffile.imwrite(path, mask)
        frames.append((t, path))
    return frames


def read_rows(path):
    return [tuple(map(int, line.split())) for line in Path(path).read_text().splitlines()]


class TestSegments(unittest.TestCase):
    def test_runs_of_consecutive_frames(self):
        self.assertEqual(_segments([3, 0, 1, 5, 6]), [(0, 1), (3, 3), (5, 6)])


class TestExportRes(unittest.TestCase):
    def cells(self, t):
        """instance 1 at x=0 in frames 0, 1, 3 (missed in 2); mother 5 at x=20 in 0-1, daughters 6, 7 in 2-3;
        an unknown identity 50001 at x=50 in every frame."""
        out = [(50001, 50)]
        out += [(1, 0)] if t != 2 else []
        out += [(5, 20)] if t <= 1 else [(6, 18), (7, 26)]
        return out

    def export(self, root):
        frames = write_masks(root / 'gt', self.cells)
        tracks = [(t, 1, 1) for t in (0, 1, 3)] + [(t, 5, 2) for t in (0, 1)] + \
                 [(t, 6, 3) for t in (2, 3)] + [(t, 7, 4) for t in (2, 3)] + [(t, 50001, 9) for t in range(4)]
        info = export_res(tracks, {'3': 2, '4': 2}, frames, root / 'res')
        return info, read_rows(root / 'res' / 'res_track.txt')

    def test_gap_becomes_single_child_segment(self):
        with tempfile.TemporaryDirectory() as folder:
            _, rows = self.export(Path(folder))
            first, second = [r for r in rows if r[1] == 0 and r[2] == 1][0], [r for r in rows if r[1] == 3][0]
            self.assertEqual(second[3], first[0])

    def test_daughters_point_at_mother(self):
        with tempfile.TemporaryDirectory() as folder:
            _, rows = self.export(Path(folder))
            mother = [r for r in rows if (r[1], r[2]) == (0, 1)][1]
            daughters = [r for r in rows if (r[1], r[2]) == (2, 3)]
            self.assertEqual([r[3] for r in daughters], [mother[0], mother[0]])

    def test_unknown_identity_removed_and_masks_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            info, rows = self.export(root)
            self.assertEqual((info['tracks'], info['segments']), (4, 5))
            self.assertEqual(sorted(p.name for p in (root / 'res').glob('*.tif')),
                             ['mask000.tif', 'mask001.tif', 'mask002.tif', 'mask003.tif'])
            m3 = tifffile.imread(root / 'res' / 'mask003.tif')
            self.assertEqual(int(m3[3, 51]), 0)                        # unknown identity gone
            self.assertEqual(set(np.unique(m3)) - {0}, {r[0] for r in rows if r[1] <= 3 <= r[2]})

    def test_existing_output_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = write_masks(root / 'gt', lambda t: [(1, 0)])
            (root / 'res').mkdir()
            with self.assertRaises(FileExistsError):
                export_res([(t, 1, 1) for t in range(4)], {}, frames, root / 'res')


class TestCanonicalGt(unittest.TestCase):
    def test_label_gap_split_and_unknown_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = write_masks(root / 'src', lambda t: [(50001, 50)] + ([(1, 0)] if t != 2 else []) + [(2, 20)])
            info = canonical_gt({1: (0, 3, 0), 2: (0, 3, 0), 50001: (0, 3, 0)}, frames, root / 'gt')
            rows = read_rows(root / 'gt' / 'TRA' / 'man_track.txt')
            self.assertEqual(sorted((b, e) for _, b, e, _ in rows), [(0, 1), (0, 3), (3, 3)])
            later = [r for r in rows if r[1] == 3][0]
            self.assertEqual(later[3], [r for r in rows if (r[1], r[2]) == (0, 1)][0][0])
            self.assertEqual(info['tracks'], 2)
            self.assertEqual(int(tifffile.imread(root / 'gt' / 'TRA' / 'man_track000.tif')[3, 51]), 0)


if __name__ == '__main__':
    unittest.main()
