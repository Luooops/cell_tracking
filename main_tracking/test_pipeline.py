"""Regression checks without downloading model weights: python -m unittest main_tracking.test_pipeline."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from main_tracking.main import build_parser
from main_tracking.pipeline import discover, process_sequence


class FakeModel:
    def eval(self, image, **kwargs):
        return (image > 0).astype(np.int32), None, None


class PipelineTests(unittest.TestCase):
    def test_mapping_empty_frames_and_short_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data"
            source.mkdir()
            for channel in [1, 2]:
                for t in [1, 2, 3]:
                    image = np.zeros((64, 64), dtype=np.uint16)
                    if channel == 1 and t != 2:
                        image[20:30, 20+t:30+t] = 1000
                    tifffile.imwrite(source / f"r01c01f01p01-ch{channel:02d}t{t:02d}.tiff", image)
            args = build_parser().parse_args(["--input-dir", str(source), "--output-root", str(root / "out"),
                "--preprocessing", "cellpose", "--min-area", "0"])
            jobs = discover(source, args.output_root)
            self.assertEqual(len(jobs), 2)
            for destination, rows in jobs.items():
                process_sequence(args, FakeModel(), destination, rows, {})
                df = pd.read_csv(destination / "instance_tracks.csv")
                if "ch01" in destination.name:
                    self.assertEqual(len(df), 2)
                    self.assertEqual(df.track_id.nunique(), 1)
                    self.assertEqual(df.time_index.tolist(), [1, 3])
                    self.assertFalse(df.passes_min_track_length.any())
                else:
                    self.assertTrue(df.empty)
                for row in rows:
                    mask = tifffile.imread(destination / row["mask_path"])
                    observed = df[df.frame_index == row["frame_index"]]
                    self.assertEqual(set(np.unique(mask)) - {0}, set(observed.instance_id))
                    with Image.open(destination / row["overlay_path"]) as overlay:
                        self.assertEqual(overlay.size, (64, 64))
            with self.assertRaisesRegex(ValueError, "not empty"):
                discover(source, args.output_root)

    def test_missing_duplicate_and_stack_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data"
            source.mkdir()
            first = source / "r01c01f01p01-ch01t01.tiff"
            second = source / "r01c01f01p01-ch01t03.tiff"
            tifffile.imwrite(first, np.zeros((10, 10), np.uint16))
            tifffile.imwrite(second, np.zeros((10, 10), np.uint16))
            with self.assertRaisesRegex(ValueError, "missing"):
                discover(source, root / "out")
            second.unlink()
            duplicate = source / "r01c01f01p01-ch01t01.tif"
            tifffile.imwrite(duplicate, np.zeros((10, 10), np.uint16))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                discover(source, root / "out")
            duplicate.unlink()
            tifffile.imwrite(first, np.zeros((2, 10, 10), np.uint16))
            with self.assertRaisesRegex(ValueError, "2D"):
                discover(source, root / "out")


if __name__ == "__main__":
    unittest.main()
