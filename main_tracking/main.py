"""Folder segmentation, tracking and overlays. See README.md for usage."""
import argparse
import os
from pathlib import Path
import sys

if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--model", default="cpsam_v2", help="cpsam_v2 or an existing local weights file")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--channel", default="ch2", help="Filename channel selection (default: ch2; matches ch02)")
    parser.add_argument("--diameter", type=float, default=25)
    parser.add_argument("--cellprob-threshold", type=float, default=1.8)
    parser.add_argument("--flow-threshold", type=float, default=1.0)
    parser.add_argument("--preprocessing", choices=["legacy", "cellpose"], default="legacy")
    parser.add_argument("--min-area", type=int, default=None, help="Default: mean area times fraction; 0 disables")
    parser.add_argument("--auto-min-area-fraction", type=float, default=0.25)
    parser.add_argument("--max-distance", type=float, default=45)
    parser.add_argument("--max-area-ratio", type=float, default=1.8)
    parser.add_argument("--max-shape-ratio", type=float, default=1.8)
    parser.add_argument("--max-lost", type=int, default=3)
    parser.add_argument("--min-track-length", type=int, default=5, help="Quality flag only; all tracks exported")
    parser.add_argument("--gap-close-max-gap", type=int, default=2)
    parser.add_argument("--gap-close-max-distance", type=float, default=30)
    parser.add_argument("--max-close-cost", type=float, default=12)
    parser.add_argument("--alpha", type=float, default=0.3)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = Path(input("Input data folder: ").strip().strip('"').strip("'"))
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if (not 0 <= args.alpha <= 1 or not 0 <= args.auto_min_area_fraction <= 1
            or args.max_lost < 0 or args.gap_close_max_gap < 0
            or args.min_track_length < 1 or args.diameter <= 0
            or args.max_distance <= 0 or args.gap_close_max_distance <= 0
            or args.max_area_ratio < 1 or args.max_shape_ratio < 1
            or args.max_close_cost < 0 or args.flow_threshold < 0
            or (args.min_area is not None and args.min_area < 0)):
        parser.error("Invalid threshold/distance/area/length parameter")
    from main_tracking.pipeline import run
    run(args)


if __name__ == "__main__":
    main()
