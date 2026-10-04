"""Re-link the masks of a main_tracking run with the pretrained Trackastra model.

Takes one sequence folder written by main_tracking/main.py (masks/, frames.csv, instance_tracks.csv, summary.json),
keeps the segmentation exactly as it is, and replaces only the association step: Trackastra scores the links between
detections of adjacent frames and its own linker turns them into tracks. The result is written in the same layout
(frames.csv, instance_tracks.csv, tracks.csv, summary.json), so track_eval/evaluate.py scores it unchanged and the
comparison with the classical tracker is on identical masks.

Trackastra is used as released -- no training on our data. Default configuration = the best of the six we tried on
the lab wells: model "general_2d", mode "greedy_nodiv" (see README.md).

Install:  pip install trackastra        (needs torch; the model weights are downloaded on first use)

Examples (from the repository root):
    python trackastra_tracking/run_trackastra.py --predictions-root main_tracking/outputs
    python trackastra_tracking/run_trackastra.py \
        --sequence-dir "main_tracking/outputs/<uuid>__r16c13/f01__p01__ch02" --input-dir "E:/aws_gt_data/<uuid>/r16c13"
then
    python track_eval/evaluate.py --gt <XML/ZIP> --predictions-root trackastra_tracking/outputs
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

MODES = ["greedy_nodiv", "greedy", "ilp"]


def csv_read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_gray(path):
    img = tifffile.imread(path) if path.suffix.lower() in (".tif", ".tiff") else None
    if img is None:
        from PIL import Image
        img = np.asarray(Image.open(path))
    if img.ndim != 2:
        raise ValueError(f"expected a single 2-D grayscale frame: {path} has shape {img.shape}")
    return img.astype(np.float32)


def find_sequences(predictions_root):
    return sorted(m.parent for m in Path(predictions_root).rglob("frames.csv")
                  if (m.parent / "instance_tracks.csv").is_file() and (m.parent / "summary.json").is_file())


def track_sequence(seq_dir, input_dir, output_root, model, mode, copy_masks, overwrite):
    seq_dir = Path(seq_dir)
    summary_in = json.loads((seq_dir / "summary.json").read_text(encoding="utf-8"))
    frames = [r for r in csv_read(seq_dir / "frames.csv") if r["status"] == "complete"]
    frames.sort(key=lambda r: int(r["frame_index"]))
    if not frames:
        raise ValueError(f"no complete frames in {seq_dir}")
    input_dir = Path(input_dir) if input_dir else Path(summary_in["input_dir"])
    if not input_dir.is_dir():
        raise FileNotFoundError(f"raw image folder not found: {input_dir} (pass --input-dir)")
    sequence_id = frames[0]["sequence_id"]
    out_dir = Path(output_root) / sequence_id
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (use --overwrite)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    masks = np.stack([tifffile.imread(seq_dir / r["mask_path"]) for r in frames]).astype(np.int32)
    imgs = np.stack([read_gray(input_dir / r["source_relative_path"]) for r in frames])
    if imgs.shape != masks.shape:
        raise ValueError(f"image stack {imgs.shape} and mask stack {masks.shape} differ")

    # Trackastra: learned association scores between detections of adjacent frames, then its own linker.
    graph, _ = model.track(imgs, masks, mode=mode)
    track_of = {}                                         # (position in `frames`, instance label) -> track id
    if mode == "greedy_nodiv":                            # every node has <= 1 predecessor and successor:
        import networkx as nx                             # a track = one connected chain of the solution graph
        for tid, comp in enumerate(nx.weakly_connected_components(graph), start=1):
            for n in comp:
                track_of[(int(graph.nodes[n]["time"]), int(graph.nodes[n]["label"]))] = tid
    else:                                                 # with divisions: CTC convention, daughters get new ids
        from trackastra.tracking import graph_to_ctc
        _, tracked = graph_to_ctc(graph, masks)
        for fi in range(len(frames)):
            for lab in np.unique(masks[fi]):
                if lab == 0: continue
                v = tracked[fi][masks[fi] == lab]; v = v[v > 0]
                if len(v): track_of[(fi, int(lab))] = int(np.bincount(v).argmax())
    next_id = max(track_of.values(), default=0) + 1

    pos = {r["image_name"]: i for i, r in enumerate(frames)}
    rows = [r for r in csv_read(seq_dir / "instance_tracks.csv") if r["image_name"] in pos]
    for r in rows:
        key = (pos[r["image_name"]], int(float(r["instance_id"])))
        if key not in track_of:                           # a detection the linker left out: its own one-frame track
            track_of[key] = next_id; next_id += 1
        r["track_id"] = track_of[key]
    length = Counter(r["track_id"] for r in rows)
    min_len = int(summary_in.get("parameters", {}).get("min_track_length", 5))
    for r in rows:
        r["track_length"] = length[r["track_id"]]
        r["passes_min_track_length"] = length[r["track_id"]] >= min_len
    rows.sort(key=lambda r: (int(r["track_id"]), int(r["frame_index"])))
    columns = list(csv_read(seq_dir / "instance_tracks.csv")[0].keys())
    csv_write(out_dir / "instance_tracks.csv", rows, columns)
    tracks_cols = ["track_id", "frame", "x", "y", "area", "major_axis_length", "minor_axis_length", "eccentricity",
                   "solidity", "label", "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"]
    csv_write(out_dir / "tracks.csv", rows, [c for c in tracks_cols if c in columns])

    if copy_masks:
        (out_dir / "masks").mkdir(exist_ok=True)
        for r in frames:
            shutil.copyfile(seq_dir / r["mask_path"], out_dir / r["mask_path"])
    else:                                                 # keep frames.csv pointing at the masks actually used
        for r in frames:
            src = (seq_dir / r["mask_path"]).resolve()
            try:
                r["mask_path"] = Path(os.path.relpath(src, out_dir.resolve())).as_posix()
            except ValueError:                            # different drive on Windows
                r["mask_path"] = str(src)
    for r in frames:
        r["overlay_path"] = ""
    csv_write(out_dir / "frames.csv", frames, list(frames[0].keys()))

    import torch, trackastra
    summary = dict(status="complete", input_dir=str(input_dir), sequence_id=sequence_id,
                   tracker="trackastra", masks_from=str(seq_dir.resolve()),
                   parameters=dict(trackastra_model=model_name(model), mode=mode, copy_masks=copy_masks),
                   segmentation_parameters=summary_in.get("parameters", {}),
                   model=dict(trackastra=getattr(trackastra, "__version__", "unknown"), torch=torch.__version__,
                              numpy=np.__version__, device=str(getattr(model, "device", "unknown"))),
                   mask_values=summary_in.get("mask_values"), coordinates=summary_in.get("coordinates"),
                   frames=len(frames), tracks=len(length), detections=len(rows), seconds=round(time.time() - t0, 3))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{sequence_id}: {len(frames)} frames, {len(rows)} detections -> {len(length)} tracks "
          f"({sum(v >= min_len for v in length.values())} with >= {min_len} frames), {summary['seconds']:.0f}s -> {out_dir}")
    return out_dir


def model_name(model):
    return getattr(model, "_pretrained_name", "pretrained")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence-dir", type=Path, help="one sequence folder of main_tracking (holds frames.csv and masks/)")
    src.add_argument("--predictions-root", type=Path, help="process every sequence found under this folder")
    ap.add_argument("--input-dir", type=Path, help="raw image folder (default: input_dir recorded in the run's summary.json)")
    ap.add_argument("--output-root", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--model", default="general_2d", help="pretrained Trackastra model: general_2d (default) or ctc")
    ap.add_argument("--mode", choices=MODES, default="greedy_nodiv",
                    help="linker: greedy_nodiv (default, no divisions; best on our wells), greedy, ilp")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--copy-masks", action="store_true", help="also copy the masks into the output folder")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    from trackastra.model import Trackastra
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"loading Trackastra '{args.model}' on {device} (mode {args.mode})")
    model = Trackastra.from_pretrained(args.model, device=device)
    model._pretrained_name = args.model

    seqs = [args.sequence_dir] if args.sequence_dir else find_sequences(args.predictions_root)
    if not seqs:
        ap.exit(2, "no sequence found (need frames.csv + instance_tracks.csv + summary.json)\n")
    out_root = args.output_root.resolve()
    for s in seqs:
        if out_root in Path(s).resolve().parents:
            continue                                      # never re-process our own outputs
        try:
            track_sequence(s, args.input_dir, out_root, model, args.mode, args.copy_masks, args.overwrite)
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            ap.exit(2, f"error in {s}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
