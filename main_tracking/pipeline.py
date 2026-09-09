"""Explicit source/mask/track mapping; masks are processed one frame at a time."""
import csv
import hashlib
import colorsys
import json
import re
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries
from main_tracking.segmentation import read_image, segment_image, save_mask
from main_tracking.tracking_v0 import SimpleCellTrackerV3, extract_detections_from_mask, gap_close_tracks, export_tracks_to_csv

PATTERN = re.compile(r"(?P<well>r\d+c\d+)(?P<field>f\d+)(?P<plane>p\d+)-(?P<channel>ch\d+)t(?P<time>\d+)", re.I)
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
FRAME_COLUMNS = ["sequence_id", "frame_index", "time_index", "image_name", "source_relative_path", "mask_path", "overlay_path", "height", "width", "status", "instances", "min_area"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_frames(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def discover(source, output_root, channel=None):
    """Validate every selected image before model loading or output creation."""
    source, output_root = source.resolve(), output_root.resolve()
    if source == output_root or source in output_root.parents:
        raise ValueError("Output root must be outside the input tree")
    if channel and not re.fullmatch(r"ch\d+", channel, re.I):
        raise ValueError("Channel must look like ch01")
    groups = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        match = PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f"Unrecognized raw image name: {path}; expected r16c13f01p01-ch01t01.tiff")
        data = match.groupdict()
        if channel and int(data["channel"][2:]) != int(channel[2:]):
            continue
        identifiers = [part for part in path.parts if UUID.fullmatch(part)]
        dataset = identifiers[-1].lower() if identifiers else source.name
        well_match = re.fullmatch(r"r(\d+)c(\d+)", data["well"], re.I)
        well = f"r{int(well_match[1]):02d}c{int(well_match[2]):02d}"
        sequence = "__".join(f"{prefix}{int(data[key][len(prefix):]):02d}" for key, prefix in [("field", "f"), ("plane", "p"), ("channel", "ch")])
        key = (f"{dataset}__{well}", sequence)
        groups.setdefault(key, []).append((int(data["time"]), path))
    if not groups:
        raise FileNotFoundError(f"No matching images in {source}")
    result = {}
    for key, files in sorted(groups.items()):
        files.sort()
        times = [t for t, _ in files]
        if len(times) != len(set(times)) or any(b != a + 1 for a, b in zip(times, times[1:])):
            raise ValueError(f"{key}: duplicate or missing time points: {times}")
        rows, shape = [], None
        for index, (t, path) in enumerate(files):
            image = read_image(path)
            if shape is not None and image.shape != shape:
                raise ValueError(f"{path}: inconsistent frame dimensions")
            shape = image.shape
            rows.append(dict(sequence_id="/".join(key), frame_index=index, time_index=t,
                image_name=path.name, source_relative_path=path.relative_to(source).as_posix(),
                mask_path=f"masks/{path.stem}_mask.tiff", overlay_path=f"overlays/{path.stem}_overlay.png",
                height=shape[0], width=shape[1], status="pending", instances="", min_area=""))
        destination = output_root.joinpath(*key)
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Output is not empty: {destination}; choose another --output-root")
        result[destination] = rows
    return result


def render_overlay(raw, mask, rows, destination, alpha):
    low, high = np.percentile(raw, [1, 99])
    gray = (np.clip((raw.astype(np.float32) - low) / max(float(high - low), 1e-8), 0, 1) * 255).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=2)
    palette = np.zeros((int(mask.max()) + 1, 3), dtype=np.uint8)
    for row in rows:
        key = f"{row['sequence_id']}:{row['track_id']}"
        hue = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") / 2**32
        palette[int(row["instance_id"])] = np.array(colorsys.hsv_to_rgb(hue, 0.75, 1)) * 255
    foreground = mask > 0
    base[foreground] = ((1 - alpha) * base[foreground] + alpha * palette[mask[foreground]]).astype(np.uint8)
    edge = find_boundaries(mask, mode="inner") & foreground
    base[edge] = palette[mask[edge]]
    overlay = Image.fromarray(base)
    draw = ImageDraw.Draw(overlay)
    for row in rows:
        draw.text((round(row["x"]), round(row["y"])), str(row["track_id"]),
                  fill="white", stroke_width=1, stroke_fill="black")
    overlay.save(destination)


def process_sequence(args, model, destination, rows, model_info):
    import tifffile
    source = args.input_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "masks").mkdir()
    (destination / "overlays").mkdir()
    start = time.perf_counter()
    report = dict(status="running", input_dir=str(source), sequence_id=rows[0]["sequence_id"],
                  parameters=vars(args), model=model_info,
                  mask_values="0=background; positive=frame-local instance_id",
                  coordinates="x=column, y=row, zero-based pixels; frame_index starts at 0; time_index from filename")
    write_json(destination / "summary.json", report)
    write_frames(destination / "frames.csv", rows)
    current = None
    try:
        tracker = SimpleCellTrackerV3(max_distance=args.max_distance, max_area_ratio=args.max_area_ratio,
            max_shape_ratio=args.max_shape_ratio, max_lost=args.max_lost,
            area_weight=5, shape_weight=3, iou_weight=2, n_history=3, gap_growth=0.35)
        for current in rows:
            raw = read_image(source / current["source_relative_path"])
            mask, threshold, _ = segment_image(model, raw, diameter=args.diameter,
                cellprob_threshold=args.cellprob_threshold, flow_threshold=args.flow_threshold,
                min_area=args.min_area, auto_min_area_fraction=args.auto_min_area_fraction,
                preprocessing=args.preprocessing)
            if mask.shape != raw.shape:
                raise ValueError("Model mask dimensions differ from source")
            save_mask(destination / current["mask_path"], mask)
            detections = extract_detections_from_mask(mask, current["frame_index"])
            tracker.update(detections)
            current.update(status="segmented", instances=len(detections), min_area=threshold)
            write_frames(destination / "frames.csv", rows)
            print(f"[{current['frame_index'] + 1}/{len(rows)}] {current['image_name']}: {len(detections)} instances", flush=True)
        tracker.finish_all()
        tracks = gap_close_tracks(tracker.get_all_tracks(), max_gap=args.gap_close_max_gap,
            max_dist=args.gap_close_max_distance, max_area_ratio=args.max_area_ratio,
            max_shape_ratio=args.max_shape_ratio, max_angle_diff_deg=120, max_close_cost=args.max_close_cost)
        df = export_tracks_to_csv(tracks, destination / "tracks.csv")
        df["instance_id"] = df["label"]
        df["frame_index"] = df["frame"]
        for column in ["sequence_id", "time_index", "image_name", "source_relative_path", "mask_path"]:
            df[column] = df["frame"].map({row["frame_index"]: row[column] for row in rows})
        df["track_length"] = df["track_id"].map({t["track_id"]: len(t["history"]) for t in tracks})
        df["passes_min_track_length"] = df["track_length"] >= args.min_track_length
        if len(df) != sum(row["instances"] for row in rows) or df.duplicated(["frame", "instance_id"]).any() or df.duplicated(["frame", "track_id"]).any():
            raise ValueError("Inconsistent instance-to-track mapping")
        df.to_csv(destination / "instance_tracks.csv", index=False, encoding="utf-8-sig")
        by_frame = {int(frame): group.to_dict("records") for frame, group in df.groupby("frame")}
        for current in rows:
            render_overlay(read_image(source / current["source_relative_path"]),
                tifffile.imread(destination / current["mask_path"]), by_frame.get(current["frame_index"], []),
                destination / current["overlay_path"], args.alpha)
            current["status"] = "complete"
        report.update(status="complete", frames=len(rows), tracks=len(tracks),
                      detections=len(df), seconds=round(time.perf_counter() - start, 3))
    except Exception as exc:
        report.update(status="failed", error=repr(exc), current_image=current["image_name"] if current else None)
        if current is not None:
            current["status"] = "failed"
        raise
    finally:
        write_frames(destination / "frames.csv", rows)
        write_json(destination / "summary.json", report)
    print(f"Saved: {destination}", flush=True)


def run(args):
    jobs = discover(args.input_dir, args.output_root, args.channel)
    import torch
    from cellpose import models
    model_source = args.model
    if model_source == "cpsam_v2":
        if "cpsam_v2" not in getattr(models, "MODEL_NAMES", []):
            raise RuntimeError("Installed Cellpose does not support cpsam_v2; update the selected environment")
    else:
        weights = Path(model_source).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        model_source = str(weights)
    gpu = not args.cpu and torch.cuda.is_available()
    print(f"Sequences: {len(jobs)} | Model: {model_source} | CUDA: {gpu}", flush=True)
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_source)
    info = dict(requested=model_source, weights=str(model.pretrained_model), device=str(model.device),
                cellpose=version("cellpose"), torch=version("torch"), numpy=version("numpy"))
    for destination, rows in jobs.items():
        process_sequence(args, model, destination, rows, info)
