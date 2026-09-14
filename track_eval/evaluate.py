"""Position-based tracking evaluation against per-image polygon GT XML/ZIP."""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from gt_process.generate_gt_masks import read_gt, read_display

UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
NAME = re.compile(r"(r\d+c\d+)(f\d+)(p\d+)-(ch\d+)t(\d+)\.[^.]+$", re.I)


def csv_read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def centroid(vertices):
    """Area centroid of polygon; degenerate outlines use vertex mean."""
    x, y = vertices.T
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = cross.sum()
    if abs(area2) < 1e-8:
        return vertices.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum(),
                     ((y + np.roll(y, -1)) * cross).sum()]) / (3 * area2)


def load_frames(gt, label):
    root = read_gt(gt)
    frames, names, conflicts = [], set(), set()
    for node in root.findall("image"):
        name = node.attrib["name"].replace("\\", "/").split("/")[-1]
        match = NAME.fullmatch(name)
        if not match or name in names:
            raise ValueError(f"Unsupported or duplicate GT image name: {name}")
        names.add(name)
        well, field, plane, channel, time = match.groups()
        sequence = f"{well.lower()}/{field.lower()}__{plane.lower()}__{channel.lower()}"
        objects, ids = [], set()
        for index, polygon in enumerate(node.findall("polygon"), 1):
            if polygon.get("label") != label:
                continue
            points = np.array([list(map(float, p.split(","))) for p in polygon.attrib["points"].split(";")])
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError(f"Invalid GT polygon: {name}, {index}")
            tid = next(((a.text or "").strip() for a in polygon.findall("attribute") if a.get("name") == "track_id"), "")
            if tid and tid in ids:
                conflicts.add((sequence, tid))
            if tid:
                ids.add(tid)
            x, y = centroid(points)
            objects.append(dict(gt_instance_id=index, gt_track_id=tid, x=float(x), y=float(y), points=points))
        frames.append(dict(name=name, xml_frame_id=node.get("id", ""), sequence=sequence,
                           time=int(time), width=int(node.attrib["width"]), height=int(node.attrib["height"]), objects=objects))
    if not frames:
        raise ValueError("GT has no per-image annotations")
    for frame in frames:
        for obj in frame["objects"]:
            obj["gt_id_conflict"] = (frame["sequence"], obj["gt_track_id"]) in conflicts
    if conflicts:
        print(f"GT warning: {len(conflicts)} tracks have duplicate IDs within a frame; excluded from continuity metrics.")
    return sorted(frames, key=lambda f: (f["sequence"], f["time"]))


def match_positions(gt_xy, pred_xy, radius):
    """Maximize number of gated matches, then minimize total distance.

    Dummy assignments prevent rejected pairs from consuming valid candidates.
    Ambiguous means either endpoint has multiple candidates within the radius.
    """
    n, m = len(gt_xy), len(pred_xy)
    if not n or not m:
        return {}, np.zeros(n, dtype=bool), np.zeros(n, dtype=int)
    distances = np.linalg.norm(np.asarray(gt_xy)[:, None, :] - np.asarray(pred_xy)[None, :, :], axis=2)
    valid = distances <= radius
    penalty = (n + 1) * (radius + 1)
    cost = np.full((n, m + n), penalty)
    cost[:, :m] = np.where(valid, distances, penalty * 3)
    row, col = linear_sum_assignment(cost)
    matches = {int(i): (int(j), float(distances[i, j])) for i, j in zip(row, col) if j < m and valid[i, j]}
    counts = valid.sum(axis=1)
    ambiguity = (counts > 1) | np.any(valid & (valid.sum(axis=0) > 1)[None, :], axis=1)
    return matches, ambiguity, counts


def prediction_index(root, dataset):
    index = {}
    for directory in sorted(root.glob(f"{dataset}__*")):
        for manifest in sorted(directory.rglob("frames.csv")):
            folder = manifest.parent
            if not (folder / "instance_tracks.csv").is_file() or not (folder / "summary.json").is_file():
                continue
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                continue
            detections = defaultdict(list)
            seen_instances, seen_tracks = set(), set()
            for row in csv_read(folder / "instance_tracks.csv"):
                key = (row["image_name"], row["instance_id"])
                identity = (row["image_name"], row["track_id"])
                if key in seen_instances or identity in seen_tracks or not row["track_id"]:
                    raise ValueError(f"Invalid prediction mapping: {folder}")
                seen_instances.add(key)
                seen_tracks.add(identity)
                row["x"], row["y"] = float(row["x"]), float(row["y"])
                if not np.isfinite([row["x"], row["y"]]).all():
                    raise ValueError(f"Nonfinite prediction position: {folder}")
                detections[row["image_name"]].append(row)
            for row in csv_read(manifest):
                if row["status"] != "complete":
                    continue
                name = row["image_name"]
                if name in index:
                    raise ValueError(f"Multiple predictions for {name}")
                index[name] = dict(folder=folder, frame=row, detections=detections[name], summary=summary)
    return index


def visualize(frame, prediction, rows, target, data_dir):
    record = prediction["frame"]
    paths = []
    if data_dir:
        paths.extend([data_dir / record["source_relative_path"], data_dir / frame["name"]])
    paths.append(Path(prediction["summary"]["input_dir"]) / record["source_relative_path"])
    source = next((p for p in paths if p.is_file()), None)
    background = "raw image"
    if source:
        base = Image.fromarray(read_display(source, frame["width"], frame["height"]))
    else:
        base = Image.new("RGB", (frame["width"], frame["height"]), (25, 25, 25))
        background = "blank (source image unavailable)"
    draw = ImageDraw.Draw(base)
    for pred in prediction["detections"]:
        x, y = pred["x"], pred["y"]
        draw.ellipse((x-4, y-4, x+4, y+4), outline="cyan", width=2)
        draw.text((x+5, y+5), "P:" + pred["track_id"], fill="cyan", stroke_width=1, stroke_fill="black")
    for obj, row in zip(frame["objects"], rows):
        color = "magenta" if row["gt_id_conflict"] else ("orange" if row["ambiguous"] else ("red" if not row["pred_track_id"] else "lime"))
        points = [tuple(p) for p in obj["points"]]
        draw.line(points + [points[0]], fill=color, width=2)
        x, y = obj["x"], obj["y"]
        draw.text((x+5, y-14), "G:" + (obj["gt_track_id"] or "?"), fill=color, stroke_width=1, stroke_fill="black")
        if row["pred_track_id"]:
            draw.line((x, y, row["pred_x"], row["pred_y"]), fill=color, width=2)
    draw.rectangle((0, 0, min(frame["width"], 1050), 30), fill="black")
    draw.text((5, 5), "G=GT P=prediction | orange=ambiguous red=unmatched green=matched cyan=prediction magenta=GT ID conflict", fill="white")
    base.save(target)
    return background


def track_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["gt_track_id"] and not row.get("gt_id_conflict", False):
            groups[(row["sequence"], row["gt_track_id"])].append(row)
    result, transitions = [], []
    for (sequence, tid), history in sorted(groups.items()):
        history.sort(key=lambda r: r["time_index"])
        matched = [r for r in history if r["pred_track_id"]]
        clean = [r for r in matched if not r["ambiguous"]]
        available = sum(r["prediction_available"] for r in history)
        fragments = sum(bool(r["pred_track_id"]) and (i == 0 or not history[i-1]["pred_track_id"]) for i, r in enumerate(history))
        events = []
        for previous, current in zip(history, history[1:]):
            eligible = bool(previous["pred_track_id"] and current["pred_track_id"])
            unambiguous = eligible and not previous["ambiguous"] and not current["ambiguous"]
            same = previous["pred_track_id"] == current["pred_track_id"] if eligible else ""
            event = dict(sequence=sequence, gt_track_id=tid, previous_image=previous["image_name"],
                current_image=current["image_name"], time_gap=current["time_index"]-previous["time_index"],
                previous_pred_id=previous["pred_track_id"], current_pred_id=current["pred_track_id"],
                eligible=eligible, unambiguous=unambiguous, same_id=same)
            events.append(event)
        transitions.extend(events)
        switches = sum(a["pred_track_id"] != b["pred_track_id"] for a, b in zip(matched, matched[1:]))
        clean_events = [e for e in events if e["unambiguous"]]
        result.append(dict(sequence=sequence, gt_track_id=tid, gt_observations=len(history),
            available_prediction_frames=available, matched_observations=len(matched), unambiguous_matches=len(clean),
            coverage=len(matched)/len(history), coverage_available=len(matched)/available if available else "",
            distinct_pred_ids=len({r["pred_track_id"] for r in matched}), matched_fragments=fragments,
            id_changes_between_matched_observations=switches,
            adjacent_pairs=sum(e["eligible"] for e in events),
            adjacent_id_changes=sum(e["eligible"] and not e["same_id"] for e in events),
            unambiguous_pairs=len(clean_events), unambiguous_id_changes=sum(not e["same_id"] for e in clean_events)))
    return result, transitions


def run(args):
    frames = load_frames(args.gt, args.label)
    ids = UUID.findall(str(args.gt.resolve()))
    dataset = args.dataset_id or (ids[-1].lower() if ids else None)
    if not dataset or not UUID.fullmatch(dataset):
        raise ValueError("Cannot identify data UUID from GT path; supply --dataset-id UUID")
    dataset = dataset.lower()
    index = prediction_index(args.predictions_root, dataset)
    missing = [f for f in frames if f["name"] not in index]
    if missing:
        print("no tracking predictions found." if len(missing) == len(frames) else "Some GT frames have no tracking predictions.")
        print(f"Missing {len(missing)}/{len(frames)} GT frames under {args.predictions_root}")
        data_dir = args.data_dir or (args.gt if args.gt.is_dir() else args.gt.parent)
        channels = sorted({NAME.fullmatch(f["name"])[4].lower() for f in missing})
        for channel in channels:
            print(f'python "{ROOT / "main_tracking/main.py"}" --input-dir "{data_dir}" --channel {channel} --output-root "{args.predictions_root}"')
        print("Use a raw-image directory for --input-dir. Existing nonempty output folders must be resolved before rerunning.")
        if not args.allow_missing or len(missing) == len(frames):
            return 2
    output = args.output_root / f"{dataset}__{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True)
    (output / "visualizations").mkdir()
    rows, frame_rows, visual_rows = [], [], []
    for frame in frames:
        prediction = index.get(frame["name"])
        preds = prediction["detections"] if prediction else []
        if prediction and (int(prediction["frame"]["width"]), int(prediction["frame"]["height"])) != (frame["width"], frame["height"]):
            raise ValueError(f"GT/prediction dimensions differ: {frame['name']}")
        matches, ambiguous, counts = match_positions([(o["x"], o["y"]) for o in frame["objects"]], [(p["x"], p["y"]) for p in preds], args.max_distance)
        local = []
        for i, obj in enumerate(frame["objects"]):
            pair = matches.get(i)
            pred = preds[pair[0]] if pair else {}
            local.append(dict(sequence=frame["sequence"], image_name=frame["name"], xml_frame_id=frame["xml_frame_id"],
                time_index=frame["time"], gt_instance_id=obj["gt_instance_id"], gt_track_id=obj["gt_track_id"],
                gt_x=obj["x"], gt_y=obj["y"], gt_id_conflict=obj["gt_id_conflict"], prediction_available=prediction is not None,
                pred_instance_id=pred.get("instance_id", ""), pred_track_id=pred.get("track_id", ""),
                pred_x=pred.get("x", ""), pred_y=pred.get("y", ""), distance=pair[1] if pair else "",
                candidate_count=int(counts[i]), ambiguous=bool(ambiguous[i])))
        rows.extend(local)
        frame_rows.append(dict(image_name=frame["name"], sequence=frame["sequence"], time_index=frame["time"],
            prediction_available=prediction is not None, gt_objects=len(local), predictions=len(preds),
            matches=len(matches), unmatched_gt=len(local)-len(matches), ambiguous_gt=int(ambiguous.sum()),
            prediction_folder=str(prediction["folder"]) if prediction else ""))
        if prediction and (args.visualize == "all" or (args.visualize == "issues" and (ambiguous.any() or len(matches) < len(local) or any(r["gt_id_conflict"] for r in local)))):
            target = output / "visualizations" / f"{Path(frame['name']).stem}_comparison.png"
            background = visualize(frame, prediction, local, target, args.data_dir)
            visual_rows.append(dict(image_name=frame["name"], file=str(target), background=background))
        print(f"{frame['name']}: GT={len(local)}, matched={len(matches)}, ambiguous={ambiguous.sum()}", flush=True)
    metrics, transitions = track_metrics(rows)
    from track_eval.metrics import export_recovery
    recovery = export_recovery(output, rows, metrics)
    columns = ["sequence", "image_name", "xml_frame_id", "time_index", "gt_instance_id", "gt_track_id", "gt_x", "gt_y", "prediction_available", "pred_instance_id", "pred_track_id", "pred_x", "pred_y", "distance", "candidate_count", "ambiguous"]
    csv_write(output / "position_matches.csv", rows, columns + ["gt_id_conflict"])
    csv_write(output / "gt_id_conflicts.csv", [r for r in rows if r["gt_id_conflict"]], columns + ["gt_id_conflict"])
    csv_write(output / "per_frame.csv", frame_rows, list(frame_rows[0]))
    csv_write(output / "per_gt_track.csv", metrics, list(metrics[0]) if metrics else ["sequence", "gt_track_id", "gt_observations"])
    csv_write(output / "transitions.csv", transitions, list(transitions[0]) if transitions else ["sequence", "gt_track_id", "previous_image", "current_image"])
    csv_write(output / "visualizations.csv", visual_rows, ["image_name", "file", "background"])
    pairs = sum(m["unambiguous_pairs"] for m in metrics)
    changes = sum(m["unambiguous_id_changes"] for m in metrics)
    report = dict(status="partial" if missing else "complete", gt=str(args.gt.resolve()), dataset=dataset,
        predictions_root=str(args.predictions_root.resolve()), max_distance=args.max_distance,
        ambiguity_rule="either endpoint has multiple candidates within radius; ID-independent position matching",
        gt_frames=len(frames), missing_prediction_frames=[f["name"] for f in missing], gt_objects=len(rows),
        matched_objects=sum(bool(r["pred_track_id"]) for r in rows),
        position_coverage=sum(bool(r["pred_track_id"]) for r in rows)/len(rows) if rows else None,
        ambiguous_gt_objects=sum(r["ambiguous"] for r in rows),
        gt_tracks=len(metrics), gt_objects_without_track_id=sum(not r["gt_track_id"] for r in rows),
        excluded_conflicting_gt_tracks=sorted({r["sequence"] + ":" + r["gt_track_id"] for r in rows if r["gt_id_conflict"]}),
        unambiguous_adjacent_pairs=pairs, unambiguous_id_changes=changes,
        unambiguous_association_accuracy=(pairs-changes)/pairs if pairs else None,
        limitations="Position-based diagnostic, not standardized IDSW/IDF1. Unmatched predictions are not false positives. GT completeness unknown. Consecutive GT observations may be temporally sparse; see time_gap. Ambiguous matches are retained but excluded from unambiguous adjacent metrics.")
    report['recovery_evaluation'] = recovery
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, item in recovery['metrics'].items():
        value = 'N/A' if item['value'] is None else f"{item['value']:.2%}"
        print(f"{name}: {value} ({item['numerator']}/{item['denominator']})")
    print(f"Evaluation saved: {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="XML/ZIP or directory containing one XML/ZIP")
    parser.add_argument("--dataset-id", help="UUID override if absent from GT path")
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "main_tracking/outputs")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--data-dir", type=Path, help="Raw images, optional for visualization and suggested CLI")
    parser.add_argument("--max-distance", type=float, default=45, help="Position gate in pixels; provisional default, calibrate manually")
    parser.add_argument("--label", default="cell")
    parser.add_argument("--visualize", choices=["issues", "all", "none"], default="issues")
    parser.add_argument("--allow-missing", action="store_true", help="Evaluate partial predictions with explicit missing-frame records")
    args = parser.parse_args()
    if not np.isfinite(args.max_distance) or args.max_distance <= 0:
        parser.error("--max-distance must be positive and finite")
    try:
        return run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
