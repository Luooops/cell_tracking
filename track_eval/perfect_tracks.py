"""Perfect-track fraction and ID switches for a tracking result (mask-overlap matching).

A second, self-contained evaluator next to evaluate.py. It answers one question: given the detections the segmenter
produced, how well were they linked? GT identities are transferred to the predicted instances by mask overlap, then

  perfect track   a GT track whose matched detections all carry ONE predicted track_id, and that predicted track_id
                  contains no detection of any other GT track
  perfect_frac    perfect tracks / GT tracks with at least one matched detection
  id_switches     number of times the predicted track_id changes between consecutive matched detections of a GT track,
                  summed over all GT tracks (a break across a missed frame counts once)
  purity          per predicted track, share of its GT-matched detections that belong to its most common GT track (mean)

Difference from evaluate.py: a frame where the segmenter missed the cell is NOT scored (the observation has no
detection to judge), but the track still has to keep the same id across it -- so missed detections cost a track only
if the tracker fails to re-join it. evaluate.py's complete_gt_track_recovery requires every GT observation to be matched.

How a predicted instance gets its GT identity (same rules as the GT cleaning we use for training data):
  1. A track_id that appears on 2+ polygons in one frame: all polygons with that id are dropped from THAT frame.
     Polygons without a track_id are ignored.
  2. A polygon is ROUGH if source == "manual" or it has <= 8 vertices (hand-drawn boxes); otherwise ACCURATE.
  3. Per frame, one-to-one, largest overlap first:
       a. accurate polygons claim instances with  overlap / instance_area >= 0.5
       b. rough polygons choose among the remaining instances with  overlap / min(instance_area, polygon_area) >= 0.5
  4. If a GT track moves more than --jump-px (300) between two consecutive frames (polygon area centroids), it is cut
     there into two GT tracks (annotation jump, not cell motion).
Predicted instances that match no polygon have unknown identity: they are not scored and do not count against the
purity of the predicted track they are in.

Usage (from the repository root):
    python track_eval/perfect_tracks.py --gt <XML/ZIP> --sequence-dir main_tracking/outputs/<uuid>__<well>/f01__p01__ch02
    python track_eval/perfect_tracks.py --gt <XML/ZIP> --predictions-root trackastra_tracking/outputs
Writes perfect_tracks.json and perfect_tracks_per_gt_track.csv next to each sequence's instance_tracks.csv
(or under --output-root).
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import polygon as draw_polygon

CONTAIN_THRESH = 0.5


# ------------------------------------------------------------------ ground truth
def read_annotation(path):
    path = Path(path)
    if path.is_dir():
        cands = [p for p in path.iterdir() if p.suffix.lower() in (".xml", ".zip")]
        if len(cands) != 1:
            raise ValueError(f"{path}: expected exactly one .xml/.zip, found {len(cands)}")
        path = cands[0]
    data = path.read_bytes()
    if data[:2] == b"PK":                                   # CVAT export: a ZIP (sometimes named .xml)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".xml")]
            if len(members) != 1:
                raise ValueError(f"{path}: expected one XML inside the archive, found {len(members)}")
            data = z.read(members[0])
    return ET.fromstring(data)


def area_centroid(pts):
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def parse_gt(root, label, rough_max_vertices):
    """{image basename: (time index, width, height, [polygon dict])}"""
    frames = {}
    for img in root.iter("image"):
        name = os.path.basename(img.get("name"))
        m = re.search(r"t(\d+)\.[A-Za-z]+$", name)
        if not m:
            raise ValueError(f"cannot read the time index from image name {name!r}")
        polys = []
        for sh in img.iter("polygon"):
            if label and sh.get("label") != label:
                continue
            tid = next(((a.text or "").strip() for a in sh.iter("attribute") if a.get("name") == "track_id"), "")
            pts = np.array([[float(v) for v in p.split(",")] for p in sh.get("points").split(";")])
            polys.append(dict(tid=tid, pts=pts, rough=sh.get("source") == "manual" or len(pts) <= rough_max_vertices,
                              centroid=area_centroid(pts)))
        frames[name] = (int(m.group(1)), int(img.get("width")), int(img.get("height")), polys)
    return frames


def jump_cuts(gt, jump_px):
    """{track_id: [time index where a new segment starts]} from unique (non-duplicated) observations."""
    seen = defaultdict(dict)
    for t, _, _, polys in gt.values():
        count = Counter(p["tid"] for p in polys)
        for p in polys:
            if p["tid"] and count[p["tid"]] == 1:
                seen[p["tid"]][t] = p["centroid"]
    cuts = defaultdict(list)
    for tid, cen in seen.items():
        for t in sorted(cen):
            if t + 1 in cen and np.hypot(cen[t + 1][0] - cen[t][0], cen[t + 1][1] - cen[t][1]) > jump_px:
                cuts[tid].append(t + 1)
    return cuts


# ------------------------------------------------------------------ matching
def _greedy(cands):
    cands.sort(reverse=True)
    used_g, used_c, out = set(), set(), {}
    for _, g, c in cands:
        if g in used_g or c in used_c:
            continue
        out[g] = c
        used_g.add(g); used_c.add(c)
    return out


def match_two_rounds(polys, rasters, mask):
    """{polygon index: instance id}; accurate polygons first, rough ones among the left-over instances."""
    area = np.bincount(mask.ravel())
    ov = {}
    for i, (rr, cc) in enumerate(rasters):
        ids, cnt = np.unique(mask[rr, cc], return_counts=True)
        for c, n in zip(ids, cnt):
            if c:
                ov[(i, int(c))] = int(n)
    acc = _greedy([(o, g, c) for (g, c), o in ov.items()
                   if not polys[g]["rough"] and o / area[c] >= CONTAIN_THRESH])
    taken = set(acc.values())
    rough = _greedy([(o, g, c) for (g, c), o in ov.items()
                     if polys[g]["rough"] and c not in taken
                     and o / min(area[c], len(rasters[g][0])) >= CONTAIN_THRESH])
    return {**acc, **rough}


# ------------------------------------------------------------------ metrics
def evaluate_tracks(assign, gt_of):
    """assign: {(frame position, instance id): predicted track id}; gt_of: same keys -> GT track key (matched only)."""
    by_gt, members, by_pred = defaultdict(list), defaultdict(set), defaultdict(list)
    for k, g in gt_of.items():
        by_gt[g].append(k)
    for k, p in assign.items():
        if k in gt_of:
            members[p].add(gt_of[k]); by_pred[p].append(gt_of[k])
    rows, switches, perfect = [], 0, 0
    for g, keys in by_gt.items():
        keys.sort()
        pids = [assign[k] for k in keys if k in assign]
        if not pids:
            continue
        sw = sum(a != b for a, b in zip(pids[:-1], pids[1:]))
        mixed = any(members[p] != {g} for p in set(pids))
        ok = len(set(pids)) == 1 and not mixed
        switches += sw; perfect += ok
        rows.append(dict(gt_track_id=g[0], gt_segment=g[1], matched_detections=len(pids), first_frame=keys[0][0],
                         last_frame=keys[-1][0], has_missed_frame=keys[-1][0] - keys[0][0] + 1 != len(keys),
                         distinct_pred_ids=len(set(pids)), id_switches=sw, shares_pred_id_with_other_gt=mixed, perfect=ok))
    purity = [Counter(v).most_common(1)[0][1] / len(v) for v in by_pred.values() if v]
    return dict(gt_tracks=len(rows), perfect_tracks=int(perfect), perfect_frac=perfect / max(len(rows), 1),
                id_switches=int(switches), mean_purity=float(np.mean(purity)) if purity else None,
                predicted_tracks_with_gt=len(by_pred)), rows


def csv_read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate_sequence(seq_dir, gt, cuts, output_root):
    seq_dir = Path(seq_dir)
    frames = sorted((r for r in csv_read(seq_dir / "frames.csv") if r["status"] == "complete"), key=lambda r: int(r["frame_index"]))
    track_of = {}
    for r in csv_read(seq_dir / "instance_tracks.csv"):
        track_of[(r["image_name"], int(float(r["instance_id"])))] = r["track_id"]
    assign, gt_of = {}, {}
    n_poly = n_dup = n_matched = n_inst = used_frames = 0
    for pos, fr in enumerate(frames):
        name = fr["image_name"]
        mask = tifffile.imread(seq_dir / fr["mask_path"]).astype(np.int64)
        for inst in np.unique(mask):
            if inst and (name, int(inst)) in track_of:
                assign[(pos, int(inst))] = track_of[(name, int(inst))]
        n_inst += int((np.unique(mask) > 0).sum())
        if name not in gt:
            continue
        used_frames += 1
        t, w, h, polys = gt[name]
        if mask.shape != (h, w):
            raise ValueError(f"{name}: mask {mask.shape} != annotation size {(h, w)}")
        count = Counter(p["tid"] for p in polys)
        use, rasters = [], []
        for p in polys:
            if not p["tid"]:
                continue
            n_poly += 1
            if count[p["tid"]] > 1:                                               # rule 1
                n_dup += 1
                continue
            rr, cc = draw_polygon(p["pts"][:, 1], p["pts"][:, 0], (h, w))
            if len(rr):
                use.append(p); rasters.append((rr, cc))
        for i, inst in match_two_rounds(use, rasters, mask).items():                # rules 2-3
            tid = use[i]["tid"]
            gt_of[(pos, inst)] = (tid, sum(t >= c for c in cuts.get(tid, ())))      # rule 4
            n_matched += 1
    if not used_frames:
        raise ValueError(f"{seq_dir}: none of its image names occur in the annotation")
    summary, rows = evaluate_tracks(assign, gt_of)
    summary.update(sequence=frames[0]["sequence_id"], frames=len(frames), frames_with_gt=used_frames,
                   gt_polygons_with_id=n_poly, gt_polygons_dropped_duplicate_id=n_dup, gt_polygons_matched=n_matched,
                   detection_recall=n_matched / max(n_poly - n_dup, 1), predicted_instances=n_inst,
                   gt_tracks_with_missed_frame=sum(r["has_missed_frame"] for r in rows))
    out = Path(output_root) / frames[0]["sequence_id"] if output_root else seq_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "perfect_tracks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "perfect_tracks_per_gt_track.csv", "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["gt_track_id"])
        w_.writeheader(); w_.writerows(rows)
    print(f"{summary['sequence']}: perfect tracks {summary['perfect_frac']:.2%} ({summary['perfect_tracks']}/{summary['gt_tracks']}), "
          f"id switches {summary['id_switches']}, purity {summary['mean_purity']:.3f} | detection recall "
          f"{summary['detection_recall']:.1%} ({n_matched}/{n_poly - n_dup}), GT tracks with a missed frame {summary['gt_tracks_with_missed_frame']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path, help="CVAT XML/ZIP, or a folder holding exactly one")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence-dir", type=Path, action="append", help="a sequence folder (frames.csv, instance_tracks.csv, masks); repeatable")
    src.add_argument("--predictions-root", type=Path, help="evaluate every sequence under this folder whose images occur in the annotation")
    ap.add_argument("--output-root", type=Path, help="where to write the result files (default: into each sequence folder)")
    ap.add_argument("--label", default="cell", help="polygon label to use ('' = all)")
    ap.add_argument("--rough-max-vertices", type=int, default=8)
    ap.add_argument("--jump-px", type=float, default=300.0)
    args = ap.parse_args()
    try:
        gt = parse_gt(read_annotation(args.gt), args.label, args.rough_max_vertices)
        cuts = jump_cuts(gt, args.jump_px)
        if args.sequence_dir:
            seqs = args.sequence_dir
        else:
            seqs = [m.parent for m in sorted(args.predictions_root.rglob("frames.csv"))
                    if (m.parent / "instance_tracks.csv").is_file()
                    and any(r["image_name"] in gt for r in csv_read(m))]
        if not seqs:
            ap.exit(2, "no matching sequence found\n")
        for s in seqs:
            evaluate_sequence(s, gt, cuts, args.output_root)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        ap.exit(2, f"Evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
