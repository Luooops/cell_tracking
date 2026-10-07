"""
lab_xml_to_ctc_cpsam.py
==========================
One-step conversion of the lab's CVAT tracking annotation (one XML -- or a
ZIP holding annotations.xml -- per well, polygons with a `track_id`
attribute) plus the raw ch02 frames into a CTC-format dataset, cleaning the
known annotation problems on the way and taking every outline from Cellpose
while keeping the annotated track ids.

Why the outlines are replaced: the polygons are not a consistent description
of the nucleus. ~90% are imported pre-labels (source="file") made with an
older Cellpose v3 model; in 9 wells they follow the nucleus (IoU 0.90-0.93
with the fine-tuned Cellpose-SAM model) but in r01c20, r11c12, r12c13 -- and
for part of r13c12, r14c03, r16c02 -- they are ~2.5x larger regions around it
(confirmed by the supervisor to be segmentation errors, not a labelling
convention). The other ~10% were drawn by hand, mostly as 4-5 point boxes.
Area, intensity, centroid and crop all come from the outline, so the default
(--replace all) uses the annotation only for which cell is which track and
takes the shape from Cellpose. --replace rough reproduces the first version
(lab_ctc_v2): imported polygons kept as drawn, only rough ones re-outlined.
--replace none runs no Cellpose at all: rules 1 and 6 only, every polygon
rasterised as drawn (rough ones first, so an imported outline wins a pixel
conflict). That keeps every annotated cell, including the ones Cellpose does
not detect, so tracks have no segmentation-made gaps -- a linking-only
evaluation set -- at the price of the outlines being the annotation's own
(2.5x too large in some wells, 4-point boxes for hand-added cells).

Rules, in order (decided 2026-10-02):
  1. Duplicate ids. If a track_id is carried by 2+ polygons in one frame,
     all polygons with that id are removed from that frame only (the track
     keeps its other frames; the frame becomes a gap). Note this differs from
     gt_to_ctc.py, which drops such a track from every frame.
  2. A polygon is rough if source == "manual" or it has <= --rough-max-vertices
     vertices; otherwise it is accurate.
  3. Cellpose segments every frame (--cellpose-model; the lab fine-tuned model
     was trained on the gray frame stacked to 3 channels, hence --input-channels 3).
  4. Matching, per frame, in two rounds:
       a. accurate polygons claim instances: overlap / instance_area >= 0.5,
          greedy by overlap, one-to-one;
       b. rough polygons choose among the instances not claimed in (a):
          overlap / min(instance_area, polygon_area) >= 0.5 -- "most of the
          instance is in the box" OR "most of the box is in the instance",
          because hand-drawn boxes are often smaller than the cell.
  5. Output mask (--replace all): a matched polygon, accurate or rough ->
     the Cellpose instance's outline; an unmatched polygon -> removed (that
     frame is a gap for the track); Cellpose instance matching nothing ->
     discarded. With --replace rough an accurate polygon is instead kept as
     drawn, matched or not, and wins any pixel conflict.
  6. Jumps. If one track_id moves more than --jump-px between two consecutive
     frames (polygon area centroids), the track is cut there: the part after
     the jump gets a new track id. No detection is removed.
Nothing else is filtered (in particular not the "4 x median displacement"
suspects -- those are mostly the ordinary tail of cell motion).

--keep-unlabelled additionally writes every Cellpose instance that got NO
track id (about 3% of them: cells the annotation misses, cells whose polygon
was removed by rule 1, debris, partial cells at the border) into the masks,
each as its own single-frame "track" with a label >= UNLABELLED_OFFSET
(50000). That is what a tracker really receives after segmentation. Their
identity is unknown, not "different from everything": a link between two of
them, or between one of them and a track that has no labelled detection in
the other frame, may well be a true link. Training code must leave such
edges out of the loss, and scoring must ignore labels >= 50000. Without the
flag the output holds labelled cells only (the linking-only dataset).

Cellpose thresholds (--cellprob-threshold, --flow-threshold; added 2026-10-03).
Defaults are Cellpose's own (0.0 / 0.4). Every annotated nucleus Cellpose
misses becomes a gap in its track (rule 5), and a gap breaks the track for an
adjacent-frame tracker, so recall matters more than usual here. Measured on 4
development wells (r08c13 r15c12 r13c12 r03c12, every 3rd frame), nucleus
recall against the annotation polygons:
    cellprob / flow    recall   extra instances/frame   merged   split
    0 / 0.4 (default)   94.6%          10.4               22        8
    -1 / 0.8            96.1%          14.7               23        9
    -2 / 0.8            96.8%          18.1               23        9
    -4 / 0.4            93.0%           7.9               22        9
flow_threshold has to be relaxed together with cellprob_threshold: at 0.4 the
extra, less regular masks fail Cellpose's flow check and recall goes down.
-2 / 0.8 cut the misses by ~40% with no more merges or splits; on those wells
the oracle ceiling went 74.2% -> 77.1% perfect tracks and a tracker trained on
default-threshold data 50.2% -> 51.9%, id switches -13%. The extra instances
mostly end up as unlabelled detections (--keep-unlabelled).

Effect of the rules on the 15 wells delivered on 2026-10-02 (default
thresholds, --replace all --keep-unlabelled = lab_ctc_v3_full):
    polygons                         111,155 in 450 frames
      accurate / rough               100,232 / 10,923 (9.8% rough)
    rule 1, duplicate ids            87 events, 175 polygons removed
                                     (r08c13 45, r12c13 40, r11c12 38, r15c12 20, r16c13 18)
    rule 4a, accurate matched        98,788 of 100,232 (98.6%); 1,286 removed
                                     (r01c20 482, r11c12 213, r13c12 165, r12c13 129)
    rule 4b, rough matched           7,193 of 10,923 (65.9%); 3,713 removed
    rule 6, jump cuts                2
    detections                       111,067 -> 105,981 (-4.6%)
    consecutive links                103,897 -> 97,562 (-6.1%)
    tracks                           6,437 -> 6,349 (88 lost all their detections)
    tracks with a missing frame      603 (9.4%) -> 1,444 (22.7%)   <- Cellpose misses
    unlabelled Cellpose instances    3,267 kept (3% of all instances)
r01c20 is hit hardest: dim, noisy frames, Cellpose finds ~72% of its nuclei,
80 of its 132 tracks get a gap.

What the cleaned data is (and is not) good for:
  * Training and end-to-end evaluation of a tracker that runs on Cellpose
    output: detections are exactly what the tracker sees in deployment, the
    identities come from the annotation.
  * Its gaps are segmentation misses, not cell behaviour. An adjacent-frame
    tracker cannot bridge them (gap closing can), so report the oracle ceiling
    alongside any score. For linking-only evaluation (tracking quality with
    segmentation taken out) use --replace none.
  * The --replace none masks are not a good segmentation ground truth for
    CTC-style metrics (DET/TRA match objects by > 50% pixel overlap): in the
    wells with the oversized old polygons DET collapses (r01c20: 0.12). For
    CTC metrics build the GT from the Cellpose outline where the cell was found
    and the annotation polygon where it was missed (gnn_tracking/lab_dev/
    full_metrics.py, build_hybrid_gt).
  * Remaining label noise not handled by these rules: ~221 suspected identity
    splits across the 15 wells (an id ends and a new id starts at the same
    spot in the next frame; r14c03 alone 76). They are kept as annotated.

Output:
    <out-root>/<well>/tNNN.tif                  raw frames, 0-based by time order
    <out-root>/<well>_GT/TRA/man_trackNNN.tif   uint16, pixel value = track label
    <out-root>/<well>_GT/TRA/man_track.txt      label start end 0   (no division in this cell line)
    <out-root>/<well>_GT/track_id_map.csv       label <-> original XML track_id (+ segment after a cut)
    <out-root>/<well>_GT/removed_polygons.csv   every polygon not carried over, with the reason
    <out-root>/build_summary.json               per-well counts -- read before training
    <out-root>/_cellpose_cache/                 raw Cellpose instance masks (safe to delete;
                                                --cache-dir reuses the masks of an earlier run)

Usage (the builds used so far; all with the lab fine-tuned Cellpose model):
    M=~/.cellpose/models/cp4_20260721_210258
    GT="D:/project new/aws_gt_data/aws_gt_data"
    # training / end-to-end evaluation set (lab_ctc_v3_full)
    python lab_xml_to_ctc_cpsam.py --gt-root "$GT" --out-root ../../lab_ctc_v3_full \\
        --cellpose-model $M --keep-unlabelled
    # same with the higher-recall Cellpose thresholds (lab_ctc_v4_cp2f08)
    python lab_xml_to_ctc_cpsam.py --gt-root "$GT" --out-root ../../lab_ctc_v4_cp2f08 \\
        --cellpose-model $M --keep-unlabelled --cellprob-threshold -2 --flow-threshold 0.8
    # linking-only set, annotation polygons as drawn, no Cellpose (lab_ctc_gt_clean)
    python lab_xml_to_ctc_cpsam.py --gt-root "$GT" --out-root ../../lab_ctc_gt_clean --replace none

--gt-root is searched for well folders (r##c##) that hold one .xml/.zip and
the frames `<well>f01p01-ch02tNN.tiff`. A finished well is skipped on a
re-run; --overwrite redoes it. --wells limits the run to some wells.
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import tifffile
from skimage.draw import polygon as draw_polygon

CONTAIN_THRESH = 0.5
UNLABELLED_OFFSET = 50000   # --keep-unlabelled: labels >= this are Cellpose instances without a GT track id


# ------------------------------------------------------------------ reading
def find_wells(gt_root):
    """{well: (folder, annotation_file)} for every r##c## folder under gt_root holding exactly one xml/zip."""
    wells = {}
    for dirpath, _, files in os.walk(gt_root):
        well = os.path.basename(dirpath)
        if not re.fullmatch(r"r\d+c\d+", well):
            continue
        ann = [f for f in files if f.lower().endswith((".xml", ".zip"))]
        if len(ann) != 1:
            raise ValueError(f"{dirpath}: expected exactly one .xml/.zip, found {len(ann)}")
        if well in wells:
            raise ValueError(f"well {well} appears twice under {gt_root}")
        wells[well] = (dirpath, os.path.join(dirpath, ann[0]))
    return dict(sorted(wells.items()))


def read_annotation(path):
    data = open(path, "rb").read()
    if data[:2] == b"PK":                                   # CVAT export: a ZIP named .xml
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
    if abs(a) < 1e-9:                                       # degenerate polygon
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def parse_frames(root, rough_max_vertices):
    """[(t, image_basename, width, height, [polygon dict])] sorted by t (time number from the file name)."""
    frames = []
    for img in root.iter("image"):
        name = os.path.basename(img.get("name"))
        m = re.search(r"t(\d+)\.tiff?$", name)
        if not m:
            raise ValueError(f"cannot read the time index from image name {name!r}")
        polys = []
        for k, sh in enumerate(img.iter("polygon")):
            tid = next(((a.text or "").strip() for a in sh.iter("attribute") if a.get("name") == "track_id"), "")
            pts = np.array([[float(v) for v in p.split(",")] for p in sh.get("points").split(";")])
            polys.append(dict(idx=k + 1, tid=tid, pts=pts, n=len(pts), source=sh.get("source"),
                              rough=sh.get("source") == "manual" or len(pts) <= rough_max_vertices,
                              centroid=area_centroid(pts)))
        frames.append((int(m.group(1)), name, int(img.get("width")), int(img.get("height")), polys))
    frames.sort(key=lambda f: f[0])
    if len({f[0] for f in frames}) != len(frames):
        raise ValueError("two images share the same time index")
    return frames


# ------------------------------------------------------------------ matching
def _greedy(cands):
    cands.sort(reverse=True)
    used_g, used_c, out = set(), set(), {}
    for _, g, c in cands:
        if g in used_g or c in used_c:
            continue
        out[g] = c
        used_g.add(g)
        used_c.add(c)
    return out


def match_two_rounds(polys, rasters, cp):
    """polys/rasters: parallel lists (rasters[i] = (rr, cc) of polygon i on its own, no overwriting).
    Returns (accurate_matches, rough_matches): {polygon list index: Cellpose instance id}."""
    cp_area = np.bincount(cp.ravel())
    ov = {}
    for i, (rr, cc) in enumerate(rasters):
        ids, cnt = np.unique(cp[rr, cc], return_counts=True)
        for c, n in zip(ids, cnt):
            if c:
                ov[(i, int(c))] = int(n)
    acc = _greedy([(o, g, c) for (g, c), o in ov.items()
                   if not polys[g]["rough"] and o / cp_area[c] >= CONTAIN_THRESH])
    taken = set(acc.values())
    rough = _greedy([(o, g, c) for (g, c), o in ov.items()
                     if polys[g]["rough"] and c not in taken
                     and o / min(cp_area[c], len(rasters[g][0])) >= CONTAIN_THRESH])
    return acc, rough


# ------------------------------------------------------------------ one well
def _track_stats(frames_of):
    """frames_of: {track key: sorted frame-number list}."""
    links = sum(sum(b - a == 1 for a, b in zip(f, f[1:])) for f in frames_of.values())
    gap = sum(any(b - a > 1 for a, b in zip(f, f[1:])) for f in frames_of.values())
    return links, gap


def process_well(well, folder, ann_path, out_root, segment, args):
    frames = parse_frames(read_annotation(ann_path), args.rough_max_vertices)
    n_frames = len(frames)
    S = Counter(n_frames=n_frames)
    removed = []                                           # (t, image, polygon idx, xml track id, reason)

    # ---- rule 6 is decided on the annotation itself (after rule 1), before any Cellpose removal ----
    seen = defaultdict(dict)                               # tid -> {t: centroid}, unique endpoints only
    orig = defaultdict(set)
    for t, name, w, h, polys in frames:
        count = Counter(p["tid"] for p in polys)
        for p in polys:
            S["polygons"] += 1
            S["rough" if p["rough"] else "accurate"] += 1
            if p["tid"]:
                orig[p["tid"]].add(t)
            if p["tid"] and count[p["tid"]] == 1:
                seen[p["tid"]][t] = p["centroid"]
        S["duplicate_events"] += sum(c > 1 for tid, c in count.items() if tid)
    cut_after = defaultdict(list)                          # tid -> frame numbers t where a new segment starts
    for tid, cen in seen.items():
        for t in sorted(cen):
            if t + 1 in cen and np.hypot(cen[t + 1][0] - cen[t][0], cen[t + 1][1] - cen[t][1]) > args.jump_px:
                cut_after[tid].append(t + 1)
                S["jump_cuts"] += 1

    def track_key(tid, t):
        return tid, sum(t >= c for c in cut_after.get(tid, ()))

    # ---- per frame: clean, segment, match, draw ----
    local, objects = [], []                                # per frame: uint16 image of local object ids; {local id: track key}
    kept = defaultdict(list)                               # track key -> frame numbers present in the output
    unlabelled = {}                                        # ("", n) -> frame number, --keep-unlabelled only
    cache_dir = os.path.join(args.cache_dir or os.path.join(out_root, "_cellpose_cache"), well)
    if args.replace != "none":
        os.makedirs(cache_dir, exist_ok=True)
    for t, name, w, h, polys in frames:
        raw_path = os.path.join(folder, name)
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(f"[{well}] raw frame not found: {raw_path}")
        count = Counter(p["tid"] for p in polys)
        use, rasters = [], []
        for p in polys:
            if not p["tid"]:
                removed.append((t, name, p["idx"], "", "no_track_id")); S["removed_no_id"] += 1
                continue
            if count[p["tid"]] > 1:                                                   # rule 1
                removed.append((t, name, p["idx"], p["tid"], "duplicate_id_in_frame")); S["removed_duplicate"] += 1
                continue
            rr, cc = draw_polygon(p["pts"][:, 1], p["pts"][:, 0], (h, w))
            if len(rr) == 0:
                removed.append((t, name, p["idx"], p["tid"], "empty_polygon")); S["removed_empty"] += 1
                continue
            use.append(p)
            rasters.append((rr, cc))

        cache = os.path.join(cache_dir, f"t{t:02d}.npy")
        if args.replace == "none":
            cp = np.zeros((h, w), np.uint16)                # no segmentation: nothing to match, nothing to replace
        elif os.path.isfile(cache):
            cp = np.load(cache)
        else:
            cp = np.asarray(segment(tifffile.imread(raw_path))).astype(np.uint16)
            np.save(cache, cp)
        if cp.shape != (h, w):
            raise ValueError(f"[{well}] t{t}: Cellpose mask {cp.shape} != image {(h, w)}")
        cp = cp.astype(np.int32)

        acc_m, rough_m = match_two_rounds(use, rasters, cp)                           # rule 4
        S["cellpose_instances"] += int(cp.max())
        S["cellpose_discarded"] += int(cp.max()) - len(rough_m) - len(acc_m)
        S["accurate_with_cellpose_match"] += len(acc_m)

        obj = {}
        inst = np.zeros(int(cp.max()) + 1, np.uint16)                                 # Cellpose instance id -> local object id
        for i, p in enumerate(use):                                                   # rule 5: Cellpose outlines first ...
            kind = "rough" if p["rough"] else "accurate"
            m = rough_m if p["rough"] else acc_m
            if args.replace == "none":
                continue
            if p["rough"] or args.replace == "all":
                if i in m:
                    inst[m[i]] = i + 1
                    obj[i + 1] = track_key(p["tid"], t)
                    S[f"{kind}_replaced"] += 1
                else:
                    removed.append((t, name, p["idx"], p["tid"], f"{kind}_no_cellpose_match")); S[f"removed_{kind}_unmatched"] += 1
        extra = {}                                                                    # local id -> ("", running number)
        if args.keep_unlabelled:
            nxt = len(use) + 1
            for c in np.unique(cp):
                if c and inst[c] == 0:
                    inst[c] = nxt
                    extra[nxt] = ("", len(unlabelled) + 1)
                    unlabelled[extra[nxt]] = t
                    nxt += 1
        img = inst[cp]
        for want_rough in (True, False):                                              # ... polygons kept as drawn on top,
            for i, p in enumerate(use):                                               #     imported outlines last (they win)
                if p["rough"] == want_rough and (args.replace == "none" or (not p["rough"] and args.replace == "rough")):
                    img[rasters[i]] = i + 1
                    obj[i + 1] = track_key(p["tid"], t)
        present = set(np.unique(img).tolist())
        for lid in list(obj):
            if lid not in present:                                                    # fully covered by other outlines
                p = use[lid - 1]
                removed.append((t, name, p["idx"], p["tid"], "covered_by_other_outlines")); S["removed_covered"] += 1
                if args.replace != "none" and (p["rough"] or args.replace == "all"):
                    S["rough_replaced" if p["rough"] else "accurate_replaced"] -= 1
                del obj[lid]
        for key in obj.values():
            kept[key].append(t)
        obj.update(extra)
        local.append(img)
        objects.append(obj)

    # ---- labels, masks, lineage ----
    keys = sorted(kept, key=lambda k: ([int(s) if s.isdigit() else s for s in re.split(r"(\d+)", k[0])], k[1]))
    label_of = {k: i + 1 for i, k in enumerate(keys)}
    if len(label_of) >= UNLABELLED_OFFSET or UNLABELLED_OFFSET + len(unlabelled) > np.iinfo(np.uint16).max:
        raise ValueError(f"[{well}] {len(label_of)} tracks + {len(unlabelled)} unlabelled instances do not fit the label ranges")
    label_of.update({k: UNLABELLED_OFFSET + k[1] for k in unlabelled})
    fi_of = {f[0]: i for i, f in enumerate(frames)}
    img_dir = os.path.join(out_root, well)
    gt_dir = os.path.join(out_root, f"{well}_GT")
    tra_dir = os.path.join(gt_dir, "TRA")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(tra_dir, exist_ok=True)
    for (t, name, w, h, _), img, obj in zip(frames, local, objects):
        lut = np.zeros(int(img.max()) + 1, np.uint16)
        for lid, key in obj.items():
            lut[lid] = label_of[key]
        tifffile.imwrite(os.path.join(tra_dir, f"man_track{fi_of[t]:03d}.tif"), lut[img])
        shutil.copy2(os.path.join(folder, name), os.path.join(img_dir, f"t{fi_of[t]:03d}.tif"))

    with open(os.path.join(gt_dir, "track_id_map.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "xml_track_id", "segment", "first_t", "last_t", "n_frames"])
        for k in keys:
            wr.writerow([label_of[k], k[0], k[1], kept[k][0], kept[k][-1], len(kept[k])])
    with open(os.path.join(gt_dir, "removed_polygons.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "image", "polygon_index_in_image", "xml_track_id", "reason"])
        wr.writerows(removed)
    with open(os.path.join(tra_dir, "man_track.txt"), "w") as f:                      # written last = "this well is finished"
        f.write("".join(f"{label_of[k]} {fi_of[kept[k][0]]} {fi_of[kept[k][-1]]} 0\n" for k in keys))
        f.write("".join(f"{label_of[k]} {fi_of[t]} {fi_of[t]} 0\n" for k, t in unlabelled.items()))

    before = {tid: sorted(ts) for tid, ts in orig.items()}
    lb, gb = _track_stats(before)
    la, ga = _track_stats(kept)
    S.update(tracks_before=len(before), tracks_after=len(kept),
             detections_before=sum(len(v) for v in before.values()), detections_after=sum(len(v) for v in kept.values()),
             links_before=lb, links_after=la, tracks_with_gap_before=gb, tracks_with_gap_after=ga,
             tracks_all_frames_before=sum(len(v) == n_frames for v in before.values()),
             tracks_all_frames_after=sum(len(v) == n_frames for v in kept.values()))
    out = dict(well=well, **{k: int(v) for k, v in S.items()})
    out["rough_match_rate"] = out.get("rough_replaced", 0) / max(out.get("rough_replaced", 0) + out.get("removed_rough_unmatched", 0), 1)
    out["accurate_match_rate"] = out.get("accurate_with_cellpose_match", 0) / max(out.get("accurate", 0), 1)
    out["replace"] = args.replace
    out["unlabelled_instances_kept"] = len(unlabelled)
    return out


# ------------------------------------------------------------------ main
def make_segmenter(model_path, gpu, channels, cellprob_threshold=0.0, flow_threshold=0.4):
    from cellpose import models
    kwargs = {"pretrained_model": os.path.expanduser(model_path)} if model_path else {}
    print(f"loading Cellpose model '{model_path or 'default (Cellpose-SAM)'}' (gpu={gpu})...", flush=True)
    model = models.CellposeModel(gpu=gpu, **kwargs)

    def segment(raw):
        a = raw.astype(np.float32)
        return model.eval(np.stack([a, a, a], -1) if channels == 3 else a,
                          cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)[0]
    return segment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", required=True, help="folder searched for r##c## well folders (annotation + raw frames)")
    ap.add_argument("--out-root", required=True, help="CTC-format output root")
    ap.add_argument("--wells", nargs="+", help="only these wells (default: all found)")
    ap.add_argument("--cellpose-model", default=None, help="Cellpose model name or path (default: the installed default)")
    ap.add_argument("--input-channels", type=int, choices=[1, 3], default=3,
                    help="3 = gray frame stacked to 3 channels (how the lab fine-tuned model was trained); 1 = plain gray")
    ap.add_argument("--cellprob-threshold", type=float, default=0.0,
                    help="Cellpose cellprob_threshold (default 0.0 = Cellpose default; lower finds dimmer nuclei)")
    ap.add_argument("--flow-threshold", type=float, default=0.4,
                    help="Cellpose flow_threshold (default 0.4 = Cellpose default; higher keeps less regular shapes). "
                         "On the lab dev wells -2 / 0.8 raised nucleus recall 94.6%% -> 96.8%%")
    ap.add_argument("--replace", choices=["all", "rough", "none"], default="all",
                    help="all = every outline comes from Cellpose (default); rough = only rough polygons are re-outlined; "
                         "none = no Cellpose, polygons as drawn (cleaning rules only)")
    ap.add_argument("--keep-unlabelled", action="store_true",
                    help=f"also write Cellpose instances that got no track id, as single-frame labels >= {UNLABELLED_OFFSET}")
    ap.add_argument("--cache-dir", default=None, help="folder of cached Cellpose masks (default: <out-root>/_cellpose_cache)")
    ap.add_argument("--rough-max-vertices", type=int, default=8, help="a polygon with this many vertices or fewer counts as rough")
    ap.add_argument("--jump-px", type=float, default=300.0, help="cut a track where it moves more than this between consecutive frames")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="redo wells already finished in --out-root")
    args = ap.parse_args()

    if args.keep_unlabelled and args.replace != "all":
        ap.error("--keep-unlabelled needs --replace all (kept polygons and extra Cellpose instances could overlap)")
    out_root = os.path.abspath(args.out_root)
    wells = find_wells(args.gt_root)
    if args.wells:
        missing = sorted(set(args.wells) - set(wells))
        if missing:
            ap.error(f"wells not found under --gt-root: {missing}")
        wells = {w: wells[w] for w in args.wells}
    done = lambda w: os.path.isfile(os.path.join(out_root, f"{w}_GT", "TRA", "man_track.txt"))
    todo = [w for w in wells if args.overwrite or not done(w)]
    print(f"{len(wells)} wells found, {len(wells) - len(todo)} already finished, {len(todo)} to run: {' '.join(todo)}")
    if not todo:
        return
    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, "build_summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = {s["well"]: s for s in json.load(f)}

    model = {}

    def segment(raw):                                       # loaded on first use: a fully cached run needs no Cellpose
        if "fn" not in model:
            model["fn"] = make_segmenter(args.cellpose_model, not args.cpu, args.input_channels,
                                         args.cellprob_threshold, args.flow_threshold)
        return model["fn"](raw)

    for i, well in enumerate(todo):
        t0 = time.time()
        txt = os.path.join(out_root, f"{well}_GT", "TRA", "man_track.txt")
        if os.path.exists(txt):
            os.remove(txt)
        s = process_well(well, *wells[well], out_root, segment, args)
        summary[well] = s
        with open(summary_path, "w") as f:
            json.dump([summary[w] for w in sorted(summary)], f, indent=2)
        print(f"[{i + 1}/{len(todo)}] {well}: {time.time() - t0:.0f}s | polygons {s['polygons']} "
              f"(rough {s.get('rough', 0)}) | duplicate-id removed {s.get('removed_duplicate', 0)} | "
              f"rough replaced {s.get('rough_replaced', 0)}, removed {s.get('removed_rough_unmatched', 0)} "
              f"({s['rough_match_rate'] * 100:.1f}% matched) | accurate replaced {s.get('accurate_replaced', 0)}, "
              f"removed {s.get('removed_accurate_unmatched', 0)} | jump cuts {s.get('jump_cuts', 0)} | "
              f"tracks {s['tracks_before']}->{s['tracks_after']} | detections {s['detections_before']}->{s['detections_after']} | "
              f"links {s['links_before']}->{s['links_after']} | tracks with a gap {s['tracks_with_gap_before']}->{s['tracks_with_gap_after']}",
              flush=True)
    print(f"\nper-well counts: {summary_path}")


if __name__ == "__main__":
    main()
