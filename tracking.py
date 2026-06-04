import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

from mask_area_filter import filter_small_instances_by_mean


# =====================================
# Utilities
# =====================================

def sort_key_by_time(path: Path):
    """
    Extract time index from filenames like:
    r01c12f01p01-ch02t01_mask.tiff
    """
    m = re.search(r"t(\d+)", path.stem)
    return int(m.group(1)) if m else None


def has_time_index(path: Path) -> bool:
    return sort_key_by_time(path) is not None


def remove_small_instances(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """
    Remove small labeled instances and relabel remaining instances to 1..N.
    """
    if min_area <= 0:
        return mask.astype(np.int32)

    clean_mask = np.zeros_like(mask, dtype=np.int32)
    new_label = 1

    for prop in regionprops(mask):
        if prop.area >= min_area:
            clean_mask[mask == prop.label] = new_label
            new_label += 1

    return clean_mask


def resolve_min_area_for_mask(min_area, frame_idx: int, mask_path: Path):
    if isinstance(min_area, dict):
        return min_area.get(mask_path.name, 0)
    if isinstance(min_area, (list, tuple)):
        if frame_idx < len(min_area):
            return min_area[frame_idx]
        return 0
    return min_area


def load_mask_sequence(
    mask_dir: str,
    min_area: int | list[int] | tuple[int, ...] | dict[str, int] | None = 0,
    auto_min_area_fraction: float = 0.25,
):
    """
    Load all mask files and stack them into (T, Y, X).
    """
    mask_dir = Path(mask_dir)

    mask_files = list(mask_dir.glob("*_mask.tif")) + list(mask_dir.glob("*_mask.tiff"))
    mask_files = [p for p in mask_files if has_time_index(p)]
    mask_files = sorted(mask_files, key=sort_key_by_time)

    if len(mask_files) == 0:
        raise FileNotFoundError(f"No mask files found in: {mask_dir}")

    masks = []
    counts_per_frame = []

    for frame_idx, p in enumerate(mask_files):
        arr = tifffile.imread(str(p))
        if arr.ndim > 2:
            arr = np.squeeze(arr)

        arr = arr.astype(np.int32)
        frame_min_area = resolve_min_area_for_mask(min_area, frame_idx, p)
        if frame_min_area is None:
            arr, _, _ = filter_small_instances_by_mean(
                arr,
                fraction_of_mean=auto_min_area_fraction,
            )
        else:
            arr = remove_small_instances(arr, min_area=int(frame_min_area))

        masks.append(arr)
        counts_per_frame.append(int(arr.max()))

    segmentation = np.stack(masks, axis=0)
    return segmentation, mask_files, counts_per_frame


def extract_detections_from_mask(mask: np.ndarray, frame_idx: int):
    """
    Extract detections from one frame mask.

    Returns a list of dicts with centroid, area, shape, and bbox features.
    """
    detections = []

    for prop in regionprops(mask):
        min_row, min_col, max_row, max_col = prop.bbox
        cy, cx = prop.centroid

        detections.append(
            {
                "frame": frame_idx,
                "label": int(prop.label),
                "y": float(cy),
                "x": float(cx),
                "area": float(prop.area),
                "major_axis_length": float(prop.major_axis_length),
                "minor_axis_length": float(prop.minor_axis_length),
                "eccentricity": float(prop.eccentricity),
                "solidity": float(prop.solidity),
                "bbox_min_row": int(min_row),
                "bbox_min_col": int(min_col),
                "bbox_max_row": int(max_row),
                "bbox_max_col": int(max_col),
            }
        )

    return detections


def euclidean_distance(y1, x1, y2, x2):
    return float(np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2))


def ratio_ok(v1: float, v2: float, max_ratio: float) -> bool:
    """
    Symmetric ratio check. Returns True if max(v1/v2, v2/v1) <= max_ratio.
    """
    if v1 <= 0 or v2 <= 0:
        return False
    ratio = max(v1 / v2, v2 / v1)
    return ratio <= max_ratio


def safe_ratio(v1: float, v2: float) -> float:
    """
    Symmetric ratio >= 1.0
    """
    if v1 <= 0 or v2 <= 0:
        return 1e6
    return max(v1 / v2, v2 / v1)


def filter_tracks_by_length(tracks, min_length=5):
    return [trk for trk in tracks if len(trk["history"]) >= min_length]


def _track_direction(history, window: int = 3):
    if len(history) < 2:
        return None

    recent = history[-(min(window, len(history) - 1) + 1):]
    dy_sum = 0.0
    dx_sum = 0.0

    for prev_det, last_det in zip(recent[:-1], recent[1:]):
        dy_sum += last_det["y"] - prev_det["y"]
        dx_sum += last_det["x"] - prev_det["x"]

    norm = float(np.hypot(dy_sum, dx_sum))
    if norm <= 1e-6:
        return None

    return dy_sum / norm, dx_sum / norm


def _direction_match(track_a, track_b, max_angle_diff_deg: float = 90.0) -> bool:
    dir_a = _track_direction(track_a["history"])
    dir_b = _track_direction(track_b["history"])

    if dir_a is None or dir_b is None:
        return True

    cos_sim = float(np.clip(dir_a[0] * dir_b[0] + dir_a[1] * dir_b[1], -1.0, 1.0))
    angle_diff = float(np.degrees(np.arccos(cos_sim)))
    return angle_diff <= max_angle_diff_deg


def _track_angle_diff_deg(track_a, track_b):
    """
    Return direction difference in degrees, or None if either track is too short.
    """
    dir_a = _track_direction(track_a["history"])
    dir_b = _track_direction(track_b["history"])

    if dir_a is None or dir_b is None:
        return None

    cos_sim = float(np.clip(dir_a[0] * dir_b[0] + dir_a[1] * dir_b[1], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_sim)))


def can_close_gap(track_a, track_b, max_gap=3, max_dist=35.0, max_area_ratio=2.0, max_shape_ratio=2.0, max_angle_diff_deg: float = 90.0):
    """
    track_a ends first, track_b starts later
    """
    hist_a = track_a["history"]
    hist_b = track_b["history"]

    end_a = hist_a[-1]
    start_b = hist_b[0]

    gap = start_b["frame"] - end_a["frame"]
    if gap <= 0 or gap > max_gap:
        return False

    dist = euclidean_distance(end_a["y"], end_a["x"], start_b["y"], start_b["x"])
    if dist > max_dist:
        return False

    if not ratio_ok(end_a["area"], start_b["area"], max_area_ratio):
        return False

    if not ratio_ok(end_a["major_axis_length"], start_b["major_axis_length"], max_shape_ratio):
        return False

    if not ratio_ok(end_a["minor_axis_length"], start_b["minor_axis_length"], max_shape_ratio):
        return False

    if not _direction_match(track_a, track_b, max_angle_diff_deg=max_angle_diff_deg):
        return False

    return True


def compute_gap_close_cost(
    track_a,
    track_b,
    max_gap=3,
    max_dist=35.0,
    max_area_ratio=2.0,
    max_shape_ratio=2.0,
    max_angle_diff_deg: float = 90.0,
    distance_weight: float = 10.0,
    area_weight: float = 4.0,
    shape_weight: float = 2.0,
    angle_weight: float = 2.0,
    gap_weight: float = 1.0,
):
    """
    Continuous cost for linking track_a -> track_b.

    Returns None when the pair fails hard gating.
    """
    hist_a = track_a["history"]
    hist_b = track_b["history"]

    end_a = hist_a[-1]
    start_b = hist_b[0]

    gap = start_b["frame"] - end_a["frame"]
    if gap <= 0 or gap > max_gap:
        return None

    dist = euclidean_distance(end_a["y"], end_a["x"], start_b["y"], start_b["x"])
    if dist > max_dist:
        return None

    if not ratio_ok(end_a["area"], start_b["area"], max_area_ratio):
        return None

    if not ratio_ok(end_a["major_axis_length"], start_b["major_axis_length"], max_shape_ratio):
        return None

    if not ratio_ok(end_a["minor_axis_length"], start_b["minor_axis_length"], max_shape_ratio):
        return None

    angle_diff = _track_angle_diff_deg(track_a, track_b)
    if angle_diff is not None and angle_diff > max_angle_diff_deg:
        return None

    area_penalty = safe_ratio(end_a["area"], start_b["area"]) - 1.0
    major_penalty = safe_ratio(end_a["major_axis_length"], start_b["major_axis_length"]) - 1.0
    minor_penalty = safe_ratio(end_a["minor_axis_length"], start_b["minor_axis_length"]) - 1.0
    shape_penalty = 0.5 * (major_penalty + minor_penalty)
    angle_penalty = 0.0 if angle_diff is None else angle_diff / max(max_angle_diff_deg, 1e-6)
    gap_penalty = gap / max(max_gap, 1)

    return (
        distance_weight * (dist / max(max_dist, 1e-6))
        + area_weight * area_penalty
        + shape_weight * shape_penalty
        + angle_weight * angle_penalty
        + gap_weight * gap_penalty
    )


def _merge_gap_close_links(tracks, links):
    successor = {src: dst for src, dst in links}
    predecessor = {dst: src for src, dst in links}
    merged_tracks = []
    visited = set()

    for i, trk in enumerate(tracks):
        if i in predecessor:
            continue

        current_idx = i
        merged = {
            **trk,
            "history": [det.copy() for det in trk["history"]],
        }
        visited.add(current_idx)

        while current_idx in successor:
            next_idx = successor[current_idx]
            if next_idx in visited:
                break

            merged["history"].extend([det.copy() for det in tracks[next_idx]["history"]])
            visited.add(next_idx)
            current_idx = next_idx

        merged_tracks.append(merged)

    for i, trk in enumerate(tracks):
        if i not in visited:
            merged_tracks.append(
                {
                    **trk,
                    "history": [det.copy() for det in trk["history"]],
                }
            )

    for new_id, trk in enumerate(merged_tracks, start=1):
        trk["track_id"] = new_id

    return merged_tracks


def gap_close_tracks(
    tracks,
    max_gap=3,
    max_dist=35.0,
    max_area_ratio=2.0,
    max_shape_ratio=2.0,
    max_angle_diff_deg: float = 90.0,
    max_close_cost: float = 12.0,
    distance_weight: float = 10.0,
    area_weight: float = 4.0,
    shape_weight: float = 2.0,
    angle_weight: float = 2.0,
    gap_weight: float = 1.0,
):
    """
    Global-cost gap closing on finished tracks using Hungarian matching.
    """
    tracks = [
        {
            **trk,
            "history": [det.copy() for det in trk["history"]],
        }
        for trk in tracks
    ]
    tracks = sorted(tracks, key=lambda t: t["history"][0]["frame"])

    if len(tracks) <= 1:
        return tracks

    current_tracks = tracks

    for gap_level in range(1, max_gap + 1):
        n_tracks = len(current_tracks)
        cost = np.full((n_tracks, n_tracks), fill_value=1e9, dtype=np.float32)

        for i, track_a in enumerate(current_tracks):
            end_frame = track_a["history"][-1]["frame"]

            for j, track_b in enumerate(current_tracks):
                if i == j:
                    continue

                start_frame = track_b["history"][0]["frame"]
                if start_frame - end_frame != gap_level:
                    continue

                pair_cost = compute_gap_close_cost(
                    track_a,
                    track_b,
                    max_gap=max_gap,
                    max_dist=max_dist,
                    max_area_ratio=max_area_ratio,
                    max_shape_ratio=max_shape_ratio,
                    max_angle_diff_deg=max_angle_diff_deg,
                    distance_weight=distance_weight,
                    area_weight=area_weight,
                    shape_weight=shape_weight,
                    angle_weight=angle_weight,
                    gap_weight=gap_weight,
                )

                if pair_cost is not None:
                    cost[i, j] = pair_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        links = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_close_cost:
                links.append((int(r), int(c)))

        if len(links) == 0:
            continue

        current_tracks = _merge_gap_close_links(current_tracks, links)

    return current_tracks

def bbox_iou(box1, box2):
    """
    box = (min_row, min_col, max_row, max_col)
    """
    r1_min, c1_min, r1_max, c1_max = box1
    r2_min, c2_min, r2_max, c2_max = box2

    inter_rmin = max(r1_min, r2_min)
    inter_cmin = max(c1_min, c2_min)
    inter_rmax = min(r1_max, r2_max)
    inter_cmax = min(c1_max, c2_max)

    inter_h = max(0, inter_rmax - inter_rmin)
    inter_w = max(0, inter_cmax - inter_cmin)
    inter_area = inter_h * inter_w

    area1 = max(0, r1_max - r1_min) * max(0, c1_max - c1_min)
    area2 = max(0, r2_max - r2_min) * max(0, c2_max - c2_min)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# =====================================
# Tracker
# =====================================

class SimpleCellTrackerV3:
    """
    Multi-frame tracker using:
    - multi-frame history based motion prediction
    - multi-frame averaged area/shape reference
    - staged Hungarian association for lost_count = 0,1,2,...
    - gap-aware distance expansion
    """

    def __init__(
        self,
        max_distance: float = 40.0,
        max_area_ratio: float = 2.0,
        max_shape_ratio: float = 2.0,
        max_lost: int = 3,
        area_weight: float = 5.0,
        shape_weight: float = 3.0,
        iou_weight: float = 8.0,
        n_history: int = 3,
        gap_growth: float = 0.35,
    ):
        self.max_distance = max_distance
        self.max_area_ratio = max_area_ratio
        self.max_shape_ratio = max_shape_ratio
        self.max_lost = max_lost
        self.area_weight = area_weight
        self.shape_weight = shape_weight
        self.iou_weight = iou_weight
        self.n_history = n_history
        self.gap_growth = gap_growth

        self.next_track_id = 1
        self.active_tracks = {}
        self.finished_tracks = {}

    def _create_track(self, det: dict):
        track_id = self.next_track_id
        self.next_track_id += 1

        self.active_tracks[track_id] = {
            "track_id": track_id,
            "vx": 0.0,
            "vy": 0.0,
            "last_frame": det["frame"],
            "lost_count": 0,
            "history": [det.copy()],
        }

    def _finish_track(self, track_id: int):
        self.finished_tracks[track_id] = self.active_tracks.pop(track_id)

    def _get_recent_history(self, trk: dict):
        hist = trk["history"]
        if len(hist) <= self.n_history:
            return hist
        return hist[-self.n_history:]

    def _get_reference_state(self, trk: dict):
        """
        Use recent history mean as a more stable reference for area/shape/bbox center.
        """
        hist = self._get_recent_history(trk)

        ref = {
            "area": float(np.mean([h["area"] for h in hist])),
            "major_axis_length": float(np.mean([h["major_axis_length"] for h in hist])),
            "minor_axis_length": float(np.mean([h["minor_axis_length"] for h in hist])),
            "eccentricity": float(np.mean([h["eccentricity"] for h in hist])),
            "solidity": float(np.mean([h["solidity"] for h in hist])),
            "bbox_min_row": int(round(np.mean([h["bbox_min_row"] for h in hist]))),
            "bbox_min_col": int(round(np.mean([h["bbox_min_col"] for h in hist]))),
            "bbox_max_row": int(round(np.mean([h["bbox_max_row"] for h in hist]))),
            "bbox_max_col": int(round(np.mean([h["bbox_max_col"] for h in hist]))),
        }
        return ref

    def _predict_position(self, trk: dict, frame_idx: int):
        """
        Multi-frame prediction:
        - if enough history, estimate velocity from recent history average displacement
        - otherwise use stored vx, vy
        """
        hist = self._get_recent_history(trk)

        if len(hist) >= 2:
            dys = []
            dxs = []
            dts = []

            for a, b in zip(hist[:-1], hist[1:]):
                dt = b["frame"] - a["frame"]
                if dt > 0:
                    dys.append((b["y"] - a["y"]) / dt)
                    dxs.append((b["x"] - a["x"]) / dt)
                    dts.append(dt)

            if len(dys) > 0:
                vy = float(np.mean(dys))
                vx = float(np.mean(dxs))
            else:
                vy = trk["vy"]
                vx = trk["vx"]
        else:
            vy = trk["vy"]
            vx = trk["vx"]

        last_obs = hist[-1]
        frame_gap = frame_idx - last_obs["frame"]

        pred_y = last_obs["y"] + vy * frame_gap
        pred_x = last_obs["x"] + vx * frame_gap

        return pred_y, pred_x, vy, vx

    def _gap_adjusted_max_distance(self, gap: int):
        """
        Allow a slightly larger search radius for larger frame gaps.
        """
        return self.max_distance * (1.0 + self.gap_growth * max(0, gap - 1))

    def _build_cost_matrix(self, track_ids, detections):
        n_tracks = len(track_ids)
        n_dets = len(detections)

        cost = np.full((n_tracks, n_dets), fill_value=1e9, dtype=np.float32)

        for i, track_id in enumerate(track_ids):
            trk = self.active_tracks[track_id]
            ref = self._get_reference_state(trk)

            for j, det in enumerate(detections):
                gap = det["frame"] - trk["last_frame"]
                if gap <= 0 or gap > self.max_lost + 1:
                    continue

                pred_y, pred_x, _, _ = self._predict_position(trk, det["frame"])
                dist = euclidean_distance(pred_y, pred_x, det["y"], det["x"])

                allowed_dist = self._gap_adjusted_max_distance(gap)
                if dist > allowed_dist:
                    continue

                if not ratio_ok(ref["area"], det["area"], self.max_area_ratio):
                    continue

                if not ratio_ok(ref["major_axis_length"], det["major_axis_length"], self.max_shape_ratio):
                    continue

                if not ratio_ok(ref["minor_axis_length"], det["minor_axis_length"], self.max_shape_ratio):
                    continue

                area_ratio = safe_ratio(ref["area"], det["area"])
                major_ratio = safe_ratio(ref["major_axis_length"], det["major_axis_length"])
                minor_ratio = safe_ratio(ref["minor_axis_length"], det["minor_axis_length"])

                shape_penalty = 0.5 * ((major_ratio - 1.0) + (minor_ratio - 1.0))

                track_box = (
                    ref["bbox_min_row"],
                    ref["bbox_min_col"],
                    ref["bbox_max_row"],
                    ref["bbox_max_col"],
                )

                det_box = (
                    det["bbox_min_row"],
                    det["bbox_min_col"],
                    det["bbox_max_row"],
                    det["bbox_max_col"],
                )

                iou = bbox_iou(track_box, det_box)
                iou_penalty = 1.0 - iou

                # normalize distance by allowed distance so different gaps are comparable
                norm_dist = dist / max(allowed_dist, 1e-6)

                # for larger gaps, reduce IoU importance a bit because overlap naturally drops
                gap_scaled_iou_weight = self.iou_weight / float(gap)

                cost[i, j] = (
                    norm_dist * 10.0
                    + self.area_weight * (area_ratio - 1.0)
                    + self.shape_weight * shape_penalty
                    + gap_scaled_iou_weight * iou_penalty
                )

        return cost

    def _match_subset(self, track_ids, detections, used_det_indices):
        """
        Hungarian matching on a subset of tracks against currently unused detections.
        """
        available_dets = [d for idx, d in enumerate(detections) if idx not in used_det_indices]
        available_det_indices = [idx for idx in range(len(detections)) if idx not in used_det_indices]

        if len(track_ids) == 0 or len(available_dets) == 0:
            return set(), set()

        cost = self._build_cost_matrix(track_ids, available_dets)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets_global = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e8:
                continue

            track_id = track_ids[r]
            det_idx = available_det_indices[c]
            det = detections[det_idx]

            trk = self.active_tracks[track_id]

            frame_gap = det["frame"] - trk["last_frame"]
            if frame_gap > 0:
                new_vy = (det["y"] - trk["history"][-1]["y"]) / frame_gap
                new_vx = (det["x"] - trk["history"][-1]["x"]) / frame_gap
                trk["vy"] = 0.5 * trk["vy"] + 0.5 * new_vy
                trk["vx"] = 0.5 * trk["vx"] + 0.5 * new_vx

            trk["last_frame"] = det["frame"]
            trk["lost_count"] = 0
            trk["history"].append(det.copy())

            matched_tracks.add(track_id)
            matched_dets_global.add(det_idx)

        return matched_tracks, matched_dets_global

    def update(self, detections):
        """
        Multi-frame staged association:
        1. match tracks with lost_count=0
        2. then lost_count=1
        3. then lost_count=2 ...
        """
        if len(self.active_tracks) == 0:
            for det in detections:
                self._create_track(det)
            return

        track_ids_all = list(self.active_tracks.keys())

        if len(detections) == 0:
            to_finish = []
            for track_id in track_ids_all:
                self.active_tracks[track_id]["lost_count"] += 1
                if self.active_tracks[track_id]["lost_count"] > self.max_lost:
                    to_finish.append(track_id)

            for track_id in to_finish:
                self._finish_track(track_id)
            return

        matched_tracks_total = set()
        matched_dets_total = set()

        # staged matching: fresh tracks first, older lost tracks later
        for lost_level in range(self.max_lost + 1):
            subset = [
                tid for tid in track_ids_all
                if tid not in matched_tracks_total
                and self.active_tracks[tid]["lost_count"] == lost_level
            ]

            matched_tracks, matched_dets = self._match_subset(
                subset,
                detections,
                matched_dets_total,
            )

            matched_tracks_total.update(matched_tracks)
            matched_dets_total.update(matched_dets)

        # unmatched tracks get older
        to_finish = []
        for track_id in track_ids_all:
            if track_id not in matched_tracks_total:
                self.active_tracks[track_id]["lost_count"] += 1
                if self.active_tracks[track_id]["lost_count"] > self.max_lost:
                    to_finish.append(track_id)

        for track_id in to_finish:
            self._finish_track(track_id)

        # unmatched detections create new tracks
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets_total:
                self._create_track(det)

    def finish_all(self):
        for track_id in list(self.active_tracks.keys()):
            self._finish_track(track_id)

    def get_all_tracks(self):
        all_tracks = list(self.finished_tracks.values()) + list(self.active_tracks.values())
        all_tracks = sorted(all_tracks, key=lambda x: x["track_id"])
        return all_tracks


# =====================================
# Export and plots
# =====================================

def export_tracks_to_csv(tracks, out_csv: str):
    rows = []

    for trk in tracks:
        track_id = trk["track_id"]
        history = trk["history"]

        for det in history:
            rows.append(
                {
                    "track_id": track_id,
                    "frame": det["frame"],
                    "x": det["x"],
                    "y": det["y"],
                    "area": det["area"],
                    "major_axis_length": det["major_axis_length"],
                    "minor_axis_length": det["minor_axis_length"],
                    "eccentricity": det["eccentricity"],
                    "solidity": det["solidity"],
                    "label": det["label"],
                    "bbox_min_row": det["bbox_min_row"],
                    "bbox_min_col": det["bbox_min_col"],
                    "bbox_max_row": det["bbox_max_row"],
                    "bbox_max_col": det["bbox_max_col"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["track_id", "frame"])
    df.to_csv(out_csv, index=False)
    return df


def save_track_length_histogram(tracks, out_png: str):
    lengths = [len(trk["history"]) for trk in tracks]

    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=30)
    plt.xlabel("Track length (frames)")
    plt.ylabel("Count")
    plt.title("Track Length Histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return lengths


def save_cells_per_frame_plot(counts_per_frame, out_png: str):
    frames = np.arange(len(counts_per_frame))

    plt.figure(figsize=(8, 5))
    plt.plot(frames, counts_per_frame, marker="o")
    plt.xlabel("Frame index")
    plt.ylabel("Number of instances")
    plt.title("Cells Per Frame")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def summarize_min_area_setting(min_area) -> str:
    if isinstance(min_area, dict):
        values = list(min_area.values())
        if len(values) == 0:
            return "dict(count=0)"
        return f"dict(count={len(values)}, mean={np.mean(values):.2f})"
    if isinstance(min_area, (list, tuple)):
        if len(min_area) == 0:
            return "sequence(count=0)"
        return f"sequence(count={len(min_area)}, mean={np.mean(min_area):.2f})"
    return str(min_area)


def save_summary(
    segmentation: np.ndarray,
    mask_files,
    counts_per_frame,
    tracks,
    track_lengths,
    out_txt: str,
    min_area,
    auto_min_area_fraction: float,
    max_distance: float,
    max_area_ratio: float,
    max_shape_ratio: float,
    max_lost: int,
    area_weight: float,
    shape_weight: float,
    min_track_length: int,
    max_angle_diff_deg: float,
    max_close_cost: float,
):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Simple Hungarian Cell Tracker V2 Summary\n")
        f.write("=======================================\n")
        f.write(f"num_frames: {segmentation.shape[0]}\n")
        f.write(f"image_height: {segmentation.shape[1]}\n")
        f.write(f"image_width: {segmentation.shape[2]}\n")
        f.write(f"num_mask_files: {len(mask_files)}\n")
        f.write("\n")
        f.write(f"min_area: {summarize_min_area_setting(min_area)}\n")
        f.write(f"auto_min_area_fraction: {auto_min_area_fraction}\n")
        f.write(f"max_distance: {max_distance}\n")
        f.write(f"max_area_ratio: {max_area_ratio}\n")
        f.write(f"max_shape_ratio: {max_shape_ratio}\n")
        f.write(f"max_lost: {max_lost}\n")
        f.write(f"area_weight: {area_weight}\n")
        f.write(f"shape_weight: {shape_weight}\n")
        f.write(f"min_track_length: {min_track_length}\n")
        f.write(f"max_angle_diff_deg: {max_angle_diff_deg}\n")
        f.write(f"max_close_cost: {max_close_cost}\n")
        f.write("\n")
        f.write(f"mean_cells_per_frame: {np.mean(counts_per_frame):.2f}\n")
        f.write(f"min_cells_per_frame: {np.min(counts_per_frame)}\n")
        f.write(f"max_cells_per_frame: {np.max(counts_per_frame)}\n")
        f.write("\n")
        f.write(f"num_tracks_after_filtering: {len(tracks)}\n")
        if len(track_lengths) > 0:
            f.write(f"mean_track_length: {np.mean(track_lengths):.2f}\n")
            f.write(f"median_track_length: {np.median(track_lengths):.2f}\n")
            f.write(f"min_track_length_observed: {np.min(track_lengths)}\n")
            f.write(f"max_track_length_observed: {np.max(track_lengths)}\n")


# =====================================
# Main pipeline
# =====================================

def run_tracking_pipeline(
    mask_dir: str,
    output_dir: str,
    min_area=500,
    auto_min_area_fraction: float = 0.25,
    max_distance: float = 45.0,
    max_area_ratio: float = 1.5,
    max_shape_ratio: float = 1.5,
    max_lost: int = 3,
    area_weight: float = 5.0,
    shape_weight: float = 3.0,
    min_track_length: int = 5,
    max_angle_diff_deg: float = 90.0,
    max_close_cost: float = 12.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading masks...")
    segmentation, mask_files, counts_per_frame = load_mask_sequence(
        mask_dir=mask_dir,
        min_area=min_area,
        auto_min_area_fraction=auto_min_area_fraction,
    )

    print(f"[INFO] Segmentation shape: {segmentation.shape}")
    print(f"[INFO] Number of mask files: {len(mask_files)}")
    print(f"[INFO] Mean cells/frame: {np.mean(counts_per_frame):.2f}")

    tracker = SimpleCellTrackerV3(
        max_distance=max_distance,
        max_area_ratio=max_area_ratio,
        max_shape_ratio=max_shape_ratio,
        max_lost=max_lost,
        area_weight=area_weight,
        shape_weight=shape_weight,
        iou_weight=2.0,
        n_history=3,
        gap_growth=0.35,
    )

    print("[INFO] Tracking...")
    for t in range(segmentation.shape[0]):
        mask = segmentation[t]
        detections = extract_detections_from_mask(mask, frame_idx=t)
        tracker.update(detections)

        if t == 0 or (t + 1) % 10 == 0 or t == segmentation.shape[0] - 1:
            print(f"[INFO] Frame {t+1}/{segmentation.shape[0]} | detections={len(detections)}")

    tracker.finish_all()
    tracks_all = tracker.get_all_tracks()
    print(f"[INFO] Number of raw tracks: {len(tracks_all)}")

    tracks_closed = gap_close_tracks(
        tracks_all,
        max_gap=2,
        max_dist=30.0,
        max_area_ratio=max_area_ratio,
        max_shape_ratio=max_shape_ratio,
        max_angle_diff_deg=min(120.0, max_angle_diff_deg + 30.0),
        max_close_cost=max_close_cost,
    )
    print(f"[INFO] Number of tracks after gap closing: {len(tracks_closed)}")

    tracks = filter_tracks_by_length(tracks_closed, min_length=min_track_length)
    print(f"[INFO] Number of filtered tracks (len >= {min_track_length}): {len(tracks)}")

    tracks_csv = output_dir / "tracks.csv"
    track_hist_png = output_dir / "track_length_histogram.png"
    cells_per_frame_png = output_dir / "cells_per_frame.png"
    summary_txt = output_dir / "summary.txt"

    print("[INFO] Exporting CSV...")
    df_tracks = export_tracks_to_csv(tracks, str(tracks_csv))
    print(df_tracks.head())

    print("[INFO] Saving plots...")
    track_lengths = save_track_length_histogram(tracks, str(track_hist_png))
    save_cells_per_frame_plot(counts_per_frame, str(cells_per_frame_png))

    print("[INFO] Saving summary...")
    save_summary(
        segmentation=segmentation,
        mask_files=mask_files,
        counts_per_frame=counts_per_frame,
        tracks=tracks,
        track_lengths=track_lengths,
        out_txt=str(summary_txt),
        min_area=min_area,
        auto_min_area_fraction=auto_min_area_fraction,
        max_distance=max_distance,
        max_area_ratio=max_area_ratio,
        max_shape_ratio=max_shape_ratio,
        max_lost=max_lost,
        area_weight=area_weight,
        shape_weight=shape_weight,
        min_track_length=min_track_length,
        max_angle_diff_deg=max_angle_diff_deg,
        max_close_cost=max_close_cost,
    )

    print("[DONE] All outputs saved to:", output_dir)


if __name__ == "__main__":
    MASK_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2/masks_out"
    OUTPUT_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2/simple_tracking_results_v2"

    run_tracking_pipeline(
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        min_area=500,
        auto_min_area_fraction=0.25,
        max_distance=45.0,
        max_area_ratio=1.8,
        max_shape_ratio=1.8,
        max_lost=3,
        area_weight=5.0,
        shape_weight=3.0,
        min_track_length=5,
        max_close_cost=12.0,
    )
