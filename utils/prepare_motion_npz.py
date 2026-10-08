"""Convert lab CTC masks/images to MotionAssociation windows; no model changes.

Run from the project root: python -m utils.prepare_motion_npz --output-root outputs/motion_tracking/npz_v001
Default split: 8 development wells for training, fold 0 (4 wells) for validation.
dataset_id denotes a WELL here, explicitly evaluating within-experiment generalization.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from skimage.measure import regionprops
import tifffile

from utils.ctc import sequence

FEATURE_NAMES = ['log1p_area', 'log1p_major_axis', 'log1p_minor_axis',
                 'eccentricity', 'solidity', 'mean_intensity', 'std_intensity']
UNKNOWN_OFFSET = 50000
DEFAULT_VAL = ['r08c13', 'r11c12', 'r03c12', 'r14c13']
DEFAULT_TRAIN = ['r15c12', 'r12c13', 'r07c02', 'r09c01', 'r13c12', 'r16c13', 'r09c09', 'r14c03']
HELD_OUT = {'r16c02', 'r01c20', 'r07c07'}


def read_detections(root, well):
    """Seven fixed feature columns; GT labels are metadata, never input features."""
    frames = sequence(root, well)
    times = [t for t, _, _ in frames]
    if times != list(range(times[0], times[-1] + 1)):
        raise ValueError(f'{well}: missing image/mask frames')
    lineage = Path(root) / f'{well}_GT' / 'TRA' / 'man_track.txt'
    if not lineage.is_file():
        raise FileNotFoundError(lineage)
    if lineage.stat().st_size:
        table = np.loadtxt(lineage, dtype=np.int64, ndmin=2)
        if table.shape[1] != 4 or np.any(table[:, 3] != 0):
            raise ValueError(f'{well}: expected no-division lab track table with four columns')
    coords, features, labels = [], [], []
    for t, raw_path, mask_path in frames:
        raw, mask = tifffile.imread(raw_path), tifffile.imread(mask_path)
        if raw.ndim != 2 or mask.shape != raw.shape:
            raise ValueError(f'{well}/{t}: image and mask must be matching 2-D arrays')
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0) or not np.isfinite(raw).all():
            raise ValueError(f'{well}/{t}: invalid labels or image pixels')
        # Per-frame deterministic normalization; no validation/test statistics fitted.
        lo, hi = np.percentile(raw, [1, 99])
        intensity = np.clip((raw.astype(np.float32) - lo) / max(float(hi-lo), 1e-6), 0, 1)
        for p in regionprops(mask, intensity_image=intensity):
            values = p.image_intensity[p.image]
            coords.append([t, p.centroid[1], p.centroid[0]])
            features.append([np.log1p(p.area), np.log1p(p.major_axis_length), np.log1p(p.minor_axis_length),
                             p.eccentricity, p.solidity, values.mean(), values.std()])
            labels.append(p.label)
    return (np.asarray(coords, dtype=np.float32).reshape(-1, 3),
            np.asarray(features, dtype=np.float32).reshape(-1, 7),
            np.asarray(labels, dtype=np.int64), times)


def previous_detections(coords, labels):
    previous = np.full(len(labels), -1, dtype=np.int64)
    last = {}
    for i, (point, label) in enumerate(zip(coords, labels)):
        if 0 < label < UNKNOWN_OFFSET:
            p = last.get(int(label), -1)
            if p >= 0 and coords[p, 0] >= point[0]:
                raise ValueError('Known identities must occur at most once per frame and be time ordered')
            previous[i] = p
            last[int(label)] = i
    return previous


def window_parents(coords, labels, previous, selected, start_frame, max_gap, distance):
    """Conservative observed-predecessor labels; never convert excluded positives to null."""
    parents = np.full(len(selected), -2, dtype=np.int64)
    local_of = {int(global_id): local for local, global_id in enumerate(selected)}
    unknown = np.flatnonzero(labels >= UNKNOWN_OFFSET)
    counts = Counter()
    for j, index in enumerate(selected):
        if labels[index] >= UNKNOWN_OFFSET:
            counts['unknown_identity'] += 1
            continue
        t = coords[index, 0]
        prior = previous[index]
        dt = t - coords[prior, 0] if prior >= 0 else float('inf')
        if prior >= 0 and dt <= max_gap:
            if prior not in local_of:
                counts['predecessor_outside_window'] += 1
                continue
            if np.linalg.norm(coords[index, 1:] - coords[prior, 1:]) > distance * dt:
                counts['positive_outside_distance_gate'] += 1
                continue
        elif t - max_gap < start_frame:
            counts['incomplete_past_context'] += 1
            continue
        # A nearer unknown detection could be the true immediate predecessor.
        unknown_dt = t - coords[unknown, 0]
        possible = (unknown_dt > 0) & (unknown_dt <= max_gap)
        if prior >= 0 and dt <= max_gap:
            possible &= unknown_dt < dt
        candidates = unknown[possible]
        if len(candidates) and np.any(np.linalg.norm(coords[candidates, 1:] - coords[index, 1:], axis=1)
                                      <= distance * (t - coords[candidates, 0])):
            counts['ambiguous_unknown_predecessor'] += 1
            continue
        if prior >= 0 and dt <= max_gap:
            parents[j] = local_of[int(prior)]
            counts['positive'] += 1
        else:
            # Full candidate time horizon present; no known or unknown candidate predecessor.
            parents[j] = -1
            counts['observed_null'] += 1
    return parents, counts


def windows(coords, times, window_frames, stride, max_tokens):
    """Keep all detections in selected whole frames; shorten oversized windows explicitly."""
    start = 0
    while start < len(times) - 1:
        end = min(start + window_frames, len(times))
        while True:
            selected = np.flatnonzero((coords[:, 0] >= times[start]) & (coords[:, 0] <= times[end-1]))
            if len(selected) <= max_tokens:
                break
            if end - start <= 2:
                raise ValueError(f'Frames {times[start]}..{times[end-1]} contain {len(selected)} detections > {max_tokens}; '
                                 'increase --max-tokens with matching training setting, or prepare audited spatial crops')
            end -= 1
        yield times[start], times[end-1], selected
        if end == len(times):
            break
        start += min(stride, end-start-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data/data')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dataset-name', default='lab_cells', help='Stable source name; IDs are source/well (well-level split)')
    parser.add_argument('--train-wells', nargs='+', default=DEFAULT_TRAIN)
    parser.add_argument('--val-wells', nargs='+', default=DEFAULT_VAL)
    parser.add_argument('--window-frames', type=int, default=6)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-gap', type=int, default=3)
    parser.add_argument('--distance', type=float, default=60)
    args = parser.parse_args(argv)
    if (args.window_frames < 2 or args.stride < 1 or args.max_tokens < 2 or args.max_gap < 1
            or not np.isfinite(args.distance) or args.distance <= 0 or not args.dataset_name.strip()):
        parser.error('Invalid window/gate/dataset parameters')
    train, val = set(args.train_wells), set(args.val_wells)
    if len(train) != len(args.train_wells) or len(val) != len(args.val_wells) or train & val:
        parser.error('Wells must be unique and train/validation must be disjoint')
    if (train | val) & HELD_OUT:
        parser.error('Original held-out test wells cannot be used to prepare training/validation')
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root == args.data_root or args.data_root in args.output_root.parents:
        parser.error('Output must be outside the source dataset')
    if args.output_root.exists():
        parser.error('Output already exists; select a new directory')
    for well in args.train_wells + args.val_wells:
        sequence(args.data_root, well)
    report = dict(status='running', parameters=vars(args), grouping_unit='well',
                  evaluation_scope='within-experiment held-out wells, not independent experiments',
                  feature_names=FEATURE_NAMES, intensity_normalization='per-frame 1st/99th percentile, clip [0,1]',
                  supervision='cleaned GT identities; ambiguous/truncated/out-of-gate positives ignored', wells={}, windows=[])
    args.output_root.mkdir(parents=True)
    try:
        for split, wells in [('train', args.train_wells), ('val', args.val_wells)]:
            for well in wells:
                coords, features, labels, times = read_detections(args.data_root, well)
                if not np.isfinite(features).all():
                    raise ValueError(f'{well}: nonfinite features')
                previous = previous_detections(coords, labels)
                counts = Counter()
                for start, end, selected in windows(coords, times, args.window_frames, args.stride, args.max_tokens):
                    parents, reasons = window_parents(coords, labels, previous, selected, start, args.max_gap, args.distance)
                    entry = dict(split=split, well=well, start=start, end=end, tokens=len(selected), counts=dict(reasons))
                    counts.update(reasons)
                    if not np.any(parents != -2):
                        entry['status'] = 'skipped_no_supervision'
                    else:
                        path = args.output_root / split / well / f't{start:03d}_t{end:03d}.npz'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, coords=coords[selected], features=features[selected], parents=parents,
                            dataset_id=np.array(f'{args.dataset_name}/{well}'), gt_labels=labels[selected],
                            source_indices=selected, feature_names=np.asarray(FEATURE_NAMES),
                            max_gap=np.int64(args.max_gap), distance=np.float32(args.distance))
                        entry.update(status='saved', path=str(path.relative_to(args.output_root)))
                        counts['saved_windows'] += 1
                    report['windows'].append(entry)
                if not counts['saved_windows']:
                    raise ValueError(f'{well}: no supervised windows produced')
                report['wells'][well] = dict(split=split, detections=len(coords), **counts)
                print(f'{split}/{well}: {dict(counts)}', flush=True)
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output_root / 'manifest.json').write_text(json.dumps(report, default=str, indent=2), encoding='utf-8')
    print(f'Saved {args.output_root}; use matching training max_gap/distance/max_tokens.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
