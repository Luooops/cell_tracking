"""
ctc_export.py
=============
Our tracking outputs and ground truth -> the Cell Tracking Challenge format that the official measures read.

CTC rules that matter here: a label must be present in EVERY frame from its begin to its end, so a track with a gap
is written as several segments, each later segment a single child of the one before (the CTC way of writing an
interrupted track); a division gives the two daughters' first segments the mother's last segment as parent.

    export_res(tracks, lineage, frames, out_dir)    predicted tracks -> <out_dir>/mask###.tif + res_track.txt
    canonical_gt(table, frames, out_dir)            a GT whose labels have gaps (scripta_data) -> a valid CTC GT
                                                    <out_dir>/TRA/man_track###.tif + man_track.txt

Detections without identity (label >= UNKNOWN_OFFSET) are removed from both by default (drop_unknown), so they are
neither a track to recover nor a false detection -- the project's rule for unknown identities. The source masks are
only read; every file is written into a new folder (zlib-compressed TIFF).
"""
from pathlib import Path

import numpy as np
import tifffile

UNKNOWN_OFFSET = 50000


def _digits(path):
    """Width of the frame number at the end of a CTC file name: man_track007.tif -> 3."""
    stem = Path(path).stem
    return len(stem) - len(stem.rstrip('0123456789'))


def _segments(frames):
    """Sorted frame numbers -> [(begin, end), ...] runs of consecutive frames."""
    frames = sorted(frames)
    runs, start = [], frames[0]
    for a, b in zip(frames[:-1], frames[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    return runs + [(start, frames[-1])]


def _write_masks(frames, relabel, out_dir, name):
    """frames: [(frame, mask path)]; relabel: {(frame, source label): new label}; other labels become 0."""
    width = _digits(frames[0][1])
    big = max(relabel.values(), default=0) > np.iinfo(np.uint16).max
    removed = 0
    for frame, path in frames:
        mask = tifffile.imread(path)
        lut = np.zeros(int(mask.max()) + 1, np.uint32 if big else np.uint16)
        for label in np.unique(mask[mask > 0]):
            new = relabel.get((frame, int(label)), 0)
            lut[label] = new
            removed += new == 0
        tifffile.imwrite(Path(out_dir) / f'{name}{frame:0{width}d}.tif', lut[mask], compression='zlib')
    return removed


def _write_table(rows, path):
    Path(path).write_text(''.join(f'{l} {b} {e} {p}\n' for l, b, e, p in rows), encoding='utf-8')


def _relabel_tracks(frames_of, parent_of, first_label=1):
    """frames_of: {track: frames}; parent_of: {track: parent track or 0}. -> (rows [(label, begin, end, parent)],
    relabel {(frame, track): label}). Tracks are split into gap-free segments; parents point at segments."""
    rows, label_at, first_row, last_row, next_label = [], {}, {}, {}, first_label
    for track in sorted(frames_of, key=lambda k: (min(frames_of[k]), k)):
        previous = None
        for begin, end in _segments(frames_of[track]):
            if previous is None:
                first_row[track] = len(rows)
            rows.append([next_label, begin, end, 0 if previous is None else rows[previous][0]])
            for f in range(begin, end + 1):
                label_at[(f, track)] = next_label
            previous, next_label = len(rows) - 1, next_label + 1
        last_row[track] = previous
    for track, row in first_row.items():             # first segment of a daughter -> the mother's last segment
        mother = parent_of.get(track, 0)
        if mother in last_row and rows[row][1] > rows[last_row[mother]][2]:
            rows[row][3] = rows[last_row[mother]][0]
    return [tuple(r) for r in rows], label_at


def export_res(tracks, lineage, frames, out_dir, drop_unknown=True):
    """tracks: rows (frame, instance label, predicted track id), e.g. from instance_tracks.csv; lineage:
    {daughter track: mother track} (lineage.json, or {} without divisions); frames: [(frame, mask path)] of the
    detection masks the tracks were made on. Writes a new out_dir; returns dict(tracks, segments, removed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    frames_of, track_of = {}, {}
    for frame, label, track in tracks:
        frame, label, track = int(frame), int(label), int(track)
        if drop_unknown and label >= UNKNOWN_OFFSET:
            continue
        frames_of.setdefault(track, []).append(frame)
        track_of[(frame, label)] = track
    rows, label_at = _relabel_tracks(frames_of, {int(d): int(m) for d, m in lineage.items()})
    relabel = {key: label_at[(key[0], track)] for key, track in track_of.items()}
    removed = _write_masks(frames, relabel, out_dir, 'mask')
    _write_table(rows, out_dir / 'res_track.txt')
    return dict(tracks=len(frames_of), segments=len(rows), removed_detections=int(removed))


def canonical_gt(table, frames, out_dir, drop_unknown=True):
    """table: {label: (begin, end, parent)} (utils.lineage.read_track_table); frames: [(frame, mask path)] of the GT
    masks. Labels split at their gaps into single-child segments, unknown identities removed, begin/end from the
    frames the label is actually present in. Writes <out_dir>/TRA/; returns dict(tracks, segments, removed)."""
    tra = Path(out_dir) / 'TRA'
    tra.mkdir(parents=True)
    frames_of = {}
    for frame, path in frames:
        for label in np.unique(tifffile.imread(path)):
            if label and not (drop_unknown and label >= UNKNOWN_OFFSET) and int(label) in table:
                frames_of.setdefault(int(label), []).append(int(frame))
    parents = {l: table[l][2] for l in frames_of}
    rows, label_at = _relabel_tracks(frames_of, parents)
    removed = _write_masks(frames, {(f, l): v for (f, l), v in label_at.items()}, tra, 'man_track')
    _write_table(rows, tra / 'man_track.txt')
    return dict(tracks=len(frames_of), segments=len(rows), removed_objects=int(removed))
