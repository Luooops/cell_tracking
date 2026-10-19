"""
gt_refine.py
============
Find places where the annotation may have given one cell a new identity, record a human decision for each, and
write a refined copy of the dataset in which the confirmed ones are joined.

Candidates come from a fixed geometric rule, never from a model (a model-chosen set would only ever correct the
model's "errors" and bias every later score in its favour):

    a known GT track A is last seen in frame t (not the last frame of the video, no children), and a known GT track
    B (no parent) is first seen in frame t + g, g in {1, 2}, closer than the tracker's gate r(t) x sqrt(g)
    (models.gnn_tracking.gating: estimated from detection positions alone)

Candidates are grouped by A: one review item = one track end with every start that could continue it.

A confirmed decision "A and B are the same cell" is written the CTC way, by man_track.txt only: B's parent becomes A
(a single child = an interrupted track, utils/lineage.py). One decision covers all of B, and chains (A -> B -> C)
follow from separate decisions. Masks and images are not changed; the refined dataset links (or copies) them.

    candidates = handover_candidates(frames, objects, table)
    table2, merges = apply_decisions(table, candidates, decisions)
    build_refined_dataset(source_root, target_root, new_tables)
"""
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

from models.gnn_tracking.gating import gating_radius

UNKNOWN_OFFSET = 50000
MAX_GAP = 2
SAME, NONE, UNSURE = 'same', 'none', 'unsure'


def handover_candidates(well, frames, objects, table):
    """frames: sorted frame numbers; objects[f] = (labels [n], positions [n, 2] as (x, y)); table: {label: (begin,
    end, parent)}. -> list of review items dict(id, well, a_label, a_frame, a_xy, radius, starts=[dict(b_label,
    b_frame, g, distance, b_xy)]) sorted by A, starts by distance."""
    pos = {f: objects[f][1] for f in frames}
    first, last = {}, {}
    for f in frames:
        for label, xy in zip(*objects[f]):
            label = int(label)
            if 0 < label < UNKNOWN_OFFSET:
                first.setdefault(label, (f, xy))
                last[label] = (f, xy)
    has_child = {p for _, _, p in table.values() if p}
    items = []
    for a, (ta, xa) in sorted(last.items()):
        if ta == frames[-1] or a in has_child:
            continue
        r = gating_radius(frames, pos, ta)[0]
        starts = []
        for b, (tb, xb) in first.items():
            g = tb - ta
            parent = table.get(b, (0, 0, 0))[2]
            d = float(np.hypot(*(np.asarray(xb) - np.asarray(xa))))
            if b != a and 1 <= g <= MAX_GAP and not parent and d < r * np.sqrt(g):
                starts.append(dict(b_label=b, b_frame=tb, g=g, distance=round(d, 1),
                                   b_xy=[round(float(v), 1) for v in xb]))
        if starts:
            items.append(dict(id=f'{well}:{a}', well=well, a_label=a, a_frame=ta,
                              a_xy=[round(float(v), 1) for v in xa], radius=round(r, 1),
                              starts=sorted(starts, key=lambda s: s['distance'])))
    return items


def check_decisions(candidates, decisions):
    """decisions: {item id: dict(choice=<b_label int> | 'none' | 'unsure', note=...)}.
    -> dict(merges [(well, a, b)], counts, undecided [ids], problems [str]); problems = unknown ids, invalid
    choices, or one start claimed by two ends."""
    by_id = {c['id']: c for c in candidates}
    merges, problems, counts = [], [], dict(same=0, none=0, unsure=0)
    claimed = {}
    for item_id, decision in decisions.items():
        if item_id not in by_id:
            problems.append(f'{item_id}: not a candidate of this review')
            continue
        choice = decision.get('choice')
        item = by_id[item_id]
        if choice in (NONE, UNSURE):
            counts[choice] += 1
            continue
        if choice is None:
            continue
        valid = {s['b_label'] for s in item['starts']}
        if not isinstance(choice, int) or choice not in valid:
            problems.append(f'{item_id}: choice {choice!r} is not one of its starts {sorted(valid)}')
            continue
        counts[SAME] += 1
        key = (item['well'], choice)
        if key in claimed:
            problems.append(f'{item["well"]}: track {choice} is claimed by ends {claimed[key]} and {item["a_label"]}')
        claimed[key] = item['a_label']
        merges.append((item['well'], item['a_label'], choice))
    undecided = [c['id'] for c in candidates if decisions.get(c['id'], {}).get('choice') is None]
    return dict(merges=sorted(merges), counts=counts, undecided=undecided, problems=problems)


def apply_merges(table, merges):
    """table: {label: (begin, end, parent)}; merges: [(a, b)] -> new table with parent(b) = a."""
    out = dict(table)
    for a, b in merges:
        begin, end, parent = out[b]
        if parent or out[a][1] >= begin:
            raise ValueError(f'cannot make {b} a continuation of {a}: parent {parent}, {a} ends {out[a][1]}, '
                             f'{b} begins {begin}')
        out[b] = (begin, end, a)
    return out


def write_track_table(table, path):
    Path(path).write_text(''.join(f'{l} {b} {e} {p}\n' for l, (b, e, p) in sorted(table.items())), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def build_refined_dataset(source_root, target_root, sequences, new_tables):
    """Every sequence of the source is reproduced under the new target_root: images and masks hard-linked (copied
    where links are impossible), man_track.txt rewritten for the wells in new_tables and linked otherwise.
    -> {'hardlink': n, 'copy': n} file counts."""
    source_root, target_root = Path(source_root), Path(target_root)
    target_root.mkdir(parents=True)
    modes = dict(hardlink=0, copy=0)
    for seq in sequences:
        for sub in (Path(seq), Path(f'{seq}_GT') / 'TRA'):
            (target_root / sub).mkdir(parents=True)
            for f in sorted((source_root / sub).iterdir()):
                if f.is_file() and not (sub.parts[-1] == 'TRA' and f.name == 'man_track.txt' and seq in new_tables):
                    modes[_link_or_copy(f, target_root / sub / f.name)] += 1
        if seq in new_tables:
            write_track_table(new_tables[seq], target_root / f'{seq}_GT' / 'TRA' / 'man_track.txt')
    return modes
