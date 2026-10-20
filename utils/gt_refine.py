"""
gt_refine.py
============
Find places where the annotation may have broken one cell's identity, record a human decision for each, and write a
refined copy of the dataset with the confirmed corrections.

Candidates come from fixed geometric rules, never from a model (a model-chosen set would only ever correct the
model's "errors" and bias every later score in its favour). Distances are judged against the tracker's gate
r(t) x sqrt(g) (models.gnn_tracking.gating: estimated from detection positions alone).

end items   a known GT track A is last seen in frame t (not the last frame of the video, no children), and a known
            GT track B (no parent) is first seen in frame t + g, g in {1, 2}, inside the gate. Listed per A with every
            such new start, plus the EXISTING tracks inside the same gate in frame t + 1 / t + 2 (neighbours).
            Every end item also carries a continuation chain: from A's last position, frame by frame, the nearest
            detection within half the gate if it has no identity (label >= 50000), allowing 1-2 missed frames;
            the chain stops at a known track (recorded when that track starts there) or when nothing is near.
            Ends with such a chain but no new start are end items too.
gap items   a known GT track X lies entirely in a gap of a known track Y: Y is absent over X's whole span and seen
            within 2 frames before and after it, both junctions inside the gate. Listed per X with every such Y.

Decisions:

    end item  <start label>  A and start B are one cell (hand-over)  -> B's parent = A        (man_track.txt only)
              swap           from frame s, A's cell carries existing track X's identity; optionally X's own cell
                             continues as start R      -> X's masks from s get a new label X' with parent A;
                                                          X ends at s - 1; R's parent = X
              chain          A continues as the unlabelled detections of its chain up to a frame -> those masks
                             get label A; optionally the track the chain meets becomes A's continuation (parent A)
    gap item  <Y label>      X is Y                     -> X's masks get label Y (Y's gap is filled; X disappears)
    any item  none / unsure                             -> nothing changes
    manual    (well, X | 'X<n>', from, to, Y | 'X<n>' | 'new')
                                                        -> X's masks in frames from..to get label Y (absent there)
                                                           or a new label; for errors no rule lists. 'X1', 'X2', ...
                                                           name new identities per well (one integer label each,
                                                           reusable across edits); 'new' is an anonymous one

Parent links are CTC single-child continuations (utils/lineage.py). Only masks touched by a relabel are rewritten;
every other file of the refined dataset is a hard link (or copy) of the source. refine_log.json lists the relabels
in the order they are applied, so outputs made on the source data can be translated (relabel_lookup / translate).
"""
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

from models.gnn_tracking.gating import gating_radius

UNKNOWN_OFFSET = 50000
MAX_GAP = 2
MAX_NEIGHBOURS = 6
SAME, SWAP, CHAIN, GAP, MANUAL, NONE, UNSURE = 'same', 'swap', 'chain', 'gap', 'manual', 'none', 'unsure'
CHAIN_REACH = 0.5          # a chain step goes to the nearest detection within this fraction of the gate


def _round(xy):
    return [round(float(v), 1) for v in xy]


def _known(label):
    return 0 < label < UNKNOWN_OFFSET


def continuation_chain(frames, objects, radius, t, xy, first):
    """Unlabelled detections that continue a track ending at frame t at position xy: frame by frame the nearest
    detection within CHAIN_REACH x the gate; unknown identities extend the chain, a known track stops it.
    -> (chain [dict(label, frame, distance)], meets: dict(label, frame) if it stopped at a track starting there)."""
    chain, meets, here, last = [], None, np.asarray(xy, dtype=float), t
    for f in range(t + 1, frames[-1] + 1):
        g = f - last
        if g > MAX_GAP or f not in objects:
            break
        labels, pos = objects[f]
        if not len(labels):
            continue
        d = np.hypot(*(pos - here).T)
        k = int(np.argmin(d))
        if d[k] >= CHAIN_REACH * radius[last] * np.sqrt(g):
            continue
        label = int(labels[k])
        if _known(label):
            if chain and first.get(label, (None,))[0] == f:
                meets = dict(label=label, frame=f)
            break
        chain.append(dict(label=label, frame=f, distance=round(float(d[k]), 1)))
        here, last = pos[k], f
    return chain, meets


def handover_candidates(well, frames, objects, table):
    """frames: sorted frame numbers; objects[f] = (labels [n], positions [n, 2] as (x, y)); table: {label: (begin,
    end, parent)}. -> list of end items dict(id, kind='end', well, a_label, a_first, a_frame, a_xy, radius,
    starts=[dict(b_label, b_frame, g, distance, b_xy)], neighbours=[dict(label, frame, g, distance)],
    chain=[dict(label, frame, distance)], chain_meets=dict(label, frame) | None)."""
    pos = {f: objects[f][1] for f in frames}
    radius = {f: gating_radius(frames, pos, f)[0] for f in frames[:-1]}
    first, last = {}, {}
    for f in frames:
        for label, xy in zip(*objects[f]):
            label = int(label)
            if _known(label):
                first.setdefault(label, (f, xy))
                last[label] = (f, xy)
    has_child = {p for _, _, p in table.values() if p}
    items = []
    for a, (ta, xa) in sorted(last.items()):
        if ta == frames[-1] or a in has_child:
            continue
        r = radius[ta]
        starts, neighbours = [], []
        for b, (tb, xb) in first.items():
            g = tb - ta
            d = float(np.hypot(*(np.asarray(xb) - np.asarray(xa))))
            if b != a and 1 <= g <= MAX_GAP and not table.get(b, (0, 0, 0))[2] and d < r * np.sqrt(g):
                starts.append(dict(b_label=b, b_frame=tb, g=g, distance=round(d, 1), b_xy=_round(xb)))
        chain, meets = continuation_chain(frames, objects, radius, ta, xa, first)
        if not starts and not chain:
            continue
        seen = {s['b_label'] for s in starts}            # tracks that begin after A ended are starts, not swap targets
        seen |= {b for b, (tb, _) in first.items() if tb > ta}
        for g in range(1, MAX_GAP + 1):
            s = ta + g
            if s not in objects:
                continue
            for label, xy in zip(*objects[s]):
                label = int(label)
                d = float(np.hypot(*(np.asarray(xy) - np.asarray(xa))))
                if _known(label) and label not in seen and first[label][0] < s and d < r * np.sqrt(g):
                    seen.add(label)
                    neighbours.append(dict(label=label, frame=s, g=g, distance=round(d, 1)))
        items.append(dict(id=f'{well}:{a}', kind='end', well=well, a_label=a, a_first=first[a][0], a_frame=ta,
                          a_xy=_round(xa), radius=round(r, 1), starts=sorted(starts, key=lambda s: s['distance']),
                          neighbours=sorted(neighbours, key=lambda n: n['distance'])[:MAX_NEIGHBOURS],
                          chain=chain, chain_meets=meets))
    return items


def gap_candidates(well, frames, objects, table):
    """-> list of gap items dict(id, kind='gap', well, x_label, x_frames, x_xy, radius, targets=[dict(label,
    before_frame, after_frame, d_before, d_after)]) for tracks X lying in the gap of a nearby track Y."""
    pos = {f: objects[f][1] for f in frames}
    at = {}
    for f in frames:
        for label, xy in zip(*objects[f]):
            if _known(int(label)):
                at.setdefault(int(label), {})[f] = xy
    radius = {f: gating_radius(frames, pos, f)[0] for f in frames[:-1]}
    linked = {p for _, _, p in table.values() if p} | {l for l, (_, _, p) in table.items() if p}
    items = []
    for x, fx in sorted(at.items()):
        if x in linked:
            continue
        span = sorted(fx)
        a, b = span[0], span[-1]
        targets = []
        for y, fy in at.items():
            if y == x or any(f in fy for f in range(a, b + 1)):
                continue
            before, after = [f for f in fy if f < a], [f for f in fy if f > b]
            if not before or not after or a - max(before) > MAX_GAP or min(after) - b > MAX_GAP:
                continue
            p, q = max(before), min(after)
            d1 = float(np.hypot(*(fx[a] - fy[p])))
            d2 = float(np.hypot(*(fy[q] - fx[b])))
            if d1 < radius[p] * np.sqrt(a - p) and d2 < radius[b] * np.sqrt(q - b):
                targets.append(dict(label=y, before_frame=p, after_frame=q, d_before=round(d1, 1), d_after=round(d2, 1)))
        if targets:
            items.append(dict(id=f'{well}:gap:{x}', kind='gap', well=well, x_label=x, x_frames=span,
                              x_xy=_round(fx[a]), radius=round(radius[max(a - 1, frames[0])], 1),
                              targets=sorted(targets, key=lambda t: t['d_before'] + t['d_after'])))
    return items


def check_decisions(candidates, decisions, manual=()):
    """decisions: {item id: dict(choice=..., swap_track, swap_frame, swap_rest, note)}; manual: [dict(well, label,
    from_frame, to_frame, target=<label> | 'new', note)]. -> dict(merges [(well, a, a_frame, b, b_frame)],
    swaps [(well, a, a_frame, x, s, r, r_frame)], moves [(well, x, from, to, target, source)], counts,
    undecided [ids], problems [str])."""
    by_id = {c['id']: c for c in candidates}
    merges, swaps, moves, problems = [], [], [], []
    counts = {SAME: 0, SWAP: 0, CHAIN: 0, GAP: 0, MANUAL: 0, NONE: 0, UNSURE: 0}
    claimed_start, claimed_track = {}, {}

    def claim(store, well, label, owner, what):
        if (well, label) in store:
            problems.append(f'{well}: {what} {label} is claimed by {store[(well, label)]} and {owner}')
        store[(well, label)] = owner

    for item_id, decision in decisions.items():
        if item_id not in by_id:
            problems.append(f'{item_id}: not a candidate of this review')
            continue
        item, choice = by_id[item_id], decision.get('choice')
        if choice is None:
            continue
        if choice in (NONE, UNSURE):
            counts[choice] += 1
            continue
        if item.get('kind') == 'gap':
            frames = item['x_frames']
            if choice not in {t['label'] for t in item['targets']}:
                problems.append(f'{item_id}: choice {choice!r} is not one of its tracks')
                continue
            counts[GAP] += 1
            claim(claimed_track, item['well'], item['x_label'], item_id, 'track')
            moves.append((item['well'], item['x_label'], frames[0], frames[-1], choice, item_id))
            continue
        starts = {s['b_label']: s['b_frame'] for s in item['starts']}
        if choice == CHAIN:
            chain, until = item.get('chain') or [], decision.get('chain_until')
            if not chain or until not in [c['frame'] for c in chain]:
                problems.append(f'{item_id}: chain end frame {until!r} is not a frame of its chain')
                continue
            taken = [c for c in chain if c['frame'] <= until]
            counts[CHAIN] += 1
            for c in taken:
                claim(claimed_track, item['well'], c['label'], item_id, 'unlabelled detection')
                moves.append((item['well'], c['label'], c['frame'], c['frame'], item['a_label'], item_id))
            # chain_link: a later start (or the track the chain meets) that continues A after the chain;
            # True = the met track (older exports)
            meets, link = item.get('chain_meets'), decision.get('chain_link')
            if link is True:
                link = meets['label'] if meets and until == chain[-1]['frame'] else 'invalid'
            if link not in (None, False):
                later = {b: f for b, f in starts.items() if f > until}
                if meets and until == chain[-1]['frame']:
                    later[meets['label']] = meets['frame']
                if link not in later:
                    problems.append(f'{item_id}: after the chain (to frame {until}) A can only continue as a start '
                                    f'that begins later: {sorted(later)}')
                    continue
                claim(claimed_start, item['well'], link, item_id, 'start')
                merges.append((item['well'], item['a_label'], item['a_frame'], link, later[link]))
        elif choice == SWAP:
            x, s, rest = decision.get('swap_track'), decision.get('swap_frame'), decision.get('swap_rest')
            if not any(n['label'] == x and n['frame'] == s for n in item.get('neighbours', [])):
                problems.append(f'{item_id}: swap track {x!r} at frame {s!r} is not one of its neighbours')
                continue
            if rest is not None and rest not in starts:
                problems.append(f'{item_id}: swap continuation {rest!r} is not one of its starts {sorted(starts)}')
                continue
            counts[SWAP] += 1
            claim(claimed_track, item['well'], x, item_id, 'existing track')
            if rest is not None:
                claim(claimed_start, item['well'], rest, item_id, 'start')
            swaps.append((item['well'], item['a_label'], item['a_frame'], x, s, rest,
                          starts.get(rest) if rest is not None else None))
        elif not isinstance(choice, int) or choice not in starts:
            problems.append(f'{item_id}: choice {choice!r} is not one of its starts {sorted(starts)}')
        else:
            counts[SAME] += 1
            claim(claimed_start, item['well'], choice, item_id, 'start')
            merges.append((item['well'], item['a_label'], item['a_frame'], choice, starts[choice]))
    for k, edit in enumerate(manual):
        well, label, f1, f2, target = (edit.get(k2) for k2 in ('well', 'label', 'from_frame', 'to_frame', 'target'))
        if not (isinstance(well, str) and (isinstance(label, int) or is_new_id(label)) and isinstance(f1, int)
                and isinstance(f2, int)
                and f1 <= f2 and (target == 'new' or isinstance(target, int) or is_new_id(target))
                and target != label):
            problems.append(f'manual edit {k + 1}: invalid {edit}')
            continue
        counts[MANUAL] += 1
        moves.append((well, label, f1, f2, target, f'manual {k + 1}'))
    undecided = [c['id'] for c in candidates if decisions.get(c['id'], {}).get('choice') is None]
    return dict(merges=sorted(merges), swaps=sorted(swaps, key=lambda s: s[:5]), moves=moves, counts=counts,
                undecided=undecided, problems=problems)


def is_new_id(target):
    """'X1', 'X2', ...: a named new identity of a manual edit."""
    return isinstance(target, str) and re.fullmatch(r'X[1-9][0-9]*', target) is not None


def refine_well(table, observed, merges, swaps, moves=(), new_ids=None):
    """Apply one well's decisions. table: {label: (begin, end, parent)}; observed: {label: sorted frames present in
    the masks}; merges: [(a, a_frame, b, b_frame)]; swaps: [(a, a_frame, x, s, r, r_frame)]; moves: [(x, from, to,
    target | 'new', source)]. Moves (gap items, manual edits) are applied first, then swaps, then hand-overs.
    -> (new table, relabels [(x, from, to | None, new label)] in order). new_ids, if a dict, receives the integer
    label given to every named new identity ('X1' -> 431). Raises ValueError on any inconsistency."""
    out = {l: tuple(v) for l, v in table.items()}
    observed = {l: sorted(fs) for l, fs in observed.items()}
    relabels, split = [], {}                          # split[x] = (s, x')
    named = {}                                        # 'X1' -> integer label
    next_label = max([l for l in table if l < UNKNOWN_OFFSET] + [0]) + 1
    # an edit may take frames from a hand-over track only where the hand-over does not look: before A's end frame
    # and after the start's first frame; swap tracks are not edited at all
    free = {}                                         # label -> (first, last) frames an edit may take, or None
    for a, a_frame, b, b_frame in merges:
        for l, lo, hi in ((a, -math.inf, a_frame - 1), (b, b_frame + 1, math.inf)):
            if l in free and free[l] is None:
                continue
            old = free.get(l, (-math.inf, math.inf))
            free[l] = (max(old[0], lo), min(old[1], hi))
    for l in {l for s in swaps for l in (s[0], s[2], s[4]) if l is not None}:
        free[l] = None

    for x, f1, f2, target, source in moves:
        if is_new_id(x):                              # a named new identity as source (e.g. a temporary X1 in a swap)
            if x not in named:
                raise ValueError(f'{source}: new id {x} is used before an edit creates it')
            x = named[x]
        if x in free and (free[x] is None or f1 < free[x][0] or f2 > free[x][1]):
            raise ValueError(f'{source}: track {x} is also part of a swap, or the edit takes the end / start frame '
                             f'of its hand-over; decide it in one place')
        moving = [f for f in observed.get(x, []) if f1 <= f <= f2]
        if not moving:
            raise ValueError(f'{source}: track {x} is not present in frames {f1}-{f2}')
        if target == 'new':
            target, next_label = next_label, next_label + 1
        elif is_new_id(target):
            if target not in named:
                named[target], next_label = next_label, next_label + 1
            target = named[target]
        elif target not in out:
            raise ValueError(f'{source}: target track {target} does not exist')
        if set(moving) & set(observed.get(target, [])):
            raise ValueError(f'{source}: track {target} is already present in some of frames {moving}')
        if any(p == x for _, _, p in out.values()):
            raise ValueError(f'{source}: track {x} has children; edit it by hand')
        observed[x] = [f for f in observed[x] if f not in moving]
        observed[target] = sorted(observed.get(target, []) + moving)
        parent_x = out.get(x, (0, 0, 0))[2]           # an unlabelled detection may have no row
        if observed[x]:
            out[x] = (observed[x][0], observed[x][-1], parent_x)
        else:
            out.pop(x, None)
        parent_t = out.get(target, (0, 0, 0))[2]
        out[target] = (observed[target][0], observed[target][-1], parent_t)
        relabels.append((x, f1, f2, target))

    def at(label, frame):
        s_new = split.get(label)
        return s_new[1] if s_new and frame >= s_new[0] else label

    for a, a_frame, x, s, rest, rest_frame in sorted(swaps, key=lambda w: w[2]):
        if x in split:
            raise ValueError(f'track {x} is split twice')
        begin, end, parent = out[x]
        before = [f for f in observed[x] if f < s]
        if not before or s not in observed[x]:
            raise ValueError(f'track {x} is not present both before and at frame {s}')
        new, next_label = next_label, next_label + 1
        out[x] = (begin, before[-1], parent)
        out[new] = (s, end, 0)
        split[x] = (s, new)
        relabels.append((x, s, None, new))
    for x, (s, new) in split.items():                 # children of x after s now hang from x'
        for label, (b, e, p) in list(out.items()):
            if p == x and b >= s:
                out[label] = (b, e, new)
    for a, a_frame, x, s, rest, rest_frame in swaps:
        out[split[x][1]] = (*out[split[x][1]][:2], at(a, a_frame))
        if rest is not None:
            r = at(rest, rest_frame)
            out[r] = (*out[r][:2], x)
    for a, a_frame, b, b_frame in merges:
        a2, b2 = at(a, a_frame), at(b, b_frame)
        if out[b2][2]:
            raise ValueError(f'track {b2} already has parent {out[b2][2]}')
        out[b2] = (*out[b2][:2], a2)
    validate_table(out)
    if new_ids is not None:
        new_ids.update(named)
    return out, relabels


def validate_table(table):
    """CTC rules for parents: the parent exists and ends before the child begins."""
    for label, (begin, end, parent) in table.items():
        if end < begin:
            raise ValueError(f'track {label}: end {end} before begin {begin}')
        if parent and (parent not in table or table[parent][1] >= begin):
            raise ValueError(f'track {label} (begins {begin}) cannot have parent {parent}'
                             f'{"" if parent not in table else f" (ends {table[parent][1]})"}')


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


def _frame_of(name):
    stem = Path(name).stem
    digits = stem[len(stem.rstrip('0123456789')):]
    return int(digits) if digits else None


def _applies(entry, frame):
    _, f1, f2, _ = entry
    return frame >= f1 and (f2 is None or frame <= f2)


def build_refined_dataset(source_root, target_root, sequences, new_tables, relabels=None):
    """Every sequence of the source is reproduced under the new target_root: images and masks hard-linked (copied
    where links are impossible); masks of frames touched by a relabel (x -> new over its frames, applied in order)
    rewritten; man_track.txt rewritten for the wells in new_tables and linked otherwise. -> file counts by mode."""
    import tifffile
    source_root, target_root, relabels = Path(source_root), Path(target_root), relabels or {}
    target_root.mkdir(parents=True)
    modes = dict(hardlink=0, copy=0, rewritten=0)
    for seq in sequences:
        for sub in (Path(seq), Path(f'{seq}_GT') / 'TRA'):
            (target_root / sub).mkdir(parents=True)
            for f in sorted((source_root / sub).iterdir()):
                if not f.is_file() or (sub.parts[-1] == 'TRA' and f.name == 'man_track.txt' and seq in new_tables):
                    continue
                frame = _frame_of(f.name)
                todo = [e for e in relabels.get(seq, [])
                        if sub.parts[-1] == 'TRA' and f.suffix == '.tif' and frame is not None and _applies(e, frame)]
                if todo:
                    mask = tifffile.imread(f)
                    out = mask.copy()
                    for x, _, _, new in todo:              # in order, so chained relabels compose
                        out[out == x] = new
                    if (out != mask).any():
                        tifffile.imwrite(target_root / sub / f.name, out, compression='zlib')
                        modes['rewritten'] += 1
                        continue
                modes[_link_or_copy(f, target_root / sub / f.name)] += 1
        if seq in new_tables:
            write_track_table(new_tables[seq], target_root / f'{seq}_GT' / 'TRA' / 'man_track.txt')
    return modes


def relabel_lookup(root):
    """{well: [(label, from_frame, to_frame | None, new_label)]} from <root>/refine_log.json, in application order
    ({} for a dataset that is not refined)."""
    path = Path(root) / 'refine_log.json'
    if not path.is_file():
        return {}
    out = {}
    for r in json.loads(path.read_text(encoding='utf-8')).get('relabels', []):
        out.setdefault(r['well'], []).append((r['label'], r['from_frame'], r.get('to_frame'), r['new_label']))
    return out


def translate(entries, frame, label):
    """Label of a source-data detection (frame, label) in the refined dataset (relabels applied in order).
    Labels that are already refined pass through unchanged."""
    for entry in entries:
        if label == entry[0] and _applies(entry, frame):
            label = entry[3]
    return label
