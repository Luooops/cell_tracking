"""
audit_handover.py
=================
Stage 0 addendum: human review of suspected identity hand-overs in the annotation, then a refined dataset.

prepare   candidates by the fixed, model-free rules of utils/gt_refine.py (a GT track ends, another starts within
          1-2 frames inside the tracker's gate; or a track lies entirely in a gap of a nearby track), one image strip
          per item; for a track end from A's first frame to one frame
          after its last candidate (centred on A while it exists; A red, new starts 1, 2, ... in colours, nearby
          existing tracks grey with their GT label) and a local review page. The review is blind: no model output
          is shown, and the items are in random order. --reuse-review-id keeps the id (and saved progress) of an
          earlier review of the same candidates.
apply     the exported decisions -> a new dataset <data_dir>/<dataset>_refine (utils/gt_refine.py): hand-overs as
          parent links in man_track.txt; swaps (A's cell given an existing track's identity) by relabelling that
          track's masks from the swap frame. Every other file is hard-linked (copied where links are impossible);
          the source dataset is never modified. refine_log.json (with the relabels) + the decisions file record
          every change.
refresh   rebuild review.html of an existing review from the current page template (candidates.json and the saved
          progress stay as they are); also adds the detection positions the page uses for its reminders.
add-links append the tracker's confident errors of an out-of-fold run on the refined dataset (audit_confident.py:
          round-1 links made between different GT cells with every member above 0.9, then same-cell edges left out
          with every member below 0.1) to an existing review as 'link' items -- is the GT right here? --, drawn and
          edited in the review's own labels; wrong labels are corrected with the page's ID edits. Saved progress is
          kept. apply of the whole review then writes a new refined dataset (--target data/scripta_data_refine2).

    python -m models.gnn_tracking.audit_handover prepare --dataset scripta_data \
        --output-root development/stage0_evaluation/handover_review_scripta_data
    (open <output-root>/review.html in a browser, decide, press Export -> decisions.json)
    python -m models.gnn_tracking.audit_handover apply --dataset scripta_data \
        --review development/stage0_evaluation/handover_review_scripta_data --decisions decisions.json
    python -m models.gnn_tracking.audit_handover add-links --dataset scripta_data \
        --review development/stage0_evaluation/handover_review_scripta_data_v6 --run-dataset scripta_data_refine \
        --run development/stage1_calibration/oof_cv_v003_lnk/oof/gnn_tracking
    python -m models.gnn_tracking.audit_handover apply --dataset scripta_data \
        --review development/stage0_evaluation/handover_review_scripta_data_v6 --decisions decisions.json \
        --target data/scripta_data_refine2
    python -m models.gnn_tracking.audit_handover refresh --dataset scripta_data \
        --review development/stage0_evaluation/handover_review_scripta_data
"""
import argparse
import datetime
import json
import random
import uuid
from pathlib import Path

import numpy as np
import tifffile

from models.gnn_tracking.audit_gating import development_wells
from utils.config import ctc_sequences, read_config, resolve_dataset
from utils.ctc import sequence
from utils.gt_refine import (build_refined_dataset, check_decisions, gap_candidates, handover_candidates, refine_well,
                             sha256)
from utils.lineage import read_track_table
from utils.run_log import log_path, log_root, run_logged

TEMPLATE = Path(__file__).with_name('handover_review.html')
TILE, GAP, HEADER = 200, 6, 22
A_COLOR = (229, 57, 53)
B_COLORS = [(30, 136, 229), (67, 160, 71), (251, 140, 0), (142, 36, 170), (0, 172, 193), (216, 27, 96)]
OTHER = (150, 150, 150)
NEIGHBOUR_TEXT = (90, 90, 90)
UNKNOWN_OFFSET = 50000


def load_well(root, well):
    """-> (frames, raws {f: image}, masks {f: labels}, objects {f: (labels, positions)})."""
    from skimage.measure import regionprops
    frames, raws, masks, objects = [], {}, {}, {}
    for f, raw_path, mask_path in sequence(root, well):
        frames.append(f)
        raws[f], masks[f] = tifffile.imread(raw_path), tifffile.imread(mask_path)
        props = sorted(regionprops(masks[f]), key=lambda p: p.label)
        objects[f] = (np.array([p.label for p in props], dtype=np.int64),
                      np.array([(p.centroid[1], p.centroid[0]) for p in props], dtype=float).reshape(-1, 2))
    return frames, raws, masks, objects


def observed_frames(objects):
    """{label (str): sorted frames in which it is present} -- lets the review page check edits as apply would."""
    out = {}
    for f in sorted(objects):
        for label in objects[f][0]:
            out.setdefault(str(int(label)), []).append(int(f))
    return out


def positions(objects):
    """{label (str): [[frame, x, y], ...]} centroids, rounded to 0.1 px -- distances shown with the page's reminders."""
    out = {}
    for f in sorted(objects):
        for label, (x, y) in zip(*objects[f]):
            out.setdefault(str(int(label)), []).append([int(f), round(float(x), 1), round(float(y), 1)])
    return out


def write_page(out, review, xy):
    """review.html = the page template with the review and the positions embedded."""
    embed = lambda data: json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    page = (TEMPLATE.read_text(encoding='utf-8').replace('/*__REVIEW_DATA__*/', embed(review))
            .replace('/*__XY_DATA__*/', embed(xy)))
    (out / 'review.html').write_text(page, encoding='utf-8')


def _crop(image, cx, cy, half):
    """Square crop centred at (cx, cy), zero-padded at the image border."""
    padded = np.pad(image, half, mode='constant')
    x, y = int(round(cx)) + half, int(round(cy)) + half
    return padded[y - half:y + half, x - half:x + half]


def _strip(path, show, centres, radius, raws, masks, colors, names, numbered, header, chain=(), half=None):
    """Draw one image strip: a tile per frame of `show`, cropped around centres[f]. Every cell carries a mark the edit
    row can refer to: labels in `colors` outlined in their colour with names[label] above; unlabelled detections in
    `chain` outlined thin red as 'A?'; other known tracks outlined grey with their GT id; other unlabelled detections
    outlined grey and tagged ?1, ?2, ... in order of first appearance (`numbered` is kept for callers, all known
    tracks are numbered). -> (frames shown, {label: frames visible}, {unlabelled label: tag})."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_dilation
    from skimage.segmentation import find_boundaries
    half = half or int(np.clip(radius * 1.4, 48, 240))
    crops = [_crop(raws[f].astype(np.float32), *centres[f], half) for f in show]
    lo, hi = np.percentile(np.concatenate([c.ravel() for c in crops]), [1, 99.7])
    font, small = ImageFont.load_default(size=14), ImageFont.load_default(size=11)
    canvas = Image.new('RGB', (len(show) * (TILE + GAP) - GAP, TILE + HEADER), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    visible, tags = {}, {}                            # label -> frames it appears in; unlabelled label -> '?k'
    for col, (f, crop) in enumerate(zip(show, crops)):
        gray = np.clip((crop - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        tile = np.array(Image.fromarray(gray).resize((TILE, TILE), Image.BILINEAR).convert('RGB'))
        labels = np.array(Image.fromarray(_crop(masks[f], *centres[f], half).astype(np.int32)).resize(
            (TILE, TILE), Image.NEAREST))
        edge = find_boundaries(labels, mode='inner')
        tile[edge & ~np.isin(labels, list(colors))] = OTHER
        marks = []
        for label in np.unique(labels):
            if label:
                visible.setdefault(int(label), []).append(int(f))
            label = int(label)
            if not label:
                continue
            region = labels == label
            ys, xs = np.nonzero(region)
            if label in colors:
                tile[binary_dilation(edge & region)] = colors[label]
                marks.append((names[label], colors[label], font, xs.mean() - 4, ys.min() - 17))
            elif label in chain:
                tile[edge & region] = A_COLOR
                marks.append(('A?', A_COLOR, small, xs.mean() - 8, ys.min() - 14))
            elif label < UNKNOWN_OFFSET:
                marks.append((str(label), NEIGHBOUR_TEXT, small, xs.mean() - 8, ys.mean() - 6))
            else:
                tag = tags.setdefault(label, f'?{len(tags) + 1}')
                marks.append((tag, NEIGHBOUR_TEXT, small, xs.mean() - 6, ys.mean() - 6))
        x0 = col * (TILE + GAP)
        canvas.paste(Image.fromarray(tile), (x0, HEADER))
        draw.text((x0 + 4, 3), header(f), fill=(40, 40, 40), font=font)
        for text, color, fnt, x, y in marks:
            draw.text((x0 + x, HEADER + max(y, 0)), text, fill=color, font=fnt, stroke_width=2,
                      stroke_fill=(255, 255, 255))
    canvas.save(path)
    return [int(f) for f in show], {str(l): fs for l, fs in sorted(visible.items())}, {str(l): t for l, t in tags.items()}


def _position(objects, f, label):
    labels, pos = objects[f]
    hit = np.nonzero(labels == label)[0]
    return pos[hit[0]] if len(hit) else None


def render_panel(item, frames, raws, masks, objects, path):
    """End item: a tile per frame from A's first frame to one frame after its last candidate, following A while it
    exists and staying at its last position afterwards. A red, new starts numbered, neighbours with their GT id."""
    chain, meets = item.get('chain') or [], item.get('chain_meets')
    last = max([s['b_frame'] for s in item['starts']] + [n['frame'] for n in item.get('neighbours', [])]
               + [c['frame'] for c in chain] + ([meets['frame']] if meets else []))
    show = [f for f in range(item['a_first'], last + 2) if f in raws]
    centres = {}
    for f in show:
        here = _position(objects, f, item['a_label']) if f <= item['a_frame'] else None
        centres[f] = here if here is not None else item['a_xy']
    colors = {item['a_label']: A_COLOR}
    colors.update({s['b_label']: B_COLORS[k % len(B_COLORS)] for k, s in enumerate(item['starts'])})
    names = {item['a_label']: 'A'}
    names.update({s['b_label']: str(k + 1) for k, s in enumerate(item['starts'])})
    if meets and meets['label'] not in colors:
        colors[meets['label']], names[meets['label']] = B_COLORS[len(item['starts']) % len(B_COLORS)], 'B'

    def header(f):
        rel = f - item['a_frame']
        return f'frame {f}  (t{rel:+d})' if rel else f'frame {f}  (t: A ends)'
    return _strip(path, show, centres, item['radius'], raws, masks, colors, names,
                  {n['label'] for n in item.get('neighbours', [])}, header, {c['label'] for c in chain})


def render_gap_panel(item, frames, raws, masks, objects, path):
    """Gap item: from 3 frames before the earliest candidate track's last frame before X to 3 frames after X;
    X red, the candidate tracks numbered. One fixed crop for all tiles (the box around every position of X and of
    the candidates in these frames), so movement between tiles is real movement."""
    first = min(t['before_frame'] for t in item['targets']) - 3
    last = max(t['after_frame'] for t in item['targets']) + 3
    show = [f for f in range(first, last + 1) if f in raws]
    labels = [item['x_label']] + [t['label'] for t in item['targets']]
    seen = np.array([p for f in show for l in labels if (p := _position(objects, f, l)) is not None])
    lo, hi = seen.min(0), seen.max(0)
    centre = (lo + hi) / 2
    half = int(np.clip(max(item['radius'] * 1.4, (hi - lo).max() / 2 + 60), 48, 320))
    centres = {f: centre for f in show}
    colors = {item['x_label']: A_COLOR}
    colors.update({t['label']: B_COLORS[k % len(B_COLORS)] for k, t in enumerate(item['targets'])})
    names = {item['x_label']: 'X'}
    names.update({t['label']: str(k + 1) for k, t in enumerate(item['targets'])})
    span = (item['x_frames'][0], item['x_frames'][-1])
    return _strip(path, show, centres, item['radius'], raws, masks, colors, names, set(),
                  lambda f: f'frame {f}' + ('  (X)' if span[0] <= f <= span[1] else ''), half=half)


def prepare(args, wells):
    out = args.output_root
    (out / 'panels').mkdir(parents=True)
    items = []
    observed, xy = {}, {}
    for well in wells:
        frames, raws, masks, objects = load_well(args.data_root, well)
        observed[well], xy[well] = observed_frames(objects), positions(objects)
        table = read_track_table(Path(args.data_root) / f'{well}_GT' / 'TRA' / 'man_track.txt')
        found = handover_candidates(well, frames, objects, table)
        for item in found:
            item['panel'] = f'panels/{well}_{item["a_label"]}.png'
            item['frames_shown'], item['visible'], item['tags'] = render_panel(item, frames, raws, masks, objects,
                                                                               out / item['panel'])
        gaps = gap_candidates(well, frames, objects, table)
        for item in gaps:
            item['panel'] = f'panels/{well}_gap_{item["x_label"]}.png'
            item['frames_shown'], item['visible'], item['tags'] = render_gap_panel(item, frames, raws, masks, objects,
                                                                                   out / item['panel'])
        items += found + gaps
        print(f'{well}: {len(found)} track ends ({sum(bool(i["starts"]) for i in found)} with new starts, '
              f'{sum(bool(i["chain"]) for i in found)} with an unlabelled continuation), {len(gaps)} tracks inside a gap',
              flush=True)
    random.Random(args.seed).shuffle(items)                 # review order carries no information
    review_id = uuid.uuid4().hex
    if args.reuse_review_id:
        old = json.loads(args.reuse_review_id.read_text(encoding='utf-8'))
        if not {i['id'] for i in old['items']} <= {i['id'] for i in items}:
            raise ValueError('--reuse-review-id: some items of the earlier review are no longer candidates')
        review_id = old['review_id']
    review = dict(review_id=review_id, dataset=args.dataset_name, data_root=str(Path(args.data_root).resolve()),
                  wells=wells, created=datetime.datetime.now().isoformat(timespec='seconds'), seed=args.seed,
                  rule='A known GT track ends at t (not the last frame, no children); a known GT track without parent '
                       'starts at t+1 or t+2 closer than the gating radius r(t) x sqrt(g) of '
                       'models.gnn_tracking.gating (positions only); existing tracks inside the same gate at t+1/t+2 '
                       'are offered as swap targets; ends followed by unlabelled detections carry a continuation '
                       'chain (nearest detection within half the gate, frame by frame). Gap items: a known track lying entirely in a gap of a known '
                       'track, both junctions inside the gate. No model output is used or shown.',
                  items=items, observed=observed)
    (out / 'candidates.json').write_text(json.dumps(review, indent=1), encoding='utf-8')
    (out / 'positions.json').write_text(json.dumps(xy, separators=(',', ':')), encoding='utf-8')
    write_page(out, review, xy)
    n_gap = sum(i.get('kind') == 'gap' for i in items)
    print(f'\n{len(items)} review items ({len(items) - n_gap} track ends, {n_gap} gap items); open {out / "review.html"}')
    return 0


def add_links(args, wells):
    """Append the tracker's confident errors of an out-of-fold run on the refined dataset to this review as 'link'
    items, in the review's own (source) labels: a refined detection is mapped to the source label of the same mask."""
    from models.gnn_tracking.audit_confident import CONTEXT, confident_false_links, confident_missed_links, draw
    from models.gnn_tracking.evaluate_answers import load_well as load_answers
    from utils.lineage import Lineage
    review_path = args.review / 'candidates.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    have = {i['id'] for i in review['items']}
    rng = random.Random(args.seed)
    new = {'false': [], 'missed': []}
    for well in wells:
        part = load_answers(args.run, args.run_root, well)
        table = read_track_table(Path(args.run_root) / f'{well}_GT' / 'TRA' / 'man_track.txt')
        found = ([dict(f, sub='false') for f in confident_false_links(part['edges'], part['relation'], table)] +
                 [dict(f, sub='missed') for f in confident_missed_links(part['edges'], part['relation'])])
        if not found:
            continue
        frames, raws, masks, objects = load_well(args.data_root, well)
        to_source = {}                                         # (frame, refined label) -> source label
        for f, _, mask_path in sequence(args.run_root, well):
            refined = tifffile.imread(mask_path)
            on = refined > 0
            for r, o in np.unique(np.stack([refined[on], masks[f][on]]), axis=1).T:
                to_source[(int(f), int(r))] = int(o)
        lineage = Lineage(table, ())
        cell = lambda label: lineage.cell_of.get(label, label)
        e = part['edges']
        for f in found:
            t, a, b, model_b = f['t'], f['a'], f['b'], f.get('model_b', f['b'])
            shown = range(t - CONTEXT, t + CONTEXT + 2)

            def members(label, exclude=None):
                if label is None or cell(label) == exclude:
                    return []
                return sorted({o for (fr, r), o in to_source.items() if fr in shown and cell(r) == cell(label)})
            sa, sb = to_source[(t, a)], to_source[(t + 1, b)]
            item_id = f"{well}:link:{f['sub']}:{t}:{sa}:{sb}"
            if item_id in have:
                continue
            blue = b if f['sub'] == 'false' else model_b
            sel = (e['round'] == 1) & (e['src_frame'] == t)
            item = dict(id=item_id, kind='link', sub=f['sub'], well=well, a_label=sa, a_frame=t, b_label=sb,
                        model_b=None if model_b is None else to_source[(t + 1, model_b)], q=f['q'], cls=f['cls'],
                        radius=round(float(e['gate'][sel][0]) if sel.any() else 100.0, 1), starts=[], neighbours=[],
                        chain=[], run_labels=dict(a=a, b=b, model_b=model_b))
            item['panel'] = f"panels/{well}_link_{f['sub']}_t{t}_{sa}_{sb}.png"
            item['frames_shown'], item['visible'], item['tags'] = draw(
                t, members(a), members(blue, exclude=cell(a)), raws, masks, objects, item['radius'],
                args.review / item['panel'])
            item['a_first'] = item['frames_shown'][0]
            new[f['sub']].append(item)
            have.add(item_id)
        print(f"{well}: {sum(f['sub'] == 'false' for f in found)} confident false links, "
              f"{sum(f['sub'] == 'missed' for f in found)} confident missed links", flush=True)
    for part in new.values():
        rng.shuffle(part)
    added = new['false'] + new['missed']                     # false links first, each part in random order
    backup = args.review / 'candidates_before_links.json'
    if not backup.exists():
        backup.write_text(review_path.read_text(encoding='utf-8'), encoding='utf-8')
    review['items'] += added
    review.setdefault('link_runs', []).append(dict(
        run=str(Path(args.run).resolve()), dataset=args.run_dataset, added=len(added),
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        rule='link items: round-1 links the tracker chose between different GT cells with every ensemble member above '
             '0.9, and round-1 same-cell edges it left out with every member below 0.1, in an out-of-fold run on the '
             'refined dataset; shown and edited in this review\'s labels'))
    review_path.write_text(json.dumps(review, indent=1), encoding='utf-8')
    xy = json.loads((args.review / 'positions.json').read_text(encoding='utf-8'))
    write_page(args.review, review, xy)
    print(f"\nadded {len(added)} link items ({len(new['false'])} false, {len(new['missed'])} missed); the review now "
          f"has {len(review['items'])} items; open {args.review / 'review.html'} (saved progress is kept)")
    return 0


def apply(args):
    review = json.loads((args.review / 'candidates.json').read_text(encoding='utf-8'))
    exported = json.loads(args.decisions.read_text(encoding='utf-8'))
    if exported.get('review_id') != review['review_id']:
        raise ValueError('decisions were exported from a different review (review_id mismatch)')
    if review['dataset'] != args.dataset_name:
        raise ValueError(f"review was made on {review['dataset']}, not {args.dataset_name}")
    result = check_decisions(review['items'], exported['decisions'], exported.get('manual', []))
    sequences = ctc_sequences(args.data_root)
    result['problems'] += [f'{src}: well {w} is not in {args.dataset_name}' for w, *_, src in result['moves']
                           if w not in sequences]
    if result['problems']:
        raise ValueError('fix these decisions in the review page first:\n  ' + '\n  '.join(result['problems']))
    if result['undecided'] and not args.allow_incomplete:
        raise ValueError(f"{len(result['undecided'])} of {len(review['items'])} items are undecided; finish the "
                         f"review or pass --allow-incomplete (a partial review is a biased correction)")
    wells = sorted({m[0] for m in result['merges']} | {s[0] for s in result['swaps']} | {m[0] for m in result['moves']})
    tables, relabels, new_ids = {}, {}, {}
    for well in wells:
        observed = {}
        for f, _, mask_path in sequence(args.data_root, well):
            for label in np.unique(tifffile.imread(mask_path)):
                observed.setdefault(int(label), []).append(f)
        tables[well], relabels[well] = refine_well(
            read_track_table(Path(args.data_root) / f'{well}_GT' / 'TRA' / 'man_track.txt'), observed,
            [m[1:] for m in result['merges'] if m[0] == well], [s[1:] for s in result['swaps'] if s[0] == well],
            [m[1:] for m in result['moves'] if m[0] == well], new_ids=new_ids.setdefault(well, {}))
    modes = build_refined_dataset(args.data_root, args.target, sequences, tables, relabels)
    (args.target / 'refine_decisions.json').write_text(args.decisions.read_text(encoding='utf-8'), encoding='utf-8')
    log = dict(source=str(Path(args.data_root).resolve()), target=str(args.target.resolve()),
               created=datetime.datetime.now().isoformat(timespec='seconds'), review=str(args.review.resolve()),
               review_id=review['review_id'], candidates_sha256=sha256(args.review / 'candidates.json'),
               decisions_sha256=sha256(args.decisions), rule=review['rule'], reviewed_wells=review['wells'],
               items=len(review['items']), counts=result['counts'], undecided=len(result['undecided']),
               change="hand-over: parent(B) = A in man_track.txt (CTC single-child continuation); swap: the existing "
                      "track X is relabelled X' in its masks from frame s, parent(X') = A, X ends at s - 1, parent of "
                      "the chosen start = X; gap item / manual edit: X's masks in the given frames get label Y (or a "
                      "new label). Images and all other masks are identical to the source.",
               files=modes,
               merges=[dict(well=w, end=a, end_frame=af, start=b, start_frame=bf) for w, a, af, b, bf in result['merges']],
               swaps=[dict(well=w, end=a, end_frame=af, track=x, from_frame=s, rest=r, rest_frame=rf)
                      for w, a, af, x, s, r, rf in result['swaps']],
               moves=[dict(well=w, track=x, from_frame=f1, to_frame=f2, target=t, source=src)
                      for w, x, f1, f2, t, src in result['moves']],
               manual=exported.get('manual', []),
               new_ids={w: ids for w, ids in sorted(new_ids.items()) if ids},
               relabels=[dict(well=w, label=x, from_frame=f1, to_frame=f2, new_label=n) for w in sorted(relabels)
                         for x, f1, f2, n in relabels[w]])
    (args.target / 'refine_log.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    c = result['counts']
    print(f"same cell {c['same']}, swap {c['swap']}, gap {c['gap']}, manual {c['manual']}, different {c['none']}, "
          f"unsure {c['unsure']}, link items: GT right {c['gt_ok']} / fixed {c['fixed']} / ambiguous {c['ambiguous']}, "
          f"undecided {len(result['undecided'])}; man_track.txt rewritten in {len(tables)} "
          f"wells; files {modes}")
    for w, ids in sorted(new_ids.items()):
        if ids:
            print(f'{w}: new identities ' + ', '.join(f'{k} -> {v}' for k, v in ids.items()))
    if modes['hardlink']:
        print('note: images and masks are hard links to the source files; never write into them')
    print('saved', args.target)
    return 0


def refresh(args):
    review = json.loads((args.review / 'candidates.json').read_text(encoding='utf-8'))
    if review['dataset'] != args.dataset_name:
        raise ValueError(f"review was made on {review['dataset']}, not {args.dataset_name}")
    cache = args.review / 'positions.json'
    if cache.exists():
        xy = json.loads(cache.read_text(encoding='utf-8'))
    else:
        from skimage.measure import regionprops
        xy = {}
        for well in review['wells']:
            objects = {}
            for f, _, mask_path in sequence(args.data_root, well):
                props = sorted(regionprops(tifffile.imread(mask_path)), key=lambda p: p.label)
                objects[f] = ([p.label for p in props], [(p.centroid[1], p.centroid[0]) for p in props])
            xy[well] = positions(objects)
        cache.write_text(json.dumps(xy, separators=(',', ':')), encoding='utf-8')
    write_page(args.review, review, xy)
    print('rebuilt', args.review / 'review.html')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'apply', 'refresh', 'add-links'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, help='Default: project config.json')
        p.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
        p.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
        if name == 'prepare':
            p.add_argument('--wells', nargs='+', help='Default: the development wells of the dataset')
            p.add_argument('--allow-held-out', action='store_true', help='Allow declared test wells')
            p.add_argument('--output-root', type=Path, required=True, help='New directory for the review')
            p.add_argument('--seed', type=int, default=0, help='Shuffle seed of the review order')
            p.add_argument('--reuse-review-id', type=Path, metavar='CANDIDATES_JSON',
                           help='Keep the review id of an earlier review of the same candidates (saved progress stays)')
        elif name == 'refresh':
            p.add_argument('--review', type=Path, required=True, help='The prepare output folder')
        elif name == 'add-links':
            p.add_argument('--review', type=Path, required=True, help='The review to extend (its prepare folder)')
            p.add_argument('--run', type=Path, required=True,
                           help='Out-of-fold folder of a run on --run-dataset (<well>/edges.npz + instance_tracks.csv)')
            p.add_argument('--run-dataset', required=True, help='Refined dataset the run was made and is judged on')
            p.add_argument('--seed', type=int, default=0, help='Shuffle seed of the added items')
        else:
            p.add_argument('--review', type=Path, required=True, help='The prepare output folder')
            p.add_argument('--decisions', type=Path, required=True, help='decisions.json exported from the page')
            p.add_argument('--target', type=Path, help='New dataset folder (default: <data_dir>/<dataset>_refine)')
            p.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    if args.command == 'prepare':
        wells = args.wells or development_wells(spec)
        if not wells or len(wells) != len(set(wells)):
            parser.error('no wells selected or duplicate wells supplied')
        if (held := sorted(set(wells) & set(spec['held_out']))) and not args.allow_held_out:
            parser.error(f'test wells need --allow-held-out: {" ".join(held)}')
        if args.output_root.exists():
            parser.error(f'output-root already exists: {args.output_root}; select a new directory')
        log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                       f'{args.output_root.resolve().name}_handover_prepare_{spec["name"]}')
        return run_logged(log, prepare, args, wells, argv=argv)
    if args.command == 'add-links':
        review = args.review / 'candidates.json'
        if not review.is_file() or not (args.review / 'positions.json').is_file():
            parser.error(f'{args.review} has no candidates.json + positions.json (run refresh on it first)')
        made_on = json.loads(review.read_text(encoding='utf-8'))
        if made_on['dataset'] != spec['name']:
            parser.error(f"the review was made on {made_on['dataset']}, not {spec['name']}")
        try:
            args.run_root = resolve_dataset(cfg, base, args.run_dataset, None)['root']
        except FileNotFoundError as exc:
            parser.error(str(exc))
        wells = sorted(p.parent.name for p in args.run.glob('*/edges.npz'))
        if not wells or not set(wells) <= set(made_on['wells']):
            parser.error(f'the run needs wells of the review: {sorted(set(wells) - set(made_on["wells"]))}')
        if held := sorted(set(wells) & set(spec['held_out'])):
            parser.error(f'test wells must not be reviewed from a run: {" ".join(held)}')
        log = log_path(log_root(args.review, base / cfg['output_root']), 'gnn_tracking',
                       f'{args.review.resolve().name}_handover_add_links_{spec["name"]}')
        return run_logged(log, add_links, args, wells, argv=argv)
    if args.command == 'refresh':
        if not (args.review / 'candidates.json').exists():
            parser.error(f'no candidates.json in {args.review}')
        return refresh(args)
    args.target =args.target or base / cfg.get('data_dir', 'data') / f'{spec["name"]}_refine'
    if args.target.exists():
        parser.error(f'target already exists: {args.target}; choose another --target')
    log = log_path(log_root(args.review, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.review.resolve().name}_handover_apply_{spec["name"]}')
    return run_logged(log, apply, args, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
