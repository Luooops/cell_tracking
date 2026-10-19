"""
audit_handover.py
=================
Stage 0 addendum: human review of suspected identity hand-overs in the annotation, then a refined dataset.

prepare   candidates by the fixed, model-free rule of utils/gt_refine.py (a GT track ends, another starts within
          1-2 frames inside the tracker's gate), one image strip per track end (frames t-1 .. last start + 1; the
          ending track A in red, the candidate starts 1, 2, ... in other colours, other cells grey) and a local
          review page. The review is blind: no model output is shown, and the items are in random order.
apply     the exported decisions -> a new dataset <data_dir>/<dataset>_refine: every sequence reproduced with
          images and masks hard-linked (copied where links are impossible), man_track.txt rewritten where a start
          was confirmed as the same cell as the end (its parent becomes that end: a CTC single-child continuation).
          The source dataset is never modified. refine_log.json + the decisions file record every change.

    python -m models.gnn_tracking.audit_handover prepare --dataset scripta_data \
        --output-root development/stage0_evaluation/handover_review_scripta_data
    (open <output-root>/review.html in a browser, decide, press Export -> decisions.json)
    python -m models.gnn_tracking.audit_handover apply --dataset scripta_data \
        --review development/stage0_evaluation/handover_review_scripta_data --decisions decisions.json
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
from utils.gt_refine import (apply_merges, build_refined_dataset, check_decisions, handover_candidates, sha256,
                             write_track_table)
from utils.lineage import read_track_table
from utils.run_log import log_path, log_root, run_logged

TEMPLATE = Path(__file__).with_name('handover_review.html')
TILE, GAP, HEADER = 220, 6, 22
A_COLOR = (229, 57, 53)
B_COLORS = [(30, 136, 229), (67, 160, 71), (251, 140, 0), (142, 36, 170), (0, 172, 193), (216, 27, 96)]
OTHER = (150, 150, 150)
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


def _crop(image, cx, cy, half):
    """Square crop centred at (cx, cy), zero-padded at the image border."""
    padded = np.pad(image, half, mode='constant')
    x, y = int(round(cx)) + half, int(round(cy)) + half
    return padded[y - half:y + half, x - half:x + half]


def render_panel(item, frames, raws, masks, path):
    """One image strip: a tile per frame from a_frame - 1 to the last start + 1, outlines drawn at tile resolution."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_dilation
    from skimage.segmentation import find_boundaries
    last_start = max(s['b_frame'] for s in item['starts'])
    show = [f for f in range(item['a_frame'] - 1, last_start + 2) if f in raws]
    cx, cy = item['a_xy']
    half = int(np.clip(item['radius'] * 1.4, 48, 240))
    crops = [_crop(raws[f].astype(np.float32), cx, cy, half) for f in show]
    lo, hi = np.percentile(np.concatenate([c.ravel() for c in crops]), [1, 99.7])
    colors = {item['a_label']: A_COLOR}
    colors.update({s['b_label']: B_COLORS[k % len(B_COLORS)] for k, s in enumerate(item['starts'])})
    names = {item['a_label']: 'A'}
    names.update({s['b_label']: str(k + 1) for k, s in enumerate(item['starts'])})
    font = ImageFont.load_default(size=14)
    canvas = Image.new('RGB', (len(show) * (TILE + GAP) - GAP, TILE + HEADER), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for col, (f, crop) in enumerate(zip(show, crops)):
        gray = np.clip((crop - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        tile = np.array(Image.fromarray(gray).resize((TILE, TILE), Image.BILINEAR).convert('RGB'))
        labels = np.array(Image.fromarray(_crop(masks[f], cx, cy, half).astype(np.int32)).resize((TILE, TILE),
                                                                                                   Image.NEAREST))
        edge = find_boundaries(labels, mode='inner')
        tile[edge & ~np.isin(labels, list(colors))] = OTHER
        marks = []
        for label, color in colors.items():
            region = labels == label
            if region.any():
                tile[binary_dilation(edge & region)] = color
                ys, xs = np.nonzero(region)
                marks.append((names[label], color, xs.mean(), ys.min()))
        x0 = col * (TILE + GAP)
        canvas.paste(Image.fromarray(tile), (x0, HEADER))
        rel = f - item['a_frame']
        draw.text((x0 + 4, 3), f'frame {f}  (t{rel:+d})' if rel else f'frame {f}  (t: A ends)', fill=(40, 40, 40),
                  font=font)
        for text, color, x, y in marks:
            draw.text((x0 + x - 4, HEADER + max(y - 17, 0)), text, fill=color, font=font, stroke_width=2,
                      stroke_fill=(255, 255, 255))
    canvas.save(path)
    return [int(f) for f in show]


def prepare(args, wells):
    out = args.output_root
    (out / 'panels').mkdir(parents=True)
    items = []
    for well in wells:
        frames, raws, masks, objects = load_well(args.data_root, well)
        table = read_track_table(Path(args.data_root) / f'{well}_GT' / 'TRA' / 'man_track.txt')
        found = handover_candidates(well, frames, objects, table)
        for item in found:
            item['panel'] = f'panels/{well}_{item["a_label"]}.png'
            item['frames_shown'] = render_panel(item, frames, raws, masks, out / item['panel'])
        items += found
        print(f'{well}: {len(found)} track ends with candidate starts '
              f'({sum(len(i["starts"]) for i in found)} pairs)', flush=True)
    random.Random(args.seed).shuffle(items)                 # review order carries no information
    review = dict(review_id=uuid.uuid4().hex, dataset=args.dataset_name, data_root=str(Path(args.data_root).resolve()),
                  wells=wells, created=datetime.datetime.now().isoformat(timespec='seconds'), seed=args.seed,
                  rule='A known GT track ends at t (not the last frame, no children); a known GT track without parent '
                       'starts at t+1 or t+2 closer than the gating radius r(t) x sqrt(g) of '
                       'models.gnn_tracking.gating (positions only). No model output is used or shown.',
                  items=items)
    (out / 'candidates.json').write_text(json.dumps(review, indent=1), encoding='utf-8')
    page = TEMPLATE.read_text(encoding='utf-8').replace(
        '/*__REVIEW_DATA__*/', json.dumps(review, separators=(',', ':')).replace('</', '<\\/'))
    (out / 'review.html').write_text(page, encoding='utf-8')
    print(f'\n{len(items)} review items ({sum(len(i["starts"]) for i in items)} pairs); open {out / "review.html"}')
    return 0


def apply(args):
    review = json.loads((args.review / 'candidates.json').read_text(encoding='utf-8'))
    exported = json.loads(args.decisions.read_text(encoding='utf-8'))
    if exported.get('review_id') != review['review_id']:
        raise ValueError('decisions were exported from a different review (review_id mismatch)')
    if review['dataset'] != args.dataset_name:
        raise ValueError(f"review was made on {review['dataset']}, not {args.dataset_name}")
    result = check_decisions(review['items'], exported['decisions'])
    if result['problems']:
        raise ValueError('fix these decisions in the review page first:\n  ' + '\n  '.join(result['problems']))
    if result['undecided'] and not args.allow_incomplete:
        raise ValueError(f"{len(result['undecided'])} of {len(review['items'])} items are undecided; finish the "
                         f"review or pass --allow-incomplete (a partial review is a biased correction)")
    by_well = {}
    for well, a, b in result['merges']:
        by_well.setdefault(well, []).append((a, b))
    tables = {w: apply_merges(read_track_table(Path(args.data_root) / f'{w}_GT' / 'TRA' / 'man_track.txt'), m)
              for w, m in by_well.items()}
    modes = build_refined_dataset(args.data_root, args.target, ctc_sequences(args.data_root), tables)
    (args.target / 'refine_decisions.json').write_text(args.decisions.read_text(encoding='utf-8'), encoding='utf-8')
    log = dict(source=str(Path(args.data_root).resolve()), target=str(args.target.resolve()),
               created=datetime.datetime.now().isoformat(timespec='seconds'), review=str(args.review.resolve()),
               review_id=review['review_id'], candidates_sha256=sha256(args.review / 'candidates.json'),
               decisions_sha256=sha256(args.decisions), rule=review['rule'], reviewed_wells=review['wells'],
               items=len(review['items']), counts=result['counts'], undecided=len(result['undecided']),
               change='man_track.txt only: parent(B) = A for every confirmed pair (CTC single-child continuation); '
                      'images and masks identical to the source',
               files=modes, merges=[dict(well=w, end=a, start=b) for w, a, b in result['merges']])
    (args.target / 'refine_log.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(f"same cell {result['counts']['same']}, different {result['counts']['none']}, unsure "
          f"{result['counts']['unsure']}, undecided {len(result['undecided'])}; man_track.txt rewritten in "
          f"{len(tables)} wells; files {modes}")
    if modes['hardlink']:
        print('note: images and masks are hard links to the source files; never write into them')
    print('saved', args.target)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'apply'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, help='Default: project config.json')
        p.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
        p.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
        if name == 'prepare':
            p.add_argument('--wells', nargs='+', help='Default: the development wells of the dataset')
            p.add_argument('--allow-held-out', action='store_true', help='Allow declared test wells')
            p.add_argument('--output-root', type=Path, required=True, help='New directory for the review')
            p.add_argument('--seed', type=int, default=0, help='Shuffle seed of the review order')
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
    args.target = args.target or base / cfg.get('data_dir', 'data') / f'{spec["name"]}_refine'
    if args.target.exists():
        parser.error(f'target already exists: {args.target}; choose another --target')
    log = log_path(log_root(args.review, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.review.resolve().name}_handover_apply_{spec["name"]}')
    return run_logged(log, apply, args, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
