"""
audit_gating.py
===============
Stage 0 gating audit: how many ground-truth links fall inside the tracker's candidate gate?

A link that the gate drops can never be predicted, whatever the network says, so this is the ceiling of every later
step. For every GT link of utils/lineage.py (continuation via the same label or a single child, mother -> daughter)
the link from frame t to t+g is inside the gate if its length is < r(t) x sqrt(g), exactly as in graph.pair_graph,
with r from gating.gating_radius (the function graph.Video.radius uses). Positions are the mask centroids, the same
numbers utils.dataset.read_well gives the tracker. No model, no weights, no image pixels.

Inference scores g = 1 (round 1) and g = 2 (round 2, track end -> track start); training also uses g = 3. Links with
g > 3 cannot be recovered at all and are counted separately.

Wells: the dataset's development wells (folds, or train + val, or every sequence that is not a declared test well).
Declared test wells are refused, so decisions taken from this audit never look at them.

    python -m models.gnn_tracking.audit_gating --dataset scripta_data --output-root development/stage0_evaluation/gating_audit_scripta_data
    python -m models.gnn_tracking.audit_gating --dataset Fluo-N2DL-HeLa --output-root development/stage0_evaluation/gating_audit_Fluo-N2DL-HeLa

Writes into a new --output-root: links.csv (one row per GT link), radius.csv (r per well and frame, with the term
that set it), anomalies.csv (lineage anomalies) and summary.json (gate recall per kind and gap, overall and per well).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops

from models.gnn_tracking import gating
from models.gnn_tracking.gating import gating_radius
from utils.config import ctc_sequences, read_config, resolve_dataset
from utils.ctc import sequence, write_csv
from utils.lineage import read_lineage
from utils.run_log import log_path, log_root, run_logged

QUANTILES = (0.5, 0.9, 0.99)
LINK_COLUMNS = ['well', 'kind', 'via', 'src_frame', 'src_label', 'dst_frame', 'dst_label', 'g', 'distance',
                'radius', 'gate', 'distance_over_gate', 'in_gate', 'radius_bound', 'spacing', 'distance_over_spacing']
RADIUS_COLUMNS = ['well', 'frame', 'radius', 'bound', 'step_q99', 'n_steps', 'spacing']


def development_wells(spec):
    """Same rule as train.py: folds, else train + val, else every sequence that is not a declared test well."""
    return ([w for fold in spec['folds'] for w in fold] or spec['train_wells'] + spec['val_wells']
            or [w for w in ctc_sequences(spec['root']) if w not in spec['held_out']])


def mask_objects(root, well):
    """{frame: (labels [n], positions [n, 2] as (x, y))}, rows sorted by label like read_well."""
    out = {}
    for frame, _, mask_path in sequence(root, well):
        props = sorted(regionprops(tifffile.imread(mask_path)), key=lambda p: p.label)
        labels = np.array([p.label for p in props], dtype=np.int64)
        pos = np.array([(p.centroid[1], p.centroid[0]) for p in props], dtype=float).reshape(-1, 2)
        out[frame] = (labels, pos)
    return out


def audit_well(root, well):
    objects = mask_objects(root, well)
    frames = sorted(objects)
    pos = {f: objects[f][1] for f in frames}
    where = {(f, int(l)): i for f in frames for i, l in enumerate(objects[f][0])}
    lineage = read_lineage(Path(root) / f'{well}_GT' / 'TRA' / 'man_track.txt', list(where))
    radius = {f: gating_radius(frames, pos, f) for f in frames[:-1]}
    radius_rows = [dict(well=well, frame=f, radius=r, **d) for f, (r, d) in radius.items()]
    rows = []
    for link in lineage.links:
        (t, a), (t2, b) = link.src, link.dst
        d = float(np.hypot(*(pos[t2][where[link.dst]] - pos[t][where[link.src]])))
        r, details = radius[t]
        gate = r * np.sqrt(link.g)
        spacing = details['spacing']
        rows.append(dict(well=well, kind=link.kind, via=link.via, src_frame=t, src_label=a, dst_frame=t2,
                         dst_label=b, g=link.g, distance=d, radius=r, gate=gate, distance_over_gate=d / gate,
                         in_gate=bool(d < gate), radius_bound=details['bound'], spacing=spacing,
                         distance_over_spacing=d / spacing if spacing else None))
    anomalies = [dict(well=well, **a) for a in lineage.anomalies]
    return rows, radius_rows, anomalies, lineage.summary()


def gap_bucket(g):
    return str(g) if g <= 3 else '>3'


def recall_table(rows):
    """{kind: {gap bucket: dict(n, in_gate, recall, distance_over_gate quantiles)}}, plus 'all' per kind."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['kind'], gap_bucket(row['g']))].append(row)
        groups[(row['kind'], 'all')].append(row)
    out = defaultdict(dict)
    for (kind, bucket), part in sorted(groups.items()):
        ratio = np.array([r['distance_over_gate'] for r in part])
        inside = sum(r['in_gate'] for r in part)
        out[kind][bucket] = dict(n=len(part), in_gate=int(inside), recall=inside / len(part),
                                 distance_over_gate={f'q{int(q * 100)}': float(np.quantile(ratio, q)) for q in QUANTILES}
                                 | {'max': float(ratio.max())})
    return dict(out)


def audit(args, wells):
    links, radii, anomalies, per_well = [], [], [], {}
    for well in wells:
        rows, radius_rows, well_anomalies, lineage_summary = audit_well(args.data_root, well)
        links += rows; radii += radius_rows; anomalies += well_anomalies
        per_well[well] = dict(lineage=lineage_summary, gate_recall=recall_table(rows),
                              radius_bound=dict(Counter(r['bound'] for r in radius_rows)),
                              radius_median=float(np.median([r['radius'] for r in radius_rows])))
        print(f"{well}: {len(rows)} links, {sum(r['in_gate'] for r in rows)} inside the gate, "
              f"radius median {per_well[well]['radius_median']:.1f} px, bounds {per_well[well]['radius_bound']}", flush=True)
    overall = recall_table(links)
    summary = dict(dataset=args.dataset_name, data_root=str(args.data_root), wells=wells,
                   rule='in gate if distance < r(t) * sqrt(g); r = models.gnn_tracking.gating.gating_radius',
                   gating_constants=dict(SAFETY=gating.SAFETY, R_MIN=gating.R_MIN, SPACING_CAP=gating.SPACING_CAP,
                                         WINDOW=gating.WINDOW, R_FALLBACK=gating.R_FALLBACK),
                   inference_gaps='round 1: g = 1; round 2: g = 2 (track end -> start); training: g <= 3',
                   gate_recall=overall, radius_bound=dict(Counter(r['bound'] for r in radii)),
                   anomalies=dict(Counter(a['kind'] for a in anomalies)), wells_detail=per_well)
    args.output_root.mkdir(parents=True)
    write_csv(args.output_root / 'links.csv', links, LINK_COLUMNS)
    write_csv(args.output_root / 'radius.csv', radii, RADIUS_COLUMNS)
    columns = ['well', 'kind', 'label'] + sorted({k for a in anomalies for k in a} - {'well', 'kind', 'label'})
    write_csv(args.output_root / 'anomalies.csv', anomalies, columns)
    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\ngate recall (all wells):')
    for kind, buckets in overall.items():
        for bucket, s in buckets.items():
            print(f"  {kind:16s} g={bucket:3s} n={s['n']:6d}  recall={s['recall']:.4f}  "
                  f"distance/gate q99={s['distance_over_gate']['q99']:.2f} max={s['distance_over_gate']['max']:.2f}")
    print(f"radius bounds: {summary['radius_bound']}; anomalies: {summary['anomalies']}")
    print('saved', args.output_root)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Default: project config.json')
    parser.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
    parser.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
    parser.add_argument('--wells', nargs='+', help='Default: the development wells of the dataset')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for the audit results')
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    wells = args.wells or development_wells(spec)
    if not wells or len(wells) != len(set(wells)):
        parser.error('no wells selected or duplicate wells supplied')
    if held := sorted(set(wells) & set(spec['held_out'])):
        parser.error(f'test wells are kept out of the audit: {" ".join(held)}')
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_audit_gating_{spec["name"]}')
    return run_logged(log, audit, args, wells, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
