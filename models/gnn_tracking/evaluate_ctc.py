"""
evaluate_ctc.py
===============
Stage 0.3 (part b): the Cell Tracking Challenge measures of a tracking run, computed by py-ctcmetrics (the official
Python implementation, pip install py-ctcmetrics) on our outputs exported to the CTC format (utils/ctc_export.py).

    DET    detection: how well the objects of every frame match the GT (here the detections are fixed, so DET ~ 1 on
           scripta_data is a self-check of the export)
    TRA    detection + linking: 1 - AOGM / AOGM_0, AOGM = weighted count of graph edits that turn the predicted
           lineage graph into the GT graph (FN node 10, split 5, add edge 1.5, FP node 1, delete edge 1, edge type 1);
           the main CTC tracking measure
    LNK    linking only: the edge part of AOGM (CTC 2023)
    BC(i)  branching correctness: F1 of divisions matched within i frames (BC(1) reported; undefined without GT
           divisions)
    CT     complete tracks: share of GT tracks reproduced from start to end without any error

GT: 'official' = <root>/source/<well>_GT (the CTC reference markers, used for Fluo-N2DL-HeLa); 'canonical' = the
dataset's own man_track masks made CTC-valid by utils.ctc_export.canonical_gt (gaps -> single-child segments,
identities >= 50000 removed), built once into --gt-cache and reused (scripta_data). 'auto' picks official when it
exists. Canonical-GT scores follow the CTC definitions on project ground truth; neither kind is comparable with the
CTC leaderboard (training data, fixed detections).

    python -m models.gnn_tracking.evaluate_ctc --dataset Fluo-N2DL-HeLa \
        --run development/stage0_evaluation/Fluo-N2DL-HeLa/gnn_division/baseline_v001/test \
        --output-root development/stage0_evaluation/ctc_Fluo-N2DL-HeLa_gnn_division_baseline_v001

Writes into a new --output-root: metrics.json (per well, mean / s.d. over wells, AOGM-pooled TRA) and per_well.csv.
The exported RES folders (res/<well>_RES) are deleted after scoring unless --keep-res.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import read_config, resolve_dataset
from utils.ctc import sequence
from utils.ctc_export import canonical_gt, export_res
from utils.gt_refine import relabel_lookup, translate
from utils.lineage import read_track_table
from utils.run_log import log_path, log_root, run_logged

METRICS = ['Valid', 'DET', 'TRA', 'LNK', 'CT', 'BC(1)']
REPORTED = ('DET', 'TRA', 'LNK', 'BC(1)', 'CT')
KEEP = ('Valid', 'DET', 'TRA', 'LNK', 'CT', 'AOGM', 'AOGM_0', 'AOGM_NS', 'AOGM_FN', 'AOGM_FP', 'AOGM_ED', 'AOGM_EA',
        'AOGM_EC', 'gt_divisions') + tuple(f'{k}({i})' for i in range(4) for k in ('BC', 'tp_div', 'fp_div', 'fn_div'))


def ctc_metrics_api():
    try:
        from ctc_metrics import evaluate_sequence, validate_sequence
    except ImportError:
        from ctc_metrics.scripts.evaluate import evaluate_sequence
        from ctc_metrics.scripts.validate import validate_sequence
    return evaluate_sequence, validate_sequence


def number(value):
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def has_division(track_table):
    """True if some parent in man_track.txt has two or more children (a CTC division)."""
    parents = [parent for _, _, parent in read_track_table(track_table).values() if parent]
    return any(parents.count(p) >= 2 for p in set(parents))


def gt_folder(args, well):
    """-> (GT folder with TRA/, 'official' | 'canonical'); a canonical GT is built once and then reused."""
    official = Path(args.data_root) / 'source' / f'{well}_GT'
    if args.gt in ('auto', 'official') and (official / 'TRA' / 'man_track.txt').is_file():
        return official, 'official'
    if args.gt == 'official':
        raise FileNotFoundError(f'no official GT at {official}')
    source = Path(args.data_root) / f'{well}_GT' / 'TRA' / 'man_track.txt'
    folder = Path(args.gt_cache) / f'{well}_GT'
    stamp = dict(source=str(source.resolve()), size=source.stat().st_size, mtime=source.stat().st_mtime_ns,
                 drop_unknown=True)
    stamp_path = folder / 'canonical.json'
    if folder.exists():
        saved = json.loads(stamp_path.read_text(encoding='utf-8')) if stamp_path.is_file() else {}
        if {k: saved.get(k) for k in stamp} != stamp:
            raise ValueError(f'{folder} exists but was not built from the current {source}; remove it or use '
                             f'another --gt-cache')
        return folder, 'canonical'
    info = canonical_gt(read_track_table(source), [(t, m) for t, _, m in sequence(args.data_root, well)], folder)
    stamp_path.write_text(json.dumps(stamp | info, indent=2), encoding='utf-8')
    print(f'  built canonical GT for {well}: {info}', flush=True)
    return folder, 'canonical'


def evaluate_well(args, well, evaluate_sequence, validate_sequence):
    run = Path(args.run) / well
    tracks = pd.read_csv(run / 'instance_tracks.csv')
    entries = relabel_lookup(args.data_root, run).get(well)
    if entries:                      # outputs made on the source data, scored on a refined dataset
        tracks['instance_id'] = [translate(entries, f, l) for f, l in zip(tracks.frame_index, tracks.instance_id)]
    lineage_path = run / 'lineage.json'
    lineage = json.loads(lineage_path.read_text(encoding='utf-8')) if lineage_path.is_file() else {}
    res = args.output_root / 'res' / f'{well}_RES'
    info = export_res(zip(tracks.frame_index, tracks.instance_id, tracks.track_id), lineage,
                      [(t, m) for t, _, m in sequence(args.data_root, well)], res)
    if not validate_sequence(str(res), threads=args.threads)['Valid']:
        raise ValueError(f'{well}: exported RES failed the CTC format check ({res}); kept for inspection')
    gt, kind = gt_folder(args, well)
    divides = has_division(gt / 'TRA' / 'man_track.txt')
    metrics = [m for m in METRICS if divides or not m.startswith('BC')]   # BC is undefined without GT divisions
    raw = evaluate_sequence(str(res), str(gt), metrics=metrics, threads=args.threads)
    if not args.keep_res:
        shutil.rmtree(res)
    return dict(gt=kind, gt_folder=str(gt.resolve()), export=info, lineage_json=lineage_path.is_file(),
                **{k: number(raw.get(k)) for k in KEEP})


def evaluate(args, wells):
    evaluate_sequence, validate_sequence = ctc_metrics_api()
    args.output_root.mkdir(parents=True)
    per_well = {}
    for well in wells:
        per_well[well] = r = evaluate_well(args, well, evaluate_sequence, validate_sequence)
        print(f"{well} ({r['gt']} GT): " + '  '.join(f'{k} {r[k]:.4f}' if r[k] is not None else f'{k} -'
                                                    for k in REPORTED), flush=True)
    summary = {}
    for k in REPORTED:
        values = [r[k] for r in per_well.values() if r[k] is not None]
        summary[k] = dict(mean=float(np.mean(values)) if values else None,
                          sd=float(np.std(values, ddof=1)) if len(values) > 1 else None, n=len(values))
    aogm, aogm0 = (sum(r[k] or 0 for r in per_well.values()) for k in ('AOGM', 'AOGM_0'))
    summary['TRA_pooled'] = 1 - min(aogm, aogm0) / aogm0 if aogm0 else None
    report = dict(run=str(Path(args.run).resolve()), dataset=args.dataset_name, wells=wells, metrics=METRICS,
                  drop_unknown_identities=True,
                  scope='CTC measure definitions (py-ctcmetrics) on fixed detections; official GT = CTC reference '
                        'markers, canonical GT = project ground truth; not comparable with the CTC leaderboard',
                  summary=summary, per_well=per_well)
    (args.output_root / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output_root / 'per_well.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['well', 'gt'] + list(KEEP))
        writer.writeheader()
        writer.writerows(dict(well=w, gt=r['gt'], **{k: r[k] for k in KEEP}) for w, r in per_well.items())
    if not args.keep_res and (args.output_root / 'res').is_dir() and not any((args.output_root / 'res').iterdir()):
        (args.output_root / 'res').rmdir()
    print('\nmean over wells: ' + '  '.join(f"{k} {summary[k]['mean']:.4f}" if summary[k]['mean'] is not None
                                            else f'{k} -' for k in REPORTED)
          + (f"  (TRA pooled over AOGM {summary['TRA_pooled']:.4f})" if summary['TRA_pooled'] is not None else ''))
    print('saved', args.output_root)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Default: project config.json')
    parser.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
    parser.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
    parser.add_argument('--run', type=Path, required=True, help='Folder with <well>/instance_tracks.csv')
    parser.add_argument('--wells', nargs='+', help='Default: every well folder of --run with instance_tracks.csv')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for the evaluation')
    parser.add_argument('--gt', choices=['auto', 'official', 'canonical'], default='auto')
    parser.add_argument('--gt-cache', type=Path, help='Canonical GT folder (default: <output-root parent>/ctc_gt/<dataset>)')
    parser.add_argument('--keep-res', action='store_true', help='Keep the exported RES folders')
    parser.add_argument('--threads', type=int, default=0, help='py-ctcmetrics processes (0 = one per CPU)')
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    args.gt_cache = args.gt_cache or args.output_root.resolve().parent / 'ctc_gt' / spec['name']
    wells = args.wells or sorted(p.parent.name for p in args.run.glob('*/instance_tracks.csv'))
    if not wells or len(wells) != len(set(wells)):
        parser.error('no wells with instance_tracks.csv found under --run, or duplicate wells supplied')
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_evaluate_ctc_{spec["name"]}')
    return run_logged(log, evaluate, args, wells, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
