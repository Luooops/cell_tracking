"""
evaluate_answers.py
===================
Stage 0.3 (part a): score the tracker's answer sheets (edges.npz from test.py --set record_edges=true, or the
out-of-fold sheets of train.py --cv --set cv_outputs=true) against the ground truth. No model, no images.

Every recorded edge gets its GT relation from utils/lineage.py: same_cell (the target is the same cell, also across
a gap or a single-child continuation), mother_daughter, unrelated, or unknown (a detection without identity, a
parent with >= 3 children). The binary question the network answers is "same cell?": positives = same_cell,
negatives = unrelated; mother_daughter edges are reported on their own and unknown edges are left out.

    calibration   ECE (equal-mass bins), NLL, Brier and the reliability curve of the ensemble p and of every member,
                  for: round1_all (every adjacent-frame candidate), round1_decision (those with DECISION_LO <= p <=
                  DECISION_HI, where the decision is not already certain), round2_solver (gap-2 edges the solver saw)
    coverage      error coverage curves (utils/calibration.py) for false links (chosen edges that are unrelated) and
                  missed links (solver edges left unchosen that are the same cell), ranked by closeness of p to 0.5
                  and by the ensemble's mutual information; a random check is the diagonal
    division      precision / recall / false positives per 1000 detections (utils/division_metrics.py); a run
                  without lineage.json predicts no division

    python -m models.gnn_tracking.evaluate_answers --dataset Fluo-N2DL-HeLa \
        --run development/stage0_evaluation/Fluo-N2DL-HeLa/gnn_tracking/baseline_v001/test \
        --output-root development/stage0_evaluation/answers_Fluo-N2DL-HeLa_gnn_tracking_baseline_v001

Writes into a new --output-root: metrics.json (pooled over wells + per well), reliability.csv, coverage.csv and
figures/answers_summary.{pdf,svg,png} with *_legend.txt. Project evaluation of association on fixed detections; not
the official CTC measures.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.gnn_tracking.edge_record import load_edges
from utils.calibration import (COVERAGE_POINTS, calibration_report, coverage_area, coverage_curve, mutual_information,
                               reliability, sigmoid)
from utils.config import read_config, resolve_dataset
from utils.division_metrics import division_scores, pooled, predicted_divisions
from utils.lineage import MOTHER_DAUGHTER, SAME_CELL, UNRELATED, Lineage, read_track_table
from utils.run_log import log_path, log_root, run_logged

DECISION_LO, DECISION_HI = 0.02, 0.98
BINS = 15
COVERAGE_AT = (1, 2, 5, 10, 20, 50)
POPULATIONS = ('round1_all', 'round1_decision', 'round2_solver')


def load_well(run, root, well):
    """-> dict(edges, relation [E] str, lineage, tracks DataFrame, divisions)."""
    edges = load_edges(Path(run) / well / 'edges.npz')
    tracks = pd.read_csv(Path(run) / well / 'instance_tracks.csv')
    lineage = Lineage(read_track_table(Path(root) / f'{well}_GT' / 'TRA' / 'man_track.txt'),
                      zip(tracks.frame_index, tracks.instance_id))
    relation = np.array([lineage.relation((t, a), (t + g, b)) for t, g, a, b in
                         zip(edges['src_frame'], edges['g'], edges['src_label'], edges['dst_label'])], dtype=object)
    lineage_path = Path(run) / well / 'lineage.json'
    has_lineage = lineage_path.is_file()
    predicted, malformed = predicted_divisions(zip(tracks.frame_index, tracks.instance_id, tracks.track_id),
                                               json.loads(lineage_path.read_text(encoding='utf-8')) if has_lineage else {})
    division = division_scores(predicted, lineage, len(tracks)) | dict(malformed=len(malformed),
                                                                      lineage_json=has_lineage)
    return dict(edges=edges, relation=relation, division=division, n_detections=len(tracks))


def masks(e, relation):
    binary = np.isin(relation, (SAME_CELL, UNRELATED))
    r1, r2 = e['round'] == 1, e['round'] == 2
    p = e['p']
    return dict(round1_all=r1 & binary, round1_decision=r1 & binary & (p >= DECISION_LO) & (p <= DECISION_HI),
                round2_solver=r2 & e['in_solver'] & binary,
                false_links=e['chosen'] & binary, missed_links=e['in_solver'] & ~e['chosen'] & binary)


def calibration_block(e, relation, sel):
    y = relation[sel] == SAME_CELL
    q = sigmoid(e['logits'][sel])
    report = calibration_report(e['p'][sel], y, BINS)
    report['members'] = [calibration_report(q[:, m], y, BINS) for m in range(q.shape[1])]
    return report


def coverage_block(e, relation, sel, error_relation):
    error = relation[sel] == error_relation
    p, q = e['p'][sel], sigmoid(e['logits'][sel])
    out = dict(items=int(sel.sum()), errors=int(error.sum()))
    curves = {}
    for name, score in (('closeness', np.minimum(p, 1 - p)), ('mutual_info', mutual_information(q))):
        curve = coverage_curve(score, error)
        curves[name] = curve
        out[name] = None if curve is None else dict(area=coverage_area(curve),
                                                    at_percent={str(k): float(curve[k]) for k in COVERAGE_AT})
    return out, curves


def mother_daughter_block(e, relation):
    out = {}
    for rnd in (1, 2):
        sel = (relation == MOTHER_DAUGHTER) & (e['round'] == rnd) & e['in_solver']
        p = e['p'][sel]
        out[f'round{rnd}'] = dict(n=int(sel.sum()), **({} if not sel.any() else dict(
            p_q10=float(np.quantile(p, .1)), p_median=float(np.median(p)), p_q90=float(np.quantile(p, .9)),
            share_p_above_half=float((p > 0.5).mean()), share_chosen=float(e['chosen'][sel].mean()))))
    return out


def concat(parts):
    keys = parts[0]['edges'].keys()
    return {k: np.concatenate([w['edges'][k] for w in parts]) for k in keys}, \
        np.concatenate([w['relation'] for w in parts])


def summarize(e, relation):
    m = masks(e, relation)
    calibration = {pop: calibration_block(e, relation, m[pop]) for pop in POPULATIONS}
    reliability_rows, coverage_rows, coverage = [], [], {}
    for pop in POPULATIONS:
        for b, row in enumerate(reliability(e['p'][m[pop]], relation[m[pop]] == SAME_CELL, BINS)):
            reliability_rows.append(dict(population=pop, bin=b, **row))
    for pop, error_relation in (('false_links', UNRELATED), ('missed_links', SAME_CELL)):
        coverage[pop], curves = coverage_block(e, relation, m[pop], error_relation)
        for name, curve in curves.items():
            if curve is not None:
                coverage_rows += [dict(population=pop, score=name, checked_percent=float(k), errors_found_percent=float(v))
                                  for k, v in zip(COVERAGE_POINTS, curve)]
    counts = {str(k): int(v) for k, v in zip(*np.unique(relation.astype(str), return_counts=True))}
    return dict(edges=int(len(relation)), relations=counts, calibration=calibration, coverage=coverage,
                mother_daughter=mother_daughter_block(e, relation)), reliability_rows, coverage_rows


def plot(output_base, reliability_rows, coverage_rows, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.metric_plots import EDGE, INK, MM, RUN_COLORS, _panel_label, _style
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.82, wspace=0.42)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=EDGE, lw=0.6, ls='--', zorder=1)
    for i, pop in enumerate(POPULATIONS):
        rows = [r for r in reliability_rows if r['population'] == pop]
        if len(rows) >= 3:
            ax.plot([r['mean_p'] for r in rows], [r['frac_pos'] for r in rows], marker='o', ms=2.2, lw=0.8,
                    color=RUN_COLORS[i], mec=INK, mew=0.3, label=pop.replace('_', ' '), zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted p', ylabel='Fraction same cell')
    ax.legend(loc='upper left')
    for ax, pop, name in ((axes[1], 'false_links', 'False links'), (axes[2], 'missed_links', 'Missed links')):
        ax.plot([0, 100], [0, 100], color=EDGE, lw=0.6, ls='--', label='random', zorder=1)
        for i, score in enumerate(('closeness', 'mutual_info')):
            rows = [r for r in coverage_rows if r['population'] == pop and r['score'] == score]
            if rows:
                ax.plot([r['checked_percent'] for r in rows], [r['errors_found_percent'] for r in rows], lw=0.9,
                        color=RUN_COLORS[i], label={'closeness': 'p near 0.5', 'mutual_info': 'disagreement'}[score])
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='Links checked (%)', ylabel=f'{name} found (%)')
        ax.legend(loc='lower right')
    for ax, letter in zip(axes, 'abc'):
        _panel_label(fig, ax, letter)
    fig.text(0.07, 0.975, title, fontsize=7.5, fontweight='bold', ha='left', va='top')
    written = []
    for suffix, kwargs in (('.pdf', {}), ('.svg', {}), ('.png', dict(dpi=600))):
        fig.savefig(output_base.with_suffix(suffix), **kwargs)
        written.append(output_base.with_suffix(suffix))
    plt.close(fig)
    legend = (f'a, Reliability of the ensemble edge probability (equal-mass bins, {BINS} per population; dashed, '
              f'perfect calibration). round1 all, every adjacent-frame candidate edge; round1 decision, those with '
              f'{DECISION_LO} <= p <= {DECISION_HI}; round2 solver, gap-2 edges offered to the solver. Positives, same '
              f'cell; negatives, unrelated cells; mother-daughter and unknown-identity edges excluded. b, c, Share of '
              f'all false links (chosen but unrelated) and missed links (offered, not chosen, same cell) found when '
              f'checking the given share of items in order of uncertainty: p near 0.5 (min(p, 1 - p)) or member '
              f'disagreement (mutual information of the ensemble); dashed, random checking. Pooled over sequences. '
              f'Project association evaluation on fixed detections, not official CTC measures.')
    legend_path = output_base.parent / f'{output_base.name}_legend.txt'
    legend_path.write_text(legend + '\n', encoding='utf-8')
    return written + [legend_path]


def evaluate(args, wells):
    parts = {w: load_well(args.run, args.data_root, w) for w in wells}
    e, relation = concat(list(parts.values()))
    overall, reliability_rows, coverage_rows = summarize(e, relation)
    overall['division'] = pooled([p['division'] for p in parts.values()])
    per_well = {}
    for w, part in parts.items():
        s, _, _ = summarize(part['edges'], part['relation'])
        per_well[w] = dict(edges=s['edges'], relations=s['relations'], division=part['division'],
                           calibration={pop: {k: s['calibration'][pop][k] for k in ('n', 'ece', 'nll', 'brier')}
                                        for pop in POPULATIONS},
                           coverage_area={pop: {k: (v or {}).get('area') for k, v in s['coverage'][pop].items()
                                                if k in ('closeness', 'mutual_info')} for pop in s['coverage']})
        cal = s['calibration']['round1_decision']
        print(f"{w}: {s['edges']} edges, round1 decision n={cal['n']} ECE={cal['ece']}, "
              f"divisions predicted {part['division']['predicted']} / GT {part['division']['gt_divisions']}", flush=True)
    report = dict(run=str(Path(args.run).resolve()), dataset=args.dataset_name, wells=wells,
                  decision_band=[DECISION_LO, DECISION_HI], bins=BINS, members=int(e['logits'].shape[1]),
                  scope='project association evaluation on recorded candidate edges; not official CTC measures',
                  overall=overall, per_well=per_well)
    out = args.output_root
    out.mkdir(parents=True)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, rows in (('reliability.csv', reliability_rows), ('coverage.csv', coverage_rows)):
        with (out / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['population'])
            writer.writeheader()
            writer.writerows(rows)
    if args.plots:
        (out / 'figures').mkdir()
        parts = Path(args.run).resolve().parts                 # <model>/<version>/test, or .../oof/<model>
        name = f'{parts[-3]} · {parts[-2]}' if parts[-1] == 'test' else f'{parts[-1]} · out-of-fold'
        for path in plot(out / 'figures' / 'answers_summary', reliability_rows, coverage_rows,
                         f'{name} · {args.dataset_name}'):
            print('saved', path, flush=True)
    c = overall['calibration']
    print('\npooled over wells:')
    for pop in POPULATIONS:
        print(f"  {pop:16s} n={c[pop]['n']:8d}  ECE={c[pop]['ece']}  NLL={c[pop]['nll']}  Brier={c[pop]['brier']}")
    for pop in ('false_links', 'missed_links'):
        cv = overall['coverage'][pop]
        print(f"  {pop:16s} errors {cv['errors']} / {cv['items']}; area closeness "
              f"{(cv['closeness'] or {}).get('area')}, disagreement {(cv['mutual_info'] or {}).get('area')}")
    d = overall['division']
    print(f"  division: predicted {d['predicted']}, GT {d['gt_divisions']}, precision {d['precision']}, "
          f"recall {d['recall']}, false positives per 1000 detections {d['false_positives_per_1000_detections']}")
    print('saved', out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Default: project config.json')
    parser.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
    parser.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
    parser.add_argument('--run', type=Path, required=True, help='Folder with <well>/edges.npz + instance_tracks.csv')
    parser.add_argument('--wells', nargs='+', help='Default: every well folder of --run that has edges.npz')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for the evaluation')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    wells = args.wells or sorted(p.parent.name for p in args.run.glob('*/edges.npz'))
    if not wells or len(wells) != len(set(wells)):
        parser.error('no wells with edges.npz found under --run, or duplicate wells supplied')
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_evaluate_answers_{spec["name"]}')
    return run_logged(log, evaluate, args, wells, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
