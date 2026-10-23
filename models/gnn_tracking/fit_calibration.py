"""
fit_calibration.py
==================
Stage 1.1: recalibrate the ensemble's edge probabilities from its out-of-fold answer sheets (no model, no images).

Every member m gets its own map of its logit z_m, p_m = sigmoid(a_m z_m + b_m); the ensemble probability is the mean
of the mapped members, as in the tracker. Methods:
    none             the tracker's p (identity maps)
    temperature      b = 0: one scale per member
    platt            scale and shift per member
    platt_by_round   separate Platt maps for round 1 (adjacent frames) and round 2 (gap-2 edges)
Maps are fitted on same_cell (1) / unrelated (0) edges: round 1 on every adjacent-frame candidate, round 2 on the
gap-2 edges the solver saw; one map for both rounds unless by round.

Evaluation is nested in the cross-validation folds of the dataset: the wells of fold k are scored with maps fitted
on the answer sheets of the other folds only, so no well is scored by a map that saw its labels. Populations as in
evaluate_answers: round1_all, round1_decision (0.02 <= p <= 0.98 of the tracker's p, so every method is scored on the
same edges) and round2_solver; error coverage of the tracker's false / missed links ranked by closeness to 0.5 and by
member disagreement of the mapped probabilities. The tracker's decisions are those of the recorded run; what a
recalibrated tracker would decide is stage 1.2 (inference on the server).

calibration.json holds the maps fitted on all folds, for the tracker. A member index pools the fold models of one
seed, and the final tracker is again another set of models: the maps are estimated on held-out answers, as in use.

    python -m models.gnn_tracking.fit_calibration --dataset scripta_data_refine \
        --run development/stage0_evaluation/scripta_data_refine/gnn_tracking/cv_v002/train/oof/gnn_tracking \
        --output-root development/stage1_calibration/platt_scripta_data_refine_gnn_tracking_cv_v002

Writes into a new --output-root: metrics.json, reliability.csv, calibration.json and figures/calibration_summary.*
with *_legend.txt. Project evaluation on recorded candidate edges; not official CTC measures.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from models.gnn_tracking.evaluate_answers import BINS, POPULATIONS, concat, load_well, masks
from utils.calibration import (apply_platt, calibration_report, coverage_area, coverage_curve, fit_platt,
                               mutual_information, reliability)
from utils.config import read_config, resolve_dataset
from utils.lineage import SAME_CELL, UNRELATED
from utils.run_log import log_path, log_root, run_logged

METHODS = ('none', 'temperature', 'platt', 'platt_by_round')


def fit_maps(e, relation, method):
    """-> {'1': [(a, b) per member], '2': [...]} fitted on the given answers."""
    members = e['logits'].shape[1]
    if method == 'none':
        return {r: [(1.0, 0.0)] * members for r in ('1', '2')}
    m = masks(e, relation)
    sets = {'1': m['round1_all'], '2': m['round2_solver']}
    if method != 'platt_by_round':
        sets = {r: m['round1_all'] | m['round2_solver'] for r in ('1', '2')}
    y = relation == SAME_CELL
    return {r: [fit_platt(e['logits'][sel, k], y[sel], shift=method != 'temperature') for k in range(members)]
            for r, sel in sets.items()}


def mapped(e, maps):
    """Member probabilities [E, M] after the maps of each edge's round."""
    q = np.empty(e['logits'].shape)
    for r in ('1', '2'):
        sel = e['round'] == int(r)
        for k, ab in enumerate(maps[r]):
            q[sel, k] = apply_platt(e['logits'][sel, k], ab)
    return q


def score(e, relation, q):
    """Calibration of the mean mapped probability on the fixed populations + error coverage."""
    p, m, y = q.mean(1), masks(e, relation), relation == SAME_CELL
    out = dict(calibration={pop: calibration_report(p[m[pop]], y[m[pop]], BINS) for pop in POPULATIONS}, coverage={})
    for pop, error_relation in (('false_links', UNRELATED), ('missed_links', SAME_CELL)):
        sel, err = m[pop], relation == error_relation
        out['coverage'][pop] = {name: coverage_area(coverage_curve(s, err[sel])) for name, s in
                                (('closeness', np.minimum(p[sel], 1 - p[sel])), ('mutual_info', mutual_information(q[sel])))}
    return out


def nested(parts, folds, method):
    """Score every well with maps fitted on the other folds' wells -> (pooled edges, relation, q, maps per fold)."""
    es, rels, qs, per_fold = [], [], [], []
    for k, fold in enumerate(folds):
        held = [w for w in fold if w in parts]
        train = [p for w, p in parts.items() if w not in fold]
        if not held or not train:
            continue
        e_fit, rel_fit = concat(train)
        maps = fit_maps(e_fit, rel_fit, method)
        per_fold.append(dict(fold=k, wells=held, maps=maps))
        e, rel = concat([parts[w] for w in held])
        es.append(e); rels.append(rel); qs.append(mapped(e, maps))
    e = {key: np.concatenate([x[key] for x in es]) for key in es[0]}
    return e, np.concatenate(rels), np.concatenate(qs), per_fold


def plot(output_base, rows, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.metric_plots import EDGE, INK, MM, RUN_COLORS, _panel_label, _style
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 62 * MM))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.82, wspace=0.35)
    for ax, pop, name in ((axes[0], 'round1_decision', 'Round 1, decision band'), (axes[1], 'round2_solver', 'Round 2, solver edges')):
        ax.plot([0, 1], [0, 1], color=EDGE, lw=0.6, ls='--', zorder=1)
        for i, method in enumerate(METHODS):
            pts = [r for r in rows if r['population'] == pop and r['method'] == method]
            ax.plot([r['mean_p'] for r in pts], [r['frac_pos'] for r in pts], marker='o', ms=2.0, lw=0.8,
                    color=RUN_COLORS[i], mec=INK, mew=0.3, label=method.replace('_', ' '), zorder=3)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted p', ylabel='Fraction same cell', title=name)
    axes[0].legend(loc='upper left')
    for ax, letter in zip(axes, 'ab'):
        _panel_label(fig, ax, letter)
    fig.text(0.1, 0.975, title, fontsize=7.5, fontweight='bold', ha='left', va='top')
    written = []
    for suffix, kwargs in (('.pdf', {}), ('.svg', {}), ('.png', dict(dpi=600))):
        fig.savefig(output_base.with_suffix(suffix), **kwargs)
        written.append(output_base.with_suffix(suffix))
    plt.close(fig)
    legend = (f'a, b, Reliability of the ensemble edge probability after per-member recalibration (equal-mass bins, '
              f'{BINS} per population; dashed, perfect calibration), nested in the cross-validation folds: every well '
              f'is scored with maps fitted on the other folds. a, adjacent-frame candidates with 0.02 <= p <= 0.98 of '
              f'the uncalibrated tracker; b, gap-2 edges offered to the solver. none, the tracker\'s probability; '
              f'temperature, one scale per member; platt, scale and shift per member; platt by round, separate maps '
              f'for the two rounds. Positives, same cell; negatives, unrelated cells. Project association evaluation '
              f'on fixed detections, not official CTC measures.')
    legend_path = output_base.parent / f'{output_base.name}_legend.txt'
    legend_path.write_text(legend + '\n', encoding='utf-8')
    return written + [legend_path]


def run(args, wells, folds):
    parts = {w: load_well(args.run, args.data_root, w) for w in wells}
    report = dict(run=str(Path(args.run).resolve()), dataset=args.dataset_name, wells=wells, folds=folds, bins=BINS,
                  members=int(next(iter(parts.values()))['edges']['logits'].shape[1]),
                  scope='project association evaluation on recorded candidate edges; not official CTC measures',
                  methods={})
    rows, final = [], {}
    for method in METHODS:
        e, rel, q, per_fold = nested(parts, folds, method)
        report['methods'][method] = dict(nested=score(e, rel, q), folds=per_fold)
        m, y = masks(e, rel), rel == SAME_CELL
        for pop in POPULATIONS:
            rows += [dict(method=method, population=pop, bin=b, **r) for b, r in enumerate(reliability(q.mean(1)[m[pop]], y[m[pop]], BINS))]
        e_all, rel_all = concat(list(parts.values()))
        final[method] = fit_maps(e_all, rel_all, method)
    out = args.output_root
    out.mkdir(parents=True)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'calibration.json').write_text(json.dumps(dict(
        run=report['run'], dataset=args.dataset_name, wells=wells, form='p_m = sigmoid(a * z_m + b); ensemble p = mean '
        'over members; maps per round ("1" adjacent frames, "2" gap-2 edges), one (a, b) per member', maps=final), indent=2),
        encoding='utf-8')
    with (out / 'reliability.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plots:
        (out / 'figures').mkdir()
        for path in plot(out / 'figures' / 'calibration_summary', rows, f'{Path(args.run).name} · out-of-fold · {args.dataset_name}'):
            print('saved', path, flush=True)
    print(f"\nnested over {len(folds)} folds ({report['members']} members); ECE / NLL / Brier, mean p vs fraction same cell:")
    for pop in POPULATIONS:
        print(f'  {pop}')
        for method in METHODS:
            c = report['methods'][method]['nested']['calibration'][pop]
            print(f"    {method:15s} n={c['n']:7d}  ECE {c['ece']:.4f}  NLL {c['nll']:.4f}  Brier {c['brier']:.4f}  "
                  f"p {c['mean_p']:.3f} / {c['frac_pos']:.3f}")
    print('  error coverage area (closeness / disagreement)')
    for method in METHODS:
        cv = report['methods'][method]['nested']['coverage']
        print(f"    {method:15s} false links {cv['false_links']['closeness']:.3f} / {cv['false_links']['mutual_info']:.3f}   "
              f"missed links {cv['missed_links']['closeness']:.3f} / {cv['missed_links']['mutual_info']:.3f}")
    print('saved', out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Default: project config.json')
    parser.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
    parser.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
    parser.add_argument('--run', type=Path, required=True, help='Out-of-fold folder with <well>/edges.npz + instance_tracks.csv')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for the calibration')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    wells = sorted(p.parent.name for p in args.run.glob('*/edges.npz'))
    folds = [[w for w in fold if w in wells] for fold in spec['folds']]
    if not wells or sum(len(f) for f in folds) < len(wells) or sum(bool(f) for f in folds) < 2:
        parser.error('every well of --run must lie in a fold of the dataset, and at least two folds are needed')
    if set(wells) & set(spec['held_out']):
        parser.error(f'test wells must not be used for calibration: {sorted(set(wells) & set(spec["held_out"]))}')
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_fit_calibration_{spec["name"]}')
    return run_logged(log, run, args, wells, folds, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
