"""Collect every final result of the paper into one folder for drawing figures and extracting numbers.
    python paper/figures/export_results.py [--out development/paper_results] [--figures DIR]
--figures: the folder with the rendered paper figures and tables (default: the paper repository's figures/out).
The output folder must not exist yet. Inputs are only read: development/final_results_v2, development/stage8_argument,
development/figure_refs, data/ (masks, for the data statistics).

Written:
  README.md               what every file is and where it comes from
  key_numbers.csv         every number the paper quotes: section, quantity, value, scope, source
  summary_by_method.csv   per method, dataset and well set: means of LNK / CT / TRA / BC, sums of the AOGM terms and
                          division counts (the values behind every table)
  ctc_per_well.csv        all CTC rows per well / video (copy of final_results_v2)
  calibration/            evaluate_answers outputs (metrics.json, reliability.csv, coverage.csv) of the main runs (MI
                          ranking on the recorded three-class MI: final_results_v2/calibration_mi3) and of
                          the retracks (no cross-frame check, thresholds 0.2-0.6)
  per_link/               per-gap link errors (figure 2) and the round-1 division probabilities (figure 5)
  data_statistics.json    what makes the data hard, from the masks: missed detections, unknown identities, nuclei per
                          frame, touching nuclei, overlap of the same cell between consecutive frames
  figures/                the rendered paper figures (pdf, png) with their statistics (*_legend.txt) and tables (.tex)
  figure_refs/            real crops and curves, and the drawing prompts
Per-well tracks (instance_tracks.csv, edges.npz, links.csv) stay in final_results_v2 (about 0.5 GB)."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from skimage.measure import regionprops

import data
import fig_division

DEV = data.ROOT / 'development'
UNKNOWN = 50000
SETS = {
    'scripta': {'dev12': [w for w in data.all_wells() if w not in data.TEST_WELLS], 'test3': data.TEST_WELLS,
                'all15': data.all_wells()},
    'dnn': {'videos13': data.DNN_VIDEOS, 'official5': data.DNN_OFFICIAL, 'other_line2': data.DNN_OTHER_LINE},
    'hela': {'seq01': ['01'], 'seq02': ['02'], 'both': ['01', '02']},
}


def summary_by_method():
    t = data.ctc_table()
    rows = []
    for (dataset, method), g in t.groupby(['dataset', 'method']):
        g = g.set_index('well')
        for name, wells in SETS[dataset].items():
            if not set(wells) <= set(g.index):
                continue
            x = g.loc[wells]
            tp, fn = x['tp_div(0)'].sum(), x['fn_div(0)'].sum()
            rows.append(dict(dataset=dataset, method=method, wells=name, n=len(x), LNK=x.LNK.mean(), CT=x.CT.mean(),
                             TRA=x.TRA.mean(), BC0=x['BC(0)'].mean(), ED=x.AOGM_ED.sum(), EA=x.AOGM_EA.sum(),
                             EC=x.AOGM_EC.sum(), tp_div=tp, fp_div=x['fp_div(0)'].sum(), fn_div=fn,
                             div_recall=tp / (tp + fn) if tp + fn else np.nan))
    return pd.DataFrame(rows)


class Numbers:
    def __init__(self):
        self.rows = []

    def add(self, section, quantity, value, scope, source):
        self.rows.append(dict(section=section, quantity=quantity, value=value, scope=scope, source=source))


def key_numbers(stats):
    k = Numbers()
    dev, test, every = SETS['scripta']['dev12'], data.TEST_WELLS, data.all_wells()
    s = lambda key, wells: data.scripta_wells(key, wells)
    d = lambda key, videos=None: data.dnn_wells(key, videos)
    ctc = 'final_results_v2/ctc_per_well.csv'

    labels = [('wells', 'wells'), ('tracks', 'GT tracks'), ('tracks_with_gap', 'tracks with a missed detection'),
              ('share_tracks_with_gap', 'share of tracks with a missed detection'),
              ('share_unknown_detections', 'share of detections without GT identity'),
              ('cells_per_frame', 'nuclei per frame, min-max'), ('touching', 'share of nuclei touching a neighbour'),
              ('pairs', 'same-cell pairs in consecutive frames'),
              ('iou_below_half', 'share of same-cell pairs with IoU < 0.5'),
              ('no_overlap', 'share of same-cell pairs with no overlap'), ('iou_median', 'median IoU of same-cell pairs'),
              ('nearest_is_other', 'share whose nearest next-frame detection is another cell')]
    for dataset, scope in (('scripta', 'Scripta (refine2), 15 wells'), ('dnn', 'DynamicNuclearNet, 13 evaluated videos')):
        for key, q in labels:
            v = stats[dataset][key]
            k.add('data', q, f'{v[0]}-{v[1]}' if key == 'cells_per_frame' else v, scope, 'data_statistics.json (masks)')

    for key in ('ours', 'ta_ft', 'ta_zs', 'ta_ft_div', 'ta_zs_div'):
        k.add('scripta', f'LNK mean, {key}', s(key, every).LNK.mean(), '15 wells (dev OOF + test)', ctc)
        k.add('scripta', f'CT mean, {key}', s(key, every).CT.mean(), '15 wells (dev OOF + test)', ctc)
        if key != 'ours':
            r = data.paired(s('ours', every).LNK, s(key, every).LNK)
            k.add('scripta', f'LNK wells ours higher vs {key}', f"{r['a_higher']}/{r['n']}", '15 wells', ctc)
            k.add('scripta', f'LNK Wilcoxon p, ours vs {key}', r['p'], '15 wells', ctc)
    for key in ('ours', 'binary', 'ta_ft', 'ta_zs', 'ta_ft_div', 'ta_zs_div', 'ctgnn0', 'ctgnn1', 'classical'):
        k.add('scripta', f'LNK / CT test wells, {key}', f"{s(key, test).LNK.mean():.4f} / {s(key, test).CT.mean():.3f}",
              '3 test wells', ctc)
    for key in ('ours', 'raw', 'binary', 'ta_ft_div', 'ta_zs_div', 'ctgnn0', 'ctgnn1'):
        k.add('divisions', f'false divisions, test wells, {key}', data.test_divisions(key)['divisions'],
              'Scripta 3 test wells (GT has none)', 'final_results_v2 lineage.json')
    for key in ('ours', 'raw', 'ta_ft_div', 'ta_zs_div'):
        k.add('divisions', f'false divisions, 15 wells, {key}', data.scripta_divisions(key, every)['divisions'],
              'Scripta 15 wells (GT has none)', 'final_results_v2 lineage.json')
    for key in ('ctgnn0', 'ctgnn1'):
        k.add('divisions', f'mothers with >= 3 daughters, test wells, {key}', data.test_divisions(key)['invalid'],
              'Scripta 3 test wells', 'final_results_v2 lineage.json')
    for key in ('ours', 'raw', 'binary', 'ta_ft_div', 'ta_zs_div', 'ta_zs_ctc'):
        x = d(key)
        tp, fn = x['tp_div(0)'].sum(), x['fn_div(0)'].sum()
        for q, v in (('LNK', x.LNK.mean()), ('BC(0)', x['BC(0)'].mean()), ('division recall', tp / (tp + fn)),
                     ('false divisions', int(x['fp_div(0)'].sum())), ('GT divisions', int(tp + fn)),
                     ('ED', int(x.AOGM_ED.sum())), ('EA', int(x.AOGM_EA.sum()))):
            k.add('dnn', f'{q}, {key}', v, 'DNN 13 videos (val17-26, test09-11)', ctc)

    g = data.gap_counts().groupby(['method', 'split']).sum(numeric_only=True)
    a = data.gap_counts().groupby('method').sum(numeric_only=True)
    gl = 'per_link/gap_links.csv'
    k.add('gaps', 'GT links by gap g=1 / g=2 / g>=3', f"{a.loc['ours', 'gt_g1']:.0f} / {a.loc['ours', 'gt_g2']:.0f} / "
          f"{a.loc['ours', 'gt_g3+']:.0f}", '15 Scripta wells', gl)
    k.add('gaps', 'GT links across >= 2 missed frames, development wells', g.loc[('ours', 'dev'), 'gt_g3+'], '12 dev wells', gl)
    k.add('gaps', 'share of g=2 links rejoined, ours', 1 - a.loc['ours', 'missed_g2'] / a.loc['ours', 'gt_g2'], '15 wells', gl)
    k.add('gaps', 'false g=2 links, ours', a.loc['ours', 'false_g2'], '15 wells', gl)
    for key in ('ours', 'ta_ft', 'ta_zs'):
        k.add('gaps', f'missed / false adjacent (g=1) links, {key}',
              f"{a.loc[key, 'missed_g1']:.0f} / {a.loc[key, 'false_g1']:.0f}", '15 wells', gl)
    k.add('gaps', 'false gap links (g>=2), Trackastra ft delta_t=3', a.loc['ta_ft_dt3', 'false_g2'] + a.loc['ta_ft_dt3', 'false_g3+'],
          '15 wells', gl)
    k.add('gaps', 'missed adjacent links, Trackastra ft delta_t=3', a.loc['ta_ft_dt3', 'missed_g1'], '15 wells', gl)

    pairs = [('cross-frame check', 'ours_nodiv', 'nogap', 'ours', 'nogap'), ('appearance', 'base', 'noapp', 'base', 'noapp'),
             ('velocity', 'base', 'novel', 'base', 'novel')]
    for name, full, abl, dfull, dabl in pairs:
        r = data.paired(s(full, dev).LNK, s(abl, dev).LNK)
        k.add('ablation', f'LNK full -> without {name}', f"{s(full, dev).LNK.mean():.4f} -> {s(abl, dev).LNK.mean():.4f} "
              f"({r['a_higher']}/{r['n']} wells higher with it, p {r['p']:.1e})", '12 dev wells, OOF', ctc)
        r = data.paired(d(dfull).LNK, d(dabl).LNK)
        k.add('ablation', f'LNK full -> without {name}', f"{d(dfull).LNK.mean():.4f} -> {d(dabl).LNK.mean():.4f} "
              f"({r['a_higher']}/{r['n']} videos higher with it, p {r['p']:.1e})", 'DNN 13 videos', ctc)
    k.add('ablation', 'EA full -> without cross-frame check',
          f"{s('ours_nodiv', dev).AOGM_EA.sum():.0f} -> {s('nogap', dev).AOGM_EA.sum():.0f}", '12 dev wells', ctc)
    k.add('ablation', 'ED full -> without cross-frame check',
          f"{s('ours_nodiv', dev).AOGM_ED.sum():.0f} -> {s('nogap', dev).AOGM_ED.sum():.0f}", '12 dev wells', ctc)
    k.add('ablation', 'BC full -> without appearance', f"{d('base')['BC(0)'].mean():.3f} -> {d('noapp')['BC(0)'].mean():.3f}",
          'DNN 13 videos', ctc)
    k.add('ablation', 'LNK Trackastra ft told (comparison for the cross-frame share of the lead)',
          s('ta_ft', dev).LNK.mean(), '12 dev wells', ctc)
    for ds, videos, scope in (('scripta', every, '15 wells (dev OOF + test)'), ('dnn', data.DNN_VIDEOS, 'DNN 13 videos')):
        r = greedy_vs_exact(ds, videos)
        k.add('solver', 'adjacent same-cell links (exact) / differing links greedy vs exact (only exact + only greedy) '
              '/ frame pairs with a difference', f"{r['links']} / {r['only_exact'] + r['only_greedy']} "
              f"({r['only_exact']} + {r['only_greedy']}) / {r['pairs_differing']} of {r['pairs']}", scope,
              'final_results_v2 edges.npz, round 1 re-decoded greedily (export_results.greedy_vs_exact)')
        for name, (n, only_ens, only_var) in single_member_vs_ensemble(ds, videos).items():
            k.add('ensemble', f'adjacent same-cell links (ensemble) / only ensemble / only {name}',
                  f'{n} / {only_ens} / {only_var}', scope,
                  'final_results_v2 edges.npz, round 1 re-decoded per member (export_results.single_member_vs_ensemble)')
    k.add('ensemble', 'false + missed edges (ED + EA), ours / Trackastra ft told',
          f"{s('ours', every).AOGM_ED.sum() + s('ours', every).AOGM_EA.sum():.0f} / "
          f"{s('ta_ft', every).AOGM_ED.sum() + s('ta_ft', every).AOGM_EA.sum():.0f}", '15 wells (dev OOF + test)', ctc)
    for key, tau in (('tau0.2', 0.2), ('tau0.3', 0.3), ('ours_nodiv', 0.4), ('tau0.5', 0.5), ('tau0.6', 0.6)):
        x = s(key, dev)
        k.add('threshold', f'tau {tau}: LNK / CT / ED / EA', f"{x.LNK.mean():.4f} / {x.CT.mean():.3f} / "
              f"{x.AOGM_ED.sum():.0f} / {x.AOGM_EA.sum():.0f}", '12 dev wells, OOF', ctc)
        y = d('ours' if key == 'ours_nodiv' else key)
        k.add('threshold', f'tau {tau}: LNK / BC', f"{y.LNK.mean():.4f} / {y['BC(0)'].mean():.3f}", 'DNN 13 videos', ctc)

    for name, key, wells, dataset, model in (('dev wells OOF', 'scripta_cv_oof_tta', dev, 'scripta', 'ours_nodiv'),
                                            ('test wells', 'scripta_test', test, 'scripta', 'ours_nodiv'),
                                            ('DNN 15 videos', 'dnn', data.DNN_VIDEOS + data.DNN_OTHER_LINE, 'dnn', 'ours'),
                                            ('HeLa (never trained on)', 'hela', None, None, None)):
        m = data.answers(key)[0]
        c, cov = m['calibration']['round1_decision'], m['coverage']
        src = f'calibration/{key}/metrics.json'
        k.add('confidence', 'undecided edges (0.02 < p < 0.98): n / mean p / observed / ECE',
              f"{c['n']} / {c['mean_p']:.3f} / {c['frac_pos']:.3f} / {c['ece']:.3f}", name, src)
        k.add('confidence', 'false links found in the 5% most uncertain links made (MI)',
              cov['false_links']['mutual_info']['at_percent']['5'], name, src)
        k.add('confidence', 'missed candidates found in the 5% most uncertain candidates left out (MI)',
              cov['missed_links']['mutual_info']['at_percent']['5'], name, src)
        k.add('confidence', 'measured false links / missed candidates',
              f"{cov['false_links']['errors']} / {cov['missed_links']['errors']}", name, src)
        if model:
            e = data.expected_errors(model, wells, dataset)
            k.add('confidence', 'expected false links / missed candidates',
                  f"{e['expected_false']:.0f} / {e['expected_missed']:.0f}", name, 'final_results_v2 links.csv, edges.npz')
    m = data.answers('scripta_cv_oof_tta')[0]['calibration']
    k.add('confidence', 'share of round-1 edges that are undecided', m['round1_decision']['n'] / m['round1_all']['n'],
          'dev wells OOF', 'calibration/scripta_cv_oof_tta/metrics.json')
    for base, tta, name in (('scripta_cv_oof_base', 'scripta_cv_oof_tta', 'dev wells OOF'), ('dnn_base', 'dnn', 'DNN')):
        for pop in ('false_links', 'missed_links'):
            k.add('confidence', f'MI coverage area, {pop}, no views -> views',
                  f"{data.answers(base)[0]['coverage'][pop]['mutual_info']['area']:.3f} -> "
                  f"{data.answers(tta)[0]['coverage'][pop]['mutual_info']['area']:.3f}", name, f'calibration/{tta}')

    h = fig_division.missed_divisions()
    for q in ('divisions', 'missed', 'near', 'continued', 'continued_flagged', 'signal', 'links', 'flagged'):
        k.add('hela', f'missed-division analysis: {q}', h[q], 'HeLa 01+02, final model',
              'fig_division.missed_divisions (edges.npz, links.csv)')
    edges = fig_division.edges()
    for population, relation in (('dnn', 'mother_daughter'), ('dnn', 'same_cell'), ('scripta', 'same_cell'),
                                 ('scripta_raw', 'same_cell'), ('hela', 'mother_daughter'), ('hela', 'same_cell')):
        p = edges.p_div[(edges.population == population) & (edges.relation == relation)].to_numpy()
        k.add('divisions', f'round-1 edges with d >= 0.5, {population}, {relation}', f'{int((p >= 0.5).sum())} of {len(p)}',
              population, 'per_link/division_edges.csv')
    for q, v in compute().items():
        k.add('compute', q, v, 'final models, server (NVIDIA GH200)', 'stage7_raw/final/checkpoints, stage7_raw/logs')
    cal = 'stage10_caliban/ctc (models/caliban, evaluate_ctc)'
    for key in ('ours', 'caliban_ft', 'caliban_zs'):
        t = s(key, test)
        k.add('caliban', f'test wells LNK / CT / ED / EA / false divisions, {key}',
              f'{t.LNK.mean():.4f} / {t.CT.mean():.3f} / {int(t.AOGM_ED.sum())} / {int(t.AOGM_EA.sum())} / '
              f'{data.test_divisions(key)["divisions"]}', '3 scripta test wells', cal if key != 'ours' else ctc)
    for name, videos in (('13 videos', None), ('test09-11', ['test09', 'test10', 'test11'])):
        for key in ('ours', 'caliban_zs'):
            t = d(key, videos)
            k.add('caliban', f'DNN LNK / CT / BC(0) / ED / EA / false divisions, {key}',
                  f'{t.LNK.mean():.4f} / {t.CT.mean():.3f} / {t["BC(0)"].mean():.3f} / {int(t.AOGM_ED.sum())} / '
                  f'{int(t.AOGM_EA.sum())} / {int(t["fp_div(0)"].sum())}', f'DNN {name}', cal if key != 'ours' else ctc)
    import re
    import torch
    run = data.CALIBAN / 'finetune_final'
    best = next(run.glob('**/runs/*/best_model.pt'))
    state = torch.load(best, map_location='cpu')['model_state_dict']
    log = (run / 'caliban_final.log').read_text(encoding='utf-8', errors='replace')
    k.add('caliban', 'parameters (fine-tuned model)', sum(v.numel() for n, v in state.items() if v.is_floating_point()
                                                          and 'running' not in n), 'Caliban ft', str(best))
    k.add('caliban', 'fine-tuning hours, 50 epochs, batch 8, one NVIDIA H100 NVL (94 GB)',
          re.search(r'Training completed in ([\d.]+) hours', log).group(1), 'Caliban ft', 'caliban_final.log')
    gaps = data.dnn_gap_counts()
    for key, c in gaps.items():
        k.add('caliban', f'DNN links by gap g1 / g2 / g3+: GT, missed, false, {key}',
              ', '.join('/'.join(str(c[f'{kind}_{b}']) for b in ('g1', 'g2', 'g3+')) for kind in ('gt', 'missed', 'false')),
              'DNN 13 videos', 'instance_tracks.csv + man_track.txt (data.dnn_gap_counts)')
    return pd.DataFrame(k.rows)


def compute():
    """Model size from the final checkpoints and wall-clock times from the server logs (the GPU model is not in the
    logs: NVIDIA GH200, per the user). Training: every member of the cross-validation and the final models (40
    epochs); the runs that shared a GPU with others took about twice as long. Tracking: the retracks with all 3 x 3
    views of the 20 evaluation videos (3 scripta test wells, 15 DNN, 2 HeLa), both tracking modes, data from the
    cache (thresholds 0.2-0.6 differ only in the decision)."""
    import re
    import torch
    stage7 = DEV / 'stage7_raw'
    count = lambda f: sum(v.numel() for v in torch.load(f, map_location='cpu', weights_only=False)['state_dict'].values()
                          if v.dtype.is_floating_point)
    params = [count(f) for f in sorted((stage7 / 'final' / 'checkpoints').glob('final_seed*.pt'))]
    seconds = lambda f: float(re.findall(r'complete after ([\d.]+) s', f.read_text(encoding='utf-8'))[-1])
    logs = stage7 / 'logs' / 'gnn_bayesian'
    train = sorted(seconds(f) / 60 for f in [*logs.glob('cv_lr5e-4_cv_folds012_train_*.log'),
                                             *logs.glob('final_final_train_finals*.log')])
    fast = [m for m in train if m < 90]
    track = [seconds(f) for f in sorted(logs.glob('final_final_retrack_none_tta_t0.*.log'))]
    return {'parameters per member / ensemble': f'{params[0]} / {sum(params)}',
            'training minutes per member (alone on a GPU) / members / all members': f'{min(fast):.0f}-{max(fast):.0f} / '
            f'{len(fast)} / {len(train)} (shared: {min(m for m in train if m >= 90):.0f}-{max(train):.0f} min)',
            'tracking seconds, 20 videos, 3 x 3 views, both modes': f'{min(track):.0f}-{max(track):.0f}'}


def greedy_vs_exact(dataset, videos, tau=0.4):
    """Does ProGCT's gain over Trackastra (greedy) come from its exact assignment? Its recorded round-1 adjacent
    candidates (g = 1, in the solver) of every frame pair are re-decoded greedily: highest p first, a link if p > tau
    and both detections are still free; the detections of a division the exact solver made stay taken. -> the
    same-cell links of the exact solver (sources with one chosen edge), the links only one decoding makes, and the
    frame pairs with any difference. Divisions and the gap-closing round are not re-decoded."""
    links = only_exact = only_greedy = pairs = differing = 0
    for video in videos:
        z = np.load(data.track_folder('ours', video, dataset) / 'edges.npz')
        m = (z['round'] == 1) & (z['g'] == 1) & z['in_solver']
        frame, src, dst, p, chosen = (z[c][m] for c in ('src_frame', 'src_label', 'dst_label', 'p', 'chosen'))
        for t in np.unique(frame):
            k = frame == t
            s, d, q, c = src[k], dst[k], p[k], chosen[k]
            n_chosen = pd.Series(s[c]).value_counts()
            dividing = set(n_chosen.index[n_chosen > 1])
            exact = {(a, b) for a, b in zip(s[c], d[c]) if a not in dividing}
            used_s, used_d = set(dividing), {b for a, b in zip(s[c], d[c]) if a in dividing}
            greedy = set()
            for i in np.argsort(-q, kind='stable'):
                if q[i] <= tau:
                    break
                if s[i] not in used_s and d[i] not in used_d:
                    greedy.add((s[i], d[i]))
                    used_s.add(s[i])
                    used_d.add(d[i])
            links += len(exact)
            only_exact += len(exact - greedy)
            only_greedy += len(greedy - exact)
            pairs += 1
            differing += exact != greedy
    return dict(links=links, only_exact=only_exact, only_greedy=only_greedy, pairs=pairs, pairs_differing=differing)


def single_member_vs_ensemble(dataset, videos, tau=0.4):
    """Does the ensemble (3 members x 3 views) make ProGCT's links, or only its confidence? The recorded round-1
    adjacent candidates (g = 1, in the solver) of every frame pair are re-decoded by the same assignment (solver.assign,
    threshold tau) with the probability of one member alone in the reference view (the recorded SAME logits, columns
    member-major within the first view), and with the 3 members without views; divisions the ensemble made stay
    taken. -> {variant: (ensemble same-cell links, links only the ensemble makes, links only the variant makes)}; the
    9 outputs together reproduce the recorded links (a check). The gap-closing round is not re-run."""
    sys.path.insert(0, str(data.ROOT))
    from models.gnn_bayesian.solver import assign
    sig = lambda x: 1 / (1 + np.exp(-x))
    variants = {'all 9 outputs (check)': lambda L: sig(L).mean(1), '3 members, no views': lambda L: sig(L[:, :3]).mean(1),
                **{f'member {m}, no views': (lambda L, m=m: sig(L[:, m])) for m in range(3)}}
    out = {name: [0, 0, 0] for name in variants}
    for video in videos:
        z = np.load(data.track_folder('ours', video, dataset) / 'edges.npz')
        sel = (z['round'] == 1) & (z['g'] == 1) & z['in_solver']
        frame, src, dst, chosen, logits = (z[c][sel] for c in ('src_frame', 'src_label', 'dst_label', 'chosen', 'logits'))
        for t in np.unique(frame):
            k = frame == t
            s, d, c, L = src[k], dst[k], chosen[k], logits[k]
            n_chosen = pd.Series(s[c]).value_counts()
            dividing = set(n_chosen.index[n_chosen > 1])
            children = {b for a, b in zip(s[c], d[c]) if a in dividing}
            ensemble = {(a, b) for a, b in zip(s[c], d[c]) if a not in dividing}
            keep = np.array([a not in dividing and b not in children for a, b in zip(s, d)])
            if not keep.any():
                continue
            rows, ri = np.unique(s[keep], return_inverse=True)
            cols, ci = np.unique(d[keep], return_inverse=True)
            for name, prob in variants.items():
                p = np.clip(prob(L[keep]), 1e-7, 1 - 1e-7)
                made = {(rows[i], cols[j]) for i, j in assign(-np.log(p), ri, ci, len(rows), len(cols),
                                                               [tau] * len(rows), tau)}
                out[name][0] += len(ensemble)
                out[name][1] += len(ensemble - made)
                out[name][2] += len(made - ensemble)
    return {name: tuple(v) for name, v in out.items()}


def data_statistics():
    """From the masks (data/ read only): Scripta refine2, all 15 wells; DynamicNuclearNet, the 13 evaluated videos."""
    out = {}
    for name, root, seqs in (('scripta', data.SCRIPTA, data.all_wells()), ('dnn', data.DNN, data.DNN_VIDEOS)):
        iou, nn_other, touch, per_frame, frames_of = [], [], [], [], {}
        known = unknown = 0
        for seq in seqs:
            prev = None
            for f in sorted((root / f'{seq}_GT' / 'TRA').glob('man_track*.tif')):
                t = int(f.stem[9:])
                m = tifffile.imread(f)
                props = {r.label: r for r in regionprops(m)}
                per_frame.append(len(props))
                known += sum(l < UNKNOWN for l in props)
                unknown += sum(l >= UNKNOWN for l in props)
                for l in props:
                    if l < UNKNOWN:
                        frames_of.setdefault((seq, l), []).append(t)
                a, b = m[:, :-1], m[:, 1:]
                c, e = m[:-1, :], m[1:, :]
                sel1, sel2 = (a != b) & (a > 0) & (b > 0), (c != e) & (c > 0) & (e > 0)
                touching = set(np.concatenate([a[sel1], b[sel1], c[sel2], e[sel2]]).tolist())
                touch += [l in touching for l in props if l < UNKNOWN]
                if prev is not None and prev[0] == t - 1:
                    pm, pp = prev[1], prev[2]
                    keys = list(props)
                    tree = cKDTree(np.array([props[l].centroid for l in keys]))
                    for l in set(pp) & set(props):
                        if l >= UNKNOWN:
                            continue
                        y0, x0 = min(pp[l].bbox[0], props[l].bbox[0]), min(pp[l].bbox[1], props[l].bbox[1])
                        y1, x1 = max(pp[l].bbox[2], props[l].bbox[2]), max(pp[l].bbox[3], props[l].bbox[3])
                        u, v = pm[y0:y1, x0:x1] == l, m[y0:y1, x0:x1] == l
                        iou.append(np.logical_and(u, v).sum() / np.logical_or(u, v).sum())
                        nn_other.append(keys[tree.query(pp[l].centroid)[1]] != l)
                prev = (t, m, props)
            print(f'  data statistics: {name} {seq}', flush=True)
        gaps = sum((max(ts) - min(ts) + 1) > len(ts) for ts in frames_of.values())
        iou = np.array(iou)
        out[name] = dict(wells=len(seqs), tracks=len(frames_of), tracks_with_gap=int(gaps),
                         share_tracks_with_gap=gaps / len(frames_of), share_unknown_detections=unknown / (known + unknown),
                         cells_per_frame=[int(min(per_frame)), int(max(per_frame))], touching=float(np.mean(touch)),
                         pairs=len(iou), iou_below_half=float((iou < 0.5).mean()), no_overlap=float((iou == 0).mean()),
                         iou_median=float(np.median(iou)), nearest_is_other=float(np.mean(nn_other)))
    out['definitions'] = (
        'tracks: GT labels < 50000; a track has a missed detection if a frame between its first and last frame has no '
        'detection of its label. pairs: the same GT label in frames t and t+1; IoU of its two masks; nearest_is_other: '
        'the detection of frame t+1 closest to the frame-t centroid has another label. touching: the mask shares a '
        'pixel border with another mask. DNN statistics over the 13 evaluated videos (its GT masks are the detections, '
        'so it has no unknown identities; gaps are frames in which the DNN annotation itself has no mask).')
    return out


README = """# Paper results (ProGCT, CVPR 2027)

Generated by `paper/figures/export_results.py` from `development/final_results_v2`, `development/stage8_argument`,
`development/figure_refs` and the masks under `data/` (all read only). Do not edit by hand; rerun the script.

| File | Content |
|---|---|
| `key_numbers.csv` | every number the paper quotes, with its scope (which wells or videos) and its source file |
| `summary_by_method.csv` | per method, dataset and well set: mean LNK, CT, TRA, BC(0); summed ED, EA, EC and divisions |
| `ctc_per_well.csv` | all CTC measures per well or video and method (copy of final_results_v2) |
| `data_statistics.json` | what makes the data hard (from the masks), with the definitions |
| `calibration/` | calibration and error coverage (evaluate_answers): metrics.json, reliability.csv, coverage.csv |
| `per_link/gap_links.csv` | per well and method: GT links, missed and false links by frame gap (figure 2) |
| `per_link/division_edges.csv` | round-1 candidate edges with their GT relation and division probability (figure 5) |
| `figures/` | rendered paper figures (pdf, png), their statistics (`*_legend.txt`) and the LaTeX tables |
| `figure_refs/` | real crops and curves for drawing, and the drawing prompts (`PROMPTS.md`) |

Method keys (as in `paper/figures/style.py`): ours = ProGCT (division mode; on Scripta identical to ours_nodiv),
binary = ProGCT with binary edges, raw = ProGCT trained on RAW264.7 only, t1 = an earlier model trained on HeLa only
(archived, not in the paper), base / noapp / novel = retracks without test-
time views / without appearance / without velocity, nogap = without the cross-frame check, tau0.2-0.6 = other linking
thresholds (0.4 is the main run); ta_ft / ta_zs = Trackastra fine-tuned / zero-shot told that there are no divisions,
ta_ft_div / ta_zs_div = allowed to divide, ta_*_dt3 = delta_t 3, ta_zs_ctc = Trackastra ctc model; ctgnn0/1 =
CellTrack-GNN seeds; classical = rule-based tracker.

Well sets: Scripta dev12 (out of fold), test3 (final models), all15; DNN videos13 (val17-26, test09-11), official5
(test07-11), other_line2 (test07-08, 3T3/HEK293); HeLa seq01, seq02.

Terms: false links are ranked among the links made; missed candidates are candidate edges left out by the solver
that join the same cell (true links without a candidate edge have no probability and are not counted). Calibration
is evaluated, not fitted. CTC measures are on fixed detections with the project ground truth, not leaderboard values.

Per-well tracks (instance_tracks.csv, lineage.json, edges.npz, links.csv) are in `development/final_results_v2`
(about 0.5 GB) and the retracks in `development/stage7_raw/{cv_lr5e-4,final}/retrack_*`.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=DEV / 'paper_results')
    parser.add_argument('--figures', type=Path, default=Path(r'D:\Papers\GNNCellTracking-Paper\figures\out'))
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f'{args.out} exists; choose a new --out')
    args.out.mkdir(parents=True)
    shutil.copy2(data.FINAL / 'ctc_per_well.csv', args.out / 'ctc_per_well.csv')
    for folder in sorted((data.FINAL / 'calibration').iterdir()):       # the evaluations the figures read
        shutil.copytree(data.calibration_folder(folder.name), args.out / 'calibration' / folder.name)
    for folder in sorted((DEV / 'stage8_argument' / 'answers').iterdir()):
        if folder.name != 'logs':
            shutil.copytree(folder, args.out / 'calibration' / f'retrack_{folder.name}')
    (args.out / 'per_link').mkdir()
    data.gap_counts().to_csv(args.out / 'per_link' / 'gap_links.csv', index=False)
    fig_division.edges().to_csv(args.out / 'per_link' / 'division_edges.csv', index=False)
    (args.out / 'figures').mkdir()
    for f in sorted(args.figures.iterdir()):
        if f.suffix in ('.pdf', '.png', '.txt', '.tex'):
            shutil.copy2(f, args.out / 'figures' / f.name)
    shutil.copytree(DEV / 'figure_refs', args.out / 'figure_refs')
    summary_by_method().to_csv(args.out / 'summary_by_method.csv', index=False)
    print('computing the data statistics from the masks (a few minutes)', flush=True)
    stats = data_statistics()
    (args.out / 'data_statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    key_numbers(stats).to_csv(args.out / 'key_numbers.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    (args.out / 'README.md').write_text(README, encoding='utf-8')
    print('written', args.out)


if __name__ == '__main__':
    main()
