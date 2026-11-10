"""Inputs of the paper figures: development/final_results_v2 (read only) and the CTC data under data/ (read only).

Design (2026-11-03): ProGCT is trained on scripta (no divisions) and the RAW264.7 videos of DynamicNuclearNet
(divisions, train60-90); HeLa is never trained on. Method keys are those of style.METHODS; every key has at most one
row per (dataset, well) in ctc_per_well.csv, whose method column holds the key.
  scripta  15 wells = 12 development wells out-of-fold (3 folds) + 3 test wells (final models), for ours and
           Trackastra ft; zero-shot models and the ProGCT trained on RAW264.7 / HeLa only ran on all 15 wells.
  dnn      15 DynamicNuclearNet videos never trained on: val17-26 and the official test videos test07-11
           (test07-08 are 3T3/HEK293, the rest RAW264.7 / val, cell line not recorded).
  hela     Fluo-N2DL-HeLa 01, 02, never trained on by ProGCT or Trackastra general_2d (Trackastra ctc was).
Our model (and its binary variant) is read in the division mode everywhere: it is never told whether a dataset has
divisions. On scripta it divides nowhere and its CTC values equal those of the no-division mode well by well.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.lineage import CONTINUATION, UNKNOWN_OFFSET, read_lineage  # noqa: E402

FINAL = ROOT / 'development' / 'final_results_v2'
DATA = ROOT / 'data'
SCRIPTA = DATA / 'scripta_data_refine2'
DNN = DATA / 'DynamicNuclearNet'
HELA = DATA / 'Fluo-N2DL-HeLa'
TEST_WELLS = ['r16c02', 'r01c20', 'r07c07']
DNN_VIDEOS = [f'val{i}' for i in range(17, 27)] + ['test09', 'test10', 'test11']   # the 13 compared videos
DNN_OFFICIAL = ['test07', 'test08', 'test09', 'test10', 'test11']                   # official DNN test videos
DNN_OTHER_LINE = ['test07', 'test08']                                               # 3T3/HEK293: no RAW264.7
CACHE = Path(__file__).resolve().parent / 'out' / 'cache'

# method key -> per-dataset folder of per-well track outputs (<folder>/<well>/instance_tracks.csv)
TRACKS = {
    'scripta': {
        'ours': 'scripta/progct/division', 'ours_nodiv': 'scripta/progct/tracking',
        'binary': 'scripta/progct_binary/division', 'raw': 'scripta/progct_raw_only/division',
        't1': 'scripta/progct_hela_only/division',
        'ta_ft_div': 'scripta/trackastra_ft/greedy', 'ta_ft': 'scripta/trackastra_ft/greedy_nodiv',
        'ta_ft_dt3': 'scripta/trackastra_ft/dt3_greedy_nodiv',
        'ta_zs_div': 'scripta/trackastra_zs/greedy', 'ta_zs': 'scripta/trackastra_zs/greedy_nodiv',
        'ta_zs_dt3': 'scripta/trackastra_zs/dt3_greedy_nodiv',
        'ctgnn0': 'scripta/celltrack_gnn/seed0', 'ctgnn1': 'scripta/celltrack_gnn/seed1', 'classical': 'scripta/classical',
    },
    'dnn': {'ours': 'dnn/progct/division', 'binary': 'dnn/progct_binary/division', 'raw': 'dnn/progct_raw_only/division',
            'ta_ft_div': 'dnn/trackastra_ft/greedy', 'ta_zs_div': 'dnn/trackastra_zs/greedy',
            'ta_zs_ctc': 'dnn/trackastra_zs_ctc/greedy'},
    'hela': {'ours': 'hela/progct/division', 'binary': 'hela/progct_binary/division', 'ta_ft_div': 'hela/trackastra_ft/greedy',
             'ta_zs_div': 'hela/trackastra_zs/greedy', 'ta_zs_ctc': 'hela/trackastra_zs_ctc/greedy'},
}
GAP_KEYS = ['ours', 'ta_ft', 'ta_zs', 'ta_ft_dt3', 'ta_zs_dt3']        # figure 2

_ctc = None


def ctc_table():
    global _ctc
    if _ctc is None:
        _ctc = pd.read_csv(FINAL / 'ctc_per_well.csv', dtype={'well': str})
        _ctc['div_recall'] = _ctc['tp_div(0)'] / (_ctc['tp_div(0)'] + _ctc['fn_div(0)'])
    return _ctc


def has(key, dataset='scripta'):
    t = ctc_table()
    return bool(((t.dataset == dataset) & (t.method == key)).any())


def wells_of(key, dataset, wells=None):
    """Per-well CTC rows of one method on one dataset (index = well)."""
    t = ctc_table()
    rows = t[(t.dataset == dataset) & (t.method == key)].set_index('well').sort_index()
    if rows.empty:
        raise KeyError(f'{key}: no {dataset} rows')
    return rows if wells is None else rows.loc[wells]


def scripta_wells(key, wells=None):
    return wells_of(key, 'scripta', wells)


def dnn_wells(key, videos=None):
    return wells_of(key, 'dnn', DNN_VIDEOS if videos is None else videos)


def hela_wells(key):
    return wells_of(key, 'hela')


def all_wells():
    return sorted(scripta_wells('ours').index)


def paired(a, b):
    """a, b: Series over the same wells -> dict(n, a_higher, b_higher, p) with a two-sided Wilcoxon signed-rank."""
    a, b = a.align(b, join='inner')
    d = a - b
    p = wilcoxon(a, b).pvalue if (d != 0).any() else float('nan')
    return dict(n=len(d), a_higher=int((d > 0).sum()), b_higher=int((d < 0).sum()), p=float(p))


# calibration_mi3: the same evaluations re-run on 2026-11-10 with the MI ranking on the recorded three-class MI (the
# paper's definition) instead of the members' binary MI; ECE and reliability are unchanged. scripta_cv_oof_g1 was not
# re-run (its tracking output is on the server) and is used for its ECE only, which the change does not touch.
BINARY_MI_ONLY = {'scripta_cv_oof_g1'}


def calibration_folder(name):
    """final_results_v2/calibration_mi3/<name>, or calibration/<name> for BINARY_MI_ONLY."""
    return FINAL / ('calibration' if name in BINARY_MI_ONLY else 'calibration_mi3') / name


def answers(name):
    """evaluate_answers output of calibration_folder(name): (metrics overall, reliability, coverage)."""
    folder = calibration_folder(name)
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))['overall']
    return metrics, pd.read_csv(folder / 'reliability.csv'), pd.read_csv(folder / 'coverage.csv')


def track_folder(key, well, dataset='scripta'):
    folder = FINAL / TRACKS[dataset][key] / well
    if not (folder / 'instance_tracks.csv').is_file():
        raise FileNotFoundError(f'{key}: no {dataset} output for {well}')
    return folder


def division_events(folder):
    """Predicted divisions of one well folder, counted as utils.lineage reads lineage: a mother with exactly two
    daughters is one division; one daughter is a continuation (not counted); three or more is no valid division
    (counted apart). A folder without lineage.json (a tracker run without divisions) has none.
    -> dict(divisions, invalid)."""
    path = Path(folder) / 'lineage.json'
    if not path.is_file():
        return dict(divisions=0, invalid=0)
    children = Counter(json.loads(path.read_text(encoding='utf-8')).values()).values()
    return dict(divisions=sum(n == 2 for n in children), invalid=sum(n >= 3 for n in children))


def divisions(key, wells, dataset='scripta'):
    """Divisions summed over wells (on scripta all of them are false: its ground truth has none)."""
    total = Counter()
    for well in wells:
        total.update(division_events(track_folder(key, well, dataset)))
    return dict(divisions=total['divisions'], invalid=total['invalid'])


def test_divisions(key):
    return divisions(key, TEST_WELLS)


def scripta_divisions(key, wells):
    return divisions(key, wells)


def expected_errors(key, wells, dataset='scripta'):
    """Expected and measured counts on the links between detections with known identity (labels < 50000), as in
    sec. 4 of the paper: expected false = sum(1 - p) over the links made; expected missed = sum p over the candidate
    edges that entered the solver and were not chosen. -> dict(expected_false, expected_missed, links, candidates)."""
    out = Counter()
    for well in wells:
        folder = track_folder(key, well, dataset)
        links = pd.read_csv(folder / 'links.csv')
        known = (links.src_label < UNKNOWN_OFFSET) & (links.dst_label < UNKNOWN_OFFSET)
        out['expected_false'] += float((1 - links.p[known]).sum())
        out['links'] += int(known.sum())
        z = np.load(folder / 'edges.npz')
        left = z['in_solver'].astype(bool) & ~z['chosen'].astype(bool) & (z['src_label'] < UNKNOWN_OFFSET) \
            & (z['dst_label'] < UNKNOWN_OFFSET)
        out['expected_missed'] += float(z['p'][left].sum())
        out['candidates'] += int(left.sum())
    return dict(out)


def predicted_links(tracks):
    """instance_tracks.csv -> {(src, dst)}: consecutive detections (frame, label) of each predicted track."""
    links = set()
    for _, g in tracks.sort_values('frame_index').groupby('track_id'):
        nodes = list(zip(g.frame_index.astype(int), g.instance_id.astype(int)))
        links.update(zip(nodes[:-1], nodes[1:]))
    return links


def gt_links(well, observations):
    """{(src, dst): g} of the ground-truth continuation links between observed detections."""
    lineage = read_lineage(SCRIPTA / f'{well}_GT' / 'TRA' / 'man_track.txt', observations)
    return {(l.src, l.dst): l.g for l in lineage.links if l.kind == CONTINUATION}


def gap_bin(g):
    return 'g1' if g == 1 else ('g2' if g == 2 else 'g3+')


def gap_counts(refresh=False):
    """Per well and method (GAP_KEYS): GT continuation links, missed and false links by frame gap (g1, g2, g3+).
    A false link joins two known detections (labels < 50000) that are not a GT continuation. Cached as csv."""
    cache = CACHE / 'gap_links.csv'
    if cache.is_file() and not refresh:
        table = pd.read_csv(cache)
        if set(table.method) == set(GAP_KEYS):
            return table
    rows = []
    for well in all_wells():
        tracks = {k: pd.read_csv(track_folder(k, well) / 'instance_tracks.csv') for k in GAP_KEYS}
        ref = tracks['ours']
        gt = gt_links(well, list(zip(ref.frame_index.astype(int), ref.instance_id.astype(int))))
        for key, df in tracks.items():
            pred = predicted_links(df)
            c = Counter()
            for link, g in gt.items():
                c[f'gt_{gap_bin(g)}'] += 1
                if link not in pred:
                    c[f'missed_{gap_bin(g)}'] += 1
            for s, d in pred:
                if s[1] < UNKNOWN_OFFSET and d[1] < UNKNOWN_OFFSET and (s, d) not in gt:
                    c[f'false_{gap_bin(d[0] - s[0])}'] += 1
            rows.append(dict(well=well, method=key, split='test' if well in TEST_WELLS else 'dev', **c))
    table = pd.DataFrame(rows).fillna(0)
    for col in [f'{k}_{b}' for k in ('gt', 'missed', 'false') for b in ('g1', 'g2', 'g3+')]:
        if col not in table:
            table[col] = 0
    CACHE.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


def fmt_p(p):
    return 'n/a' if not np.isfinite(p) else (f'{p:.1e}' if p < 1e-3 else f'{p:.3f}')
