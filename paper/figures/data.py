"""Inputs of the paper figures: development/final_results (read only) and the CTC data under data/ (read only).

Method keys are those of style.METHODS. The 15-well scripta comparison joins out-of-fold results on the 12 dev wells
with the final model on the 3 test wells, for ours and for Trackastra ft; Trackastra zs ran on all 15 wells.
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

FINAL = ROOT / 'development' / 'final_results'
DATA = ROOT / 'data'
SCRIPTA = DATA / 'scripta_data_refine2'
TEST_WELLS = ['r16c02', 'r01c20', 'r07c07']
CACHE = Path(__file__).resolve().parent / 'out' / 'cache'

# method key -> [(method name in ctc_per_well.csv, split or None for every split)]
CTC_ROWS = {
    'ours': [('C ours final', 'test'), ('ours f001 + TTA (OOF)', 'dev_oof')],
    'ta_ft': [('F Trackastra fine-tuned', 'test'), ('Trackastra fine-tuned (OOF)', 'dev_oof')],
    'ta_zs': [('E Trackastra zero-shot general_2d', None)],
    'ta_zs_ctc': [('Trackastra zero-shot ctc', None)],
    'ta_ft_dt3': [('Trackastra ft_final delta_t 3', 'test')] + [(f'Trackastra ft_fold{k} delta_t 3', 'dev_oof') for k in range(3)],
    'ta_zs_dt3': [('Trackastra zs_general_2d delta_t 3', None)],
    'binary': [('B binary edges', 'test')],
    'v003': [('A gnn_tracking v003', 'test')],
    'classical': [('D classical', 'test')],
    'ctgnn0': [('G CellTrack-GNN seed0', 'test')],
    'ctgnn1': [('G CellTrack-GNN seed1', 'test')],
    'base': [('ours f001, no TTA (OOF)', 'dev_oof')],
    'noapp': [('ours OOF, noappearance', 'dev_oof')],
    'novel': [('ours OOF, novelocity', 'dev_oof')],
}
HELA_ROWS = {
    'ours': 'ours LOSO + TTA', 'base': 'ours LOSO, no TTA', 'noapp': 'ours LOSO, noappearance',
    'novel': 'ours LOSO, novelocity', 'ta_zs': 'Trackastra zero-shot general_2d', 'ta_zs_ctc': 'Trackastra zero-shot ctc',
}

# method key -> folders of per-well track outputs (<folder>/<well>/instance_tracks.csv), first match wins
_DT3 = 'scripta/supplementary_trackastra_dt3'
TRACKS = {
    'ours': ['scripta/C_ours_final/tracking', 'scripta/ours_cv_oof_tta/tracking'],
    'ta_ft': ['scripta/F_trackastra_ft/final/tracks'] + [f'scripta/F_trackastra_ft/fold{k}/tracks' for k in range(3)],
    'ta_zs': ['scripta/E_trackastra_zs_general_2d/tracks'],
    'ta_ft_dt3': [f'{_DT3}/ft_final/tracks'] + [f'{_DT3}/ft_fold{k}/tracks' for k in range(3)],
    'ta_zs_dt3': [f'{_DT3}/zs_general_2d/tracks'],
}

_ctc = None


def ctc_table():
    global _ctc
    if _ctc is None:
        _ctc = pd.read_csv(FINAL / 'ctc_per_well.csv')
        _ctc['div_recall'] = _ctc['tp_div(0)'] / (_ctc['tp_div(0)'] + _ctc['fn_div(0)'])
    return _ctc


def scripta_wells(key, wells=None):
    """Per-well CTC rows of one method on scripta (index = well)."""
    t = ctc_table()
    parts = [t[(t.dataset == 'scripta') & (t.method == m) & ((t.split == s) if s else True)] for m, s in CTC_ROWS[key]]
    rows = pd.concat(parts).set_index('well')
    if rows.index.duplicated().any():
        raise ValueError(f'{key}: duplicated wells')
    return rows if wells is None else rows.loc[wells]


def hela_wells(key):
    t = ctc_table()
    return t[(t.dataset == 'hela') & (t.method == HELA_ROWS[key])].set_index('well').sort_index()


def all_wells():
    return sorted(scripta_wells('ours').index)


def paired(a, b):
    """a, b: Series over the same wells -> dict(n, a_better, b_better, p) with a two-sided Wilcoxon signed-rank."""
    a, b = a.align(b, join='inner')
    d = a - b
    p = wilcoxon(a, b).pvalue if (d != 0).any() else float('nan')
    return dict(n=len(d), a_higher=int((d > 0).sum()), b_higher=int((d < 0).sum()), p=float(p))


def answers(name):
    """evaluate_answers output in final_results/calibration/<name>: (metrics overall, reliability, coverage)."""
    folder = FINAL / 'calibration' / name
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))['overall']
    return metrics, pd.read_csv(folder / 'reliability.csv'), pd.read_csv(folder / 'coverage.csv')


def track_folder(key, well):
    for folder in TRACKS[key]:
        if (FINAL / folder / well / 'instance_tracks.csv').is_file():
            return FINAL / folder / well
    raise FileNotFoundError(f'{key}: no output for {well}')


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
    """Per well and method: GT continuation links, missed and false links by frame gap (g1, g2, g3+).
    A false link joins two known detections (labels < 50000) that are not a GT continuation. Cached as csv."""
    cache = CACHE / 'gap_links.csv'
    if cache.is_file() and not refresh:
        return pd.read_csv(cache)
    rows = []
    for well in all_wells():
        tracks = {k: pd.read_csv(track_folder(k, well) / 'instance_tracks.csv') for k in TRACKS}
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
