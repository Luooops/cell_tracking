"""Figure 5 (one column): does the model decide divisions from the cells or from recognising the dataset?
Our model is never told whether a dataset has divisions.
a  p_division of every adjacent-frame candidate edge (round 1, where divisions are decided), by the ground-truth
   relation of the edge: share of the edges in each probability band, and the share >= tau_D.
   DNN: final model (scripta + RAW264.7) on the 13 compared videos it never saw; scripta: 3-fold OOF on the 12
   development wells (no divisions in the GT) of our model and of the same model trained on RAW264.7 only (never saw
   scripta, so it cannot know that scripta has none); HeLa: final model, never trained on HeLa.
b  false divisions on the scripta test wells (data.division_events) of every method that may divide."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from utils.lineage import MOTHER_DAUGHTER, SAME_CELL, UNRELATED, read_lineage

SOURCES = {                       # population -> (dataset, method key, wells, dataset root)
    'dnn': ('dnn', 'ours', data.DNN_VIDEOS, data.DNN),
    'scripta': ('scripta', 'ours', None, data.SCRIPTA),
    'scripta_raw': ('scripta', 'raw', None, data.SCRIPTA),
    'hela': ('hela', 'ours', ['01', '02'], data.HELA),
}
ROWS = [('dnn', MOTHER_DAUGHTER, 'DNN, mother$\\rightarrow$daughter'),
        ('dnn', SAME_CELL, 'DNN, same cell'),
        ('scripta', SAME_CELL, 'Scripta, same cell'),
        ('scripta_raw', SAME_CELL, 'Scripta, same cell,\nmodel trained on RAW264.7 only'),
        ('hela', MOTHER_DAUGHTER, 'HeLa, mother$\\rightarrow$daughter'),
        ('hela', SAME_CELL, 'HeLa, same cell')]
COUNTED = ['ours', 'raw', 't1', 'binary', 'ta_ft_div', 'ta_zs_div', 'ctgnn0', 'ctgnn1']
SHORT = {'ours': 'ProGCT', 'raw': 'ProGCT, RAW264.7 only', 't1': 'ProGCT, HeLa only', 'binary': 'ProGCT, binary',
         'ta_ft_div': 'Trackastra ft', 'ta_zs_div': 'Trackastra zs', 'ctgnn0': 'CellTrack-GNN s0',
         'ctgnn1': 'CellTrack-GNN s1'}
THRESHOLD = 0.5                    # the division threshold of the tracker (track.DIV_THRESH)
ACCENT = '#C8553D'
BANDS = [(0, 1e-4, '<10$^{-4}$', '#ECECEC'), (1e-4, 1e-2, '10$^{-4}$–10$^{-2}$', '#C9C9C9'),
         (1e-2, THRESHOLD, '10$^{-2}$–$\\tau_D$', '#8F8F8F'), (THRESHOLD, np.inf, '$\\geq\\tau_D$', ACCENT)]


def edges(refresh=False):
    """Round-1 candidate edges of every well: population, well, relation, p_div. Cached as csv."""
    cache = data.CACHE / 'division_edges.csv'
    if cache.is_file() and not refresh:
        table = pd.read_csv(cache)
        if set(table.population) == set(SOURCES):
            return table
    parts = []
    for population, (dataset, key, wells, root) in SOURCES.items():
        wells = wells or [w for w in data.all_wells() if w not in data.TEST_WELLS]
        for well in wells:
            well_dir = data.track_folder(key, well, dataset)
            z = np.load(well_dir / 'edges.npz')
            tracks = pd.read_csv(well_dir / 'instance_tracks.csv')
            lineage = read_lineage(root / f'{well}_GT' / 'TRA' / 'man_track.txt',
                                   zip(tracks.frame_index.astype(int), tracks.instance_id.astype(int)))
            keep = z['round'] == 1
            t, g, a, b = (z[k][keep].astype(int) for k in ('src_frame', 'g', 'src_label', 'dst_label'))
            relation = [lineage.relation((ti, ai), (ti + gi, bi)) for ti, gi, ai, bi in zip(t, g, a, b)]
            parts.append(pd.DataFrame(dict(population=population, well=well, relation=relation, p_div=z['p_div'][keep])))
    table = pd.concat(parts, ignore_index=True)
    data.CACHE.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


def share_text(share):
    return '0' if share == 0 else f'{share:.0%}' if share >= 0.1 else f'{share:.2%}'


def precision_recall(table, population):
    t = table[(table.population == population) & (table.relation != 'unknown')]
    positive = (t.relation == MOTHER_DAUGHTER).to_numpy()
    p = t.p_div.to_numpy()
    return positive[p >= THRESHOLD].mean(), (positive & (p >= THRESHOLD)).sum() / positive.sum(), positive.sum(), len(t)


def main(refresh=False):
    style.setup()
    table = edges(refresh)
    fig = plt.figure(figsize=(style.COLUMN, 80 * style.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.3], left=0.4, right=0.87, bottom=0.095, top=0.9, hspace=0.5)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    lines = ['Figure 5 (one column). Divisions are decided from the cells, not from recognising the dataset. Our model '
             'is never told whether a dataset has divisions.',
             'a: d (p_division) of every adjacent-frame candidate edge (round 1, where divisions are decided), by the '
             'ground-truth relation of the edge, as the share of edges in each probability band; right: share >= '
             'tau_D. DNN: final model (trained on scripta + RAW264.7 train60-90) on the 13 compared videos '
             '(val17-26, test09-11), never trained on. Scripta: 12 development wells (no divisions in the GT), '
             '3-fold OOF of our model, and the same model trained on RAW264.7 only (never saw scripta). HeLa: final '
             'model, never trained on HeLa. Edges whose relation is unknown (unlabelled detections) are left out.']
    for r, (population, relation, name) in enumerate(ROWS):
        p = table.p_div[(table.population == population) & (table.relation == relation)].to_numpy()
        left = 0.0
        for lo, hi, _, colour in BANDS:
            share = ((p >= lo) & (p < hi)).mean()
            ax_a.barh(r, share, left=left, height=0.64, color=colour, lw=0)
            left += share
        above = (p >= THRESHOLD).mean()
        ax_a.text(1.04, r, share_text(above), transform=ax_a.get_yaxis_transform(), ha='left', va='center',
                  fontsize=6, color=ACCENT if above >= 0.1 else 'black', fontweight='bold' if above >= 0.1 else 'normal')
        lines.append(f'  {name.replace(chr(10), " ")}: {len(p)} edges, {(p >= THRESHOLD).sum()} >= {THRESHOLD} '
                     f'({above:.3%}), max {p.max():.3g}, median {np.median(p):.3g}')
    for population in SOURCES:
        p = table.p_div[(table.population == population) & (table.relation == UNRELATED)].to_numpy()
        lines.append(f'  {population}, unrelated edges: {len(p)}, {(p >= THRESHOLD).sum()} >= {THRESHOLD}, '
                     f'max {p.max():.3g}')
    for population in ('dnn', 'hela'):
        precision, recall, n_pos, n = precision_recall(table, population)
        lines.append(f'  {population}, edges with d >= {THRESHOLD} as mother -> daughter edges ({n_pos} GT '
                     f'mother-daughter edges of {n}): precision {precision:.3f}, recall {recall:.3f} (edge level, '
                     'before the one-to-one matching of the tracker).')
    ax_a.text(1.04, -0.8, '$\\geq\\tau_D$', transform=ax_a.get_yaxis_transform(), ha='left', va='center', fontsize=6)
    ax_a.set_yticks(range(len(ROWS)), [name for *_, name in ROWS], fontsize=5.8)
    ax_a.invert_yaxis()
    ax_a.set(xlim=(0, 1), xticks=[0, 0.5, 1])
    ax_a.set_xticklabels(['0', '50%', '100%'])
    ax_a.set_xlabel('Share of candidate edges', labelpad=1.5)
    ax_a.tick_params(axis='y', length=0)
    ax_a.spines['left'].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for *_, colour in BANDS]
    ax_a.legend(handles, [band[2] for band in BANDS], ncol=4, loc='lower left', bbox_to_anchor=(-0.05, 1.06),
                handlelength=0.9, handletextpad=0.35, columnspacing=0.8, fontsize=5.5, borderaxespad=0)
    ax_a.text(-0.06, 1.1, '$d_{ij}$:', transform=ax_a.transAxes, ha='right', va='bottom', fontsize=6)
    style.panel(ax_a, 'a', x=-0.68, y=1.08)

    counts = {key: data.test_divisions(key) for key in COUNTED}
    floor = 0.6
    for y, key in enumerate(COUNTED):
        value = max(counts[key]['divisions'], floor)
        ax_b.plot([floor, value], [y, y], color=style.color(key), lw=1.5, solid_capstyle='butt')
        ax_b.plot(value, y, 'o', ms=3.6, color=style.color(key), mec=style.EDGE, mew=0.3, clip_on=False, zorder=3)
        ax_b.text(value * 1.4, y, str(counts[key]['divisions']), ha='left', va='center', fontsize=6,
                  fontweight='bold' if key in ('ours', 'raw') else 'normal')
    ax_b.set_xscale('log')
    ax_b.set_xlim(floor, 1500)
    ax_b.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    ax_b.minorticks_off()
    ax_b.set_yticks(range(len(COUNTED)), [SHORT[key] for key in COUNTED], fontsize=6)
    for tick in ax_b.get_yticklabels()[:2]:
        tick.set_fontweight('bold')
    ax_b.set_ylim(len(COUNTED) - 0.5, -0.5)
    ax_b.tick_params(axis='y', length=0)
    ax_b.spines['left'].set_visible(False)
    ax_b.xaxis.grid(True, color=style.GRID, lw=0.4)
    ax_b.set_axisbelow(True)
    ax_b.set_xlabel('False divisions on the Scripta test wells', labelpad=1.5)
    style.panel(ax_b, 'b', x=-0.68, y=1.0)
    wells = data.all_wells()
    every = {key: data.scripta_divisions(key, wells)['divisions'] for key in ('ours', 'raw', 't1', 'ta_ft_div', 'ta_zs_div')}
    lines.append('b: false divisions on the scripta test wells r16c02, r01c20, r07c07 (a mother with exactly two '
                 'daughters; the GT has none), every method allowed to divide (Trackastra: greedy). A count of 0 sits '
                 'at the left end of the log axis. ' + ', '.join(f'{style.plain(key)} {counts[key]["divisions"]}'
                                                                 for key in COUNTED)
                 + f'. CellTrack-GNN also gives {counts["ctgnn0"]["invalid"]} / {counts["ctgnn1"]["invalid"]} mothers '
                   'three or more daughters.')
    lines.append('  All 15 wells (dev OOF + test): ' + ', '.join(f'{style.plain(key)} {v}' for key, v in every.items())
                 + '. Told that there are no divisions (greedy_nodiv), Trackastra has none by construction.')
    style.save(fig, 'fig5_division_evidence', '\n'.join(lines))


if __name__ == '__main__':
    main()
