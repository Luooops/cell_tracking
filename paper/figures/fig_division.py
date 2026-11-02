"""Figure 5 (full width): does the model decide divisions from the cells or from recognising the dataset?
Our model is never told whether a dataset has divisions.
a  p_division of every adjacent-frame candidate edge (round 1, where divisions are decided), by the ground-truth
   relation of the edge: share of the edges with p_division >= x (log axes). HeLa: LOSO + TTA (each sequence scored by
   the model trained without it); scripta: 3-fold OOF + TTA on the 12 development wells (no divisions in the GT) of
   our model and of the same model trained on HeLa only (never saw scripta, so it cannot know scripta has none).
   A model that only recognised the dataset would raise p_division on every HeLa edge.
b  HeLa: precision and recall of the edges with p_division >= x as mother -> daughter edges.
c  false divisions on the scripta test wells (data.division_events) of every method that may divide."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from utils.lineage import MOTHER_DAUGHTER, SAME_CELL, UNRELATED, read_lineage

HELA = data.DATA / 'Fluo-N2DL-HeLa'
SOURCES = {                       # population -> (folders in final_results, dataset root)
    'hela': (['hela/ours_loso_tta/division'], HELA),
    'scripta': (['scripta/ours_cv_oof_tta/division'], data.SCRIPTA),
    'scripta_t1': (['scripta/T1_hela_only/dev/division'], data.SCRIPTA),
}
GROUPS = [('hela', MOTHER_DAUGHTER, 'HeLa, mother-daughter', '#C8553D', '-'),
          ('hela', SAME_CELL, 'HeLa, same cell', '#2A9D8F', '-'),
          ('scripta', SAME_CELL, 'Scripta, same cell', '#3B6FB6', '-'),
          ('scripta_t1', SAME_CELL, 'Scripta, same cell,\nmodel trained on HeLa only', '#7FA7D9', '--')]
COUNTED = ['ours', 't1', 'binary', 'ta_ft_div', 'ta_zs_div', 'ctgnn0', 'ctgnn1']
SHORT = {'ours': 'ProGCT', 't1': 'ProGCT,\nHeLa only', 'binary': 'ProGCT,\nbinary', 'ta_ft_div': 'TA ft',
         'ta_zs_div': 'TA zs', 'ctgnn0': 'CT-GNN s0', 'ctgnn1': 'CT-GNN s1'}
THRESHOLD = 0.5                    # the division threshold of the tracker (track.DIV_THRESH)


def edges(refresh=False):
    """Round-1 candidate edges of every well: population, well, relation, p_div. Cached as csv."""
    cache = data.CACHE / 'division_edges.csv'
    if cache.is_file() and not refresh:
        table = pd.read_csv(cache)
        if set(table.population) == set(SOURCES):
            return table
    parts = []
    for population, (folders, root) in SOURCES.items():
        for folder in folders:
            for well_dir in sorted((data.FINAL / folder).iterdir()):
                well = well_dir.name
                z = np.load(well_dir / 'edges.npz')
                tracks = pd.read_csv(well_dir / 'instance_tracks.csv')
                lineage = read_lineage(root / f'{well}_GT' / 'TRA' / 'man_track.txt',
                                       zip(tracks.frame_index.astype(int), tracks.instance_id.astype(int)))
                keep = z['round'] == 1
                t, g, a, b = (z[k][keep].astype(int) for k in ('src_frame', 'g', 'src_label', 'dst_label'))
                relation = [lineage.relation((ti, ai), (ti + gi, bi)) for ti, gi, ai, bi in zip(t, g, a, b)]
                parts.append(pd.DataFrame(dict(population=population, well=well, relation=relation,
                                               p_div=z['p_div'][keep])))
    table = pd.concat(parts, ignore_index=True)
    data.CACHE.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


def main(refresh=False):
    style.setup()
    table = edges(refresh)
    fig = plt.figure(figsize=(style.FULL, 50 * style.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.3], left=0.075, right=0.995, bottom=0.25, top=0.93,
                          wspace=0.38)
    ax_s, ax_pr, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    x = np.logspace(-6, 0, 200)
    lines = ['Figure 5. Divisions are decided from the cells, not from recognising the dataset. Our model is never '
             'told whether a dataset has divisions.',
             'a: p_division of every adjacent-frame candidate edge (round 1, where divisions are decided), by the '
             'ground-truth relation of the edge; share of edges with p_division >= x. HeLa: LOSO + TTA (each sequence '
             'scored by the model trained without it). Scripta: 12 development wells, 3-fold OOF + TTA (no divisions '
             'in the GT), of our model and of the same model trained on HeLa only (never saw scripta). Edges whose '
             'relation is unknown (unlabelled detections) are left out.',
             f'  Share >= {THRESHOLD} (the division threshold) and the largest value:']
    for population, relation, name, colour, ls in GROUPS:
        p = table.p_div[(table.population == population) & (table.relation == relation)].to_numpy()
        share = np.array([(p >= v).mean() for v in x])
        ax_s.plot(x, np.where(share > 0, share, np.nan), color=colour, ls=ls, lw=0.9, label=name)
        lines.append(f'  {name.replace(chr(10), " ")}: {len(p)} edges, {(p >= THRESHOLD).sum()} >= {THRESHOLD} '
                     f'({(p >= THRESHOLD).mean():.3%}), max {p.max():.3g}')
    for population in SOURCES:
        p = table.p_div[(table.population == population) & (table.relation == UNRELATED)].to_numpy()
        lines.append(f'  {population}, unrelated edges: {len(p)}, {(p >= THRESHOLD).sum()} >= {THRESHOLD}, '
                     f'max {p.max():.3g}')
    ax_s.axvline(THRESHOLD, color='#9A9A9A', lw=0.5, ls=':', zorder=0)
    ax_s.set(xscale='log', yscale='log', xlim=(1e-6, 1), ylim=(1e-6, 1.5), xlabel='p$_{division}$ threshold x',
             ylabel='Share of edges $\\geq$ x')
    ax_s.legend(loc='lower left', fontsize=4.8, handlelength=1.8, borderaxespad=0.2)
    style.panel(ax_s, 'a', x=-0.17)

    hela = table[(table.population == 'hela') & (table.relation != 'unknown')]
    positive = (hela.relation == MOTHER_DAUGHTER).to_numpy()
    p = hela.p_div.to_numpy()
    grid = np.linspace(0.05, 0.95, 91)
    precision = [positive[p >= v].mean() if (p >= v).any() else np.nan for v in grid]
    recall = [(positive & (p >= v)).sum() / positive.sum() for v in grid]
    ax_pr.plot(grid, precision, color='black', lw=0.9, label='Precision')
    ax_pr.plot(grid, recall, color='#C8553D', lw=0.9, label='Recall')
    ax_pr.axvline(THRESHOLD, color='#9A9A9A', lw=0.5, ls=':', zorder=0)
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0, 0.5, 1], xlabel='p$_{division}$ threshold',
              ylabel='HeLa mother-daughter edges')
    ax_pr.legend(loc='lower left', handlelength=1.4)
    style.panel(ax_pr, 'b', x=-0.25)
    k = int(np.argmin(np.abs(grid - THRESHOLD)))
    lines.append(f'b: HeLa, edges with p_division >= x as mother -> daughter edges ({positive.sum()} GT '
                 f'mother-daughter edges of {len(hela)}). At {THRESHOLD}: precision {precision[k]:.3f}, '
                 f'recall {recall[k]:.3f} (edge level, before the one-to-one matching of the tracker).')

    counts = {key: data.test_divisions(key)['divisions'] for key in COUNTED}
    xs = np.arange(len(COUNTED))
    ax_c.bar(xs, [max(counts[key], 0.6) for key in COUNTED], 0.7, color=[style.color(key) for key in COUNTED],
             edgecolor=style.EDGE, lw=0.3)
    for i, key in enumerate(COUNTED):
        ax_c.text(i, max(counts[key], 0.6) * 1.15, str(counts[key]), ha='center', va='bottom', fontsize=5.5)
    ax_c.set_yscale('log')
    ax_c.set_ylim(0.5, 1000)
    ax_c.set_xticks(xs, [SHORT[key] for key in COUNTED], rotation=35, ha='right', rotation_mode='anchor', fontsize=5.5)
    ax_c.set_ylabel('False divisions (test wells)')
    ax_c.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax_c.set_axisbelow(True)
    style.panel(ax_c, 'c', x=-0.13)
    wells = data.all_wells()
    every = {key: data.scripta_divisions(key, wells)['divisions'] for key in ('ours', 't1', 'ta_ft_div', 'ta_zs_div')}
    lines.append('c: false divisions on the scripta test wells r16c02, r01c20, r07c07 (a mother with exactly two '
                 'daughters; the GT has none), every method allowed to divide (Trackastra: greedy). Bars of 0 are drawn '
                 'at the bottom of the log axis. ' + ', '.join(f'{style.plain(key)} {counts[key]}' for key in COUNTED)
                 + f'. CellTrack-GNN also gives {data.test_divisions("ctgnn0")["invalid"]} / '
                 f'{data.test_divisions("ctgnn1")["invalid"]} mothers three or more daughters.')
    lines.append('  All 15 wells (dev OOF + test): ' + ', '.join(f'{style.plain(key)} {v}' for key, v in every.items())
                 + '. Told that there are no divisions (greedy_nodiv), Trackastra has none by construction.')
    style.save(fig, 'fig5_division_evidence', '\n'.join(lines))


if __name__ == '__main__':
    main()
