"""Figure S4 (one column): is the division decision about the cell or about the dataset?
p_division of every adjacent-frame candidate edge (round 1, where divisions are decided) of our model, by the
ground-truth relation of the edge: HeLa LOSO + TTA (each sequence scored by the model trained without it) and
scripta 3-fold OOF + TTA (no divisions in its ground truth).
a  share of the edges with p_division >= x (log axes): a model that only recognised 'this is HeLa' would raise
   p_division on every HeLa edge; a cell-level decision keeps it low on the HeLa continuations.
b  HeLa: precision and recall of the edges with p_division >= x as mother -> daughter edges."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from utils.lineage import MOTHER_DAUGHTER, SAME_CELL, UNRELATED, read_lineage

HELA = data.DATA / 'Fluo-N2DL-HeLa'
SOURCES = {                       # population -> (folder in final_results, dataset root)
    'hela': ('hela/ours_loso_tta/division', HELA),
    'scripta_cv': ('scripta/ours_cv_oof_tta/division', data.SCRIPTA),
}
GROUPS = [('hela', MOTHER_DAUGHTER, 'HeLa, mother-daughter', '#C8553D', '-'),
          ('hela', SAME_CELL, 'HeLa, same cell', '#2A9D8F', '-'),
          ('hela', UNRELATED, 'HeLa, unrelated', '#2A9D8F', ':'),
          ('scripta_cv', SAME_CELL, 'Scripta, same cell', '#3B6FB6', '-'),
          ('scripta_cv', UNRELATED, 'Scripta, unrelated', '#3B6FB6', ':')]
THRESHOLD = 0.5                    # the division threshold of the tracker (track.DIV_THRESH)


def edges(refresh=False):
    """Round-1 candidate edges of every well: population, well, relation, p_div. Cached as csv."""
    cache = data.CACHE / 'division_edges.csv'
    if cache.is_file() and not refresh:
        return pd.read_csv(cache)
    parts = []
    for population, (folder, root) in SOURCES.items():
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
    fig = plt.figure(figsize=(style.COLUMN, 50 * style.MM))
    gs = fig.add_gridspec(1, 2, left=0.13, right=0.98, bottom=0.22, top=0.93, wspace=0.42)
    ax_s, ax_pr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    x = np.logspace(-6, 0, 200)
    lines = ['Figure S4. p_division of every adjacent-frame candidate edge of our model (round 1, where divisions '
             'are decided), by the ground-truth relation of the edge. HeLa: LOSO + TTA (each sequence scored by the '
             'model trained without it); scripta: 3-fold OOF + TTA, 12 development wells (no divisions in the GT).',
             'Edges whose relation is unknown (unlabelled detections, mothers with 3+ children) are left out.',
             f'a: share of edges with p_division >= x. Share >= {THRESHOLD} (the division threshold) and the largest '
             'value per group:']
    for population, relation, name, colour, ls in GROUPS:
        p = table.p_div[(table.population == population) & (table.relation == relation)].to_numpy()
        share = np.array([(p >= v).mean() for v in x])
        ax_s.plot(x, np.where(share > 0, share, np.nan), color=colour, ls=ls, lw=0.9, label=name)
        lines.append(f'  {name}: {len(p)} edges, {(p >= THRESHOLD).sum()} >= {THRESHOLD} '
                     f'({(p >= THRESHOLD).mean():.2%}), max {p.max():.3g}')
    ax_s.axvline(THRESHOLD, color='#9A9A9A', lw=0.5, ls='--', zorder=0)
    ax_s.set(xscale='log', yscale='log', xlim=(1e-6, 1), ylim=(1e-6, 1.5), xlabel='p$_{division}$ threshold x',
             ylabel='Share of edges $\\geq$ x')
    ax_s.legend(loc='lower left', fontsize=4.8, handlelength=1.6, borderaxespad=0.2)
    style.panel(ax_s, 'a', x=-0.25)

    hela = table[(table.population == 'hela') & (table.relation != 'unknown')]
    positive = (hela.relation == MOTHER_DAUGHTER).to_numpy()
    p = hela.p_div.to_numpy()
    grid = np.linspace(0.05, 0.95, 91)
    precision = [positive[p >= v].mean() if (p >= v).any() else np.nan for v in grid]
    recall = [(positive & (p >= v)).sum() / positive.sum() for v in grid]
    ax_pr.plot(grid, precision, color='black', lw=0.9, label='Precision')
    ax_pr.plot(grid, recall, color='#C8553D', lw=0.9, label='Recall')
    ax_pr.axvline(THRESHOLD, color='#9A9A9A', lw=0.5, ls='--', zorder=0)
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0, 0.5, 1], xlabel='p$_{division}$ threshold', ylabel='HeLa mother-daughter edges')
    ax_pr.legend(loc='lower left', handlelength=1.4)
    style.panel(ax_pr, 'b', x=-0.25)
    k = int(np.argmin(np.abs(grid - THRESHOLD)))
    lines.append(f'b: HeLa, edges with p_division >= x as mother -> daughter edges ({positive.sum()} GT '
                 f'mother-daughter edges of {len(hela)}). At {THRESHOLD}: precision {precision[k]:.3f}, '
                 f'recall {recall[k]:.3f} (edge level, before the one-to-one matching of the tracker).')
    style.save(fig, 'figS4_division_edges', '\n'.join(lines))


if __name__ == '__main__':
    main()
