"""Main comparison on scripta.
fig1_main_comparison (paper, one column): a, b  LNK and CT on all 15 videos, paired: ours vs Trackastra ft and zs,
      each with divisions allowed (greedy, the setting of ours: not told whether there are divisions) and with
      divisions forbidden (greedy_nodiv, marked ‡) (dev wells out-of-fold, test wells final).
figS4_test_wells (supplement, one column): LNK and CT on the 3 test wells for every method (mean and the wells),
      the points behind the paper's test-well table."""
import matplotlib.pyplot as plt
import numpy as np

import data
import style

PAIRED = ['ours', 'ta_ft_div', 'ta_zs_div', 'ta_ft', 'ta_zs']
TEST = ['ours', 'binary', 'ta_ft_div', 'ta_zs_div', 'ta_ft', 'ta_zs', 'ctgnn', 'classical']   # ctgnn: both seeds
SHORT = {'ours': 'ProGCT', 'binary': 'ProGCT, binary', 'ta_ft_div': 'TA ft', 'ta_zs_div': 'TA zs', 'ta_ft': 'TA ft‡',
         'ta_zs': 'TA zs‡', 'ctgnn': 'CellTrack-GNN', 'classical': 'Classical‡'}
HIGHLIGHT = '#E3ECF8'                     # band behind ProGCT (user: show that ours is best)


def highlight(ax, i):
    ax.axvspan(i - 0.45, i + 0.45, color=HIGHLIGHT, zorder=0, lw=0)
    lab = ax.get_xticklabels()[i]
    lab.set_color(style.color('ours'))
    lab.set_fontweight('bold')


def test_values(k, metric):
    if k == 'ctgnn':                     # CellTrack-GNN: per-well mean of its two training seeds
        return np.mean([data.scripta_wells(s, data.TEST_WELLS)[metric].values for s in ('ctgnn0', 'ctgnn1')], axis=0)
    return data.scripta_wells(k, data.TEST_WELLS)[metric].values


def paired_panel(ax, metric, wells):
    values = {k: data.scripta_wells(k, wells)[metric] for k in PAIRED}
    x = np.arange(len(PAIRED))
    for well in wells:
        ax.plot(x, [values[k][well] for k in PAIRED], color='#DADADA', lw=0.4, zorder=1)
    for i, k in enumerate(PAIRED):
        v = values[k]
        test = v.index.isin(data.TEST_WELLS)
        ax.scatter(np.full(len(v), i)[~test], v[~test], s=8, color=style.light(k), edgecolor=style.color(k), lw=0.5,
                   zorder=2)
        ax.scatter(np.full(len(v), i)[test], v[test], s=10, marker='D', color=style.light(k, 0.3),
                   edgecolor=style.color(k), lw=0.7, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=style.color('ours') if k == 'ours' else style.EDGE,
                lw=1.6 if k == 'ours' else 1.0, zorder=4)
    ax.set_xticks(x, [SHORT[k] for k in PAIRED], rotation=35, ha='right', rotation_mode='anchor')
    highlight(ax, 0)
    ax.set_xlim(-0.5, len(PAIRED) - 0.5)
    ax.set_ylabel(metric)
    ax.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)
    return values


def test_panel(ax, metric):
    x = np.arange(len(TEST))
    means = {}
    for i, k in enumerate(TEST):
        v = test_values(k, metric)
        means[k] = v.mean()
        ck = 'ctgnn0' if k == 'ctgnn' else k
        ax.scatter(np.full(len(v), i), v, s=10, marker='D', color=style.light(ck), edgecolor=style.color(ck),
                   lw=0.6, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=style.color('ours') if k == 'ours' else style.EDGE,
                lw=1.6 if k == 'ours' else 1.0, zorder=4)
    ax.set_xticks(x, [SHORT[k] for k in TEST], rotation=45, ha='right', rotation_mode='anchor', fontsize=5.5)
    highlight(ax, 0)
    ax.set_xlim(-0.6, len(TEST) - 0.4)
    ax.set_ylabel(f'{metric} (test wells)')
    ax.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)
    return means


def paired():
    """Paper figure (one column): a, b on all 15 videos."""
    wells = data.all_wells()
    fig = plt.figure(figsize=(style.COLUMN, 42 * style.MM))
    gs = fig.add_gridspec(1, 2, left=0.16, right=0.99, bottom=0.24, top=0.9, wspace=0.48)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    stats = {}
    for ax, metric, letter in ((axes[0], 'LNK', 'a'), (axes[1], 'CT', 'b')):
        values = paired_panel(ax, metric, wells)
        style.panel(ax, letter, x=-0.3)
        stats[metric] = (values, {k: data.paired(values['ours'], values[k]) for k in PAIRED[1:]})
    axes[0].set_ylim(0.88, 1.0)
    axes[1].set_ylim(0.3, 0.95)
    axes[0].set_title('15 videos, paired', loc='left', fontsize=6.5)
    lines = ['Main comparison on scripta (CTC measures on fixed detections, project GT).',
             'a, b: 15 videos paired; dev wells (circles) = 3-fold out-of-fold, test wells (diamonds) = final '
             'models; black bar = mean. Two-sided Wilcoxon signed-rank, ours vs each:']
    for metric, (values, tests_) in stats.items():
        means = ', '.join(f'{style.label(k)} {values[k].mean():.4f}' for k in PAIRED)
        lines.append(f'  {metric}: means {means}')
        for k, s in tests_.items():
            lines.append(f'    ours vs {style.label(k)}: ours higher in {s["a_higher"]}/{s["n"]} videos, '
                         f'lower in {s["b_higher"]}, p = {data.fmt_p(s["p"])}')
    lines.append('TA = Trackastra; unmarked: divisions allowed (greedy), as for ours; ‡ divisions forbidden '
                 '(Trackastra greedy_nodiv).')
    style.save(fig, 'fig1_main_comparison', '\n'.join(lines))


def test_wells():
    """Supplementary figure (one column): LNK and CT on the 3 test wells for every method (Table 1 as points)."""
    fig = plt.figure(figsize=(style.COLUMN, 48 * style.MM))
    gs = fig.add_gridspec(1, 2, left=0.16, right=0.99, bottom=0.36, top=0.9, wspace=0.5)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    tests = {}
    for ax, metric, letter, lo in ((axes[0], 'LNK', 'a', 0.78), (axes[1], 'CT', 'b', 0.0)):
        tests[metric] = test_panel(ax, metric)
        ax.set_ylim(lo, 1.0 if metric == 'LNK' else 0.9)
        style.panel(ax, letter, x=-0.3)
    axes[0].set_title('3 test wells', loc='left', fontsize=6.5)
    lines = ['Test wells r16c02, r01c20, r07c07, every method; black bar = mean, diamonds = wells. TA = Trackastra; '
             '‡ divisions forbidden (the classical tracker never divides); CellTrack-GNN: per-well mean of its two '
             'training seeds. ProGCT highlighted (band, bold label, blue mean bar).']
    for metric, means in tests.items():
        lines.append(f'  {metric}: ' + ', '.join(f'{SHORT[k]} {v:.4f}' for k, v in means.items()))
    style.save(fig, 'figS4_test_wells', '\n'.join(lines))


def main():
    style.setup()
    paired()
    test_wells()


if __name__ == '__main__':
    main()
