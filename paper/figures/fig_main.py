"""Figure 1 (full width): main comparison on scripta.
a, b  LNK and CT on all 15 wells, paired by well: ours vs Trackastra ft and zs (dev wells out-of-fold, test wells final).
c, d  LNK and CT on the 3 test wells for every method (mean and the wells)."""
import matplotlib.pyplot as plt
import numpy as np

import data
import style

PAIRED = ['ours', 'ta_ft', 'ta_zs']
TEST = ['ours', 'binary', 'v003', 'ta_ft', 'ta_zs', 'ctgnn0', 'ctgnn1', 'classical']
SHORT = {'ours': 'Ours', 'binary': 'Ours, binary', 'v003': 'GNN v003', 'ta_ft': 'TA ft', 'ta_zs': 'TA zs',
         'ctgnn0': 'CT-GNN s0', 'ctgnn1': 'CT-GNN s1', 'classical': 'Classical'}


def paired_panel(ax, metric, wells):
    values = {k: data.scripta_wells(k, wells)[metric] for k in PAIRED}
    x = np.arange(len(PAIRED))
    for well in wells:
        ax.plot(x, [values[k][well] for k in PAIRED], color='#C8C8C8', lw=0.4, zorder=1)
    for i, k in enumerate(PAIRED):
        v = values[k]
        test = v.index.isin(data.TEST_WELLS)
        ax.scatter(np.full(len(v), i)[~test], v[~test], s=7, color=style.color(k), edgecolor=style.EDGE, lw=0.3, zorder=2)
        ax.scatter(np.full(len(v), i)[test], v[test], s=8, marker='D', color=style.color(k), edgecolor='black',
                   lw=0.5, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color='black', lw=1.0, zorder=4)
    ax.set_xticks(x, [style.label(k).replace('Trackastra ', 'TA ') for k in PAIRED])
    ax.set_xlim(-0.5, len(PAIRED) - 0.5)
    ax.set_ylabel(metric)
    ax.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)
    return values


def test_panel(ax, metric):
    x = np.arange(len(TEST))
    means = {}
    for i, k in enumerate(TEST):
        v = data.scripta_wells(k, data.TEST_WELLS)[metric]
        means[k] = v.mean()
        ax.scatter(np.full(len(v), i), v.values, s=9, marker='D', color=style.color(k), edgecolor='black', lw=0.4,
                   zorder=3)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color='black', lw=1.0, zorder=4)
    ax.set_xticks(x, [SHORT[k] for k in TEST], rotation=35, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.6, len(TEST) - 0.4)
    ax.set_ylabel(f'{metric} (test wells)')
    ax.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)
    return means


def main():
    style.setup()
    wells = data.all_wells()
    fig = plt.figure(figsize=(style.FULL, 52 * style.MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.45, 1.45], left=0.055, right=0.995, bottom=0.24, top=0.92,
                          wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    stats = {}
    for ax, metric, letter in ((axes[0], 'LNK', 'a'), (axes[1], 'CT', 'b')):
        values = paired_panel(ax, metric, wells)
        style.panel(ax, letter, x=-0.3)
        stats[metric] = (values, {k: data.paired(values['ours'], values[k]) for k in PAIRED[1:]})
    axes[0].set_ylim(0.88, 1.0)
    axes[1].set_ylim(0.3, 0.95)
    tests = {}
    for ax, metric, letter, lo in ((axes[2], 'LNK', 'c', 0.78), (axes[3], 'CT', 'd', 0.0)):
        tests[metric] = test_panel(ax, metric)
        ax.set_ylim(lo, 1.0 if metric == 'LNK' else 0.9)
        style.panel(ax, letter, x=-0.2)
    axes[0].set_title('15 wells, paired', loc='left', fontsize=6.5)
    axes[2].set_title('3 test wells', loc='left', fontsize=6.5)

    lines = ['Figure 1. Main comparison on scripta (refine2 GT; CTC measures on fixed detections, project GT).',
             'a, b: 15 wells paired by well; dev wells (circles) = 3-fold out-of-fold, test wells (diamonds) = final '
             'models; black bar = mean. Two-sided Wilcoxon signed-rank, ours vs each:']
    for metric, (values, tests_) in stats.items():
        means = ', '.join(f'{style.label(k)} {values[k].mean():.4f}' for k in PAIRED)
        lines.append(f'  {metric}: means {means}')
        for k, s in tests_.items():
            lines.append(f'    ours vs {style.label(k)}: ours higher in {s["a_higher"]}/{s["n"]} wells, '
                         f'lower in {s["b_higher"]}, p = {data.fmt_p(s["p"])}')
    lines.append('c, d: test wells r16c02, r01c20, r07c07; black bar = mean, diamonds = wells. TA = Trackastra, '
                 'CT-GNN = CellTrack-GNN (s0/s1 = training seeds; shared labels counted strictly, see RESULTS.md).')
    for metric, means in tests.items():
        lines.append(f'  {metric}: ' + ', '.join(f'{style.label(k)} {v:.4f}' for k, v in means.items()))
    style.save(fig, 'fig1_main_comparison', '\n'.join(lines))


if __name__ == '__main__':
    main()
