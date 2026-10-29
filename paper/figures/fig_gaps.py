"""Figure 2 (one column): link errors by frame gap g on the 15 scripta wells.
a  missed GT continuations (% of the GT links of that gap);  b  false links (count).
Ours vs Trackastra ft and zs at delta_t 1 (its intended use) and at delta_t 3 (hatched)."""
import matplotlib.pyplot as plt
import numpy as np

import data
import style

METHODS = ['ours', 'ta_ft', 'ta_zs', 'ta_ft_dt3', 'ta_zs_dt3']
BINS = ['g1', 'g2', 'g3+']
TICKS = ['g = 1', 'g = 2', 'g ≥ 3']


def bars(ax, values, log_floor):
    width = 0.8 / len(METHODS)
    for j, k in enumerate(METHODS):
        x = np.arange(len(BINS)) + (j - (len(METHODS) - 1) / 2) * width
        v = np.array([values[k][b] for b in BINS], float)
        ax.bar(x, np.maximum(v, log_floor), width, color=style.color(k), edgecolor=style.EDGE, lw=0.3,
               hatch='////' if k.endswith('dt3') else None, label=style.label(k))
        for xi, vi in zip(x, v):
            if vi == 0:
                ax.text(xi, log_floor * 1.3, '0', ha='center', va='bottom', fontsize=5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(BINS)), TICKS)
    ax.yaxis.grid(True, color=style.GRID, lw=0.4, which='major')
    ax.set_axisbelow(True)


def main():
    style.setup()
    plt.rcParams['hatch.linewidth'] = 0.4
    t = data.gap_counts()
    total = t.groupby('method').sum(numeric_only=True)
    missed = {k: {b: 100 * total.loc[k, f'missed_{b}'] / total.loc[k, f'gt_{b}'] for b in BINS} for k in METHODS}
    false = {k: {b: total.loc[k, f'false_{b}'] for b in BINS} for k in METHODS}

    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 58 * style.MM))
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.3, top=0.93, wspace=0.42)
    bars(axes[0], missed, 0.1)
    axes[0].set_ylabel('Missed links (% of GT)')
    axes[0].set_ylim(0.1, 200)
    bars(axes[1], false, 1)
    axes[1].set_ylabel('False links')
    axes[1].set_ylim(1, 1e5)
    for ax, letter in zip(axes, 'ab'):
        style.panel(ax, letter, x=-0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.55, 0.0), handlelength=1.2,
               columnspacing=0.8)

    gt = total.loc['ours']
    ours = t[t.method == 'ours'].set_index('well')
    lines = ['Figure 2. Link errors by frame gap g, 15 scripta wells (dev wells out-of-fold, test wells final).',
             f'GT continuation links: g=1 {int(gt.gt_g1)}, g=2 {int(gt.gt_g2)}, g>=3 {int(gt["gt_g3+"])}. '
             'A missed link is a GT continuation not in the output; a false link joins two known detections '
             '(labels < 50000) that are not a GT continuation. Counts per link, not AOGM.',
             'Trackastra at delta_t 3 (hatched) for illustration only; comparisons use delta_t 1.']
    for k in METHODS:
        lines.append(f'  {style.plain(k)}: missed ' +', '.join(f'{b} {int(total.loc[k, f"missed_{b}"])}' for b in BINS)
                     + '; false ' + ', '.join(f'{b} {int(total.loc[k, f"false_{b}"])}' for b in BINS))
    lines.append('Paired per well, ours vs Trackastra ft (delta_t 1), Wilcoxon (fewer errors is better):')
    other = t[t.method == 'ta_ft'].set_index('well')
    for col in ('missed_g1', 'missed_g2', 'false_g1', 'false_g2'):
        s = data.paired(other[col], ours[col])
        lines.append(f'  {col}: ours fewer in {s["a_higher"]}/{s["n"]} wells, more in {s["b_higher"]}, '
                     f'p = {data.fmt_p(s["p"])}')
    style.save(fig, 'fig2_gap_errors', '\n'.join(lines))


if __name__ == '__main__':
    main()
