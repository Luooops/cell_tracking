"""Figure 3 (full width): what the link probabilities are worth.
a  reliability of the edges in the decision band (0.02 < p < 0.98), equal-count bins, Wilson 95% intervals;
b, c  error coverage: share of the false (b) / missed (c) links found when checking the given share of the
      chosen (b) / not chosen (c) candidate links, ranked by mutual information (solid) or closeness to 0.5 (dashed).
Ours: scripta 3-fold OOF + TTA, scripta test (final model), HeLa LOSO + TTA."""
import matplotlib.pyplot as plt
import numpy as np

import data
import style

SOURCES = {'scripta_cv': 'scripta_cv_oof_tta', 'scripta_test': 'scripta_test_C', 'hela': 'hela_loso_tta'}
SCORES = {'mutual_info': ('-', 'MI'), 'closeness': ('--', 'closeness')}
MAX_CHECKED = 20


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    style.setup()
    loaded = {k: data.answers(v) for k, v in SOURCES.items()}
    fig = plt.figure(figsize=(style.FULL, 55 * style.MM))
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, bottom=0.2, top=0.9, wspace=0.32)
    ax_r, ax_f, ax_m = (fig.add_subplot(gs[0, i]) for i in range(3))

    ax_r.plot([0, 1], [0, 1], color='#9A9A9A', lw=0.6, ls=':', zorder=1)
    for key, (metrics, reliability, _) in loaded.items():
        name, col = style.DATASETS[key]
        r = reliability[reliability.population == 'round1_decision']
        lo, hi = wilson(r.frac_pos * r.n, r.n)
        ax_r.errorbar(r.mean_p, r.frac_pos, yerr=[np.clip(r.frac_pos - lo, 0, None), np.clip(hi - r.frac_pos, 0, None)], fmt='o-', ms=2.2, lw=0.7,
                      elinewidth=0.5, capsize=0, color=col, label=name)
    ax_r.set(xlim=(0, 1), ylim=(0, 1), xlabel='Predicted p (same cell)', ylabel='Fraction same cell')
    ax_r.set_aspect('equal')
    ax_r.legend(loc='upper left', handlelength=1.4)
    ax_r.set_title('Decision band, 0.02 < p < 0.98', loc='left', fontsize=6.5)

    for ax, population, title in ((ax_f, 'false_links', 'False links (among chosen links)'),
                                  (ax_m, 'missed_links', 'Missed links (among candidates not chosen)')):
        ax.plot([0, MAX_CHECKED], [0, MAX_CHECKED], color='#9A9A9A', lw=0.6, ls=':', label='Random order')
        for key, (_, _, coverage) in loaded.items():
            name, col = style.DATASETS[key]
            for score, (ls, _) in SCORES.items():
                c = coverage[(coverage.population == population) & (coverage.score == score)]
                c = c[c.checked_percent <= MAX_CHECKED]
                ax.plot(c.checked_percent, c.errors_found_percent, ls=ls, color=col, lw=0.8)
        ax.set(xlim=(0, MAX_CHECKED), ylim=(0, 100), xlabel='Links checked (%)', ylabel='Errors found (%)')
        ax.set_title(title, loc='left', fontsize=6.5)
        ax.yaxis.grid(True, color=style.GRID, lw=0.4)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color='black', ls=ls, lw=0.8, label=f'ranked by {lab}') for ls, lab in SCORES.values()]
    handles.append(plt.Line2D([], [], color='#9A9A9A', ls=':', lw=0.6, label='random order'))
    ax_m.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 0.22), handlelength=2)
    for ax, letter in zip((ax_r, ax_f, ax_m), 'abc'):
        style.panel(ax, letter, x=-0.18)

    lines = ['Figure 3. Confidence of the link probabilities (evaluate_answers; edges of all candidate links).',
             'a: decision band = edges with 0.02 < p < 0.98 (round 1, before the solver); 15 equal-count bins; '
             'Wilson 95% intervals. b, c: share of errors found when checking links in order of uncertainty; '
             f'x axis cut at {MAX_CHECKED}%; area = normalised area under the full curve (1 = perfect, 0.5 = random).']
    for key, (m, _, _) in loaded.items():
        band, cov = m['calibration']['round1_decision'], m['coverage']
        lines.append(f'  {style.DATASETS[key][0]} ({SOURCES[key]}): band n {band["n"]}, mean p {band["mean_p"]:.3f}, '
                     f'fraction same {band["frac_pos"]:.3f}, ECE {band["ece"]:.3f}; all edges ECE '
                     f'{m["calibration"]["round1_all"]["ece"]:.4f}')
        for population in ('false_links', 'missed_links'):
            c = cov[population]
            lines.append(f'    {population}: {c["errors"]} errors among {c["items"]} links; area MI '
                         f'{c["mutual_info"]["area"]:.3f}, closeness {c["closeness"]["area"]:.3f}; found at 5% checked: '
                         f'MI {c["mutual_info"]["at_percent"]["5"]:.1f}%, closeness {c["closeness"]["at_percent"]["5"]:.1f}%')
    style.save(fig, 'fig3_confidence', '\n'.join(lines))


if __name__ == '__main__':
    main()
