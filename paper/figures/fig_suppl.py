"""Supplementary figures.
S1 (full width)  cue ablation: ours without the appearance or the velocity input, scripta OOF and HeLa LOSO (no TTA).
S2 (one column)  HeLa: ours (LOSO + TTA) vs Trackastra zero-shot, per sequence.
S3 (one column)  test-time augmentation: tracking unchanged, error localisation (MI coverage area) better."""
import matplotlib.pyplot as plt
import numpy as np

import data
import style

ABLATION = ['base', 'noapp', 'novel']
ABL_LABEL = {'base': 'Full', 'noapp': 'No appearance', 'novel': 'No velocity'}


def dots(ax, x, values, key, marker='o'):
    ax.scatter(np.full(len(values), x), values, s=7, marker=marker, color=style.color(key), edgecolor=style.EDGE,
               lw=0.3, zorder=3)
    ax.plot([x - 0.25, x + 0.25], [np.mean(values)] * 2, color='black', lw=1.0, zorder=4)


def grid(ax):
    ax.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)


def ablation():
    fig = plt.figure(figsize=(style.FULL, 48 * style.MM))
    gs = fig.add_gridspec(1, 5, left=0.065, right=0.995, bottom=0.25, top=0.88, wspace=0.55)
    panels = [('scripta', 'LNK'), ('scripta', 'CT'), ('hela', 'LNK'), ('hela', 'CT'), ('hela', 'BC(0)')]
    lines = ['Figure S1. Cue ablation (no TTA): the trained model with the appearance or the velocity input removed '
             'at test time (--ablate); scripta 12 dev wells out-of-fold, HeLa LOSO sequences 01/02. '
             'Black bar = mean. Paired Wilcoxon over the 12 scripta wells:']
    for i, (dataset, metric) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        values = {k: (data.scripta_wells(k) if dataset == 'scripta' else data.hela_wells(k))[metric] for k in ABLATION}
        for j, k in enumerate(ABLATION):
            dots(ax, j, values[k], k)
        ax.set_xticks(range(len(ABLATION)), [ABL_LABEL[k] for k in ABLATION], rotation=30, ha='right',
                      rotation_mode='anchor')
        ax.set_ylabel(metric)
        ax.set_title('Scripta' if dataset == 'scripta' else 'HeLa', loc='left', fontsize=6.5)
        grid(ax)
        style.panel(ax, 'abcde'[i], x=-0.3)
        means = ', '.join(f'{ABL_LABEL[k]} {values[k].mean():.4f}' for k in ABLATION)
        lines.append(f'  {dataset} {metric}: {means}')
        if dataset == 'scripta':
            for k in ABLATION[1:]:
                s = data.paired(values['base'], values[k])
                lines.append(f'    full vs {ABL_LABEL[k]}: full higher in {s["a_higher"]}/{s["n"]}, p = {data.fmt_p(s["p"])}')
    style.save(fig, 'figS1_cue_ablation', '\n'.join(lines))


def hela():
    keys = ['ours', 'ta_zs', 'ta_zs_ctc']
    metrics = [('LNK', 'LNK'), ('CT', 'CT'), ('BC(0)', 'BC(0)'), ('div_recall', 'Division recall')]
    fig, ax = plt.subplots(figsize=(style.COLUMN, 50 * style.MM))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.27, top=0.95)
    width = 0.26
    lines = ['Figure S2. HeLa (Fluo-N2DL-HeLa 01/02), CTC measures on fixed detections. Ours = LOSO (trained on the '
             'other sequence + scripta) with TTA; Trackastra zero-shot general_2d (HeLa not in its training data); '
             'Trackastra ctc (hatched) was trained on all CTC GT including HeLa and is shown for reference only. '
             'Bars = mean of 01 and 02, dots = sequences. Division recall at tolerance 0 = tp / (tp + fn).']
    for j, k in enumerate(keys):
        t = data.hela_wells(k)
        for i, (col, _) in enumerate(metrics):
            x = i + (j - 1) * width
            ax.bar(x, t[col].mean(), width, color=style.color(k), edgecolor=style.EDGE, lw=0.3,
                   hatch='////' if k == 'ta_zs_ctc' else None, label=style.label(k) if i == 0 else None)
            ax.scatter([x - 0.05, x + 0.05], t[col].values, s=4, color='white', edgecolor='black', lw=0.3, zorder=3)
        lines.append(f'  {style.plain(k)}: ' + '; '.join(
            f'{name} {" / ".join(f"{v:.3f}" for v in t[col])}' for col, name in metrics)
            + f'; false divisions {" / ".join(str(int(v)) for v in t["fp_div(0)"])}')
    ax.set_xticks(range(len(metrics)), [name for _, name in metrics])
    ax.set_ylim(0.5, 1.02)
    grid(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, handlelength=1.2)
    style.save(fig, 'figS2_hela', '\n'.join(lines))


def tta():
    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 48 * style.MM), gridspec_kw=dict(width_ratios=[1, 1.3]))
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.3, top=0.9, wspace=0.5)
    ax = axes[0]
    lines = ['Figure S3. Test-time augmentation (flips; members x views averaged).',
             'a: LNK per scripta dev well (out-of-fold) and HeLa sequence, without vs with TTA. '
             'b: normalised area under the error-coverage curve when links are ranked by mutual information '
             '(1 = perfect, 0.5 = random).']
    for dataset, getter, marker in (('scripta', data.scripta_wells, 'o'), ('hela', data.hela_wells, 's')):
        base = getter('base')['LNK']
        with_tta = (getter('ours')['LNK'] if dataset == 'hela' else data.scripta_wells('ours').loc[base.index]['LNK'])
        ax.scatter(base, with_tta, s=8, marker=marker, color=style.DATASETS['hela' if dataset == 'hela' else 'scripta_cv'][1],
                   edgecolor=style.EDGE, lw=0.3, label='Scripta' if dataset == 'scripta' else 'HeLa')
        s = data.paired(with_tta, base)
        test = f', p = {data.fmt_p(s["p"])}' if s['n'] > 5 else ''
        lines.append(f'  {dataset} LNK: no TTA {base.mean():.4f}, TTA {with_tta.mean():.4f}; TTA higher in '
                     f'{s["a_higher"]}/{s["n"]}, lower in {s["b_higher"]}{test}')
    ax.plot([0.94, 1], [0.94, 1], color='#9A9A9A', lw=0.6, ls=':')
    ax.set(xlim=(0.945, 0.995), ylim=(0.945, 0.995), xlabel='LNK, no TTA', ylabel='LNK, TTA')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', handletextpad=0.2)
    style.panel(ax, 'a', x=-0.35)

    ax = axes[1]
    groups = [('scripta_cv_oof', 'Scripta'), ('hela_loso', 'HeLa')]
    x = 0
    ticks = []
    for prefix, name in groups:
        for population, short in (('false_links', 'false'), ('missed_links', 'missed')):
            for j, (suffix, key) in enumerate((('base', 'base'), ('tta', 'ours'))):
                m, _, _ = data.answers(f'{prefix}_{suffix}')
                area = m['coverage'][population]['mutual_info']['area']
                ax.bar(x + (j - 0.5) * 0.38, area, 0.38, color=style.color(key), edgecolor=style.EDGE, lw=0.3,
                       label=('No TTA', 'TTA')[j] if x == 0 else None)
                lines.append(f'  {name} {short} links, {("no TTA", "TTA")[j]}: MI area {area:.3f}')
            ticks.append((x, f'{name}\n{short}'))
            x += 1
    ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=5.5)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel('MI coverage area')
    grid(ax)
    ax.legend(loc='upper left', ncol=2, handlelength=1)
    style.panel(ax, 'b', x=-0.3)
    style.save(fig, 'figS3_tta', '\n'.join(lines))


def main():
    style.setup()
    plt.rcParams['hatch.linewidth'] = 0.4
    ablation()
    hela()
    tta()


if __name__ == '__main__':
    main()
