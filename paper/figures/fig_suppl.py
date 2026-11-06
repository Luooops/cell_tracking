"""Supplementary figures.
S1 (full width)  cue ablation: ours without the appearance or the velocity input, scripta OOF and the 13 DNN videos
                 (no TTA). Needs the --retrack runs (keys base, noapp, novel); skipped while they are missing.
S2 (one column)  HeLa, never trained on: ours and its binary variant vs Trackastra, per sequence.
S3 (one column)  test-time augmentation: tracking unchanged, error localisation (MI coverage area). Needs the --retrack
                 runs as S1."""
import matplotlib.pyplot as plt
import numpy as np

import data
import style

ABLATION = ['base', 'noapp', 'novel']
ABL_LABEL = {'base': 'Full', 'noapp': 'No appearance', 'novel': 'No velocity'}
DEV = None                      # set in main: the 12 development wells


def dots(ax, x, values, key, marker='o'):
    ax.scatter(np.full(len(values), x), values, s=8, marker=marker, color=style.light(key), edgecolor=style.color(key),
               lw=0.5, zorder=3)
    ax.plot([x - 0.25, x + 0.25], [np.mean(values)] * 2, color=style.EDGE, lw=1.0, zorder=4)


def grid(ax):
    ax.yaxis.grid(True, color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)


def ablation():
    fig = plt.figure(figsize=(style.FULL, 48 * style.MM))
    gs = fig.add_gridspec(1, 5, left=0.065, right=0.995, bottom=0.25, top=0.88, wspace=0.55)
    panels = [('scripta', 'LNK'), ('scripta', 'CT'), ('dnn', 'LNK'), ('dnn', 'CT'), ('dnn', 'BC(0)')]
    lines = ['Figure S1. Cue ablation (no TTA): the trained model with the appearance or the velocity input removed '
             'at test time (--ablate); scripta 12 dev wells out-of-fold, DNN 13 videos (final model). '
             'Black bar = mean. Paired Wilcoxon over the wells / videos:']
    for i, (dataset, metric) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        values = {k: (data.scripta_wells(k, DEV) if dataset == 'scripta' else data.dnn_wells(k))[metric] for k in ABLATION}
        for j, k in enumerate(ABLATION):
            dots(ax, j, values[k], k)
        ax.set_xticks(range(len(ABLATION)), [ABL_LABEL[k] for k in ABLATION], rotation=30, ha='right',
                      rotation_mode='anchor')
        ax.set_ylabel(metric)
        ax.set_title(style.DS if dataset == 'scripta' else 'DNN', loc='left', fontsize=6.5)
        grid(ax)
        style.panel(ax, 'abcde'[i], x=-0.3)
        means = ', '.join(f'{ABL_LABEL[k]} {values[k].mean():.4f}' for k in ABLATION)
        lines.append(f'  {dataset} {metric}: {means}')
        if metric != 'BC(0)':
            for k in ABLATION[1:]:
                s = data.paired(values['base'], values[k])
                lines.append(f'    full vs {ABL_LABEL[k]}: full higher in {s["a_higher"]}/{s["n"]}, p = {data.fmt_p(s["p"])}')
    style.save(fig, 'figS1_cue_ablation', '\n'.join(lines))


def hela():
    keys = ['ours', 'binary', 'ta_ft_div', 'ta_zs_div', 'ta_zs_ctc']
    metrics = [('LNK', 'LNK'), ('CT', 'CT'), ('BC(0)', 'BC(0)'), ('div_recall', 'Division recall')]
    fig, ax = plt.subplots(figsize=(style.COLUMN, 56 * style.MM))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.3, top=0.96)
    width = 0.16
    lines = ['Figure S2. HeLa (Fluo-N2DL-HeLa 01/02), never trained on, CTC measures on fixed detections. Ours and '
             'ours binary = final models (scripta + RAW264.7) with TTA; Trackastra ft = fine-tuned from general_2d on '
             'the same data; Trackastra zero-shot general_2d (HeLa not in its training data); '
             'Trackastra ctc (hatched) was trained on all CTC GT including HeLa and is shown for reference only. '
             'Bars = mean of 01 and 02, dots = sequences. Division recall at tolerance 0 = tp / (tp + fn).']
    for j, k in enumerate(keys):
        t = data.hela_wells(k)
        for i, (col, _) in enumerate(metrics):
            x = i + (j - 2) * width
            ctc = k == 'ta_zs_ctc'
            ax.bar(x, t[col].mean(), width * 0.9, color=style.light(k, 0.75 if ctc else 0.55),
                   edgecolor=style.color('ta_ft_dt3' if ctc else k), lw=0.6, hatch='//////' if ctc else None,
                   label=style.label(k) if i == 0 else None)
            ax.scatter([x - 0.04, x + 0.04], t[col].values, s=4, color='white', edgecolor=style.EDGE, lw=0.4, zorder=3)
        lines.append(f'  {style.plain(k)}: ' + '; '.join(
            f'{name} {" / ".join(f"{v:.3f}" for v in t[col])}' for col, name in metrics)
            + f'; false divisions {" / ".join(str(int(v)) for v in t["fp_div(0)"])}')
    ax.set_xticks(range(len(metrics)), [name for _, name in metrics])
    ax.set_ylim(0.0, 1.02)
    grid(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=1.2, columnspacing=0.8)
    style.save(fig, 'figS2_hela', '\n'.join(lines))


def tta():
    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 42 * style.MM), gridspec_kw=dict(width_ratios=[1, 1.3]))
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.9, wspace=0.5)
    ax = axes[0]
    lines = ['Figure S3. Test-time augmentation (flips; members x views averaged).',
             'a: LNK per scripta dev well (out-of-fold) and DNN video (13), without vs with TTA. '
             'b: normalised area under the error-coverage curve when links are ranked by mutual information '
             '(1 = perfect, 0.5 = random).']
    for dataset, getter, marker in (('scripta', lambda k: data.scripta_wells(k, DEV), 'o'), ('dnn', data.dnn_wells, 's')):
        base = getter('base')['LNK']
        with_tta = getter('ours').loc[base.index]['LNK']
        col = style.DATASETS['dnn' if dataset == 'dnn' else 'scripta_cv'][1]
        ax.scatter(base, with_tta, s=9, marker=marker, color=style.light(col), edgecolor=col, lw=0.5,
                   label=style.DS if dataset == 'scripta' else 'DNN')
        s = data.paired(with_tta, base)
        test = f', p = {data.fmt_p(s["p"])}' if s['n'] > 5 else ''
        lines.append(f'  {dataset} LNK: no TTA {base.mean():.4f}, TTA {with_tta.mean():.4f}; TTA higher in '
                     f'{s["a_higher"]}/{s["n"]}, lower in {s["b_higher"]}{test}')
    ax.plot([0.94, 1], [0.94, 1], color='#9A9A9A', lw=0.6, ls=':')
    ax.set(xlim=(0.945, 1.0), ylim=(0.945, 1.0), xlabel='LNK, no TTA', ylabel='LNK, TTA')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', handletextpad=0.2)
    style.panel(ax, 'a', x=-0.35)

    ax = axes[1]
    groups = [('scripta_cv_oof', style.DS), ('dnn', 'DNN')]
    x = 0
    ticks = []
    for prefix, name in groups:
        for population, short in (('false_links', 'false'), ('missed_links', 'missed')):
            for j, (suffix, key) in enumerate((('base', 'base'), ('tta', 'ours'))):
                m, _, _ = data.answers(f'{prefix}_{suffix}' if prefix != 'dnn' or suffix == 'base' else 'dnn')
                area = m['coverage'][population]['mutual_info']['area']
                ax.bar(x + (j - 0.5) * 0.38, area, 0.34, color=style.light(key, 0.85 if key == 'base' else 0.45),
                       edgecolor=style.color(key), lw=0.6,
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
    global DEV
    DEV = [w for w in data.all_wells() if w not in data.TEST_WELLS]
    style.setup()
    plt.rcParams['hatch.linewidth'] = 0.4
    hela()
    if data.has('base') and data.has('base', 'dnn'):
        ablation()
        tta()
    else:
        print('figures S1 and S3 skipped: no --retrack results (base, noapp, novel) in final_results_v2 yet')


if __name__ == '__main__':
    main()
