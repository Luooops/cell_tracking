"""Publication-style summary figures for evaluate.py results (identity association metrics).

One evaluation (one run) or several evaluations of the same sequences (model comparison) -> a four-panel figure:
    a  identity metrics: bar = mean across sequences, error bar = s.d., dots = individual sequences
    b  link recall by temporal gap dt (pooled over sequences; shows bridging of missed detections)
    c  outcome of every GT track: recovered / fragmented / mixed / missing (pooled, % of GT tracks)
    d  ID switches per 100 GT links (single run: per sequence; comparison: mean +/- s.d., dots = sequences)
Exports <name>.pdf (editable text), .svg, .png (600 dpi), <name>_source_data.csv with every plotted value and
<name>_legend.txt (figure legend: statistics, n, metric scope).
The metrics are this project's association-only scores on fixed CTC detections, not official CTC TRA/SEG.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

UNKNOWN_OFFSET = 50000
RATE_METRICS = [('perfect', 'Perfect tracks', 1.0), ('purity', 'Purity', 100.0),
                ('identity_link_recall', 'Link recall', 100.0), ('observation_coverage', 'Coverage', 100.0)]
OUTCOMES = [('recovered', 'Recovered'), ('fragmented', 'Fragmented'), ('mixed', 'Mixed'), ('missing', 'Missing')]
MODEL_NAMES = {'gnn_tracking': 'GNN', 'gnn_division': 'GNN-division', 'trackastra': 'Trackastra',
               'motion_tracking': 'Motion', 'classical_tracking': 'Classical'}

# Light, low-saturation family (NMI-pastel style); dark edges keep pale bars readable on white.
RUN_COLORS = ['#B4C0E4', '#F0C0CC', '#A8D8D0', '#F2D7A6', '#CFC0E8', '#BFDDB0', '#E4CCD8', '#D8D8D8']
OUTCOME_COLORS = {'recovered': '#A8D8C0', 'fragmented': '#F2D7A6', 'mixed': '#F0B8C0', 'missing': '#D8D8D8'}
SINGLE_COLOR = '#B4C0E4'
EDGE, INK, DOTS = '#606060', '#272727', '#4D4D4D'
MM = 1 / 25.4


def run_label(info, fallback):
    if info.get('model'):
        name = MODEL_NAMES.get(info['model'], info['model'])
        return f"{name} · {info['version']}" if info.get('version') else name
    return fallback


def load_evaluation(folder, label=None):
    """An evaluate.py output folder -> dict(label, wells={well: report}, outcomes={well: Counter})."""
    folder = Path(folder)
    reports = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    info_path = folder / 'run.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.is_file() else {}
    outcomes = {}
    for well in reports:
        counts = Counter()
        with (folder / well / 'per_gt_track.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['recovered'] == 'True':
                    counts['recovered'] += 1
                elif row['mixing'] == 'True':
                    counts['mixed'] += 1
                elif int(row['missing_predictions']):
                    counts['missing'] += 1
                else:
                    counts['fragmented'] += 1
        outcomes[well] = counts
    return dict(label=label or run_label(info, folder.name), wells=reports, outcomes=outcomes)


def _rates(run, key, scale):
    return np.array([run['wells'][w][key] * scale for w in run['wells'] if run['wells'][w].get(key) is not None])


def _links(report):
    return sum(v['total'] for v in report['identity_link_recall_by_dt'].values())


def _switch_rates(run):
    return {w: 100.0 * r['id_switches'] / _links(r) for w, r in run['wells'].items() if _links(r)}


def _gap_recall(run):
    """{dt label: (pooled recall %, n links)}; gaps >= 4 frames are pooled into one bin."""
    pooled = Counter(), Counter()
    for report in run['wells'].values():
        for dt, v in report['identity_link_recall_by_dt'].items():
            key = dt if int(dt) < 4 else '≥4'
            pooled[0][key] += v['retained']
            pooled[1][key] += v['total']
    return {k: (100.0 * pooled[0][k] / pooled[1][k], pooled[1][k]) for k in pooled[1]}


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.right': False,
        'axes.spines.top': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'legend.frameon': False, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK, 'text.color': INK})


def _bars_with_dots(ax, x, values, width, color, seed):
    mean = values.mean() if len(values) else np.nan
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    ax.bar(x, mean, width, color=color, edgecolor=EDGE, linewidth=0.5, zorder=2)
    if len(values) > 1:
        ax.errorbar(x, mean, yerr=sd, fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
    jitter = np.random.default_rng(seed).uniform(-width * 0.22, width * 0.22, len(values))
    ax.scatter(x + jitter, values, s=5, color=DOTS, linewidths=0, alpha=0.85, zorder=5)
    return mean, sd


def _panel_label(fig, ax, letter):
    box = ax.get_position()
    fig.text(box.x0 - 0.045, box.y1 + 0.035, letter, fontsize=8, fontweight='bold', va='bottom', ha='left')


def plot_metrics(runs, output_base, title=None, before_save=None):
    """runs: list of load_evaluation() dicts (1 = single run, >1 = comparison). Returns written paths.
    before_save(fig, output_base) runs after the final layout draw (e.g. a panel-alignment audit)."""
    _style()
    import matplotlib.pyplot as plt
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    compare = len(runs) > 1
    colors = RUN_COLORS if compare else [SINGLE_COLOR]
    if compare and len(runs) > len(RUN_COLORS):
        raise ValueError(f'At most {len(RUN_COLORS)} runs per comparison figure')
    source = []

    fig = plt.figure(figsize=(180 * MM, 112 * MM))
    grid = fig.add_gridspec(2, 3, left=0.07, right=0.985, bottom=0.11 if compare else 0.07,
                            top=0.86 if title else 0.9, wspace=0.42, hspace=0.62)
    ax_a, ax_b = fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2])
    ax_c, ax_d = fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2])

    # a: identity metrics
    n_runs = len(runs)
    width = 0.8 / n_runs
    for m, (key, name, scale) in enumerate(RATE_METRICS):
        for r, run in enumerate(runs):
            x = m + (r - (n_runs - 1) / 2) * width
            values = _rates(run, key, scale)
            mean, sd = _bars_with_dots(ax_a, x, values, width * 0.92, colors[r % len(colors)], seed=31 * m + r)
            if not compare:
                top = max(mean + sd, values.max()) if len(values) else mean
                ax_a.text(x, top + 2.5, f'{mean:.1f}', ha='center', va='bottom', fontsize=6)
            source.append(dict(panel='a', run=run['label'], metric=name, statistic='mean', value=mean, sd=sd,
                               n=len(values)))
            source += [dict(panel='a', run=run['label'], metric=name, statistic=f'sequence {w}',
                            value=run['wells'][w][key] * scale, sd='', n=1)
                       for w in run['wells'] if run['wells'][w].get(key) is not None]
    ax_a.set_xticks(range(len(RATE_METRICS)), [n for _, n, _ in RATE_METRICS])
    ax_a.set_ylim(0, 110)
    ax_a.set_yticks(range(0, 101, 20))
    ax_a.set_ylabel('Score (%)')
    ax_a.set_xlim(-0.6, len(RATE_METRICS) - 0.4)
    ax_a.tick_params(axis='x', length=0)

    # b: link recall by gap
    gap_keys = sorted({k for run in runs for k in _gap_recall(run)}, key=lambda k: 99 if k == '≥4' else int(k))
    for r, run in enumerate(runs):
        recall = _gap_recall(run)
        for g, key in enumerate(gap_keys):
            if key not in recall:
                continue
            x = g + (r - (n_runs - 1) / 2) * width
            value, n = recall[key]
            ax_b.bar(x, value, width * 0.92, color=colors[r % len(colors)], edgecolor=EDGE, linewidth=0.5, zorder=2)
            source.append(dict(panel='b', run=run['label'], metric=f'link recall dt={key}', statistic='pooled',
                               value=value, sd='', n=n))
    totals = Counter()
    for run in runs[:1]:
        for key, (_, n) in _gap_recall(run).items():
            totals[key] = n
    ax_b.set_xticks(range(len(gap_keys)), [f'{k}\n(n={totals[k]})' for k in gap_keys])
    ax_b.set_xlabel('Gap between GT observations (frames)')
    ax_b.set_ylabel('Link recall (%)')
    ax_b.set_ylim(0, 105)
    ax_b.set_yticks(range(0, 101, 20))
    ax_b.set_xlim(-0.6, len(gap_keys) - 0.4)
    ax_b.tick_params(axis='x', length=0)

    # c: GT track outcomes (100% stacked)
    if compare:
        groups = [(run['label'], sum(run['outcomes'].values(), Counter())) for run in runs]
    else:
        run = runs[0]
        groups = [(w, run['outcomes'][w]) for w in run['wells']] + [('All', sum(run['outcomes'].values(), Counter()))]
    for i, (name, counts) in enumerate(groups):
        total = sum(counts.values()) or 1
        bottom = 0.0
        for key, label in OUTCOMES:
            share = 100.0 * counts[key] / total
            ax_c.bar(i, share, 0.62, bottom=bottom, color=OUTCOME_COLORS[key], edgecolor=EDGE, linewidth=0.4,
                     label=label if i == 0 else None, zorder=2)
            if share >= 7:
                ax_c.text(i, bottom + share / 2, f'{share:.0f}', ha='center', va='center', fontsize=5.5)
            bottom += share
            source.append(dict(panel='c', run=name, metric=f'GT tracks {label.lower()}', statistic='% of GT tracks',
                               value=share, sd='', n=total))
    ax_c.set_xticks(range(len(groups)), [g.replace(' · ', '\n') for g, _ in groups])
    ax_c.set_xlim(-0.6, len(groups) - 0.4)
    ax_c.set_ylim(0, 100)
    ax_c.set_yticks(range(0, 101, 20))
    ax_c.set_ylabel('GT tracks (%)')
    ax_c.tick_params(axis='x', length=0)
    ax_c.legend(ncol=4, loc='lower right', bbox_to_anchor=(1.0, 1.01), handlelength=0.9, handleheight=0.9,
                columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

    # d: ID switches per 100 GT links
    if compare:
        for r, run in enumerate(runs):
            values = np.array(list(_switch_rates(run).values()))
            mean, sd = _bars_with_dots(ax_d, r, values, 0.62, colors[r % len(colors)], seed=97 + r)
            source.append(dict(panel='d', run=run['label'], metric='ID switches per 100 links', statistic='mean',
                               value=mean, sd=sd, n=len(values)))
        ax_d.set_xticks([])                     # runs are identified by colour (legend in a)
        ax_d.set_xlabel('Runs (colours as in a)')
    else:
        rates = _switch_rates(runs[0])
        for i, (well, value) in enumerate(rates.items()):
            ax_d.bar(i, value, 0.62, color=SINGLE_COLOR, edgecolor=EDGE, linewidth=0.5, zorder=2)
            source.append(dict(panel='d', run=runs[0]['label'], metric='ID switches per 100 links',
                               statistic=f'sequence {well}', value=value, sd='', n=_links(runs[0]['wells'][well])))
        ax_d.set_xticks(range(len(rates)), list(rates))
        n_runs_d = len(rates)
    ax_d.set_xlim(-0.6, (n_runs if compare else n_runs_d) - 0.4)
    ax_d.set_ylim(bottom=0)
    ax_d.set_ylabel('ID switches per 100 GT links')
    ax_d.tick_params(axis='x', length=0)

    if not compare and len(groups) > 8:
        for t in ax_c.get_xticklabels() + ax_d.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('right')
            t.set_rotation_mode('anchor')
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.yaxis.grid(True, color='#E6E6E6', linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    if compare:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[r], edgecolor=EDGE, linewidth=0.5)
                   for r in range(n_runs)]
        ax_a.legend(handles, [run['label'] for run in runs], ncol=min(n_runs, 4), loc='lower right',
                    bbox_to_anchor=(1.0, 1.01), handlelength=0.9, handleheight=0.9, columnspacing=1.0,
                    handletextpad=0.4, borderaxespad=0)
    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), 'abcd'):
        _panel_label(fig, ax, letter)
    n_seq = len(next(iter(runs))['wells'])
    legend = (f"{'Comparison of ' + str(n_runs) + ' runs' if compare else runs[0]['label']} on the same "
              f"{n_seq} sequences ({', '.join(runs[0]['wells'])}). "
              f"a, Perfect tracks (GT tracks carried by one predicted track that contains no other GT track), "
              f"purity, link recall (consecutive GT observations kept under one predicted ID) and coverage (GT "
              f"observations with a prediction); bars, mean across sequences; error bars, s.d.; dots, individual "
              f"sequences (n = {n_seq}). b, Link recall by frame gap between consecutive GT observations, pooled "
              f"over sequences; n, number of GT links. c, Outcome of every GT track "
              f"({'pooled per run' if compare else 'per sequence and pooled'}): recovered (perfect), fragmented "
              f"(several predicted IDs), mixed (shares a predicted track with another GT track), missing "
              f"(unpredicted observations); numbers, % of GT tracks. d, ID switches per 100 GT links"
              f"{'; bars, mean across sequences; error bars, s.d.; dots, sequences' if compare else ' per sequence'}. "
              f"Association-only scores on fixed CTC detections; mask labels ≥{UNKNOWN_OFFSET} (unknown "
              f"identity) are excluded. Not the official CTC TRA/SEG measures.")
    if title:
        fig.text(0.07, 0.975, title, fontsize=7.5, fontweight='bold', ha='left', va='top')

    fig.canvas.draw()
    if before_save:
        before_save(fig, output_base)
    written = []
    for suffix, kwargs in (('.pdf', {}), ('.svg', {}), ('.png', dict(dpi=600))):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    legend_path = output_base.parent / f'{output_base.name}_legend.txt'
    legend_path.write_text(legend + '\n', encoding='utf-8')
    written.append(legend_path)
    data_path = output_base.parent / f'{output_base.name}_source_data.csv'
    with data_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['panel', 'run', 'metric', 'statistic', 'value', 'sd', 'n'])
        writer.writeheader()
        writer.writerows(source)
    return written + [data_path]
