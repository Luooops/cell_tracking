"""Shared figure style for the paper (CVPR two-column layout): sizes, fonts, one fixed colour per method, saving."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
COLUMN = 82.5 * MM          # \linewidth of one column (3.25 in)
FULL = 174.6 * MM           # \textwidth (6.875 in)
OUT = Path(__file__).resolve().parent / 'out'

EDGE = '#404040'
GRID = '#E6E6E6'

# label, colour (fixed for every figure)
METHODS = {
    'ours': ('ProGCT', '#3B6FB6'),
    'binary': ('ProGCT, binary', '#8FB3E3'),
    # Trackastra links with or without divisions as the user chooses: 'div' = greedy (divisions allowed, not told
    # whether the data has any: the setting of our model); no div. = greedy_nodiv (told that scripta has none)
    'ta_ft_div': ('Trackastra ft', '#C2452D'),
    'ta_zs_div': ('Trackastra zs', '#EE9A86'),
    'ta_ft': ('Trackastra ft‡', '#E07B39'),            # ‡ = divisions forbidden (greedy_nodiv), as in the paper
    'ta_zs': ('Trackastra zs‡', '#F2B880'),
    'ta_ft_dt3': ('Trackastra ft‡, $\\Delta t$=3', '#B5541E'),   # the Delta t = 3 runs also use greedy_nodiv
    'ta_zs_dt3': ('Trackastra zs‡, $\\Delta t$=3', '#D9965A'),
    'ta_zs_ctc': ('Trackastra ctc', '#F7D9B8'),
    'ctgnn0': ('CellTrack-GNN s0', '#9C7FCB'),
    'ctgnn1': ('CellTrack-GNN s1', '#C5B3E6'),
    'classical': ('Classical', '#A6A6A6'),
    't1': ('ProGCT, HeLa only', '#7FA7D9'),
    'raw': ('ProGCT, RAW264.7 only', '#1F4E8C'),
    'ours_nodiv': ('ProGCT, no div.', '#3B6FB6'),
    'binary_nodiv': ('ProGCT, binary, no div.', '#8FB3E3'),
    'noapp': ('No appearance', '#2A9D8F'),
    'novel': ('No velocity', '#C8607A'),
    'base': ('No TTA', '#6C93CC'),
}
DS = 'ScriptaLive'           # name of the proprietary dataset in every figure (the paper's \ds macro)
DATASETS = {                # populations of the confidence figure
    'scripta_cv': (f'{DS} dev. (OOF)', '#3B6FB6'),
    'scripta_test': (f'{DS} test', '#79A6DC'),
    'dnn': ('DNN', '#2A9D8F'),
    'hela': ('HeLa (unseen)', '#C9822F'),
}


def label(key):
    return METHODS[key][0]


def plain(key):
    """Label without mathtext, for the legend text files."""
    return label(key).replace('$\\Delta t$', 'delta_t')


def color(key):
    return METHODS[key][1]


def light(key, amount=0.55):
    """Light fill: a method's colour (or any colour) mixed with white. The method colour itself is the edge."""
    rgb = np.array(mcolors.to_rgb(METHODS[key][1] if key in METHODS else key))
    return tuple(rgb + (1 - rgb) * amount)


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 6, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2, 'axes.spines.right': False, 'axes.spines.top': False,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.titleweight': 'normal', 'lines.linewidth': 0.8, 'mathtext.default': 'regular'})


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', ha='right', va='bottom')


def save(fig, name, legend):
    """out/<name>.pdf + .png, and out/<name>_legend.txt with the statistics and a caption draft."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)
    (OUT / f'{name}_legend.txt').write_text(legend.strip() + '\n', encoding='utf-8')
    print('written', OUT / f'{name}.pdf')
