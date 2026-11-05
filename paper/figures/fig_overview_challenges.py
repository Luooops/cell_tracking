"""Main figure of the paper.

a  Task: a segmented time-lapse in, tracks, lineages and link confidence out (a real DynamicNuclearNet clip with a
   division, tracked by ProGCT).
b  Current learned trackers vs ProGCT, one column per contribution: the setting the user of a current tracker must
   choose (temporal window, division switch, threshold / temperature) and what it costs, above the ProGCT component
   that replaces it (frame-gap graph with cross-frame check, three-class edges, calibrated decisions).
c  Results: one real example and the measured evidence per contribution.

Vivid architecture channel (single candidate), drawn with Matplotlib so that real microscopy, real masks and real
model outputs sit inside an editable vector layout. Every microscopy tile is a crop of data/ (read only); outlines
are ground-truth mask contours; colours and arrows of tracker output are links the named tracker actually made
(development/final_results_v2); every number comes from development/paper_results/key_numbers.csv or the gap-link
table of figure 2. Icons and module glyphs are schematic.

    python paper/figures/fig_overview_challenges.py [--out DIR]
writes fig.svg / fig.pdf / fig.png, structure.json, facts.json, audit.json and parts/ (one SVG per semantic part)
to DIR (default development/overview_design/challenges).
"""
import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import ndimage as ndi
from skimage import measure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
from utils.lineage import UNKNOWN_OFFSET  # noqa: E402

ROOT = D.ROOT
RESULTS = ROOT / 'development' / 'paper_results'

# ---------------------------------------------------------------- layout (mm, y downwards)
W, H = 174.6, 89.4
A_X, A_W = 0.4, 44.6                                  # panel a
RX = A_X + A_W + 1.6                                  # right block: b (two bands) and c (results band)
GAP = 1.6
CW = (W - 0.4 - RX - 2 * GAP) / 3
KEYS = ['gap', 'div', 'trust']
COLX = {k: RX + i * (CW + GAP) for i, k in enumerate(KEYS)}
NUM = {'gap': 1, 'div': 2, 'trust': 3}
TITLES = {'gap': 'Missed detections', 'div': 'Cell division', 'trust': 'Link reliability'}
HEAD_Y = 2.6                                          # column titles
BAND = dict(cur=(5.4, 24.2), pro=(26.4, 50.6), res=(55.2, H - 0.4))
RES_HEAD_Y = 53.0                                     # panel c title line
CARD_PAD = 2.6                                        # band top -> card top (room for the band's tab)

BLUE, BLUE_D, BLUE_BAND = '#3B6FB6', '#24497F', '#E1EBF8'
RED_D, RED_BAND, RED_EDGE, RED_TXT = '#A23B2E', '#FBEEEB', '#D9A79C', '#8E3326'
ORANGE = '#E07B39'                                    # Trackastra (style.METHODS ta_ft)
GOLD = '#E3A21A'
INK, MUTED = '#1F2A33', '#5E6B75'
GREEN_GT = '#7FD35B'
RED_M, PURPLE, MAGENTA = '#D64545', '#8A6BC4', '#C2479A'
PT = 72 / 25.4
# identity colours of panel a (no role colour: no red, orange, gold, magenta except the dividing family)
ID_COLORS = ['#4E79A7', '#59A14F', '#76B7B2', '#9C755F', '#8CD17D', '#A0CBE8', '#2F4B7C', '#B6992D', '#D4A6C8',
             '#6B6ECF', '#79706E', '#86A0CC']

# crop centred (row, col) so that both divisions of the window show both daughters at frame 38
EX_TASK = dict(video='val20', mother=15, frames=(31, 34, 38), tail=12, crop=136, centre=(80, 420))
EX_GAP = dict(well='r08c13', t=17, label=453)
EX_DIV = dict(video='val18', mother=83)          # another held-out division than panel a's
EX_TRUST = dict(well='r15c12', x0=0, y0=1020, w=520, h=240)
TRUST_FRAMES = (11, 19)

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=str(ROOT / 'development' / 'overview_design' / 'challenges'))
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'image.composite_image': False, 'mathtext.default': 'regular', 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold'})
fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')
FACTS = {}
STRUCT = dict(parts={}, connections=[])
_gid = [0]


def gid(artist, part):
    _gid[0] += 1
    artist.set_gid(f'{part}__{_gid[0]}')
    return artist


def txt(part, x, y, s, fs=6, c=INK, b=False, ha='left', va='center', it=False, z=9, **kw):
    return gid(ax.text(x, y, s, fontsize=fs, color=c, fontweight='bold' if b else 'normal', ha=ha, va=va,
                       fontstyle='italic' if it else 'normal', zorder=z, **kw), part)


def rbox(part, x, y, w, h, fc, ec='none', r=1.2, lw=0.5, ls='-', z=0):
    return gid(ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec,
                                           lw=lw, ls=ls, zorder=z)), part)


def arrow(part, a, b, c=BLUE, lw=0.8, rad=0.0, ls='-', z=7, head=4.5, shrink=0):
    return gid(ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=head, color=c, lw=lw, linestyle=ls,
                                            connectionstyle=f'arc3,rad={rad}', shrinkA=shrink, shrinkB=shrink,
                                            zorder=z)), part)


def line(part, xs, ys, c, lw=0.6, ls='-', z=6):
    return gid(ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle='round')[0], part)


def centred(part, x, cy, s, fs, c='white', b=True, z=9):
    """Optical vertical centring: baseline half a Times cap height (0.662 em) below the centre."""
    return txt(part, x, cy + 0.662 * fs / PT / 2, s, fs=fs, c=c, b=b, ha='center', va='baseline', z=z)


def text_width(s, fs, b=False, it=False):
    t = ax.text(0, 0, s, fontsize=fs, fontweight='bold' if b else 'normal', fontstyle='italic' if it else 'normal')
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def runs(part, x, y, pieces, ha='left'):
    """Text of several styles on one line: pieces = [(text, dict(fs, c, b, it))]; returns the end x."""
    widths = [text_width(s, st.get('fs', 6), st.get('b', False), st.get('it', False)) for s, st in pieces]
    x0 = {'center': x - sum(widths) / 2, 'right': x - sum(widths)}.get(ha, x)
    for (s, st), w in zip(pieces, widths):
        txt(part, x0, y, s, fs=st.get('fs', 6), c=st.get('c', INK), b=st.get('b', False), it=st.get('it', False))
        x0 += w
    return x0


def tab(part, x, y, s, fc, fs=6.0):
    """Filled label tab sitting on the top edge of a band."""
    w = text_width(s, fs, b=True) + 2.2
    rbox(part, x, y - 1.2, w, 2.4, fc, r=0.6, z=8)
    centred(part, x + w / 2, y, s, fs)
    return w


def letter(x, y, s):
    txt('panel_letters', x, y, s, fs=8.5, b=True, c=INK)


# ---------------------------------------------------------------- image helpers
def scripta_img(well, t):
    return tifffile.imread(D.SCRIPTA / well / f't{t:03d}.tif').astype(float)


def scripta_mask(well, t):
    return tifffile.imread(D.SCRIPTA / f'{well}_GT' / 'TRA' / f'man_track{t:03d}.tif')


def dnn_img(video, t):
    return tifffile.imread(D.DNN / video / f't{t:03d}.tif').astype(float)


def dnn_mask(video, t):
    return tifffile.imread(D.DNN / f'{video}_GT' / 'TRA' / f'man_track{t:03d}.tif')


def centroid(mask, label):
    yy, xx = np.nonzero(mask == label)
    return float(yy.mean()), float(xx.mean())


class Tile:
    """A raw crop placed at (x, y, w, h) mm; px -> mm mapping for vector overlays. z: base drawing order, so that
    stacked tiles keep their outlines under the tiles in front."""

    def __init__(self, part, image, cy, cx, x, y, w, h, hpx, lo_hi=None, sigma=0.0, z=3, edge='#8A949C'):
        self.sigma, self.z = sigma, z
        wpx = hpx * w / h
        y0 = int(round(np.clip(cy - hpx / 2, 0, image.shape[0] - hpx)))
        x0 = int(round(np.clip(cx - wpx / 2, 0, image.shape[1] - wpx)))
        self.y0, self.x0, self.hpx, self.wpx = y0, x0, int(round(hpx)), int(round(wpx))
        crop = image[y0:y0 + self.hpx, x0:x0 + self.wpx]
        lo, hi = lo_hi if lo_hi is not None else np.percentile(crop, [1, 99.7])
        self.x, self.y, self.w, self.h, self.part = x, y, w, h, part
        gid(ax.imshow(np.clip((crop - lo) / (hi - lo), 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='lanczos',
                      extent=(x, x + w, y + h, y), zorder=z), part)
        gid(ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=edge, lw=0.35, zorder=z + 0.8)), part)
        self.rect = ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='none', zorder=0))

    def clip(self, artist):
        artist.set_clip_path(self.rect)
        return artist

    def mm(self, py, px):
        return self.x + (px - self.x0) / self.wpx * self.w, self.y + (py - self.y0) / self.hpx * self.h

    def outline(self, mask, label, c, lw=0.7, ls='-'):
        """Ground-truth mask contour on a 4x bilinear upsampling (smoothed for coarse masks)."""
        sub = (mask[self.y0:self.y0 + self.hpx, self.x0:self.x0 + self.wpx] == label).astype(float)
        if not sub.any():
            return
        up = 4
        sub = ndi.zoom(np.pad(sub, 1), up, order=1)
        if self.sigma:
            sub = ndi.gaussian_filter(sub, self.sigma * up)
        for cc in measure.find_contours(sub, 0.5):
            cc = (cc + 0.5) / up - 1
            xs = np.clip(self.x + cc[:, 1] / self.wpx * self.w, self.x, self.x + self.w)
            ys = np.clip(self.y + cc[:, 0] / self.hpx * self.h, self.y, self.y + self.h)
            self.clip(line(self.part, xs, ys, c, lw=lw, ls=ls, z=self.z + 0.5))

    def ring(self, mask, label, c, lw=0.7, ls='-', pad=1.25):
        """Highlight ellipse fitted to a coarse mask (moments), enlarged by pad."""
        props = measure.regionprops((mask == label).astype(np.uint8))[0]
        x, y = self.mm(*props.centroid)
        sx, sy = self.w / self.wpx, self.h / self.hpx
        a, b = props.major_axis_length * pad, max(props.minor_axis_length * pad, 6)
        self.clip(gid(ax.add_patch(Ellipse((x, y), a * sx, b * sy, angle=90 - np.degrees(props.orientation),
                                           fc='none', ec=c, lw=lw, ls=ls, zorder=self.z + 0.5)), self.part))

    def label(self, s, corner='tl', c='white', fs=5.0):
        x = self.x + 0.6 if corner[1] == 'l' else self.x + self.w - 0.6
        y = self.y + 1.2 if corner[0] == 't' else self.y + self.h - 1.2
        txt(self.part, x, y, s, fs=fs, c=c, ha='left' if corner[1] == 'l' else 'right', z=self.z + 0.9,
            bbox=dict(boxstyle='round,pad=0.12,rounding_size=0.3', fc=(0, 0, 0, 0.5), ec='none'))


def window_levels(images, cy, cx, r):
    vals = np.concatenate([im[int(max(0, cy - r)):int(cy + r), int(max(0, cx - r)):int(cx + r)].ravel() for im in images])
    return np.percentile(vals, [1, 99.7])


# ---------------------------------------------------------------- facts from the results
numbers = pd.read_csv(RESULTS / 'key_numbers.csv')


def number(section, quantity, scope=None):
    """One value of key_numbers.csv; scope (substring) picks the row when a quantity is listed for several data sets."""
    row = numbers[(numbers.section == section) & (numbers.quantity == quantity)]
    if scope is None and len(row) > 1:
        scope = {'data': 'Scripta', 'confidence': 'dev wells OOF'}.get(section)
    if scope is not None:
        row = row[row.scope.str.contains(scope, regex=False)]
    if len(row) != 1:
        raise KeyError(f'{section} / {quantity}')
    return row.iloc[0]['value']


gaps = D.gap_counts().groupby('method')[['gt_g1', 'gt_g2', 'gt_g3+', 'missed_g2']].sum()
ablation_gap = str(number('ablation', 'LNK full -> without cross-frame check', '12 dev wells'))
F = dict(
    rejoined=float(number('gaps', 'share of g=2 links rejoined, ours')),
    false_gap_dt3=int(float(number('gaps', 'false gap links (g>=2), Trackastra ft delta_t=3'))),
    nocheck_wells=ablation_gap.split('(')[1].split(' ')[0],
    bc=(float(number('dnn', 'BC(0), ours')), float(number('dnn', 'BC(0), ta_ft_div'))),
    false_div_dnn=(int(number('dnn', 'false divisions, ours')), int(number('dnn', 'false divisions, binary')),
                   int(number('dnn', 'false divisions, ta_ft_div'))),
    false_div_scripta=(int(number('divisions', 'false divisions, 15 wells, ours')),
                       int(number('divisions', 'false divisions, 15 wells, ta_ft_div'))),
    cover_false=float(number('confidence', 'false links found in the 5% most uncertain links made (MI)')),
    ece=str(number('confidence', 'undecided edges (0.02 < p < 0.98): n / mean p / observed / ECE')).split('/')[-1].strip(),
)
count_err = []
for scope in ('dev wells OOF', 'test wells', 'DNN 15 videos'):
    meas = [float(v) for v in number('confidence', 'measured false links / missed candidates', scope).split('/')]
    exp = [float(v) for v in number('confidence', 'expected false links / missed candidates', scope).split('/')]
    count_err += [abs(e - m) / m for e, m in zip(exp, meas)]
F['count_err'] = max(count_err)
FACTS['numbers'] = {k: (list(v) if isinstance(v, tuple) else (v if isinstance(v, (str, int)) else float(v)))
                    for k, v in F.items()}
FACTS['numbers']['ablation_cross_frame_check'] = ablation_gap


def tracks_of(key, well, dataset='scripta'):
    return pd.read_csv(D.track_folder(key, well, dataset) / 'instance_tracks.csv')


# ================================================================= frames: panel a card, bands, titles, letters
rbox('a_task', A_X, 0.6, A_W, H - 1.0, '#F7F8FA', ec='#D5DBE1', r=1.6, lw=0.5)
letter(A_X + 1.0, HEAD_Y, 'a')
txt('a_task', A_X + 4.2, HEAD_Y, 'Task: link nuclei through time', fs=6.8, b=True)
letter(RX - 0.2, HEAD_Y, 'b')
for k in KEYS:
    x = COLX[k] + CW / 2
    w = text_width(TITLES[k], 6.8, b=True) + 4.0
    x0 = x - w / 2 + (1.0 if k == 'gap' else 0)
    gid(ax.add_patch(Circle((x0 + 1.5, HEAD_Y), 1.5, fc=INK, ec='none', zorder=8)), f'{k}_title')
    centred(f'{k}_title', x0 + 1.5, HEAD_Y, str(NUM[k]), 6.0)
    txt(f'{k}_title', x0 + 3.8, HEAD_Y, TITLES[k], fs=6.8, b=True)

for band, fc, ec in (('cur', RED_BAND, RED_EDGE), ('pro', BLUE_BAND, BLUE), ('res', 'white', '#C9D3DB')):
    y0, y1 = BAND[band]
    rbox(f'band_{band}', RX - 0.6, y0, W - 0.2 - (RX - 0.6), y1 - y0, fc, ec=ec, r=1.6, lw=0.9 if band == 'pro' else 0.5)
tab('band_cur', RX + 0.4, BAND['cur'][0], 'Current trackers', RED_D)
tab('band_pro', RX + 0.4, BAND['pro'][0], 'ProGCT', BLUE)
letter(RX - 0.2, RES_HEAD_Y, 'c')
txt('panel_letters', RX + 3.0, RES_HEAD_Y, 'Results', fs=6.8, b=True, c=INK)


def card(part, k, band, fc='white', ec='none', lw=0.5):
    y0, y1 = BAND[band]
    rbox(part, COLX[k], y0 + CARD_PAD, CW, y1 - y0 - CARD_PAD - 0.8, fc, ec=ec, r=1.2, lw=lw, z=1)
    return COLX[k], y0 + CARD_PAD, CW, y1 - y0 - CARD_PAD - 0.8


# ================================================================= panel a: the task on a real clip
def panel_task():
    video, mother, frames, tail, crop = (EX_TASK[k] for k in ('video', 'mother', 'frames', 'tail', 'crop'))
    tr = tracks_of('ours', video, 'dnn')
    f = D.track_folder('ours', video, 'dnn')
    lin = {int(k): int(v) for k, v in json.loads((f / 'lineage.json').read_text()).items()}
    links = pd.read_csv(f / 'links.csv')
    gt = pd.read_csv(D.DNN / f'{video}_GT' / 'TRA' / 'man_track.txt', sep=' ', header=None, names=['l', 's', 'e', 'p'])
    te = int(gt[gt.l == mother].iloc[0].e)                       # mother's last frame
    cy, cx = EX_TASK['centre']
    ims = [dnn_img(video, t) for t in frames]
    lv = window_levels(ims, cy, cx, crop / 2)
    # input: three stacked frames with all detections outlined (no identity yet)
    s, off = 26.0, 3.4
    x0, y0 = A_X + 3.0, 7.2
    tiles = []
    for i, t in enumerate(frames):
        tile = Tile('a_task', ims[i], cy, cx, x0 + i * off, y0 + i * off, s, s, crop, lo_hi=lv, sigma=0.8, z=3 + 2 * i,
                    edge='white')
        m = dnn_mask(video, t)
        for lab in np.unique(m[tile.y0:tile.y0 + tile.hpx, tile.x0:tile.x0 + tile.wpx]):
            if lab:
                tile.ring(m, lab, '#D7DEE5', lw=0.35, pad=1.1)
        tile.label(f't = {t}', 'tl', fs=4.8) if i == len(frames) - 1 else None
        tiles.append(tile)
    yb = y0 + s + 2 * off
    txt('a_task', A_X + A_W / 2, yb + 2.3, 'input: segmented frames', fs=5.6, c=INK, b=True, ha='center')
    txt('a_task', A_X + A_W / 2, yb + 4.8, 'nuclei vanish, jump, touch and divide', fs=5.4, c=MUTED, ha='center',
        it=True)
    arrow('a_task', (A_X + A_W / 2, yb + 6.2), (A_X + A_W / 2, yb + 10.2), c=BLUE, lw=1.0, head=5.5)
    txt('a_task', A_X + A_W / 2 + 1.4, yb + 8.2, 'ProGCT', fs=5.8, c=BLUE, b=True)
    # output: tracks (identity colours, tails) on the last frame, and the lineage of the same cells
    ty = yb + 11.6
    th = 26.0
    out = Tile('a_task', ims[-1], cy, cx, A_X + 1.6, ty, 22.0, th, crop * th / 22.0, lo_hi=lv, sigma=0.8, z=3)
    # every nucleus visible in the tile (any mask pixel inside) gets a colour, a ring, a tail and a lineage line
    m_last = dnn_mask(video, frames[-1])
    visible = {int(l) for l in np.unique(m_last[out.y0:out.y0 + out.hpx, out.x0:out.x0 + out.wpx]) if l}
    last = tr[tr.frame_index == frames[-1]]
    inside = last[last.instance_id.isin(visible)].sort_values('x')
    assert len(inside) == len(visible), 'a visible nucleus has no track'
    t0, t1 = frames[-1] - tail, frames[-1]
    span = tr.groupby('track_id').frame_index.agg(['min', 'max'])
    mother_tid = int(tr[(tr.frame_index == te) & (tr.instance_id == mother)].track_id.iloc[0])
    # lineage units: a track whose parent divides inside the window [t0, t1] joins its parent's family; the
    # family is one colour (sisters share it) and its lines stay adjacent; the highlighted division is magenta
    xpos = dict(zip(inside.track_id.astype(int), inside.x))
    units = {}
    for tid in xpos:
        par = lin.get(tid)
        key = par if par is not None and span.loc[par, 'max'] >= t0 else tid
        units.setdefault(key, []).append(tid)
    order = sorted(units, key=lambda k_: np.mean([xpos[t_] for t_ in units[k_]]))
    colours, unit_colour, ci = {}, {}, 0
    for key in order:
        unit_colour[key] = MAGENTA if key == mother_tid else ID_COLORS[ci % len(ID_COLORS)]
        ci += key != mother_tid
        for t_ in units[key]:
            colours[t_] = unit_colour[key]
    assert ci <= len(ID_COLORS), 'more lineage units than identity colours'
    for r in inside.itertuples():
        tid = int(r.track_id)
        out.ring(m_last, int(r.instance_id), colours[tid], lw=0.8, pad=1.1)
        hist = tr[(tr.track_id.isin([tid, lin.get(tid, -1)])) & (tr.frame_index > frames[-1] - tail)
                  & (tr.frame_index <= frames[-1])].sort_values('frame_index')
        pts = [out.mm(y_, x_) for x_, y_ in zip(hist.x, hist.y)]
        if len(pts) > 1:
            out.clip(line('a_task', [p[0] for p in pts], [p[1] for p in pts], colours[tid], lw=0.6, z=out.z + 0.6))
    out.label('tracks', 'tl', fs=4.8)
    # lineage: time runs downwards, one line per track; the dividing family forks
    lx0, lx1, ly0, ly1 = A_X + 25.0, A_X + A_W - 1.6, ty + 1.0, ty + th - 1.0
    yt = lambda t_: ly0 + (t_ - t0) / (t1 - t0) * (ly1 - ly0)  # noqa: E731
    shown = [t_ for key in order for t_ in sorted(units[key], key=xpos.get)]   # every track of the tile
    xs = dict(zip(shown, np.linspace(lx0 + 0.8, lx1 - 1.4, len(shown))))
    step = (lx1 - 1.4 - lx0 - 0.8) / max(1, len(shown) - 1)
    for tid in shown:
        a_, b_ = max(t0, span.loc[tid, 'min']), min(t1, span.loc[tid, 'max'])
        line('a_task', [xs[tid], xs[tid]], [yt(a_), yt(b_)], colours[tid], lw=0.8, z=5)
    forks = []
    for key in order:
        if key in units[key]:                                     # a single track, no division in the window
            continue
        kids = units[key]
        px = np.mean([xs[t_] for t_ in kids]) if len(kids) > 1 else xs[kids[0]] - step / 2
        fy = yt(span.loc[key, 'max'])
        line('a_task', [px, px], [yt(max(t0, span.loc[key, 'min'])), fy], unit_colour[key], lw=0.8, z=5)
        for t_ in kids:
            line('a_task', [px, xs[t_]], [fy, yt(span.loc[t_, 'min'])], unit_colour[key], lw=0.8, z=5)
        assert all(c_ in xpos for c_, p_ in lin.items() if p_ == key), 'a daughter is outside the tile'
        for t_ in (c_ for c_, p_ in lin.items() if p_ == key and c_ not in xpos):   # daughter outside the tile
            line('a_task', [px, px - step * 0.35], [fy, yt(span.loc[t_, 'min'])], unit_colour[key], lw=0.6,
                 ls=(0, (1.2, 0.8)), z=5)
        forks.append(int(key))
    d = links[(links.kind == 'division') & (links.src_label == mother) & (links.src_frame == te)].p
    if len(d):
        out.label(f'division d = {d.min():.2f}', 'br', c='#F6BFE3', fs=4.8)
    txt('a_task', (lx0 + lx1) / 2, ty - 0.9, 'lineage', fs=5.0, c=MUTED, ha='center')
    arrow('a_task', (lx1 + 0.2, ly0), (lx1 + 0.2, ly1), c=MUTED, lw=0.4, head=3)
    txt('a_task', lx1 - 0.6, ly1 + 1.4, 'time', fs=4.8, c=MUTED, ha='center')
    yo = ty + th + 2.6
    txt('a_task', A_X + A_W / 2, yo, 'tracks + lineages + link confidence', fs=5.6, c=INK, b=True, ha='center')
    txt('a_task', A_X + A_W / 2, yo + 2.5, 'a link joins one nucleus in frames t and t+g', fs=5.2, c=MUTED,
        ha='center')
    FACTS['a'] = dict(video=video, mother=mother, frames=list(frames), tracks_shown=[int(t_) for t_ in shown],
                      divisions_in_window=forks, highlighted_mother_track=mother_tid)


# ================================================================= b: current trackers (settings) vs ProGCT
def icon_slider(part, x, y, w):
    line(part, [x, x + w], [y, y], RED_D, lw=0.7)
    for i, lab in enumerate(['1', '2', '3']):
        xx = x + i * w / 2
        line(part, [xx, xx], [y - 0.7, y + 0.7], RED_D, lw=0.5)
        txt(part, xx, y + 2.9, lab, fs=5.2, c=RED_TXT, ha='center')
    gid(ax.add_patch(FancyBboxPatch((x + w / 2 - 0.9, y - 1.4), 1.8, 2.8, boxstyle='round,pad=0,rounding_size=0.4',
                                    fc='white', ec=RED_D, lw=0.7, zorder=7)), part)
    txt(part, x + w / 2, y - 2.6, 'Δt', fs=5.2, c=RED_D, b=True, ha='center')


def icon_switch(part, x, y, w):
    rbox(part, x, y - 1.4, w, 2.8, '#F3D3CD', ec=RED_D, r=1.4, lw=0.7, z=6)
    gid(ax.add_patch(Circle((x + w - 1.4, y), 1.05, fc='white', ec=RED_D, lw=0.7, zorder=7)), part)
    txt(part, x + 0.5, y + 0.05, 'on', fs=5.0, c=RED_D, b=True)
    txt(part, x + w / 2, y + 2.8, 'off?', fs=4.8, c=RED_TXT, ha='center')


def icon_dial(part, x, y, r):
    gid(ax.add_patch(Arc((x, y), 2 * r, 2 * r, theta1=0, theta2=180, ec=RED_D, lw=0.7, zorder=6)), part)
    for a_ in np.linspace(0, np.pi, 5):
        line(part, [x + r * np.cos(a_), x + 0.75 * r * np.cos(a_)], [y - r * np.sin(a_), y - 0.75 * r * np.sin(a_)],
             RED_D, lw=0.45)
    line(part, [x, x + 0.8 * r * np.cos(2.2)], [y, y - 0.8 * r * np.sin(2.2)], RED_D, lw=0.9, z=7)
    gid(ax.add_patch(Circle((x, y), 0.4, fc=RED_D, ec='none', zorder=7)), part)
    txt(part, x, y + 1.6, 'τ, T', fs=5.0, c=RED_D, b=True, ha='center')


CUR = {
    'gap': ('Temporal window Δt', ['small Δt: a miss breaks the track;',
                                          'large Δt: links skip frames in', 'which the cell is present'],
            icon_slider),
    'div': ('Division switch', ['on: divisions appear where cells', 'never divide;',
                                'off: no lineages at all'], icon_switch),
    'trust': ('Threshold and temperature', ['class-weighted scores are not', 'probabilities; a fitted temperature',
                                            'needs annotated tracks'], icon_dial),
}


def node(part, x, y, c=BLUE, r=0.85, ghost=False):
    gid(ax.add_patch(Circle((x, y), r, fc='white' if ghost else c, ec=c, lw=0.6,
                            ls=(0, (1.2, 0.8)) if ghost else '-', zorder=7)), part)


def glyph_gap(part, x, y, w, h):
    """Frames t, t+1, t+2; the top cell is missed at t+1 (red ghost) and rejoined by an arc with the gap g as input."""
    cx = [x + w * 0.16, x + w * 0.42, x + w * 0.68]
    for k, lab in enumerate(['t', 't+1', 't+2']):
        rbox(part, cx[k] - 2.8, y + 2.8, 5.6, 6.2, 'white', ec='#B9C9DE', r=0.8, lw=0.4, z=2)
        txt(part, cx[k], y + h - 0.2, lab, fs=5.0, c=MUTED, ha='center')
    ys = [y + 4.2, y + 6.0, y + 7.8]
    node(part, cx[0], ys[0])
    node(part, cx[1], ys[0], c=RED_M, ghost=True)
    node(part, cx[2], ys[0])
    for k in (1, 2):
        for j in range(3):
            node(part, cx[j], ys[k], c='#7F9CC4', r=0.75)
        line(part, [cx[0], cx[2]], [ys[k], ys[k]], '#7F9CC4', lw=0.5, z=5)
    arrow(part, (cx[0], ys[0] - 0.9), (cx[2], ys[0] - 0.9), c=BLUE, lw=0.9, rad=-0.3, head=4.5)
    txt(part, cx[2] + 3.4, ys[0], 'g = 2', fs=5.4, c=BLUE, b=True)


def glyph_div(part, x, y, w, h):
    """Three class probabilities of one edge (division highlighted) and the lineage it implies."""
    labs, ps, cs = ['unrel.', 'same', 'division'], [0.04, 0.10, 0.86], ['#C8D3E0', BLUE, MAGENTA]
    base = y + h - 2.0
    bw, sp, x0 = 4.4, 6.6, x + w * 0.06
    for k in range(3):
        bx = x0 + k * sp
        rbox(part, bx, base - 7.0 * ps[k] - 0.05, bw, 7.0 * ps[k] + 0.05, cs[k], r=0.3, z=3)
        txt(part, bx + bw / 2, base + 1.3, labs[k], fs=4.8, c=MUTED, ha='center')
    line(part, [x0 - 0.8, x0 + 2 * sp + bw + 0.8], [base, base], '#7F9CC4', lw=0.4)
    txt(part, x0 + 2 * sp + bw / 2, base - 7.0 * ps[2] - 1.2, 'p', fs=5.2, c=MAGENTA, ha='center', it=True)
    mx, my = x + w * 0.70, y + h * 0.45
    node(part, mx, my, c=MAGENTA, r=1.0)
    for dy in (-2.4, 2.4):
        arrow(part, (mx + 1.0, my + dy * 0.3), (mx + 5.6, my + dy), c=MAGENTA, lw=0.7, head=4)
        node(part, mx + 6.5, my + dy, c=MAGENTA, r=0.85)


def glyph_calib(part, x, y, w, h):
    """Reliability diagram (p against observed frequency, on the diagonal) with tau, and the 3 x 3 ensemble whose
    disagreement (MI) flags a link."""
    ox, oy, s = x + w * 0.10, y + h - 1.6, 8.0
    rbox(part, ox, oy - s, s, s, 'white', ec='#9BB0CC', r=0.2, lw=0.4, z=2)
    line(part, [ox, ox + s], [oy, oy - s], MUTED, lw=0.4, ls=(0, (1, 1)))
    for v, o in ((0.12, 0.10), (0.3, 0.33), (0.5, 0.48), (0.7, 0.72), (0.88, 0.9)):
        node(part, ox + v * s, oy - o * s, r=0.45)
    line(part, [ox + 0.4 * s] * 2, [oy, oy - s], GOLD, lw=0.6)
    txt(part, ox + 0.4 * s, oy - s - 0.9, 'τ', fs=5.4, c='#9A6A00', ha='center')
    txt(part, ox - 1.0, oy - s / 2, 'observed', fs=4.8, c=MUTED, ha='center', rotation=90)
    ex = x + w * 0.46
    for k in range(3):
        rbox(part, ex + k * 0.9, y + 0.8 + k * 0.9, 6.4, 4.0, '#DCE7F5' if k < 2 else 'white', ec=BLUE, r=0.5,
             lw=0.45, z=3 + k)
    txt(part, ex + 5.0, y + 4.6, '3 × 3', fs=5.2, c=BLUE, ha='center', b=True)
    txt(part, ex + 12.4, y + 3.6, 'MI', fs=6.4, c='#9A6A00', ha='center', b=True)
    txt(part, ex + 6.6, y + 8.4, 'MI = disagreement', fs=4.9, c='#9A6A00', ha='center')


PRO = {
    'gap': ('Frame-gap graph', ['frames (t, t+g) with g as input; a 2nd', 'pass rejoins tracks across a miss'], glyph_gap),
    'div': ('Three-class edges', ['division is a class, learnt on dividing', 'and non-dividing data: no switch'],
            glyph_div),
    'trust': ('Calibrated decisions', ['p is a probability: link if p > 0.4', '(error costs); MI flags doubtful links'],
              glyph_calib),
}


def panel_b():
    for i, k in enumerate(KEYS):
        name, lines_, icon = CUR[k]
        x, y, w, h = card(f'{k}_current', k, 'cur', 'white', RED_EDGE)
        if icon is icon_dial:
            icon(f'{k}_current', x + 5.0, y + h / 2 + 0.8, 3.6)
        else:
            icon(f'{k}_current', x + 1.8, y + h / 2 + 0.6, 6.2)
        tx = x + 10.0
        txt(f'{k}_current', tx, y + 2.3, name, fs=6.2, b=True, c=RED_D)
        for j, s in enumerate(lines_):
            txt(f'{k}_current', tx, y + 5.0 + j * 2.4, s, fs=5.4, c=RED_TXT)
        name, lines_, glyph = PRO[k]
        x, y2, w, h2 = card(f'{k}_progct', k, 'pro', 'white', BLUE, lw=0.9)
        glyph(f'{k}_progct', x + 0.8, y2 + 0.6, w - 1.6, 10.6)
        ty = y2 + 13.4
        gid(ax.add_patch(Circle((x + 2.4, ty), 1.4, fc=BLUE, ec='none', zorder=8)), f'{k}_progct')
        centred(f'{k}_progct', x + 2.4, ty, str(NUM[k]), 5.8)
        txt(f'{k}_progct', x + 4.4, ty, name, fs=6.8, b=True, c=BLUE_D)
        for j, s in enumerate(lines_):
            txt(f'{k}_progct', x + w / 2, ty + 2.8 + j * 2.4, s, fs=5.5, c=INK, ha='center')
        # setting -> component (replaced by), component -> result
        xc = x + w / 2
        arrow(f'{k}_progct', (xc, y + h + 0.05), (xc, y2 - 0.05), c=BLUE, lw=0.8, head=4.5)
        arrow(f'{k}_result', (xc, y2 + h2 + 0.05), (xc, BAND['res'][0] + CARD_PAD - 0.05), c=BLUE, lw=0.8, head=4.5)
        if i < 2:
            nx = COLX[KEYS[i + 1]]
            arrow('band_pro', (x + w + 0.05, y2 + 5.5), (nx - 0.05, y2 + 5.5), c=BLUE, lw=0.8, head=4.5)
            STRUCT['connections'].append(dict(source=f'{k}_progct', target=f'{KEYS[i + 1]}_progct', kind='pipeline'))
        for a, b in (('current', 'progct'), ('progct', 'result')):
            STRUCT['connections'].append(dict(source=f'{k}_{a}', target=f'{k}_{b}', kind='reading order'))


# ================================================================= c: results
RES_IMG_H = 20.5


def result_text(k, head, sub, y):
    part = f'{k}_result'
    """Plain-language outcome: highlighted key words (blue, bold) + rest, and one explanatory line; the numbers are
    in the experiments section."""
    pcs = [(t, dict(fs=7.2, b=True, c=BLUE) if st.get('big') else dict(fs=6.6, c=INK, b=st.get('b', False)))
           for t, st in head]
    runs(part, COLX[k] + CW / 2, y, pcs, ha='center')
    txt(part, COLX[k] + CW / 2, y + 2.9, sub, fs=5.9, c=MUTED, ha='center', it=True)


def result_gap(x, y, w):
    well, t, label = EX_GAP['well'], EX_GAP['t'], EX_GAP['label']
    ours = tracks_of('ours', well)
    gt = D.gt_links(well, list(zip(ours.frame_index.astype(int), ours.instance_id.astype(int))))
    dst = [d for (s_, d), g in gt.items() if s_ == (t, label) and g == 2][0]
    links = pd.read_csv(D.track_folder('ours', well) / 'links.csv')
    row = links[(links['round'] == 2) & (links.src_frame == t) & (links.src_label == label) & (links.dst_label == dst[1])]
    assert len(row) == 1, 'ProGCT did not rejoin the example'
    p = float(row.p.iloc[0])
    assert ((t, label), dst) not in D.predicted_links(tracks_of('ta_ft', well))
    m0, m2 = scripta_mask(well, t), scripta_mask(well, t + 2)
    (ay, ax_), (by, bx) = centroid(m0, label), centroid(m2, dst[1])
    cy, cx = (ay + by) / 2, (ax_ + bx) / 2
    ims = [scripta_img(well, t + k) for k in range(3)]
    lv = window_levels(ims, cy, cx, 200)
    tw = (w - 1.2) / 3
    ry, rh = y + 3.4, RES_IMG_H - 3.4
    rt = [Tile('gap_result', ims[k], cy, cx, x + k * (tw + 0.6), ry, tw, rh, 260, lo_hi=lv) for k in range(3)]
    for k, lab in enumerate(['t', 't+1', 't+2']):
        rt[k].label(lab, 'tl', fs=4.8)
    rt[0].outline(m0, label, BLUE, lw=0.9)
    rt[2].outline(m2, dst[1], BLUE, lw=0.9)
    rt[1].outline(m0, label, '#FF6B6B', lw=0.8, ls=(0, (1.6, 1.2)))
    centred('gap_result', rt[1].x + rt[1].w / 2, rt[1].y + rt[1].h - 1.3, 'missed', 4.8, c='#FF8A80', z=rt[1].z + 1)
    a, b = rt[0].mm(ay, ax_), rt[2].mm(by, bx)
    arrow('gap_result', (a[0], ry + 0.6), (b[0], ry + 0.6), c=BLUE, lw=0.9, rad=-0.16, head=4.5)
    for pnt in (a, b):
        line('gap_result', [pnt[0], pnt[0]], [ry + 0.6, pnt[1] - 1.2], BLUE, lw=0.5, ls=(0, (1, 0.8)), z=7)
    txt('gap_result', (a[0] + b[0]) / 2 + 6.0, y + 0.9, f'p = {p:.2f}', fs=5.0, c=BLUE_D, ha='left')
    FACTS['gap'] = dict(well=well, t=t, label=label, dst=list(dst), p=p)
    return [('Rejoined', dict(big=True)), (' across a missed frame', {})], \
        'the 2nd pass links t to t+2 like any other link'


def result_div(x, y, w):
    video, mother = EX_DIV['video'], EX_DIV['mother']
    gt = pd.read_csv(D.DNN / f'{video}_GT' / 'TRA' / 'man_track.txt', sep=' ', header=None, names=['l', 's', 'e', 'p'])
    te = int(gt[gt.l == mother].iloc[0].e)
    kids = gt[gt.p == mother]
    f = D.track_folder('ours', video, 'dnn')
    tr = pd.read_csv(f / 'instance_tracks.csv')
    tid = {(int(a), int(b)): int(c) for a, b, c in zip(tr.frame_index, tr.instance_id, tr.track_id)}
    lin = {int(k): int(v) for k, v in json.loads((f / 'lineage.json').read_text()).items()}
    for k in kids.itertuples():
        assert lin.get(tid[(int(k.s), int(k.l))]) == tid[(te, mother)], 'ProGCT missed the division'
    links = pd.read_csv(f / 'links.csv')
    dl = links[(links.kind == 'division') & (links.src_frame == te) & (links.src_label == mother)]
    mk, mk1 = dnn_mask(video, te), dnn_mask(video, te + 1)
    cy, cx = centroid(mk, mother)
    frames = [te - 3, te, te + 1]
    ims = [dnn_img(video, t) for t in frames]
    lv = window_levels(ims, cy, cx, 60)
    tw = (w - 1.2) / 3
    tiles = [Tile('div_result', ims[k], cy, cx, x + k * (tw + 0.6), y, tw, RES_IMG_H, 64, lo_hi=lv, sigma=0.8)
             for k in range(3)]
    for k, lab in enumerate(['t−3', 't', 't+1']):
        tiles[k].label(lab, 'tl', fs=4.8)
    tiles[0].ring(dnn_mask(video, frames[0]), mother, '#EA7DC7', lw=0.7)
    tiles[1].ring(mk, mother, '#EA7DC7', lw=0.7)
    for k in kids.itertuples():
        tiles[2].ring(mk1, int(k.l), '#EA7DC7', lw=0.8)
    if len(dl):
        tiles[2].label(f'd = {dl.p.min():.2f}', 'bl', c='#F6BFE3', fs=4.8)
    FACTS['div'] = dict(video=video, mother=mother, last_frame=te, daughters=[int(l) for l in kids.l],
                        division_p=[float(v) for v in dl.p])
    return [('Divisions found', dict(big=True)), (', none invented', {})], \
        'one model for videos with and without mitoses'


def result_trust(x, y, w):
    well, x0, y0, ww, hh = (EX_TRUST[k] for k in ('well', 'x0', 'y0', 'w', 'h'))
    assert well not in D.TEST_WELLS
    f = D.track_folder('ours', well)
    tr = pd.read_csv(f / 'instance_tracks.csv')
    pos = {(int(a), int(b)): (yy, xx) for a, b, xx, yy in zip(tr.frame_index, tr.instance_id, tr.x, tr.y)}
    gt = D.gt_links(well, list(pos))
    links = pd.read_csv(f / 'links.csv')
    im = scripta_img(well, 15)
    lv = np.percentile(im[y0:y0 + hh, x0:x0 + ww], [1, 99.7])
    tile = Tile('trust_result', im, y0 + hh / 2, x0 + ww / 2, x, y, w, RES_IMG_H, hh, lo_hi=lv)
    counts = dict(links=0, flagged=0, false=0, flagged_false=0)
    for q in links.itertuples():
        s_, d = (q.src_frame, q.src_label), (q.dst_frame, q.dst_label)
        if s_ not in pos or d not in pos or not TRUST_FRAMES[0] <= q.src_frame < TRUST_FRAMES[1]:
            continue
        (ya, xa), (yb, xb) = pos[s_], pos[d]
        if not (tile.x0 <= xa < tile.x0 + tile.wpx and tile.y0 <= ya < tile.y0 + tile.hpx):
            continue
        pa, pb = tile.mm(ya, xa), tile.mm(yb, xb)
        counts['links'] += 1
        if q.uncertain:
            tile.clip(line('trust_result', [pa[0], pb[0]], [pa[1], pb[1]], GOLD, lw=1.3, z=6))
            counts['flagged'] += 1
        else:
            tile.clip(line('trust_result', [pa[0], pb[0]], [pa[1], pb[1]], '#7FE3E8', lw=0.35, z=5))
        if q.src_label < UNKNOWN_OFFSET and q.dst_label < UNKNOWN_OFFSET and (s_, d) not in gt:
            counts['false'] += 1
            counts['flagged_false'] += bool(q.uncertain)
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            if tile.x + 0.5 < mx < tile.x + tile.w - 0.5 and tile.y + 0.5 < my < tile.y + tile.h - 0.5:
                txt('trust_result', mx, my, '×', fs=6.0, c='#FF5A4E', b=True, ha='center')
    rbox('trust_result', x + 0.5, y + 0.5, 23.0, 2.4, (0, 0, 0, 0.55), r=0.4, z=8)
    end = runs('trust_result', x + 1.0, y + 1.7, [('flagged (high MI)', dict(fs=5.0, c=GOLD, b=True))])
    txt('trust_result', end + 1.2, y + 1.7, '× false', fs=5.0, c='#FF8A80', b=True)
    FACTS['trust'] = dict(well=well, frame_shown=15, region_px=[x0, y0, ww, hh], counts_in_tile=counts)
    return [('Errors flagged', dict(big=True)), (' before review', {})], \
        'the most uncertain links hold most false links'


def panel_c():
    for k, fn in (('gap', result_gap), ('div', result_div), ('trust', result_trust)):
        x, y, w, h = card(f'{k}_result', k, 'res', 'white', '#AFC3DF')
        head, sub = fn(x + 1.0, y + 1.0, w - 2.0)
        result_text(k, head, sub, y + 1.0 + RES_IMG_H + 3.0)


panel_task()
panel_b()
panel_c()

# ---------------------------------------------------------------- save
STRUCT['parts'] = {'a_task': dict(panel='a', kind='task: input frames, ProGCT, tracks and lineage'),
                   'panel_letters': dict(kind='panel letters')}
for k in KEYS:
    for kind in ('title', 'current', 'progct', 'result'):
        STRUCT['parts'][f'{k}_{kind}'] = dict(contribution=NUM[k], title=TITLES[k], kind=kind,
                                              panel='c' if kind == 'result' else 'b')
for band in ('cur', 'pro', 'res'):
    STRUCT['parts'][f'band_{band}'] = dict(kind='band and its tab')
STRUCT['size_mm'] = [W, H]
fig.savefig(OUT / 'fig.svg')


def flatten_for_inkscape(path):
    """Matplotlib wraps every artist in <g id="figure_1"><g id="axes_1">, so Inkscape selects the whole figure on a
    click. Unwrap those two groups in place (paint order unchanged) and give each artist an Inkscape label (its part
    name), so a click selects one object and the Objects panel lists the parts by name."""
    svg_ns, ink_ns = 'http://www.w3.org/2000/svg', 'http://www.inkscape.org/namespaces/inkscape'
    ET.register_namespace('', svg_ns)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    ET.register_namespace('inkscape', ink_ns)
    tree = ET.parse(path)
    root = tree.getroot()

    def unwrap(parent, gid_):
        for i, child in enumerate(list(parent)):
            if child.get('id') == gid_:
                parent.remove(child)
                for j, grand in enumerate(list(child)):
                    parent.insert(i + j, grand)
                return True
            if unwrap(child, gid_):
                return True
        return False

    for gid_ in ('figure_1', 'axes_1'):
        assert unwrap(root, gid_), f'{gid_} not found'
    sodi_ns = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
    ET.register_namespace('sodipodi', sodi_ns)
    for el in root.iter():
        if '__' in el.get('id', ''):
            el.set(f'{{{ink_ns}}}label', el.get('id').split('__')[0])
        elif el.get('id') == 'patch_1':                   # white page background: locked, so clicks pass through
            el.set(f'{{{ink_ns}}}label', 'background (locked)')
            el.set(f'{{{sodi_ns}}}insensitive', 'true')
    tree.write(path, encoding='utf-8', xml_declaration=True)


flatten_for_inkscape(OUT / 'fig.svg')
fig.savefig(OUT / 'fig.pdf')
fig.savefig(OUT / 'fig.png', dpi=600)
(OUT / 'facts.json').write_text(json.dumps(FACTS, indent=1), encoding='utf-8')
(OUT / 'structure.json').write_text(json.dumps(STRUCT, indent=1), encoding='utf-8')


# ---------------------------------------------------------------- text collision audit (renderer boxes)
def audit():
    r = fig.canvas.get_renderer()
    boxes = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        boxes.append((t.get_gid().split('__')[0], t.get_text(), bb.x0 / fig.dpi * 25.4, H - bb.y1 / fig.dpi * 25.4,
                      bb.x1 / fig.dpi * 25.4, H - bb.y0 / fig.dpi * 25.4, t.get_fontsize()))
    problems = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if min(a[4], b[4]) - max(a[2], b[2]) > 0.15 and min(a[5], b[5]) - max(a[3], b[3]) > 0.15:
                problems.append(f'overlap: {a[1]!r} / {b[1]!r}')
    for part, s, x0, y0, x1, y1, fs in boxes:
        k = part.split('_')[0]
        if k in COLX and (x0 < COLX[k] + 0.2 or x1 > COLX[k] + CW - 0.2):
            problems.append(f'outside column {k}: {s!r} ({x0:.1f}-{x1:.1f} vs {COLX[k]:.1f}-{COLX[k] + CW:.1f})')
        if part == 'a_task' and (x0 < A_X + 0.2 or x1 > A_X + A_W - 0.2):
            problems.append(f'outside panel a: {s!r}')
    return dict(texts=len(boxes), min_font_pt=min(b[6] for b in boxes), problems=problems)


report = audit()
(OUT / 'audit.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- parts: real SVG objects grouped by gid prefix
def export_parts():
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.parse(OUT / 'fig.svg').getroot()
    ns = '{http://www.w3.org/2000/svg}'
    groups = {}
    for el in root.iter():
        g = el.get('id', '')
        if '__' in g:
            groups.setdefault(g.split('__')[0], []).append(el)
    (OUT / 'parts').mkdir(exist_ok=True)
    for old in (OUT / 'parts').glob('*.svg'):
        old.unlink()
    index = []
    defs = root.find(f'{ns}defs')
    for part, els in sorted(groups.items()):
        new = ET.Element(root.tag, dict(root.attrib))
        if defs is not None:
            new.append(copy.deepcopy(defs))
        for e in els:
            new.append(copy.deepcopy(e))
        ET.ElementTree(new).write(OUT / 'parts' / f'{part}.svg', encoding='utf-8', xml_declaration=True)
        index.append(dict(part=part, file=f'parts/{part}.svg', objects=len(els),
                          texts=sum(1 for e in els for _ in e.iter(f'{ns}text')),
                          images=sum(1 for e in els for _ in e.iter(f'{ns}image'))))
    (OUT / 'parts' / 'index.json').write_text(json.dumps(index, indent=1), encoding='utf-8')


export_parts()
print('written', OUT)
