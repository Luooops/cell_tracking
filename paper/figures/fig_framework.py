"""Framework figure of ProGCT: how one calibrated posterior per decision is computed and used.

Row 1 (scoring):  input frames -> (1) frame-gap graph -> graph network (standard) -> (2) three-class edges
Row 2 (deciding): (3) ensemble of 3 models x 3 views -> (3) calibrated decisions -> (1) cross-frame check -> outputs
A plain flowchart: every card alike, no step numbers, no highlighted frames.
Features are drawn as small flat cubes; the first cube of a node token shows a real masked appearance crop
(utils.dataset.appearance_crop of a Scripta nucleus). Everything else is schematic. No equations.

    python paper/figures/fig_framework.py [--out DIR]
writes fig.svg (Inkscape-friendly: no wrapper groups, objects labelled by part), fig.pdf, fig.png, audit.json to
DIR (default development/overview_design/framework).
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from skimage import measure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

ROOT = D.ROOT
W, H = 174.6, 74.0
BLUE, BLUE_D, BLUE_L, BLUE_BAND = '#3B6FB6', '#24497F', '#DCE7F5', '#EEF3FA'
GREY_E, GREY_F, GREY_T = '#B9C2CA', '#F2F4F6', '#6B7680'
INK, MUTED = '#1F2A33', '#5E6B75'
TEAL, TEAL_L, PURPLE, MAGENTA, GOLD, GOLD_T, RED_M = ('#1F9E8F', '#D3EEEA', '#8A6BC4', '#C2479A', '#E3A21A',
                                                    '#9A6A00', '#D64545')
NODE_C = '#7F9CC4'
PT = 72 / 25.4

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=str(ROOT / 'development' / 'overview_design' / 'framework'))
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'image.composite_image': False, 'mathtext.default': 'regular', 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
                     'mathtext.bf': 'Times New Roman:bold'})
fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')
_gid = [0]


def gid(artist, part):
    _gid[0] += 1
    artist.set_gid(f'{part}__{_gid[0]}')
    return artist


def txt(part, x, y, s, fs=6, c=INK, b=False, ha='left', va='center', it=False, z=9, **kw):
    return gid(ax.text(x, y, s, fontsize=fs, color=c, fontweight='bold' if b else 'normal', ha=ha, va=va,
                       fontstyle='italic' if it else 'normal', zorder=z, **kw), part)


def centred(part, x, cy, s, fs, c='white', b=True, z=9):
    return txt(part, x, cy + 0.662 * fs / PT / 2, s, fs=fs, c=c, b=b, ha='center', va='baseline', z=z)


def rbox(part, x, y, w, h, fc, ec='none', r=1.2, lw=0.5, ls='-', z=0, alpha=1.0):
    return gid(ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec,
                                           lw=lw, ls=ls, zorder=z, alpha=alpha)), part)


def arrow(part, a, b, c=BLUE, lw=0.8, rad=0.0, ls='-', z=7, head=4.5, shrink=0):
    return gid(ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=head, color=c, lw=lw, linestyle=ls,
                                            connectionstyle=f'arc3,rad={rad}', shrinkA=shrink, shrinkB=shrink,
                                            zorder=z)), part)


def line(part, xs, ys, c, lw=0.6, ls='-', z=6):
    return gid(ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle='round')[0], part)


def poly(part, pts, fc, ec='none', lw=0.4, z=4):
    return gid(ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle='round')), part)


def node(part, x, y, c=BLUE, r=0.8, ghost=False, z=7):
    gid(ax.add_patch(Circle((x, y), r, fc='white' if ghost else c, ec=c, lw=0.6,
                            ls=(0, (1.2, 0.8)) if ghost else '-', zorder=z)), part)


def shade(c, k):
    """Lighter (k > 0) or darker (k < 0) version of a colour."""
    rgb = np.array(mcolors.to_rgb(c))
    return tuple(rgb + (1 - rgb) * k) if k > 0 else tuple(rgb * (1 + k))


def cube(part, x, y, s, c, z=4, image=None):
    """Flat isometric cube: front face (x, y, s), top and right faces offset by d. image: shown on the front face."""
    d = 0.35 * s
    poly(part, [(x, y), (x + d, y - d), (x + s + d, y - d), (x + s, y)], shade(c, 0.35), ec=shade(c, -0.35), lw=0.3,
         z=z)
    poly(part, [(x + s, y), (x + s + d, y - d), (x + s + d, y + s - d), (x + s, y + s)], shade(c, -0.18),
         ec=shade(c, -0.35), lw=0.3, z=z)
    if image is not None:
        gid(ax.imshow(image, cmap='gray', vmin=0, vmax=1, extent=(x, x + s, y + s, y), zorder=z + 0.1,
                      interpolation='lanczos'), part)
        poly(part, [(x, y), (x + s, y), (x + s, y + s), (x, y + s)], 'none', ec=shade(c, -0.35), lw=0.3, z=z + 0.2)
    else:
        poly(part, [(x, y), (x + s, y), (x + s, y + s), (x, y + s)], c, ec=shade(c, -0.35), lw=0.3, z=z)


def tokens(part, x, y, s, colours, gap=0.15, z=4, image=None):
    """A row of cubes; returns the x after the last cube (front faces)."""
    for i, c in enumerate(colours):
        cube(part, x + i * (s + gap), y, s, c, z=z + i * 0.01, image=image if i == 0 else None)
    return x + len(colours) * (s + gap) + 0.35 * s


def slab(part, x, y, w, h, depth, c, z=2):
    """Flat cuboid seen from above-front: top face (parallelogram w x h, sheared by h) and thin front/side faces."""
    sh = 0.55 * h
    top = [(x + sh, y), (x + sh + w, y), (x + w, y + h), (x, y + h)]
    poly(part, top, c, ec=shade(c, -0.4), lw=0.35, z=z)
    poly(part, [(x, y + h), (x + w, y + h), (x + w, y + h + depth), (x, y + h + depth)], shade(c, -0.15),
         ec=shade(c, -0.4), lw=0.35, z=z)
    poly(part, [(x + w, y + h), (x + sh + w, y), (x + sh + w, y + depth), (x + w, y + h + depth)], shade(c, -0.28),
         ec=shade(c, -0.4), lw=0.35, z=z)
    return sh


def text_width(s, fs, b=False):
    t = ax.text(0, 0, s, fontsize=fs, fontweight='bold' if b else 'normal')
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def card(part, x, y, w, h, title):
    """Every card alike (plain flowchart: no numbers, no highlighted frames)."""
    rbox(part, x, y, w, h, 'white', ec=GREY_E, r=1.4, lw=0.45, z=1)
    txt(part, x + 1.2, y + 2.4, title, fs=6.6, b=True, c=GREY_T)


def note(part, x, y, s, fs=5.3, c=MUTED, ha='left', it=False, b=False, **kw):
    return txt(part, x, y, s, fs=fs, c=c, ha=ha, it=it, b=b, **kw)


def appearance_crop_of(well, t, label):
    from utils.dataset import appearance_crop
    m = tifffile.imread(D.SCRIPTA / f'{well}_GT' / 'TRA' / f'man_track{t:03d}.tif')
    im = tifffile.imread(D.SCRIPTA / well / f't{t:03d}.tif').astype(float)
    p = measure.regionprops((m == label).astype(np.uint8))[0]
    r0, c0, r1, c1 = p.bbox
    return np.asarray(appearance_crop(im[r0:r1, c0:c1], p.image), float)


from matplotlib.transforms import Affine2D  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================== real data
# one example runs through the figure: Scripta r15c12, frames 8 -> 9, cell 435 (development well, out of fold)
EXW, EXT, EXS = 'r15c12', 8, 435
SCR = D.SCRIPTA
FIN = D.FINAL


def s_img(well, t):
    return tifffile.imread(SCR / well / f't{t:03d}.tif').astype(float)


def s_mask(well, t):
    return tifffile.imread(SCR / f'{well}_GT' / 'TRA' / f'man_track{t:03d}.tif')


def d_img(video, t):
    return tifffile.imread(D.DNN / video / f't{t:03d}.tif').astype(float)


def d_mask(video, t):
    return tifffile.imread(D.DNN / f'{video}_GT' / 'TRA' / f'man_track{t:03d}.tif')


def norm(a, lo_hi):
    lo, hi = lo_hi
    return np.clip((a - lo) / (hi - lo), 0, 1)


ex_dir = FIN / 'scripta' / 'progct' / 'division' / EXW
EZ = np.load(ex_dir / 'edges.npz')
ETR = pd.read_csv(ex_dir / 'instance_tracks.csv')
EPOS = {(int(a), int(b)): (x_, y_) for a, b, x_, y_ in zip(ETR.frame_index, ETR.instance_id, ETR.x, ETR.y)}
SX, SY = EPOS[(EXT, EXS)]
R1M = (EZ['round'] == 1) & (EZ['src_frame'] == EXT)
CAND = R1M & (EZ['src_label'] == EXS)
GATE = float(EZ['gate'][CAND][0])                       # r * sqrt(g) in pixels for g = 1
CHOSEN = int(EZ['dst_label'][CAND & (EZ['chosen'] == 1)][0])
Q9 = 1 / (1 + np.exp(-EZ['logits'][CAND & (EZ['dst_label'] == CHOSEN)][0].astype(float)))   # [views x members]
P_EX, MI_EX = float(EZ['p'][CAND & (EZ['dst_label'] == CHOSEN)][0]), float(EZ['mi'][CAND & (EZ['dst_label'] == CHOSEN)][0])
assert abs(Q9.mean() - P_EX) < 1e-3
Q9 = Q9.reshape(3, 3)                                   # rows: views, columns: members (edge_scores loops views x models)
# MI threshold of the review queue: 95th percentile over the known-identity links made in the development wells
_mis = []
for w_ in [w_ for w_ in D.all_wells() if w_ not in D.TEST_WELLS]:
    _l = pd.read_csv(D.track_folder('ours', w_) / 'links.csv')
    _mis.append(_l.mi[(_l.src_label < 50000) & (_l.dst_label < 50000)].values)
MI_THR = float(np.quantile(np.concatenate(_mis), 0.95))
REL = pd.read_csv(ROOT / 'development' / 'paper_results' / 'calibration' / 'scripta_cv_oof_tta' / 'reliability.csv')
REL = REL[REL.population == 'round1_decision']
assert int(REL.n.sum()) == 9030
FACTS = dict(example=dict(well=EXW, t=EXT, source=EXS, gate_px=GATE, chosen=CHOSEN, p=P_EX, mi=MI_EX,
                          nine=Q9.round(3).tolist(), mi_threshold=MI_THR))


class Tile:
    """A real crop at (x, y, w, h) mm centred on pixel (cx, cy); px -> mm mapping; overlays clipped to the tile."""

    def __init__(self, part, image, cx, cy, x, y, w, h, hpx, lo_hi, z=3):
        self.part, self.x, self.y, self.w, self.h, self.z = part, x, y, w, h, z
        wpx = hpx * w / h
        self.y0 = int(round(np.clip(cy - hpx / 2, 0, image.shape[0] - hpx)))
        self.x0 = int(round(np.clip(cx - wpx / 2, 0, image.shape[1] - wpx)))
        self.hpx, self.wpx = int(round(hpx)), int(round(wpx))
        crop = image[self.y0:self.y0 + self.hpx, self.x0:self.x0 + self.wpx]
        gid(ax.imshow(norm(crop, lo_hi), cmap='gray', vmin=0, vmax=1, interpolation='lanczos',
                      extent=(x, x + w, y + h, y), zorder=z), part)
        gid(ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='#8A949C', lw=0.35, zorder=z + 0.8)), part)
        self.rect = ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='none', zorder=0))

    def mm(self, px, py):
        return self.x + (px - self.x0) / self.wpx * self.w, self.y + (py - self.y0) / self.hpx * self.h

    def inside(self, px, py):
        return self.x0 <= px < self.x0 + self.wpx and self.y0 <= py < self.y0 + self.hpx

    def clip(self, artist):
        artist.set_clip_path(self.rect)
        return artist

    def outline(self, mask, label, c, lw=0.7, ls='-', sigma=0.0):
        sub = (mask[self.y0:self.y0 + self.hpx, self.x0:self.x0 + self.wpx] == label).astype(float)
        if not sub.any():
            return
        sub = ndi.zoom(np.pad(sub, 1), 4, order=1)
        if sigma:
            sub = ndi.gaussian_filter(sub, sigma * 4)
        for cc in measure.find_contours(sub, 0.5):
            cc = (cc + 0.5) / 4 - 1
            self.clip(line(self.part, self.x + cc[:, 1] / self.wpx * self.w, self.y + cc[:, 0] / self.hpx * self.h,
                           c, lw=lw, ls=ls, z=self.z + 0.5))

    def label(self, s, corner='tl', c='white', fs=4.8):
        xx = self.x + 0.5 if corner[1] == 'l' else self.x + self.w - 0.5
        yy = self.y + 1.1 if corner[0] == 't' else self.y + self.h - 1.1
        txt(self.part, xx, yy, s, fs=fs, c=c, ha='left' if corner[1] == 'l' else 'right', z=self.z + 0.9,
            bbox=dict(boxstyle='round,pad=0.1,rounding_size=0.3', fc=(0, 0, 0, 0.5), ec='none'))


def skewed_slab(part, image01, x, y, w, h, sh, depth, z):
    """A real image on the top face of a flat cuboid: the image is sheared like the face (Inkscape keeps it)."""
    tr_ = Affine2D.from_values(w, 0, -sh, h, x + sh, y) + ax.transData
    im = ax.imshow(image01, cmap='gray', vmin=0, vmax=1, extent=(0, 1, 1, 0), zorder=z, interpolation='lanczos')
    im.set_transform(tr_)
    gid(im, part)
    c = '#3A424B'
    poly(part, [(x, y + h), (x + w, y + h), (x + w, y + h + depth), (x, y + h + depth)], shade(c, -0.1),
         ec=shade(c, -0.4), lw=0.35, z=z)
    poly(part, [(x + w, y + h), (x + sh + w, y), (x + sh + w, y + depth), (x + w, y + h + depth)], shade(c, -0.25),
         ec=shade(c, -0.4), lw=0.35, z=z)
    poly(part, [(x + sh, y), (x + sh + w, y), (x + w, y + h), (x, y + h)], 'none', ec='#C9D1D8', lw=0.4, z=z + 0.2)
    return lambda u, v: (x + sh + w * u - sh * v, y + h * v)


CROP = appearance_crop_of(EXW, EXT, EXS)

# =============================================================== layout
R1, R2 = (3.0, 35.0), (40.0, 72.0)          # rows (y0, y1)
C1 = [(0.6, 20.6), (22.8, 72.0), (74.2, 116.6), (118.8, 174.0)]
C2 = [(0.6, 40.0), (42.2, 92.0), (94.2, 128.0), (130.2, 174.0)]


def hcard(row, col):
    (y0, y1), (x0, x1) = row, col
    return x0, y0, x1 - x0, y1 - y0


WIN = 240                                    # px window around the example
ims_ex = [s_img(EXW, t) for t in (EXT, EXT + 1)]
LV = np.percentile(np.concatenate([i[int(SY - WIN):int(SY + WIN), int(SX - WIN):int(SX + WIN)].ravel()
                                   for i in ims_ex]), [1, 99.7])

# --------------------------------------------------------------- row 1, card 1: input (two real frames)
x, y, w, h = hcard(R1, C1[0])
card('input', x, y, w, h, 'Input')
for k, (dy, lab) in enumerate(((5.4, 'frame t'), (16.2, 'frame t+1'))):
    t = EXT + k
    y0_, x0_ = int(SY - WIN / 2), int(SX - WIN * 0.9)
    crop = ims_ex[k][y0_:y0_ + WIN, x0_:x0_ + int(WIN * 1.8)]
    m = s_mask(EXW, t)[y0_:y0_ + WIN, x0_:x0_ + int(WIN * 1.8)]
    to = skewed_slab('input', norm(crop, LV), x + 1.4, y + dy, 13.0, 6.0, 3.3, 0.9, z=2 + 2 * k)
    for lab_ in np.unique(m):
        if lab_ == 0:
            continue
        for cc in measure.find_contours(ndi.zoom((m == lab_).astype(float), 2, order=1), 0.5):
            u, v = (cc[:, 1] / 2) / m.shape[1], (cc[:, 0] / 2) / m.shape[0]
            pts = np.array([to(a_, b_) for a_, b_ in zip(u, v)])
            line('input', pts[:, 0], pts[:, 1], '#E8EEF2', lw=0.25, z=2.5 + 2 * k)
    note('input', x + w / 2, y + dy + 8.3, lab, fs=5.4, c=INK, ha='center')
note('input', x + w / 2, y + h - 3.0, 'segmented nuclei', fs=5.2, ha='center', it=True)
note('input', x + w / 2, y + h - 1.0, 'pairs t, t+g; g = 1, 2', fs=5.0, ha='center', it=True)

# --------------------------------------------------------------- row 1, card 2: (1) frame-gap graph on real frames
x, y, w, h = hcard(R1, C1[1])
card('graph', x, y, w, h, 'Frame-gap graph')
ts = 13.6
tl = Tile('graph', ims_ex[0], SX, SY, x + 1.6, y + 5.2, ts, ts, 230, LV)
tr = Tile('graph', ims_ex[1], SX, SY, x + 1.6 + ts + 3.6, y + 5.2, ts, ts, 230, LV)
tl.label('t')
tr.label('t+1')
f8 = {b: p_ for (a, b), p_ in EPOS.items() if a == EXT and tl.inside(*p_)}
f9 = {b: p_ for (a, b), p_ in EPOS.items() if a == EXT + 1 and tr.inside(*p_)}
# in-frame context: the source's 4 nearest neighbours in frame t (as the network's neighbour edges)
allf8 = {b: p_ for (a, b), p_ in EPOS.items() if a == EXT}
nn = sorted((np.hypot(p_[0] - SX, p_[1] - SY), b) for b, p_ in allf8.items() if b != EXS)[:4]
for _, b in nn:
    if b in f8:
        a_, b_ = tl.mm(SX, SY), tl.mm(*f8[b])
        tl.clip(line('graph', [a_[0], b_[0]], [a_[1], b_[1]], PURPLE, lw=0.7, ls=(0, (1.5, 1)), z=5))
for b, p_ in f8.items():
    node('graph', *tl.mm(*p_), c=BLUE if b == EXS else '#A9BEDD', r=0.75 if b == EXS else 0.55, z=6)
for b, p_ in f9.items():
    node('graph', *tr.mm(*p_), c='#A9BEDD', r=0.55, z=6)
# the gate around the source's position, drawn in frame t+1
gc = tr.mm(SX, SY)
gr_mm = GATE / tr.wpx * tr.w
tr.clip(gid(ax.add_patch(Circle(gc, gr_mm, fc=TEAL_L, ec=TEAL, lw=0.6, ls=(0, (2, 1)), alpha=0.35, zorder=4)),
            'graph'))
tr.clip(gid(ax.add_patch(Circle(gc, gr_mm, fc='none', ec=TEAL, lw=0.6, ls=(0, (2, 1)), zorder=5)), 'graph'))
node('graph', *gc, c='white', r=0.55, ghost=True, z=6)
for d_ in EZ['dst_label'][CAND]:
    if int(d_) in f9:
        a_, b_ = tl.mm(SX, SY), tr.mm(*f9[int(d_)])
        line('graph', [a_[0], b_[0]], [a_[1], b_[1]], BLUE, lw=0.6, z=6.5)
note('graph', tr.x + ts / 2, y + 20.4, r'gate $\mathit{r}\sqrt{\mathit{g}}$', fs=5.0, c=TEAL, ha='center')
note('graph', tl.x + ts / 2, y + 20.4, '4 neighbours', fs=5.0, c=PURPLE, ha='center')
tx = x + 35.4
note('graph', tx, y + 6.0, 'node', fs=5.0, c=MUTED)
tokens('graph', tx, y + 7.6, 2.4, [BLUE] + ['#A9BEDD'] * 3, image=CROP)
note('graph', tx, y + 11.3, 'crop + size, motion', fs=4.8, c=MUTED)
note('graph', tx, y + 14.4, 'edge', fs=5.0, c=MUTED)
tokens('graph', tx, y + 16.0, 2.4, ['#A9BEDD', '#A9BEDD', BLUE, '#A9BEDD'])
note('graph', tx + 2 * 2.55 + 1.2, y + 20.3, 'g', fs=5.6, c=BLUE, b=True, ha='center', it=True)
note('graph', tx, y + 22.3, 'geometry + gap', fs=4.8, c=MUTED)
note('graph', x + w / 2, y + h - 3.6, 'one graph per frame pair (t, t+g): candidates', fs=5.2, c=INK, ha='center')
note('graph', x + w / 2, y + h - 1.4, 'within the gate, the gap g as an input', fs=5.2, c=INK, ha='center')

# --------------------------------------------------------------- row 1, card 3: graph network (standard)
x, y, w, h = hcard(R1, C1[2])
card('network', x, y, w, h, 'Graph network (standard)')
gid(ax.imshow(CROP, cmap='gray', vmin=0, vmax=1, extent=(x + 2.0, x + 6.4, y + 11.4, y + 7.0), zorder=4,
              interpolation='lanczos'), 'network')
poly('network', [(x + 7.4, y + 6.4), (x + 12.6, y + 7.8), (x + 12.6, y + 10.6), (x + 7.4, y + 12.0)], '#C9D1D8',
     ec=GREY_T, lw=0.4)
note('network', x + 10.0, y + 9.2, 'CNN', fs=5.0, c=INK, ha='center')
arrow('network', (x + 6.6, y + 9.2), (x + 7.3, y + 9.2), c=GREY_T, lw=0.6, head=3.5)
tokens('network', x + 13.6, y + 7.9, 2.2, ['#C9D1D8'] * 3)
note('network', x + 2.0, y + 14.4, 'appearance embedding', fs=4.9, c=GREY_T)
mx, my = x + 23.0, y + 5.6
for k in range(2):
    rbox('network', mx + k * 1.2, my + k * 1.2, 15.4, 11.0, 'white', ec=GREY_E, r=0.8, lw=0.45, z=3 + k)
pts = [(mx + 4.2, my + 5.0), (mx + 7.8, my + 3.4), (mx + 11.6, my + 5.6), (mx + 6.2, my + 9.0), (mx + 11.0, my + 9.8)]
for a_, b_ in ((0, 1), (1, 2), (0, 3), (3, 4), (2, 4), (1, 3)):
    line('network', [pts[a_][0] + 1.2, pts[b_][0] + 1.2], [pts[a_][1] + 1.2, pts[b_][1] + 1.2], '#9AA7B3', lw=0.45,
         z=5)
for p_ in pts:
    node('network', p_[0] + 1.2, p_[1] + 1.2, c='#8A97A4', r=0.7, z=6)
note('network', mx + 8.7, my + 14.6, 'attention message passing × 2', fs=4.9, c=GREY_T, ha='center')
arrow('network', (x + 21.2, y + 9.2), (mx - 0.2, y + 9.2), c=GREY_T, lw=0.6, head=3.5)
note('network', x + w / 2, y + h - 6.0, 'cells see their candidates and their neighbours;', fs=5.2, c=GREY_T,
     ha='center')
note('network', x + w / 2, y + h - 3.6, 'one feature vector per candidate edge', fs=5.2, c=GREY_T, ha='center')
note('network', x + w / 2, y + h - 1.4, 'appearance as in CellTrack-GNN', fs=4.8, c=GREY_T, ha='center', it=True)

# --------------------------------------------------------------- row 1, card 4: (2) three-class edges, real edges
x, y, w, h = hcard(R1, C1[3])
card('head', x, y, w, h, 'Three-class edges')


def edge_example(x0, title, pair_imgs, pair_masks, labels, probs, lo_hi, sigma):
    """Two thumbnails (t, t+1) with the cells outlined, and the three real class probabilities of the edge."""
    note('head', x0, y + 5.4, title, fs=5.0, c=INK, b=True)
    th = 5.6
    for k in range(2):
        im_, m_ = pair_imgs[k], pair_masks[k]
        yy, xx = np.nonzero(np.isin(m_, labels[0] if k == 0 else labels[1]))
        t_ = Tile('head', im_, xx.mean(), yy.mean(), x0 + k * (th + 0.5), y + 6.8, th, th,
                  max(48, 1.6 * max(np.ptp(yy), np.ptp(xx))), lo_hi)
        for lab_ in (labels[0] if k == 0 else labels[1]):
            t_.outline(m_, lab_, MAGENTA if probs[2] > 0.5 else BLUE, lw=0.6, sigma=sigma)
        t_.label('t' if k == 0 else 't+1', fs=4.6)
    base = y + 20.0
    bx0 = x0 + 2 * th + 1.8
    for k, (lab_, pv, c) in enumerate(zip(('unrel.', 'same', 'div.'), probs, ('#C8D3E0', BLUE, MAGENTA))):
        bx = bx0 + k * 4.4
        rbox('head', bx, base - 9.0 * pv - 0.05, 2.8, 9.0 * pv + 0.05, c, r=0.2, z=3)
        note('head', bx + 1.4, base + 1.2, lab_, fs=4.6, c=MUTED, ha='center')
    line('head', [bx0 - 0.5, bx0 + 3 * 4.4 - 1.0], [base, base], NODE_C, lw=0.4)


# continuation: a confident Scripta link of the example region; division: a held-out DynamicNuclearNet mother
cont = 145
cm = R1M & (EZ['src_label'] == cont) & (EZ['chosen'] == 1)
pc = float(EZ['p'][cm][0]), float(EZ['p_div'][cm][0])
edge_example(x + 2.0, 'continuation', ims_ex, [s_mask(EXW, EXT), s_mask(EXW, EXT + 1)], ([cont], [cont]),
             (max(0.0, 1 - pc[0] - pc[1]), pc[0], pc[1]), LV, 0.0)
dz = np.load(FIN / 'dnn' / 'progct' / 'division' / 'val18' / 'edges.npz')
dm = (dz['round'] == 1) & (dz['src_label'] == 83) & (dz['p_div'] > 0.5)
dt = int(dz['src_frame'][dm][0])
dp = float(dz['p'][dm][0]), float(dz['p_div'][dm][0])
dims = [d_img('val18', dt), d_img('val18', dt + 1)]
dlv = np.percentile(np.concatenate([i.ravel() for i in dims]), [1, 99.8])
edge_example(x + 28.6, 'division (held-out video)', dims, [d_mask('val18', dt), d_mask('val18', dt + 1)],
             ([83], [int(v) for v in dz['dst_label'][dm]]), (max(0.0, 1 - dp[0] - dp[1]), dp[0], dp[1]), dlv, 0.8)
FACTS['head'] = dict(continuation=dict(well=EXW, t=EXT, cell=cont, p_same=pc[0], p_div=pc[1]),
                     division=dict(video='val18', t=dt, mother=83, daughters=[int(v) for v in dz['dst_label'][dm]],
                                   p_same=dp[0], p_div=dp[1]))
rbox('head', x + 2.2, y + 22.6, w - 4.4, 7.4, BLUE_BAND, ec='#BFD0E8', r=0.8, lw=0.4, z=2)
note('head', x + 3.2, y + 24.4, 'one model for videos with and without mitoses: no dataset', fs=5.2, c=INK)
note('head', x + 3.2, y + 26.8, 'flag, no division switch, no class weights (outputs stay probabilities)',
     fs=5.0, c=INK)

# --------------------------------------------------------------- row 2, card 1: (3) ensemble, the real 9 predictions
x, y, w, h = hcard(R2, C2[0])
card('ensemble', x, y, w, h, 'Ensemble')
note('ensemble', x + 2.4, y + 5.4, 'models', fs=4.8, c=MUTED)
note('ensemble', x + 8.0, y + 5.4, '3 mirrored views', fs=4.8, c=MUTED)
for k in range(3):
    yy = y + 7.0 + k * 5.6
    rbox('ensemble', x + 2.4, yy, 4.2, 4.2, BLUE_L, ec=BLUE, r=0.5, lw=0.45, z=3)
    centred('ensemble', x + 4.5, yy + 2.1, str(k + 1), 5.2, c=BLUE_D)
    for j in range(3):
        hh = float(Q9[j, k])
        bx = x + 8.0 + j * 1.9
        rbox('ensemble', bx, yy, 1.4, 4.2, 'white', ec='#C8D3E0', r=0.0, lw=0.3, z=3)
        rbox('ensemble', bx, yy + 4.2 * (1 - hh), 1.4, 4.2 * hh, BLUE, r=0.0, z=4)
bx_ = x + 14.4
line('ensemble', [bx_, bx_ + 0.6, bx_ + 0.6, bx_], [y + 7.0, y + 7.0, y + 22.4, y + 22.4], NODE_C, lw=0.45)
for cy_, col in ((y + 11.0, BLUE_D), (y + 19.0, GOLD_T)):
    line('ensemble', [bx_ + 0.6, bx_ + 3.2], [y + 14.7, cy_], col, lw=0.5)
note('ensemble', bx_ + 3.8, y + 9.8, 'mean', fs=5.0, c=BLUE_D)
txt('ensemble', bx_ + 3.8, y + 12.4, f'p = {P_EX:.2f}', fs=6.4, c=BLUE, b=True)
note('ensemble', bx_ + 3.8, y + 17.8, 'disagreement', fs=5.0, c=GOLD_T)
txt('ensemble', bx_ + 3.8, y + 20.4, f'MI = {MI_EX:.2f}', fs=6.4, c=GOLD_T, b=True)
note('ensemble', x + w / 2, y + h - 3.6, 'one real candidate edge: models disagree,', fs=5.2, c=INK, ha='center')
note('ensemble', x + w / 2, y + h - 1.4, 'so MI flags it although p > τ', fs=5.2, c=INK, ha='center')

# --------------------------------------------------------------- row 2, card 2: (3) calibrated decisions
x, y, w, h = hcard(R2, C2[1])
card('decide', x, y, w, h, 'Calibrated decisions')
ox, oy, s = x + 3.6, y + 16.4, 9.0
rbox('decide', ox, oy - s, s, s, 'white', ec='#9BB0CC', r=0.2, lw=0.4, z=2)
line('decide', [ox, ox + s], [oy, oy - s], MUTED, lw=0.4, ls=(0, (1, 1)))
for r_ in REL.itertuples():
    gid(ax.add_patch(Circle((ox + r_.mean_p * s, oy - r_.frac_pos * s), 0.32, fc=BLUE, ec='white', lw=0.2,
                            zorder=6)), 'decide')
line('decide', [ox + 0.4 * s] * 2, [oy, oy - s], GOLD, lw=0.7)
txt('decide', ox + 0.4 * s, oy - s - 1.0, r'$\tau$ = 0.4', fs=5.2, c=GOLD_T, ha='center')
note('decide', ox + s / 2, oy + 1.3, 'predicted p', fs=4.8, c=MUTED, ha='center')
note('decide', ox - 1.0, oy - s / 2, 'observed', fs=4.8, c=MUTED, ha='center', rotation=90)
# the links ProGCT made in the example region, frame t -> t+1, on frame t+1
dt_ = Tile('decide', ims_ex[1], SX, SY, x + 16.0, y + 5.0, 31.0, 14.6, 200, LV)
dt_.label('t → t+1')
made = R1M & (EZ['chosen'] == 1)
nflag = 0
for k in np.where(made)[0]:
    src, dst = (EXT, int(EZ['src_label'][k])), (EXT + 1, int(EZ['dst_label'][k]))
    if src not in EPOS or dst not in EPOS or not dt_.inside(*EPOS[src]):
        continue
    a_, b_ = dt_.mm(*EPOS[src]), dt_.mm(*EPOS[dst])
    flag = EZ['mi'][k] >= MI_THR
    nflag += flag
    node('decide', *a_, c='white', r=0.45, ghost=True, z=6)
    dt_.clip(arrow('decide', a_, b_, c=GOLD if flag else '#7FB2F0', lw=1.0 if flag else 0.8, head=4))
dt_.label('flagged: high MI', 'br', c='#F4C86A')
FACTS['decide'] = dict(flagged_in_tile=int(nflag))
note('decide', x + w / 2, y + h - 6.0, 'link if p > τ; τ follows from the error costs of', fs=5.2, c=INK,
     ha='center')
note('decide', x + w / 2, y + h - 3.6, 'the evaluation (no tuning); one-to-one assignment', fs=5.2, c=INK,
     ha='center')
note('decide', x + w / 2, y + h - 1.4, 'per frame pair, divisions included', fs=5.2, c=INK, ha='center')

# --------------------------------------------------------------- row 2, card 3: (1) cross-frame check, real gap
x, y, w, h = hcard(R2, C2[2])
card('check', x, y, w, h, 'Cross-frame check')
gw, gt_, gl = 'r08c13', 17, 453
gtr = pd.read_csv(D.track_folder('ours', gw) / 'instance_tracks.csv')
glinks = pd.read_csv(D.track_folder('ours', gw) / 'links.csv')
grow = glinks[(glinks['round'] == 2) & (glinks.src_frame == gt_) & (glinks.src_label == gl)].iloc[0]
gm0, gm2 = s_mask(gw, gt_), s_mask(gw, gt_ + 2)
yy0, xx0 = np.nonzero(gm0 == gl)
yy2, xx2 = np.nonzero(gm2 == int(grow.dst_label))
gcx, gcy = (xx0.mean() + xx2.mean()) / 2, (yy0.mean() + yy2.mean()) / 2
gims = [s_img(gw, gt_ + k) for k in range(3)]
glv = np.percentile(np.concatenate([i[int(gcy - 150):int(gcy + 150), int(gcx - 150):int(gcx + 150)].ravel()
                                    for i in gims]), [1, 99.7])
tw_ = (w - 2.0 - 1.2) / 3
gtiles = [Tile('check', gims[k], gcx, gcy, x + 1.0 + k * (tw_ + 0.6), y + 8.0, tw_, 11.0, 210, glv) for k in range(3)]
for k, lab in enumerate(['t', 't+1', 't+2']):
    gtiles[k].label(lab, fs=4.6)
gtiles[0].outline(gm0, gl, BLUE, lw=0.8)
gtiles[2].outline(gm2, int(grow.dst_label), BLUE, lw=0.8)
gtiles[1].outline(gm0, gl, '#FF6B6B', lw=0.7, ls=(0, (1.6, 1.2)))
pa = gtiles[0].mm(xx0.mean(), yy0.mean())
pb = gtiles[2].mm(xx2.mean(), yy2.mean())
arrow('check', (pa[0], y + 7.6), (pb[0], y + 7.6), c=BLUE, lw=0.9, rad=-0.25, head=4.5)
txt('check', x + w / 2, y + 20.6, f'rejoined: g = 2, p = {grow.p:.2f}', fs=5.0, c=BLUE, b=True, ha='center')
centred('check', gtiles[1].x + tw_ / 2, y + 17.8, 'missed', 4.6, c='#FF8A80', b=True)
FACTS['check'] = dict(well=gw, t=gt_, label=gl, dst=int(grow.dst_label), p=float(grow.p))
note('check', x + w / 2, y + h - 6.0, 'a 2nd pass joins tracks that end', fs=5.2, c=INK, ha='center')
note('check', x + w / 2, y + h - 3.6, 'at t to tracks that start at t+2,', fs=5.2, c=INK, ha='center')
note('check', x + w / 2, y + h - 1.4, 'with the same p and τ', fs=5.2, c=INK, ha='center')

# --------------------------------------------------------------- row 2, card 4: outputs, real tracks of the region
x, y, w, h = hcard(R2, C2[3])
card('output', x, y, w, h, 'Output')
ot = Tile('output', s_img(EXW, 16), SX, SY, x + 1.4, y + 5.0, 18.0, 14.6, 260, LV)
ot.label('tracks, frames 10–16', fs=4.6)
olinks = pd.read_csv(ex_dir / 'links.csv')
ids = ['#4E79A7', '#59A14F', '#9C755F', '#76B7B2', '#B6992D', '#A0CBE8', '#8CD17D', '#2F4B7C']
tid_of = {(int(a), int(b)): int(c) for a, b, c in zip(ETR.frame_index, ETR.instance_id, ETR.track_id)}
colour, ci = {}, 0
for q in olinks.itertuples():
    if not 10 <= q.src_frame < 16:
        continue
    s_, d_ = (q.src_frame, q.src_label), (q.dst_frame, q.dst_label)
    if s_ not in EPOS or d_ not in EPOS or not ot.inside(*EPOS[s_]):
        continue
    tid = tid_of[s_]
    if tid not in colour:
        colour[tid] = ids[ci % len(ids)]
        ci += 1
    a_, b_ = ot.mm(*EPOS[s_]), ot.mm(*EPOS[d_])
    flag = q.mi >= MI_THR
    ot.clip(line('output', [a_[0], b_[0]], [a_[1], b_[1]], GOLD if flag else colour[tid], lw=1.1 if flag else 0.7,
                 z=5.5 if flag else 5))
cx0 = x + 21.0
for k, (lab, val) in enumerate((('link', 'p, MI flag'), ('track', 'weakest link'), ('video', 'expected errors'))):
    yy = y + 7.0 + k * 4.4
    rbox('output', cx0, yy - 1.6, w - (cx0 - x) - 1.4, 3.4, '#FFF7E6', ec='#EBCB8B', r=0.6, lw=0.35, z=2)
    note('output', cx0 + 0.7, yy, lab, fs=5.0, c=INK, b=True)
    note('output', cx0 + 7.4, yy, val, fs=5.0, c=GOLD_T)
note('output', cx0 + (w - (cx0 - x) - 1.4) / 2, y + 19.6, 'confidence', fs=5.0, c=INK, ha='center')
note('output', x + w / 2, y + h - 6.0, 'tracks and lineages; every decision', fs=5.2, c=INK, ha='center')
note('output', x + w / 2, y + h - 3.6, 'carries its probability; gold = flagged', fs=5.2, c=INK, ha='center')
note('output', x + w / 2, y + h - 1.4, 'links that reviewers check first', fs=5.2, c=INK, ha='center')

# --------------------------------------------------------------- flow arrows
for a_, b_ in zip(C1[:-1], C1[1:]):
    arrow('flow', (a_[1] + 0.1, (R1[0] + R1[1]) / 2), (b_[0] - 0.1, (R1[0] + R1[1]) / 2), c=BLUE, lw=0.9, head=5)
for a_, b_ in zip(C2[:-1], C2[1:]):
    arrow('flow', (a_[1] + 0.1, (R2[0] + R2[1]) / 2), (b_[0] - 0.1, (R2[0] + R2[1]) / 2), c=BLUE, lw=0.9, head=5)
gy = (R1[1] + R2[0]) / 2
xe = (C1[3][0] + C1[3][1]) / 2
xs_ = (C2[0][0] + C2[0][1]) / 2
line('flow', [xe, xe, xs_, xs_], [R1[1] + 0.1, gy, gy, gy], BLUE, lw=0.9, z=6)
arrow('flow', (xs_, gy - 0.05), (xs_, R2[0] - 0.1), c=BLUE, lw=0.9, head=5)
txt('flow', (xe + xs_) / 2, gy - 1.25, 'for every candidate edge: three class probabilities, 3 models × 3 views',
    fs=5.0, c=BLUE_D, ha='center', it=True)
(OUT / 'facts.json').write_text(json.dumps(FACTS, indent=1), encoding='utf-8')

# =============================================================== save, audit, Inkscape clean-up
fig.savefig(OUT / 'fig.svg')
fig.savefig(OUT / 'fig.pdf')
fig.savefig(OUT / 'fig.png', dpi=600)


def audit():
    r = fig.canvas.get_renderer()
    boxes = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        boxes.append((t.get_gid().split('__')[0], t.get_text(), bb.x0 / fig.dpi * 25.4, H - bb.y1 / fig.dpi * 25.4,
                      bb.x1 / fig.dpi * 25.4, H - bb.y0 / fig.dpi * 25.4, t.get_fontsize()))
    probs = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if min(a[4], b[4]) - max(a[2], b[2]) > 0.15 and min(a[5], b[5]) - max(a[3], b[3]) > 0.15:
                probs.append(f'overlap: {a[1]!r} / {b[1]!r}')
    cards = {'input': (R1, C1[0]), 'graph': (R1, C1[1]), 'network': (R1, C1[2]), 'head': (R1, C1[3]),
             'ensemble': (R2, C2[0]), 'decide': (R2, C2[1]), 'check': (R2, C2[2]), 'output': (R2, C2[3])}
    for part, s, x0, y0, x1, y1, fs in boxes:
        if part in cards:
            (cy0, cy1), (cx0, cx1) = cards[part]
            if x0 < cx0 + 0.2 or x1 > cx1 - 0.2 or y0 < cy0 + 0.1 or y1 > cy1 - 0.1:
                probs.append(f'outside {part}: {s!r} ({x0:.1f}-{x1:.1f}, {y0:.1f}-{y1:.1f})')
    return dict(texts=len(boxes), min_font_pt=min(b[6] for b in boxes), problems=probs)


report = audit()
(OUT / 'audit.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
print(json.dumps(report, indent=1))


def flatten_for_inkscape(path):
    """Unwrap Matplotlib's figure_1 / axes_1 groups and label every object with its part (Inkscape Objects panel);
    the white page background is locked."""
    svg_ns, ink_ns = 'http://www.w3.org/2000/svg', 'http://www.inkscape.org/namespaces/inkscape'
    sodi_ns = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
    for p_, u_ in (('', svg_ns), ('xlink', 'http://www.w3.org/1999/xlink'), ('inkscape', ink_ns), ('sodipodi', sodi_ns)):
        ET.register_namespace(p_, u_)
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
    for el in root.iter():
        if '__' in el.get('id', ''):
            el.set(f'{{{ink_ns}}}label', el.get('id').split('__')[0])
        elif el.get('id') == 'patch_1':
            el.set(f'{{{ink_ns}}}label', 'background (locked)')
            el.set(f'{{{sodi_ns}}}insensitive', 'true')
    tree.write(path, encoding='utf-8', xml_declaration=True)


flatten_for_inkscape(OUT / 'fig.svg')
print('written', OUT)
