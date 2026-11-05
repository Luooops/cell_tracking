"""Framework figure of ProGCT: how one calibrated posterior per decision is computed and used.

Row 1 (scoring):  frame-pair graphs (t, t+g), g = 1, 2, 3 in training -> graph network (appearance and geometry
                  streams in parallel, fused in the node and edge features, message passing, edge head) ->
                  three-class edges
Row 2 (deciding): ensemble of 3 models x 3 views -> calibrated decisions -> constrained gap closing (g = 2) -> outputs
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
# Row 1 (user, 2026-11-05): input -> two parallel branches, stacked (frame-gap graph above: the graph and its geometric
# features, computed from the detections; appearance embedding below: the learned vector e of each cell) -> graph
# network (where both meet: node = [geometry, e], candidate edge = [geometry, g, e_i . e_j]) -> three-class edges.
# Feature colours are the same in every card: geometry light blue, the gap g blue, appearance grey. Each branch ends
# in its output features at its right edge, where the flow joins.
R1, R2 = (3.0, 35.0), (40.0, 72.0)          # rows (y0, y1)
R1T, R1B = (3.0, 22.4), (23.6, 35.0)        # the two branches of row 1 (the graph needs the height)
C1 = [(0.6, 17.6), (19.8, 72.8), (75.0, 121.0), (123.2, 174.0)]
C2 = [(0.6, 34.0), (36.2, 86.0), (88.2, 128.0), (130.2, 174.0)]
GR, GEO = '#C9D1D8', '#A9BEDD'                             # appearance / geometric features (g: BLUE)


def blocks(part, x0, y0, groups, s=2.0, gap=0.7, dims=None):
    """A feature vector drawn as concatenated groups of cubes, each group named under it: groups = [(colours, label)].
    dims: the width of each group, written under its name (user: geometry and g separately).
    Returns the x after the last cube."""
    xx = x0
    for k_, (colours, lab) in enumerate(groups):
        tokens(part, xx, y0, s, list(colours))
        width = len(colours) * (s + 0.15) - 0.15
        if lab in ('g', 'e'):
            note(part, xx + width / 2, y0 + s + 1.3, lab, fs=4.8, c=BLUE if lab == 'g' else INK, b=lab == 'g',
                 it=True, ha='center')
        elif lab == 'eij':
            txt(part, xx + width / 2, y0 + s + 1.3, r'$\mathit{e_i}$·$\mathit{e_j}$', fs=4.4, c=INK, ha='center')
        else:
            note(part, xx + width / 2, y0 + s + 1.3, lab, fs=4.2, c=MUTED, ha='center')
        if dims:
            note(part, xx + width / 2, y0 + s + 2.9, dims[k_], fs=4.0, c=GREY_T, ha='center')
        xx += width + gap
    return xx - gap + 0.35 * s


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
for k, (dy, lab) in enumerate(((5.4, 'frame t'), (16.2, 'frame t+g'))):
    t = EXT + k
    y0_, x0_ = int(SY - WIN / 2), int(SX - WIN * 0.9)
    crop = ims_ex[k][y0_:y0_ + WIN, x0_:x0_ + int(WIN * 1.8)]
    m = s_mask(EXW, t)[y0_:y0_ + WIN, x0_:x0_ + int(WIN * 1.8)]
    to = skewed_slab('input', norm(crop, LV), x + 1.2, y + dy, 11.6, 5.6, 3.0, 0.9, z=2 + 2 * k)
    for lab_ in np.unique(m):
        if lab_ == 0:
            continue
        for cc in measure.find_contours(ndi.zoom((m == lab_).astype(float), 2, order=1), 0.5):
            u, v = (cc[:, 1] / 2) / m.shape[1], (cc[:, 0] / 2) / m.shape[0]
            pts = np.array([to(a_, b_) for a_, b_ in zip(u, v)])
            line('input', pts[:, 0], pts[:, 1], '#E8EEF2', lw=0.25, z=2.5 + 2 * k)
    note('input', x + w / 2, y + dy + 8.0, lab, fs=5.2, c=INK, ha='center')
note('input', x + w / 2, y + h - 3.0, 'segmented nuclei', fs=4.8, ha='center', it=True)
note('input', x + w / 2, y + h - 1.0, 'g = 1, 2, 3', fs=4.8, ha='center', it=True)


def sqrt_label(part, cx, yb, lead, under, fs, c):
    """'lead' + a LaTeX-style radical over 'under', centred at cx on baseline yb: the texts stay editable SVG text
    (Times New Roman) and the radical sign with its bar is a path (mathtext's \\sqrt shows as 'p' in Inkscape)."""
    wl, wu, wr = sum(text_width(s, fs) for s, _ in lead), text_width(under, fs), 1.0
    x0 = cx - (wl + 0.3 + wr + wu + 0.15) / 2
    xx = x0
    for s, it_ in lead:                                     # lead: [(text, italic)]
        txt(part, xx, yb, s, fs=fs, c=c, it=it_, va='baseline')
        xx += text_width(s, fs)
    xr, top = x0 + wl + 0.3, yb - 0.66 * fs / PT - 0.25    # a thin space; bar just above the italic letter
    line(part, [xr, xr + 0.22, xr + 0.5, xr + wr, xr + wr + wu + 0.15],
         [yb - 0.45, yb - 0.58, yb + 0.3, top, top], c, lw=0.35, z=9)
    txt(part, xr + wr, yb, under, fs=fs, c=c, it=True, va='baseline')


# --------------------------------------------------------------- row 1, branch 1: frame-gap graph (geometry)
# models.gnn_tracking.graph.pair_graph: one graph per pair (t, t+g); candidate edges = pairs closer than r*sqrt(g);
# message edges = candidates + each cell's 4 nearest same-frame cells; geometric node / edge features, g included.
# Example (user: the first one was crowded): r15c12 frame 8, cell GS, chosen for 3 well-separated candidates in
# different directions and 4 neighbours within 1.4 gates (same well and frames as the rest of the figure).
GS = 68
GX, GY = EPOS[(EXT, GS)]
CAND_G = R1M & (EZ['src_label'] == GS)
GATE_G = float(EZ['gate'][CAND_G][0])
x, y, w, h = hcard(R1T, C1[1])
card('graph', x, y, w, h, 'Frame-gap graph')
ts, off = 11.0, 1.8
ty0 = y + 4.4
allf8 = {b: p_ for (a, b), p_ in EPOS.items() if a == EXT}
nn = sorted((np.hypot(p_[0] - GX, p_[1] - GY), b) for b, p_ in allf8.items() if b != GS)[:4]
HPX = max(2.6 * GATE_G, 2.3 * nn[-1][0])                   # all 4 neighbours and the whole gate inside the tiles
gims = [s_img(EXW, EXT + k) for k in range(4)]
GLV = np.percentile(np.concatenate([i[int(GY - HPX):int(GY + HPX), int(GX - HPX):int(GX + HPX)].ravel()
                                    for i in gims[:2]]), [1, 99.7])
tl = Tile('graph', gims[0], GX, GY, x + 1.4, ty0, ts, ts, HPX, GLV)
tl.label('t', fs=4.4)
note('graph', tl.x + ts / 2, ty0 + ts + 1.7, '4 neighbours', fs=4.4, c=PURPLE, ha='center')   # under frame t
sx0 = x + 1.4 + ts + 4.0
for k, z_ in ((3, 1.0), (2, 2.0)):                          # frames t+3, t+2 behind t+1; each layer (image and frame)
    Tile('graph', gims[k], GX, GY, sx0 + (k - 1) * off, ty0 + (k - 1) * off, ts, ts, HPX, GLV, z=z_)
tr = Tile('graph', gims[1], GX, GY, sx0, ty0, ts, ts, HPX, GLV, z=3.5)                 # lies wholly below the next
tr.label('t+1', fs=4.4)
for k in range(3):                                          # each layer is one pair graph: g at its top-right corner
    txt('graph', sx0 + k * off + ts + 0.5, ty0 + k * off + 0.7, f'g = {k + 1}', fs=4.6, c=BLUE, b=True)
tw_title = text_width('Frame-gap graph', 6.6, True)         # gate label on the title line, above frame t+1 (user)
sqrt_label('graph', max(x + 1.2 + tw_title + 6.0, sx0 + ts / 2), y + 3.1, [('gate ', False), ('r', True)], 'g',
           4.6, TEAL)
assert all(tl.inside(*allf8[b]) for _, b in nn), 'a neighbour is outside the tile'
src_mm = tl.mm(GX, GY)
for _, b in nn:                                             # only the cells the drawing explains: the source,
    b_ = tl.mm(*allf8[b])                                   # its 4 in-frame neighbours (message edges) ...
    tl.clip(line('graph', [src_mm[0], b_[0]], [src_mm[1], b_[1]], PURPLE, lw=0.6, ls=(0, (1.5, 1)), z=5))
    node('graph', *b_, c='#C9B6EE', r=0.42, z=6)
node('graph', *src_mm, c=BLUE, r=0.55, z=6)
gc = tr.mm(GX, GY)                                          # ... and its candidates in t+1, inside the g = 1 gate
gr_mm = GATE_G / tr.wpx * tr.w
tr.clip(gid(ax.add_patch(Circle(gc, gr_mm, fc=TEAL_L, ec=TEAL, lw=0.5, ls=(0, (2, 1)), alpha=0.3, zorder=4.2)),
            'graph'))
tr.clip(gid(ax.add_patch(Circle(gc, gr_mm, fc='none', ec=TEAL, lw=0.5, ls=(0, (2, 1)), zorder=5)), 'graph'))
node('graph', *gc, c='white', r=0.4, ghost=True, z=6)
for d_ in EZ['dst_label'][CAND_G]:                          # the source's real candidate edges (g = 1)
    p_ = EPOS.get((EXT + 1, int(d_)))
    if p_ is not None and tr.inside(*p_):
        b_ = tr.mm(*p_)
        line('graph', [src_mm[0], b_[0]], [src_mm[1], b_[1]], BLUE, lw=0.5, z=6.5)
        node('graph', *b_, c='#A9BEDD', r=0.42, z=6.6)
# output: the geometric features of every node and candidate edge (g among them), at the right edge
ox = x + w - 9.4
note('graph', ox + 7.6, y + 2.6, 'node  N × 8', fs=4.6, c=MUTED, ha='right')       # N cells of t, t+g
blocks('graph', ox, y + 4.6, [((GEO, GEO), 'geometry')], dims=['8'])
note('graph', ox + 7.6, y + 11.4, 'edge  C × 13', fs=4.6, c=MUTED, ha='right')      # C candidate edges
blocks('graph', ox, y + 13.2, [((GEO, GEO), 'geometry'), ((BLUE,), 'g')], gap=1.2, dims=['12', '1'])
arrow('graph', (sx0 + 2 * off + ts + 6.2, y + 14.2), (ox - 0.6, y + 14.2), c=GREY_T, lw=0.6, head=3.5)

# --------------------------------------------------------------- row 1, branch 2: appearance embedding
# models.gnn_tracking.model.AppearanceNet: masked crop -> CNN -> e (32-d, unit length); trained with a contrastive
# loss; e enters the node features and, as e_i . e_j, the candidate-edge features (graph network card).
x, y, w, h = hcard(R1B, C1[1])
card('appear', x, y, w, h, 'Appearance embedding')
ay = y + 7.0
gid(ax.imshow(CROP, cmap='gray', vmin=0, vmax=1, extent=(x + 1.6, x + 5.6, ay + 2.0, ay - 2.0), zorder=4,
              interpolation='lanczos'), 'appear')
note('appear', x + 1.4, ay + 3.3, 'crop 64 × 64', fs=4.4, c=MUTED)
poly('appear', [(x + 7.2, ay - 2.2), (x + 11.6, ay - 1.1), (x + 11.6, ay + 1.1), (x + 7.2, ay + 2.2)], GR,
     ec=GREY_T, lw=0.4)
note('appear', x + 9.4, ay, 'CNN', fs=4.8, c=INK, ha='center')
arrow('appear', (x + 5.8, ay), (x + 7.1, ay), c=GREY_T, lw=0.6, head=3.5)
ox = x + w - 9.0                                            # output e at the right edge, like the graph's features
arrow('appear', (x + 11.8, ay), (ox - 0.5, ay), c=GREY_T, lw=0.6, head=3.5)
note('appear', (x + 11.8 + ox) / 2, ay - 1.4, 'one vector per cell', fs=4.6, c=INK, ha='center')
note('appear', (x + 11.8 + ox) / 2, ay + 1.6, 'contrastive loss in training', fs=4.4, c=GREY_T, it=True,
     ha='center')
note('appear', ox + 7.6, ay - 2.8, 'node  N × 32', fs=4.6, c=MUTED, ha='right')
blocks('appear', ox, ay - 1.0, [((GR, GR, GR), 'e')])

# --------------------------------------------------------------- row 1, card 3: graph network (the branches meet)
# node encoder on [geometry, e] -> 2 x attention message passing over candidate + in-frame edges -> edge head
# MLP(h_i, h_j, candidate-edge features: geometry, g, e_i . e_j) -> 3 logits.
x, y, w, h = hcard(R1, C1[2])
card('network', x, y, w, h, 'Graph network')
ya, yg = y + 8.0, y + 16.4                                  # node row, edge row: every arrow runs straight
note('network', x + 1.4, ya - 2.8, 'node  N × 40', fs=4.6, c=MUTED)    # the two branches' blocks, concatenated
nx_ = blocks('network', x + 1.4, ya - 1.0, [((GEO, GEO), 'geometry'), ((GR,), 'e')], gap=1.2,
             dims=['8', '32'])
note('network', x + 1.4, yg - 2.8, 'edge  C × 16', fs=4.6, c=MUTED)
ex_ = blocks('network', x + 1.4, yg - 1.0, [((GEO, GEO), 'geometry'), ((BLUE,), 'g'), ((GR,), 'sim.')], gap=1.2,
             dims=['12', '1', '3'])
mx, my, mw, mh = x + 16.8, y + 4.8, 11.8, 12.6              # message passing: tall enough for both rows
for k in range(2):
    rbox('network', mx + k * 0.9, my + k * 0.9, mw, mh, 'white', ec=GREY_E, r=0.8, lw=0.45, z=3 + k)
pts = [(mx + 2.8, my + 4.6), (mx + 5.9, my + 2.8), (mx + 9.0, my + 5.0), (mx + 4.6, my + 9.0), (mx + 8.6, my + 9.8)]
for a_, b_ in ((0, 1), (1, 2), (0, 3), (3, 4), (2, 4), (1, 3)):
    line('network', [pts[a_][0] + 0.9, pts[b_][0] + 0.9], [pts[a_][1] + 0.9, pts[b_][1] + 0.9], '#9AA7B3', lw=0.45,
         z=5)
for p_ in pts:
    node('network', p_[0] + 0.9, p_[1] + 0.9, c='#8A97A4', r=0.6, z=6)
note('network', mx + mw / 2 + 0.45, my + mh + 2.4, 'attention message', fs=4.4, c=GREY_T, ha='center')
note('network', mx + mw / 2 + 0.45, my + mh + 4.3, 'passing × 2', fs=4.4, c=GREY_T, ha='center')
arrow('network', (nx_ + 0.2, ya), (mx - 0.2, ya), c=GREY_T, lw=0.6, head=3.5)                  # nodes -> GNN
hx, hw = x + 32.2, 12.6
rbox('network', hx, ya - 3.2, hw, 6.2, '#F2F4F6', ec=GREY_E, r=0.8, lw=0.45, z=3)
note('network', hx + hw / 2, ya - 1.6, 'edge head', fs=4.8, c=INK, b=True, ha='center')
txt('network', hx + hw / 2, ya + 1.0, r'$h_i$, $h_j$, edge', fs=4.6, c=INK, ha='center')
arrow('network', (mx + mw + 1.0, ya), (hx - 0.2, ya), c=GREY_T, lw=0.6, head=3.5)              # GNN -> head
# the edge features also reach the head directly (skip): branch below the edge row, right, up into the head
bx, sy = ex_ + 1.6, y + 24.0
arrow('network', (ex_ + 0.2, yg), (mx - 0.2, yg), c=GREY_T, lw=0.6, head=3.5)                  # edges -> GNN
gid(ax.add_patch(Circle((bx, yg), 0.35, fc=GREY_T, ec='none', zorder=7)), 'network')
line('network', [bx, bx, hx + 2.0], [yg, sy, sy], GREY_T, lw=0.6)
arrow('network', (hx + 2.0, sy), (hx + 2.0, ya + 3.1), c=GREY_T, lw=0.6, head=3.5)
note('network', hx - 2.4, sy - 0.9, 'skip', fs=4.2, c=GREY_T, it=True, ha='center')
# output, bottom right: three logits per candidate edge (unrelated, same, division)
ax_ = hx + 8.6
arrow('network', (ax_, ya + 3.1), (ax_, y + 24.2), c=GREY_T, lw=0.6, head=3.5)                 # head -> logits
for k, (c, s) in enumerate((('#C8D3E0', 'u'), (BLUE, 's'), (MAGENTA, 'd'))):
    rbox('network', ax_ - 3.4 + k * 2.6, y + 24.4, 1.6, 1.6, c, r=0.2, z=3)
    note('network', ax_ - 2.6 + k * 2.6, y + 27.4, s, fs=4.4, c=MUTED, ha='center')
note('network', ax_ - 4.2, y + 25.2, 'logits', fs=4.4, c=MUTED, ha='right')
for k, s in enumerate(('node: [geometry 8, e 32] → N × 40 → h: N × 64',              # dimensions (model.py):
                       'edge: [geometry 12, g 1, sim. 3] → C × 16',                   # blocks side by side
                       r'head: [$h_i$, $h_j$, edge] → C × 144 → C × 3')):          # are concatenated
    txt('network', x + 1.4, y + 25.9 + k * 2.15, s, fs=4.2, c=GREY_T)
# --------------------------------------------------------------- row 1, card 4: three-class edges, real edges
x, y, w, h = hcard(R1, C1[3])
card('head', x, y, w, h, 'Three-class edges')


def edge_example(x0, title, pair_imgs, pair_masks, labels, probs, lo_hi, sigma):
    """Two thumbnails (t, t+1) with the cells outlined, and the three real class probabilities of the edge as bars
    (u unrelated, s same cell, d division) as tall as the thumbnails."""
    note('head', x0, y + 5.4, title, fs=4.8, c=INK, b=True)
    tw, th = 6.4, 11.4
    for k in range(2):
        im_, m_ = pair_imgs[k], pair_masks[k]
        yy, xx = np.nonzero(np.isin(m_, labels[0] if k == 0 else labels[1]))
        t_ = Tile('head', im_, xx.mean(), yy.mean(), x0 + k * (tw + 0.5), y + 6.8, tw, th,
                  max(48, 1.6 * np.ptp(yy), 1.5 * np.ptp(xx) * th / tw), lo_hi)
        for lab_ in (labels[0] if k == 0 else labels[1]):
            t_.outline(m_, lab_, MAGENTA if probs[2] > 0.5 else BLUE, lw=0.6, sigma=sigma)
        t_.label('t' if k == 0 else 't+1', fs=4.4)
    base = y + 6.8 + th
    bx0 = x0 + 2 * tw + 0.5 + 1.4
    for k, (lab_, pv, c) in enumerate(zip(('u', 's', 'd'), probs, ('#C8D3E0', BLUE, MAGENTA))):
        bx = bx0 + k * 2.6
        rbox('head', bx, base - (th - 1.0) * pv - 0.05, 1.8, (th - 1.0) * pv + 0.05, c, r=0.2, z=3)
        note('head', bx + 0.9, base + 1.2, lab_, fs=4.6, c=MUTED, ha='center')
    line('head', [bx0 - 0.4, bx0 + 3 * 2.6 - 0.4], [base, base], NODE_C, lw=0.4)


# continuation: a confident Scripta link of the example region; division: a held-out DynamicNuclearNet mother
cont = 145
cm = R1M & (EZ['src_label'] == cont) & (EZ['chosen'] == 1)
pc = float(EZ['p'][cm][0]), float(EZ['p_div'][cm][0])
edge_example(x + 1.2, 'continuation', ims_ex, [s_mask(EXW, EXT), s_mask(EXW, EXT + 1)], ([cont], [cont]),
             (max(0.0, 1 - pc[0] - pc[1]), pc[0], pc[1]), LV, 0.0)
dz = np.load(FIN / 'dnn' / 'progct' / 'division' / 'val18' / 'edges.npz')
dm = (dz['round'] == 1) & (dz['src_label'] == 83) & (dz['p_div'] > 0.5)
dt = int(dz['src_frame'][dm][0])
dp = float(dz['p'][dm][0]), float(dz['p_div'][dm][0])
dims = [d_img('val18', dt), d_img('val18', dt + 1)]
dlv = np.percentile(np.concatenate([i.ravel() for i in dims]), [1, 99.8])
edge_example(x + 25.4, 'division (held-out)', dims, [d_mask('val18', dt), d_mask('val18', dt + 1)],
             ([83], [int(v) for v in dz['dst_label'][dm]]), (max(0.0, 1 - dp[0] - dp[1]), dp[0], dp[1]), dlv, 0.8)
FACTS['head'] = dict(continuation=dict(well=EXW, t=EXT, cell=cont, p_same=pc[0], p_div=pc[1]),
                     division=dict(video='val18', t=dt, mother=83, daughters=[int(v) for v in dz['dst_label'][dm]],
                                   p_same=dp[0], p_div=dp[1]))
rbox('head', x + 1.2, y + 21.4, w - 2.4, 8.6, BLUE_BAND, ec='#BFD0E8', r=0.8, lw=0.4, z=2)
for k, s in enumerate(('one model for videos with and without mitoses:', 'no dataset flag, no division switch,',
                       'no class weights (outputs stay probabilities)')):
    note('head', x + w / 2, y + 23.2 + k * 2.3, s, fs=4.9, c=INK, ha='center')

# --------------------------------------------------------------- row 2, card 1: (3) ensemble, the real 9 predictions
x, y, w, h = hcard(R2, C2[0])
card('ensemble', x, y, w, h, 'Ensemble')
note('ensemble', x + 1.6, y + 5.4, 'models', fs=4.8, c=MUTED)
note('ensemble', x + 7.6, y + 5.4, '3 mirrored views', fs=4.8, c=MUTED)
for k in range(3):                                          # compact (user): rows fill the card height
    yy = y + 7.0 + k * 5.6
    rbox('ensemble', x + 1.6, yy, 4.6, 4.6, BLUE_L, ec=BLUE, r=0.5, lw=0.45, z=3)
    centred('ensemble', x + 3.9, yy + 2.3, str(k + 1), 5.4, c=BLUE_D)
    for j in range(3):
        hh = float(Q9[j, k])
        bx = x + 7.6 + j * 2.3
        rbox('ensemble', bx, yy, 1.8, 4.6, 'white', ec='#C8D3E0', r=0.0, lw=0.3, z=3)
        rbox('ensemble', bx, yy + 4.6 * (1 - hh), 1.8, 4.6 * hh, BLUE, r=0.0, z=4)
bx_ = x + 15.0
line('ensemble', [bx_, bx_ + 0.6, bx_ + 0.6, bx_], [y + 7.0, y + 7.0, y + 22.8, y + 22.8], NODE_C, lw=0.45)
for cy_, col in ((y + 10.6, BLUE_D), (y + 19.2, GOLD_T)):
    line('ensemble', [bx_ + 0.6, bx_ + 3.0], [y + 14.9, cy_], col, lw=0.5)
note('ensemble', bx_ + 3.6, y + 9.4, 'mean', fs=5.0, c=BLUE_D)
txt('ensemble', bx_ + 3.6, y + 12.0, 'p', fs=6.4, c=BLUE, b=True, it=True)     # no example values: the
note('ensemble', bx_ + 3.6, y + 18.0, 'disagreement', fs=5.0, c=GOLD_T)              # bars show them (user)
txt('ensemble', bx_ + 3.6, y + 20.6, 'MI', fs=6.4, c=GOLD_T, b=True)
for k, s in enumerate(('one real candidate edge:', 'models disagree, so MI flags it', 'although p > τ')):
    note('ensemble', x + w / 2, y + h - 6.0 + k * 2.3, s, fs=5.0, c=INK, ha='center')

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
txt('decide', ox + 0.4 * s, oy - s - 1.0, 'τ = 0.4', fs=5.2, c=GOLD_T, ha='center')
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

# --------------------------------------------------------------- row 2, card 3: constrained gap closing, real gap
x, y, w, h = hcard(R2, C2[2])
card('check', x, y, w, h, 'Constrained gap closing')
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
txt('check', x + w / 2, y + 20.6, 'rejoined: g = 2', fs=5.0, c=BLUE, b=True, ha='center')
centred('check', gtiles[1].x + tw_ / 2, y + 17.8, 'missed', 4.6, c='#FF8A80', b=True)
FACTS['check'] = dict(well=gw, t=gt_, label=gl, dst=int(grow.dst_label), p=float(grow.p))
note('check', x + w / 2, y + h - 6.0, 'a 2nd pass joins tracks that end', fs=5.2, c=INK, ha='center')
note('check', x + w / 2, y + h - 3.6, 'at t to tracks that start at t+2,', fs=5.2, c=INK, ha='center')
note('check', x + w / 2, y + h - 1.4, 'with the same p and τ', fs=5.2, c=INK, ha='center')

# --------------------------------------------------------------- row 2, card 4: outputs, real tracks of the region
x, y, w, h = hcard(R2, C2[3])
card('output', x, y, w, h, 'Output')
ot = Tile('output', s_img(EXW, 16), SX, SY, x + 1.4, y + 5.0, 18.0, 14.6, 260, LV)
ot.label('tracks', fs=4.6)
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
ym, yt_, yb_ = (R1[0] + R1[1]) / 2, (R1T[0] + R1T[1]) / 2, (R1B[0] + R1B[1]) / 2
xf, xj = (C1[0][1] + C1[1][0]) / 2, (C1[1][1] + C1[2][0]) / 2          # fork after the input, join before the network
line('flow', [C1[0][1] + 0.1, xf], [ym, ym], BLUE, lw=0.9, z=6)
line('flow', [xf, xf], [yt_, yb_], BLUE, lw=0.9, z=6)
for yy in (yt_, yb_):
    arrow('flow', (xf, yy), (C1[1][0] - 0.1, yy), c=BLUE, lw=0.9, head=5)
    line('flow', [C1[1][1] + 0.1, xj], [yy, yy], BLUE, lw=0.9, z=6)
line('flow', [xj, xj], [yt_, yb_], BLUE, lw=0.9, z=6)
arrow('flow', (xj, ym), (C1[2][0] - 0.1, ym), c=BLUE, lw=0.9, head=5)
arrow('flow', (C1[2][1] + 0.1, ym), (C1[3][0] - 0.1, ym), c=BLUE, lw=0.9, head=5)
for a_, b_ in zip(C2[:-1], C2[1:]):
    arrow('flow', (a_[1] + 0.1, (R2[0] + R2[1]) / 2), (b_[0] - 0.1, (R2[0] + R2[1]) / 2), c=BLUE, lw=0.9, head=5)
gy = (R1[1] + R2[0]) / 2
xe = (C1[-1][0] + C1[-1][1]) / 2
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
    cards = {'input': (R1, C1[0]), 'graph': (R1T, C1[1]), 'appear': (R1B, C1[1]), 'network': (R1, C1[2]),
             'head': (R1, C1[3]),
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
