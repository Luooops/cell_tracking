"""Figure 0 (full width): method overview of the GNN tracker (models/gnn_tracking + models/gnn_bayesian, final config).
a  one frame pair (t, t+g) of a dev well -> candidate edges inside the gating radius r*sqrt(g) + same-frame kNN context
b  edge network: masked crops -> appearance embedding, node / edge features, 2 attention message-passing layers,
   3-way edge head (unrelated / same / division)
c  ensemble members x mirrored views averaged -> posterior p and mutual information
d  two-round min-cost assignment with cost-derived thresholds, then per-link / per-track confidence
Image and detections are real (scripta refine2, dev well r03c12, frames 10-11); the probability bars of c are
schematic."""
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

import data
import style
from models.gnn_tracking.gating import gating_radius
from utils.dataset import appearance_crop

WELL, T = 'r03c12', 10
BOX = (1680, 1320, 420)                 # row, col, size of the shown window (pixels of the 2160 x 2160 frame)
W, H = style.FULL / style.MM, 74.0      # canvas in mm

C_T, C_N = '#3B6FB6', '#2A9D8F'         # frame t / frame t+g nodes
CLS = {'unrelated': '#BDBDBD', 'same': '#3B6FB6', 'division': '#C8553D'}
INK, SOFT, FILL = '#262626', '#7F7F7F', '#F4F6FA'
NET = {'app': '#E8EEF8', 'mp': '#DCE6F5', 'head': '#CFDDF2', 'loss': '#FBEFE6'}


# ---------------------------------------------------------------- drawing helpers (canvas units = mm)
def box(ax, x, y, w, h, text='', fc=FILL, ec=SOFT, fs=6, lw=0.5, r=0.8, color=INK, weight='normal', **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, weight=weight,
                zorder=3, linespacing=1.15, **kw)


def arrow(ax, a, b, color=INK, lw=0.6, style_='-|>', rad=0.0, ls='-', ms=5, z=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style_, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=z))


def title(ax, x, y, letter, text):
    ax.text(x, y, letter, fontsize=8, weight='bold', ha='left', va='baseline', color='black')
    ax.text(x + 3.2, y, text, fontsize=7, ha='left', va='baseline', color='black')


def bars(ax, x, y, w, h, p, gap=0.25):
    """Three vertical probability bars (unrelated / same / division) inside the box (x, y, w, h)."""
    bw = (w - 2 * gap) / 3
    ax.add_line(plt.Line2D([x - 0.3, x + w + 0.3], [y, y], color=SOFT, lw=0.4, zorder=3))
    for k, (name, v) in enumerate(zip(CLS, p)):
        ax.add_patch(Rectangle((x + k * (bw + gap), y), bw, max(v, 0.02) * h, fc=CLS[name], ec='none', zorder=3))


def inset(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


# ---------------------------------------------------------------- real data of panel a / b
def frame(t):
    root = data.SCRIPTA
    image = tifffile.imread(root / WELL / f't{t:03d}.tif')
    mask = tifffile.imread(root / f'{WELL}_GT' / 'TRA' / f'man_track{t:03d}.tif')
    return image, mask


def window(a):
    y0, x0, s = BOX
    return a[y0:y0 + s, x0:x0 + s]


def contour_rgb(image, mask, colour):
    img = window(image).astype(float)
    lo, hi = np.percentile(img, [1, 99.5])
    rgb = np.repeat(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)[..., None], 3, axis=2)
    m = window(mask)
    edge = find_boundaries(m, mode='inner')
    for _ in range(2):
        edge = find_boundaries(np.where(edge, 0, m), mode='inner') | edge
    rgb[edge & (m > 0)] = np.array([int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return rgb


def centroids(mask):
    """{label: (x, y)} of the cells whose centroid lies in the window (window coordinates, pixels)."""
    y0, x0, s = BOX
    out = {}
    for r in regionprops(mask):
        cy, cx = r.centroid
        if y0 <= cy < y0 + s and x0 <= cx < x0 + s:
            out[r.label] = (cx - x0, cy - y0)
    return out


def all_positions():
    pos = {}
    for t in range(T - 4, T + 6):
        _, mask = frame(t)
        pos[t] = np.array([(r.centroid[1], r.centroid[0]) for r in regionprops(mask)])
    return list(pos), pos


def cell_crops(image, mask, labels):
    out = []
    for r in regionprops(mask):
        if r.label in labels:
            y0, x0, y1, x1 = r.bbox
            out.append(appearance_crop(image[y0:y1, x0:x1].astype(np.float32), r.image).astype(float))
    return out


# ---------------------------------------------------------------- panels
def panel_a(fig, ax):
    title(ax, 0.5, H - 3.5, 'a', 'Frame-pair graph')
    (im0, m0), (im1, m1) = frame(T), frame(T + 1)
    for k, (im, m, c, name) in enumerate(((im0, m0, C_T, 'frame $t$'), (im1, m1, C_N, 'frame $t{+}g$'))):
        a = inset(fig, 1.0 + k * 20.5, 46.5, 19.5, 19.5)
        a.imshow(contour_rgb(im, m, c), interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(c); s.set_linewidth(0.8)
        ax.text(1.0 + k * 20.5 + 9.75, 45.4, name, ha='center', va='top', fontsize=6, color=c)
    arrow(ax, (12.0, 41.5), (12.0, 37.5), color=SOFT, lw=0.6)
    ax.text(13.2, 39.5, 'velocities: GT (training),\nown links so far (inference)', fontsize=4.8,
            color=SOFT, va='center', style='italic', linespacing=1.1)

    # graph of the same window: nodes at the real centroids, candidate edges inside the real gating radius
    frames, pos = all_positions()
    r = gating_radius(frames, pos, T)[0]
    p0, p1 = centroids(m0), centroids(m1)
    s, X0, Y0, S = BOX[2], 1.5, 4.6, 38.5        # window -> canvas: x mm = X0 + px / s * S (y flipped)
    to = lambda p: (X0 + p[0] / s * S, Y0 + S * 0.85 - p[1] / s * S * 0.85)
    ax.add_patch(Rectangle((X0, Y0), S, S * 0.85, fc='white', ec=GRIDC, lw=0.4, zorder=1))
    l0, l1 = list(p0), list(p1)
    q0, q1 = np.array([p0[k] for k in l0]), np.array([p1[k] for k in l1])
    for q, offset in ((q0, 0), (q1, 1)):          # same-frame kNN context edges (k = 4 in the model; 2 drawn)
        _, idx = cKDTree(q).query(q, k=3)
        for i, js in enumerate(idx):
            for j in js[1:]:
                ax.add_line(plt.Line2D(*zip(to(q[i]), to(q[j])), color='#C9C9C9', lw=0.35, ls=(0, (1.5, 1.2)),
                                       zorder=2))
    focus = int(np.argmin(np.hypot(*(q0 - s / 2).T)))
    for i in range(len(q0)):
        d = np.hypot(*(q1 - q0[i]).T)
        for j in np.where(d < r)[0]:
            same = l0[i] == l1[j]
            hl = i == focus
            ax.add_line(plt.Line2D(*zip(to(q0[i]), to(q1[j])), color=INK if hl else '#8C8C8C',
                                   lw=0.75 if hl else 0.45, zorder=3, alpha=1 if (hl or same) else 0.8))
    c = to(q0[focus])
    ax.add_patch(Circle(c, r / s * S * 0.92, fc='none', ec=C_T, lw=0.5, ls=(0, (2, 1.2)), zorder=2))
    R, ang = r / s * S * 0.92, np.deg2rad(250)
    tip = (c[0] + R * np.cos(ang), c[1] + R * np.sin(ang))
    ax.add_line(plt.Line2D([c[0], tip[0]], [c[1], tip[1]], color=C_T, lw=0.5, zorder=4))
    ax.text(tip[0] - 0.6, tip[1] + 2.2, r'$r\sqrt{g}$', fontsize=6, color=C_T, zorder=6, ha='right', va='center',
            bbox=dict(fc='white', ec='none', pad=0.2))
    for q, col, fc in ((q0, C_T, C_T), (q1, C_N, 'white')):
        for p in q:
            ax.add_patch(Circle(to(p), 0.85, fc=fc, ec=col, lw=0.7, zorder=4))
    # legend
    ly = 1.3
    ax.add_patch(Circle((2.5, ly + 0.3), 0.7, fc=C_T, ec=C_T, lw=0.6)); ax.text(3.7, ly + 0.3, '$t$', va='center', fontsize=5.5)
    ax.add_patch(Circle((7.0, ly + 0.3), 0.7, fc='white', ec=C_N, lw=0.6)); ax.text(8.2, ly + 0.3, '$t{+}g$', va='center', fontsize=5.5)
    ax.add_line(plt.Line2D([13.5, 16.5], [ly + 0.3] * 2, color='#8C8C8C', lw=0.5))
    ax.text(17.1, ly + 0.3, 'candidate', va='center', fontsize=5.5)
    ax.add_line(plt.Line2D([27.5, 30.5], [ly + 0.3] * 2, color='#C9C9C9', lw=0.5, ls=(0, (1.5, 1.2))))
    ax.text(31.1, ly + 0.3, 'kNN context', va='center', fontsize=5.5)
    return r


GRIDC = '#D9D9D9'


def panel_b(fig, ax, x0):
    title(ax, x0, H - 3.5, 'b', 'Edge network (one model for gaps $g$ = 1-3)')
    # inputs (column 1)
    im, m = frame(T)
    labels = list(centroids(m))[:3]
    for k, c in enumerate(cell_crops(im, m, labels)):
        a = inset(fig, x0 + 1.0 + k * 4.6, 53.5 - k * 1.4, 6.5, 6.5)
        a.imshow(c, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(SOFT); s.set_linewidth(0.4)
    ax.text(x0 + 9.0, 62.6, 'masked crops 64$\\times$64', ha='center', fontsize=5.3, color=SOFT)
    box(ax, x0, 33.0, 18.0, 12.0, 'node features (8)\n$x, y$, area, intensity,\nframe flag, velocity', fs=5.0)
    box(ax, x0, 14.0, 18.0, 16.5, 'edge features (13)\n$\\Delta x, \\Delta y, d/r$, area ratio\ncompetition (4)\n'
        'velocity residual (4)\ngap $(g{-}1)/2$', fs=5.0)

    # appearance branch (row 1)
    xa = x0 + 20.5
    box(ax, xa, 52.0, 12.5, 10.0, 'AppearanceNet\n3 conv blocks\n$e_i\\in\\mathbb{S}^{31}$', fc=NET['app'], fs=5.2)
    arrow(ax, (x0 + 17.0, 56.0), (xa, 57.0))
    xs = xa + 14.5
    box(ax, xs, 52.0, 17.0, 10.0, '$s_{ij}=e_i^{\\top}e_j$\n+ softmax$_{j}(s_{ij}/\\tau)$\n+ softmax$_{i}(s_{ij}/\\tau)$',
        fc=NET['app'], fs=5.2)
    arrow(ax, (xa + 12.5, 57.0), (xs, 57.0))
    arrow(ax, (xs + 8.5, 52.0), (xs + 8.5, 47.5), color=SOFT)
    ax.text(xs + 9.1, 49.6, '3 app. features\n$\\rightarrow f_{ij}$ (16)', fontsize=4.8, color=SOFT, va='center')

    # encoder + message passing (row 2)
    xm = xa
    box(ax, xm, 33.0, 7.0, 12.0, 'node\nenc.\n$h_i^{0}$', fc=NET['mp'], fs=5.2)
    arrow(ax, (x0 + 18.0, 39.0), (xm, 39.0))
    arrow(ax, (xa + 3.5, 52.0), (xa + 3.5, 45.0), color=SOFT)                 # e_i joins the node input
    ax.text(xa + 4.1, 48.5, '$e_i$', fontsize=5.3, color=SOFT, va='center')
    xp = xm + 9.0
    box(ax, xp, 14.0, 15.5, 31.0, '', fc=NET['mp'])
    ax.text(xp + 7.75, 43.6, 'attention message\npassing $\\times$2', ha='center', va='top', fontsize=5.2, zorder=3)
    ax.text(xp + 7.75, 19.2, '4 heads; softmax\nover incoming\nmessages', ha='center', va='center', fontsize=4.8,
            color=SOFT, zorder=3)
    gx, gy = xp + 7.75, 31.0
    nodes = [(-4.6, 3.2), (-4.6, -2.6), (0.0, 0.3), (4.6, 3.2), (4.6, -2.6)]
    for a in (0, 1, 3, 4):
        arrow(ax, (gx + nodes[a][0], gy + nodes[a][1]), (gx + nodes[a][0] * 0.22, gy + 0.3 + nodes[a][1] * 0.22),
              color=C_T if a < 2 else C_N, lw=0.5, ms=3.5)
    for k, (dx, dy) in enumerate(nodes):
        col = C_T if k <= 2 else C_N
        ax.add_patch(Circle((gx + dx, gy + dy), 1.0 if k == 2 else 0.8, fc=col if k <= 2 else 'white', ec=col, lw=0.6,
                            zorder=5))
    ax.text(gx, gy - 6.0, '$\\sum_j \\alpha_{ij}\\, v_{ij}$', ha='center', fontsize=5.3, zorder=5)
    arrow(ax, (xm + 7.0, 39.0), (xp, 39.0))
    arrow(ax, (x0 + 18.0, 22.0), (xp, 22.0))
    ax.text(x0 + 18.4, 23.0, '$f_{ij}$', fontsize=5.3, color=SOFT)

    # edge head + output (horizontal probability bars)
    xh = xp + 17.5
    box(ax, xh, 31.0, 13.0, 12.0, 'edge MLP\n$[h_i, h_j, f_{ij}]$\n$\\rightarrow$ softmax', fc=NET['head'], fs=5.2)
    arrow(ax, (xp + 15.5, 37.0), (xh, 37.0))
    arrow(ax, (xp + 15.5, 22.0), (xh + 4.0, 31.0), color=SOFT, rad=0.3)
    arrow(ax, (xh + 6.5, 31.0), (xh + 6.5, 27.5))
    for k, (name, short, v) in enumerate(zip(CLS, ('unrelated', 'same', 'division'), (0.12, 0.83, 0.05))):
        y = 24.5 - k * 3.0
        ax.text(xh + 6.3, y, short, ha='right', va='center', fontsize=4.8, color=SOFT if k == 0 else CLS[name])
        ax.add_patch(Rectangle((xh + 6.8, y - 1.0), max(v, 0.03) * 6.2, 2.0, fc=CLS[name], ec='none', zorder=3))
    ax.add_line(plt.Line2D([xh + 6.8] * 2, [14.6, 26.2], color=SOFT, lw=0.4))
    ax.text(xh + 6.5, 13.6, '$p(c \\mid i \\rightarrow j)$', ha='center', va='top', fontsize=5.3)

    # training objective
    box(ax, x0, 1.8, xh + 13.0 - x0, 8.2, '', fc=NET['loss'], ec='#D9A27C')
    ax.text(x0 + 1.2, 7.9, 'Training', fontsize=5.6, weight='bold', va='center')
    ax.text(x0 + 10.0, 7.9, '$\\mathcal{L}=\\mathrm{CE}_{3\\text{-way}}$ (no class weights)'
            ' $+\\,\\lambda\\,\\mathrm{InfoNCE}(s_{ij}/\\tau)$', fontsize=5.0, va='center')
    ax.text(x0 + 1.2, 4.1, 'GT velocities, mirrored videos, unknown IDs masked, scripta + HeLa jointly',
            fontsize=4.8, va='center', color=SOFT)
    return xh + 13.0


def panel_c(ax, x0):
    title(ax, x0, H - 3.5, 'c', 'Bayesian averaging')
    probs = [[(0.10, 0.86, 0.04), (0.18, 0.78, 0.04), (0.07, 0.90, 0.03)],
             [(0.30, 0.66, 0.04), (0.14, 0.82, 0.04), (0.22, 0.74, 0.04)],
             [(0.12, 0.84, 0.04), (0.09, 0.88, 0.03), (0.25, 0.71, 0.04)]]
    cw, ch, gx, gy = 5.2, 5.4, 1.3, 2.0
    gx0, gy0 = x0 + 4.0, 38.0
    for v, view in enumerate(('id', '$\\leftrightarrow$', '$\\updownarrow$')):
        ax.text(gx0 + v * (cw + gx) + cw / 2, gy0 + 3 * (ch + gy) - 0.3, view, ha='center', fontsize=5.5, color=SOFT)
    for mi in range(3):
        ax.text(gx0 - 0.8, gy0 + (2 - mi) * (ch + gy) + ch / 2 - 0.3, f'$\\theta_{mi + 1}$', ha='right', va='center',
                fontsize=5.5, color=SOFT)
        for v in range(3):
            x, y = gx0 + v * (cw + gx), gy0 + (2 - mi) * (ch + gy)
            ax.add_patch(Rectangle((x - 0.4, y - 0.4), cw + 0.8, ch + 0.2, fc='white', ec=GRIDC, lw=0.35, zorder=2))
            bars(ax, x, y, cw, ch - 0.8, probs[mi][v], gap=0.3)
    ax.text(gx0 + 1.5 * cw + gx, gy0 + 3 * (ch + gy) + 2.6, '$M$ members $\\times$ $V$ views', ha='center', fontsize=5.3)
    arrow(ax, (gx0 + 1.5 * cw + gx, gy0 - 0.8), (gx0 + 1.5 * cw + gx, 33.5))
    mean = np.mean(np.array(probs).reshape(-1, 3), 0)
    bars(ax, gx0 + 4.0, 23.5, 9.0, 9.0, mean)
    ax.text(gx0 + 8.5, 21.8, '$\\bar p=\\frac{1}{MV}\\sum_{m,v} p_{m,v}$', ha='center', va='top', fontsize=5.3)
    ax.text(gx0 + 8.5, 16.3, '$\\mathrm{MI}=H(\\bar p)-\\overline{H(p_{m,v})}$', ha='center', va='top', fontsize=5.3)
    ax.text(gx0 + 8.5, 11.0, 'posterior under the\ntraining mix, used as is\n(no dataset prior)', ha='center',
            va='top', fontsize=4.8, color=SOFT, linespacing=1.15)
    return gx0 + 3 * cw + 2 * gx


def panel_d(ax, x0):
    title(ax, x0, H - 3.5, 'd', 'Two-round assignment')
    cols = [x0 + 4.0, x0 + 15.0, x0 + 26.0]
    ys = [61.0, 55.0, 49.0, 43.0]
    for c, name in zip(cols, ('$t$', '$t{+}1$', '$t{+}2$')):
        ax.text(c, 64.6, name, ha='center', fontsize=6, color=SOFT)
    P = lambda f, k: (cols[f], ys[k])
    for (a, b), (c, d), kind in (((0, 0), (1, 0), 'same'), ((0, 2), (1, 2), 'division'), ((0, 2), (1, 3), 'division'),
                                 ((1, 0), (2, 0), 'same'), ((1, 2), (2, 2), 'same'), ((1, 3), (2, 3), 'same')):
        arrow(ax, P(a, b), P(c, d), color=CLS[kind], lw=0.9, style_='-', z=3)
    arrow(ax, P(0, 1), P(2, 1), color=CLS['same'], lw=0.9, ls=(0, (2.2, 1.2)), style_='-', rad=-0.22, z=3)
    ax.add_patch(Circle(P(1, 1), 1.05, fc='none', ec=GRIDC, lw=0.6, ls=(0, (1.5, 1)), zorder=5))
    ax.text(cols[1], ys[1] - 2.2, 'missed', ha='center', va='top', fontsize=4.8, color=SOFT, style='italic')
    for f, k in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)):
        ax.add_patch(Circle(P(f, k), 1.05, fc=C_T if f == 0 else 'white', ec=C_T if f == 0 else C_N, lw=0.7, zorder=5))
    lx = x0 + 0.8
    ax.add_line(plt.Line2D([lx, lx + 3], [38.6] * 2, color=CLS['same'], lw=0.9))
    ax.text(lx + 3.6, 38.6, 'round 1, $g$=1', va='center', fontsize=5.0)
    ax.add_line(plt.Line2D([lx + 17, lx + 20], [38.6] * 2, color=CLS['same'], lw=0.9, ls=(0, (2.2, 1.2))))
    ax.text(lx + 20.6, 38.6, 'round 2, $g$=2', va='center', fontsize=5.0)
    ax.add_line(plt.Line2D([lx, lx + 3], [35.4] * 2, color=CLS['division'], lw=0.9))
    ax.text(lx + 3.6, 35.4, 'division (round 1)', va='center', fontsize=5.0)
    box(ax, x0, 17.5, 33.5, 14.8, '', fc=FILL)
    ax.text(x0 + 16.75, 30.2, 'one-to-one min-cost matching', ha='center', va='center', fontsize=5.3)
    ax.text(x0 + 16.75, 26.6, 'cost $-\\log\\bar p$; link only if $\\bar p_{\\mathrm{same}}>0.4$',
            ha='center', va='center', fontsize=5.1)
    ax.text(x0 + 16.75, 23.2, 'two daughters if $\\bar p_{\\mathrm{div}}\\geq 0.5$', ha='center', va='center', fontsize=5.1)
    ax.text(x0 + 16.75, 19.8, 'thresholds from LNK / BC costs, not tuned', ha='center', va='center', fontsize=4.8,
            color=SOFT)
    box(ax, x0, 1.8, 33.5, 13.4, '', fc=NET['head'], ec='#8FA9D6')
    ax.text(x0 + 1.2, 12.9, 'Output: tracks + confidence', fontsize=5.6, weight='bold', va='center')
    ax.text(x0 + 1.2, 9.6, 'per link: $\\bar p$, MI', fontsize=5.1, va='center')
    ax.text(x0 + 1.2, 6.6, 'per track: $\\prod_{\\ell}\\bar p_\\ell$, weakest link', fontsize=5.1, va='center')
    ax.text(x0 + 1.2, 3.6, '$\\mathbb{E}$[false] $=\\sum_\\ell(1-\\bar p_\\ell)$, $\\mathbb{E}$[missed]', fontsize=5.1,
            va='center')


def main():
    style.setup()
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    r = panel_a(fig, ax)
    xb = 44.0
    end_b = panel_b(fig, ax, xb)
    xc = end_b + 2.0
    end_c = panel_c(ax, xc)
    xd = W - 34.0
    panel_d(ax, xd)
    for x in (xb - 1.5, xc - 1.0, xd - 1.8):
        ax.add_line(plt.Line2D([x, x], [1.5, H - 1.5], color=GRIDC, lw=0.4))
    arrow(ax, (41.0, 28.0), (xb - 0.3, 28.0), lw=0.9, ms=6)                     # graph -> network
    arrow(ax, (end_b - 4.0, 43.0), (xc + 1.6, 44.2), lw=0.9, ms=6)              # edge posteriors -> averaging
    arrow(ax, (end_c - 1.0, 28.0), (xd - 0.3, 25.0), lw=0.9, ms=6)              # averaged posterior -> solver

    legend = f"""Figure 0. Method overview (schematic; image, detections and gating radius real).
a: scripta refine2, dev well {WELL}, frames {T} and {T + 1}, window rows {BOX[0]}-{BOX[0] + BOX[2]}, cols {BOX[1]}-{BOX[1] + BOX[2]};
   candidate edges = pairs closer than the adaptive gating radius r = {r:.1f} px (g = 1; r*sqrt(g) for gaps);
   dotted = same-frame nearest neighbours (the model uses k = 4, 2 are drawn). Highlighted: the candidates of one cell.
b: TrackerGNN3 (models/gnn_bayesian/model.py on models/gnn_tracking/model.py): AppearanceNet 3 conv blocks -> 32-d unit
   embedding; node features 8 + 32; edge features 13 geometric + 3 appearance; 2 attention layers, 4 heads, hidden 64;
   edge MLP -> softmax over unrelated / same / division. Loss: cross-entropy without class weights + contrastive
   (InfoNCE, tau = 0.1) appearance loss on same-cell edges.
c: probability bars are schematic. Final configuration: 3 seeds x 3 views (identity, horizontal, vertical mirror),
   mean softmax; mutual information as disagreement. Prior 'none'.
d: round 1 adjacent frames with divisions (p_div >= 0.5), continuation threshold 0.4 (from the AOGM edge costs of LNK);
   round 2 bridges one missed frame between track ends and starts (threshold 0.4, no divisions).
"""
    style.save(fig, 'fig0_overview', legend)


if __name__ == '__main__':
    main()
