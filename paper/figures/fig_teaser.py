"""Figure 1 (full width, page 1): the argument of the paper in one figure.
Three rows, one per decision a tracker takes: current learned trackers leave it to a user setting (left, with the
failure it causes), ProGCT takes it on one calibrated posterior (right, with the evidence). A strip at the bottom shows
how ProGCT computes the posteriors; the numbers in circles tie each module to its row.
The cells of rows 1 and 2 left and of row 1 right are schematic; every number is read from development/final_results_v2
(data.py); the DNN cells of row 2 are real (val19, frames 15-16); the curves of row 3 are real (scripta development
wells, out of fold, TTA)."""
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from skimage.measure import regionprops

import data
import style

W, H = style.FULL / style.MM, 74.0              # canvas in mm
INK, SOFT, MUTED = '#262626', '#7F7F7F', '#A6A6A6'
BLUE, PURPLE, TEAL, RED, GREY = '#3B6FB6', '#8E6BBF', '#2A9D8F', '#C8553D', '#BDBDBD'
OLD_BG, NEW_BG = '#F3F3F3', '#EAF1FA'
CLS = [('unrelated', GREY), ('same', BLUE), ('division', RED)]
DIVISION = ('val19', 15, 34)                    # DNN video, frame, mother label: a division ProGCT found
X_LAB, X_OLD, X_MID, X_NEW, X_END = 0.6, 30.0, 84.5, 90.5, W - 0.4
X_OLD_TEXT, X_NEW_TEXT = X_OLD + 23.0, X_NEW + 30.0
ROWS = [(48.5, 62.5), (33.0, 47.0), (17.5, 31.5)]  # (bottom, top) of each row, mm
FS = 5.7                                        # body text


def rbox(ax, x, y, w, h, fc, ec='none', lw=0.5, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                zorder=z))


def arrow(ax, a, b, color=INK, lw=0.7, rad=0.0, ms=5, z=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=z))


def number(ax, x, y, n, r=1.5, fs=6):
    ax.add_patch(Circle((x, y), r, fc=INK, ec='none', zorder=6))
    ax.text(x, y - 0.05, str(n), ha='center', va='center', fontsize=fs, color='white', weight='bold', zorder=7)


def cell(ax, x, y, fc, ec=None, ghost=False, w=3.4, h=2.6, z=5):
    if ghost:
        ax.add_patch(Ellipse((x, y), w, h, fc='white', ec=MUTED, lw=0.6, ls=(0, (1.6, 1.0)), zorder=z))
    else:
        ax.add_patch(Ellipse((x, y), w, h, fc=fc, ec=ec or fc, lw=0.6, zorder=z))


def lines(ax, x, y, items, dy=2.55):
    """items: [(text, colour, bold)] from the top line at y downwards."""
    for k, (text, colour, bold) in enumerate(items):
        ax.text(x, y - k * dy, text, ha='left', va='top', fontsize=FS, color=colour,
                weight='bold' if bold else 'normal')


def mark(ax, x, y, ok):
    ax.text(x, y, '✓' if ok else '✗', ha='center', va='center', fontsize=7.5, color=BLUE if ok else RED,
            weight='bold', zorder=7, family='DejaVu Sans')


def real_crop(video, t, label, half=26):
    """(crop at t, crop at t+1) around the cell `label` of frame t, normalised together."""
    mask = tifffile.imread(data.DNN / f'{video}_GT' / 'TRA' / f'man_track{t:03d}.tif')
    cy, cx = [r.centroid for r in regionprops(mask) if r.label == label][0]
    y0, x0 = int(cy) - half, int(cx) - half
    out = [tifffile.imread(data.DNN / video / f't{k:03d}.tif').astype(float)[y0:y0 + 2 * half, x0:x0 + 2 * half]
           for k in (t, t + 1)]
    lo, hi = np.percentile(np.concatenate([o.ravel() for o in out]), [1, 99.7])
    return [np.clip((o - lo) / (hi - lo + 1e-9), 0, 1) for o in out]


def division_probs(video, t, mother):
    """Mean over the two daughter edges of (p_unrelated, p_same, p_division) in ProGCT's record."""
    z = np.load(data.track_folder('ours', video, 'dnn') / 'edges.npz')
    sel = (z['round'] == 1) & (z['src_frame'] == t) & (z['src_label'] == mother) & z['chosen'].astype(bool)
    p, d = z['p'][sel].mean(), z['p_div'][sel].mean()
    return np.array([max(1 - p - d, 0), p, d])


def inset(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def facts():
    """Every number shown in the figure, from final_results_v2."""
    dev = [w for w in data.all_wells() if w not in data.TEST_WELLS]
    gaps = data.gap_counts().groupby('method').sum(numeric_only=True)
    cv_all = data.answers('scripta_cv_oof_tta')[0]
    cv = cv_all['coverage']
    nogap = data.paired(data.scripta_wells('ours_nodiv', dev).LNK, data.scripta_wells('nogap', dev).LNK)
    return dict(
        gap_found=1 - gaps.loc['ours', 'missed_g2'] / gaps.loc['ours', 'gt_g2'],
        dt3_false=int(gaps.loc['ta_ft_dt3', 'false_g2'] + gaps.loc['ta_ft_dt3', 'false_g3+']),
        nogap_wells=(nogap['a_higher'], nogap['n']), nogap_p=nogap['p'],
        ta_false_div=data.scripta_divisions('ta_ft_div', data.all_wells())['divisions'],
        ours_false_div=data.scripta_divisions('ours', data.all_wells())['divisions'],
        bc=data.dnn_wells('ours')['BC(0)'].mean(), bc_ta=data.dnn_wells('ta_ft_div')['BC(0)'].mean(),
        ece=cv_all['calibration']['round1_decision']['ece'],
        cov_false=cv['false_links']['mutual_info']['at_percent']['5'],
        cov_missed=cv['missed_links']['mutual_info']['at_percent']['5'])


def header(ax):
    ax.text(W / 2, H - 2.6, 'Every tracking decision should rest on one calibrated probability, '
            'not on a setting chosen for the dataset', ha='center', va='center', fontsize=7.5, weight='bold')
    y = 65.6
    ax.text(X_LAB, y, 'Decision', fontsize=6.5, weight='bold', va='center')
    ax.text(X_OLD + 1.5, y + 0.9, 'Current learned trackers', fontsize=6.5, weight='bold', va='center', color='#595959')
    ax.text(X_OLD + 1.5, y - 1.7, 'a user setting decides', fontsize=5.8, va='center', color=SOFT, style='italic')
    ax.text(X_NEW + 1.5, y + 0.9, 'ProGCT', fontsize=6.5, weight='bold', va='center', color=BLUE)
    ax.text(X_NEW + 1.5, y - 1.7, 'one calibrated posterior decides', fontsize=5.8, va='center', color=BLUE,
            style='italic')


def row_frame(ax, k, question, setting):
    lo, hi = ROWS[k]
    rbox(ax, X_OLD, lo, X_MID - X_OLD, hi - lo, OLD_BG)
    rbox(ax, X_NEW, lo, X_END - X_NEW, hi - lo, NEW_BG)
    arrow(ax, (X_MID + 0.8, (lo + hi) / 2), (X_NEW - 0.8, (lo + hi) / 2), color=BLUE, lw=1.1, ms=7)
    number(ax, X_LAB + 1.6, hi - 2.6, k + 1)
    ax.text(X_LAB + 4.0, hi - 1.2, question, fontsize=6.3, weight='bold', va='top', linespacing=1.15)
    ax.text(X_LAB + 4.0, lo + 1.0, setting, fontsize=5.6, va='bottom', color=SOFT, style='italic', linespacing=1.15)
    return lo, hi


def row_gaps(ax, f):
    lo, hi = ROWS[0]
    xs = [X_OLD + 4.5, X_OLD + 11.0, X_OLD + 17.5]
    for x, text in zip(xs, ('$t$', '$t{+}1$', '$t{+}2$')):
        ax.text(x, hi - 0.9, text, ha='center', va='top', fontsize=5.3, color=SOFT)
    y = hi - 4.6                    # adjacent-frame linking: the track breaks at the missed detection
    cell(ax, xs[0], y, BLUE); cell(ax, xs[1], y, None, ghost=True); cell(ax, xs[2], y, PURPLE)
    ax.add_line(plt.Line2D([xs[0] + 1.7, xs[1] - 1.7], [y, y], color=BLUE, lw=0.9, zorder=4))
    mark(ax, xs[2] + 3.3, y, False)
    y = lo + 3.6                    # window linking: skips a cell that is present
    for x in xs:
        cell(ax, x, y, BLUE if x == xs[0] else '#C9D6EA', ec=BLUE if x == xs[0] else '#9DB3D6')
    arrow(ax, (xs[0] + 1.0, y + 1.0), (xs[2] - 1.0, y + 1.0), color=RED, lw=0.9, rad=-0.38, ms=5)
    mark(ax, xs[2] + 3.3, y, False)
    lines(ax, X_OLD_TEXT, hi - 1.6, [('Adjacent frames only:', INK, True),
                                     ('a missed cell breaks the track', INK, False)])
    lines(ax, X_OLD_TEXT, lo + 7.3, [('Temporal window ($\\Delta t$=3):', INK, True),
                                     ('links past cells that are visible', INK, False),
                                     (f'{f["dt3_false"]:,} false gap links', SOFT, False)])
    xs = [X_NEW + 5.0, X_NEW + 12.5, X_NEW + 20.0]   # ProGCT: the frame-gap graph (t, t+2) rejoins the track
    y = lo + 5.0
    cell(ax, xs[0], y, BLUE); cell(ax, xs[1], y, None, ghost=True); cell(ax, xs[2], y, BLUE)
    arrow(ax, (xs[0] + 1.3, y + 1.3), (xs[2] - 1.3, y + 1.3), color=BLUE, lw=1.0, rad=-0.45, ms=5)
    ax.text(xs[1], y + 5.2, '$p(\\mathrm{same}\\,|\\,g{=}2) > \\tau$', ha='center', va='bottom', fontsize=5.5,
            color=BLUE)
    for x, text in zip(xs, ('$t$', '$t{+}1$, missed', '$t{+}2$')):
        ax.text(x, y - 1.8, text, ha='center', va='top', fontsize=5.3, color=SOFT)
    mark(ax, xs[2] + 3.3, y, True)
    a, n = f['nogap_wells']
    lines(ax, X_NEW_TEXT, hi - 1.6, [('Frame-gap graph: scores $(t, t{+}g)$, $g$ as input,', INK, True),
                                     ('so a cell is rejoined across a missed frame', INK, False),
                                     (f'rejoins {f["gap_found"]:.0%} of the links across a missed frame;', BLUE, True),
                                     (f'without it, LNK falls in {a}/{n} wells', BLUE, False)])


def row_divisions(fig, ax, f):
    lo, hi = ROWS[1]
    sx, x0 = X_OLD + 1.8, X_OLD + 11.6
    for on, y in ((True, hi - 4.4), (False, lo + 4.4)):
        rbox(ax, sx, y - 1.0, 4.2, 2.0, RED if on else MUTED, r=1.0, z=3)
        ax.add_patch(Circle((sx + (3.2 if on else 1.0), y), 0.75, fc='white', ec='none', zorder=4))
        ax.text(sx + 4.8, y, 'ON' if on else 'OFF', fontsize=5.0, va='center', color=RED if on else SOFT,
                weight='bold')
    y = hi - 4.4                    # ON, data without mitoses: a false division
    cell(ax, x0, y, BLUE, w=2.8, h=2.2)
    for dy in (1.5, -1.5):
        ax.add_line(plt.Line2D([x0 + 1.4, x0 + 4.4], [y, y + dy], color=RED, lw=0.8, zorder=4))
        cell(ax, x0 + 5.5, y + dy, BLUE, w=2.3, h=1.8)
    mark(ax, x0 + 9.0, y, False)
    y = lo + 4.4                    # OFF, dividing data: one daughter loses its mother
    cell(ax, x0, y, TEAL, w=2.8, h=2.2)
    ax.add_line(plt.Line2D([x0 + 1.4, x0 + 4.4], [y, y + 1.4], color=TEAL, lw=0.8, zorder=4))
    cell(ax, x0 + 5.5, y + 1.5, TEAL, w=2.3, h=1.8)
    cell(ax, x0 + 5.5, y - 1.5, PURPLE, w=2.3, h=1.8)
    mark(ax, x0 + 9.0, y, False)
    lines(ax, X_OLD_TEXT, hi - 1.6, [('Switch ON, no mitoses:', INK, True),
                                     (f'{f["ta_false_div"]} false divisions', INK, False)])
    lines(ax, X_OLD_TEXT, lo + 6.8, [('Switch OFF, mitoses:', INK, True),
                                     ('lineages are lost', INK, False)])
    video, t, mother = DIVISION     # ProGCT: a real division with its three-class probabilities
    for k, (img, name) in enumerate(zip(real_crop(video, t, mother), ('$t$', '$t{+}1$'))):
        a = inset(fig, X_NEW + 1.8 + k * 9.4, lo + 1.3, 8.8, 8.8)
        a.imshow(img, cmap='gray', interpolation='bilinear')
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(SOFT); s.set_linewidth(0.4)
        ax.text(X_NEW + 6.2 + k * 9.4, lo + 10.5, name, ha='center', va='bottom', fontsize=5.3, color=SOFT)
    probs = division_probs(video, t, mother)
    bx, by, bw, bh = X_NEW + 21.6, lo + 2.6, 1.4, 6.6
    for k, ((name, colour), v) in enumerate(zip(CLS, probs)):
        ax.add_patch(Rectangle((bx + k * (bw + 0.5), by), bw, max(v, 0.03) * bh, fc=colour, ec='none', zorder=4))
        ax.text(bx + k * (bw + 0.5) + bw / 2, by - 0.4, name[0].upper(), ha='center', va='top', fontsize=4.8,
                color=SOFT if name == 'unrelated' else colour)
    ax.add_line(plt.Line2D([bx - 0.4, bx + 3 * bw + 1.4], [by, by], color=SOFT, lw=0.4, zorder=4))
    ax.text(bx + 2.8, by + bh + 0.5, f'$d$ = {probs[2]:.2f}', ha='center', va='bottom', fontsize=5.3, color=RED)
    lines(ax, X_NEW_TEXT, hi - 1.6, [('Three-class edge: unrelated / same / division,', INK, True),
                                     ('one model for both kinds of data, no dataset flag', INK, False),
                                     (f'{f["ours_false_div"]} false divisions without mitoses,', BLUE, True),
                                     (f'BC {f["bc"]:.3f} with mitoses (Trackastra {f["bc_ta"]:.3f})', BLUE, False)])


def row_confidence(fig, ax, f):
    lo, hi = ROWS[2]
    gx, gy, gw = X_OLD + 2.5, hi - 5.6, 16.5      # a score with no meaning and a threshold to tune
    ax.add_patch(Rectangle((gx, gy - 0.8), gw, 1.6, fc='white', ec=MUTED, lw=0.5, zorder=3))
    ax.add_patch(Rectangle((gx, gy - 0.8), gw * 0.7, 1.6, fc='#C9C9C9', ec='none', zorder=3))
    ax.text(gx + gw * 0.7, gy + 1.4, 'score 0.7 = ?', ha='center', va='bottom', fontsize=5.5, color=INK)
    for x in (0.35, 0.5, 0.62):
        ax.add_line(plt.Line2D([gx + gw * x] * 2, [gy - 2.9, gy - 1.0], color=RED, lw=0.6, ls=(0, (1.5, 1.0)),
                               zorder=4))
    ax.text(gx + gw * 0.5, gy - 3.3, '$\\tau$ = ? (ad hoc or tuned)', ha='center', va='top', fontsize=5.5, color=RED)
    lines(ax, X_OLD_TEXT, hi - 1.6, [('Scores are not probabilities:', INK, True),
                                     ('class weights shift them; the', INK, False),
                                     ('threshold is ad hoc or tuned,', INK, False),
                                     ('calibration fitted on labels', INK, False)])
    metrics, reliability, coverage = data.answers('scripta_cv_oof_tta')   # ProGCT: real curves
    r = reliability[reliability.population == 'round1_decision']
    c = coverage[(coverage.population == 'missed_links') & (coverage.score == 'mutual_info')]
    c = c[c.checked_percent <= 20]
    for k, (title, xl, xlim, ylim, curve, diag) in enumerate((
            ('calibration', 'predicted $p$', (0, 1), (0, 1), (r.mean_p, r.frac_pos), ([0, 1], [0, 1])),
            ('missed candidates', '% left-out checked', (0, 20), (0, 100),
             (c.checked_percent, c.errors_found_percent),
             ([0, 20], [0, 20])))):
        x = X_NEW + 3.4 + k * 13.6
        a = inset(fig, x, lo + 4.5, 8.0, 5.8)
        a.plot(*diag, color=MUTED, lw=0.5, ls=':')
        a.plot(*curve, '-', color=BLUE, lw=0.9)
        a.set(xlim=xlim, ylim=ylim, xticks=[], yticks=list(ylim))
        a.tick_params(labelsize=4.6, length=1.2, pad=0.6)
        for v, ha in zip(xlim, ('left', 'right')):
            ax.text(x + (0 if ha == 'left' else 8.0), lo + 4.1, f'{v:g}', ha=ha, va='top', fontsize=4.6, color=INK)
        ax.text(x + 4.0, lo + 2.3, xl, ha='center', va='top', fontsize=4.6, color=SOFT)
        ax.text(x + 4.0, lo + 10.7, title, ha='center', va='bottom', fontsize=5.0, color=SOFT)
    lines(ax, X_NEW_TEXT, hi - 1.6, [('No class weights: calibrated on unseen wells', INK, True),
                                     (f'(ECE {f["ece"]:.3f}); $\\tau$ = 0.4 from the LNK costs, not tuned', INK, False),
                                     (f'5% most uncertain: {f["cov_false"]:.0f}% of false links made,', BLUE, True),
                                     (f'{f["cov_missed"]:.0f}% of missed candidates; counts within 10%', BLUE, False)])


def pipeline(ax):
    lo, hi = 1.0, 14.4
    ax.text(X_LAB, hi - 1.4, 'How ProGCT\ncomputes it', fontsize=6.3, weight='bold', va='top', linespacing=1.15)
    ax.text(X_LAB, lo + 0.4, 'detections in,\ntracks and\nconfidence out', fontsize=5.3, color=SOFT, va='bottom',
            style='italic', linespacing=1.1)
    steps = [([1], 'Frame-gap graph', 'detections of $t$, $t{+}g$;\ngate $r\\sqrt{g}$, $g$ as input'),
             ([2], 'Three-class GNN', 'no class weights; Scripta +\nRAW264.7 (DNN), no flag'),
             ([3], 'Ensemble', '3 networks $\\times$ 3 mirror\nviews; mean $p$, MI'),
             ([1, 2, 3], 'Assignment', '$\\tau$ = 0.4, $\\tau_D$ = 0.5; then\ncross-frame check, $g$ = 2'),
             ([3], 'Tracks + confidence', '$p$, flag per link and track;\nexpected errors per video')]
    gap = 2.6
    w = (X_END - X_OLD - gap * (len(steps) - 1)) / len(steps)
    for k, (nums, name, text) in enumerate(steps):
        x = X_OLD + k * (w + gap)
        rbox(ax, x, lo, w, hi - lo, 'white', ec='#9DB3D6', lw=0.6)
        ax.text(x + 1.4, hi - 2.6, name, fontsize=6.0, weight='bold', va='center', color=INK)
        ax.text(x + 1.4, lo + 4.3, text, fontsize=5.3, va='center', color='#404040', linespacing=1.25)
        for j, n in enumerate(nums):
            number(ax, x + w - 1.8 - (len(nums) - 1 - j) * 3.0, hi - 2.6, n, r=1.2, fs=5.2)
        if k:
            arrow(ax, (x - gap + 0.3, (lo + hi) / 2), (x - 0.3, (lo + hi) / 2), color=BLUE, lw=0.8, ms=5)


def main():
    style.setup()
    f = facts()
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    header(ax)
    row_frame(ax, 0, 'Link across a\nmissed cell?', 'setting:\ntemporal window')
    row_frame(ax, 1, 'Did the cell\ndivide?', 'setting:\ndivision switch')
    row_frame(ax, 2, 'Can this link\nbe trusted?', 'setting: threshold,\ncalibration fit')
    row_gaps(ax, f)
    row_divisions(fig, ax, f)
    row_confidence(fig, ax, f)
    ax.add_line(plt.Line2D([X_LAB, X_END], [16.1, 16.1], color='#D9D9D9', lw=0.5))
    pipeline(ax)
    a, n = f['nogap_wells']
    legend = f"""Figure 1 (teaser). Cells of rows 1-2 left and row 1 right: schematic. Numbers (final_results_v2):
row 1: false gap links (g >= 2) of fine-tuned Trackastra with delta_t = 3, 15 scripta wells: {f['dt3_false']};
  ProGCT rejoins {f['gap_found']:.1%} of the 1060 GT links across one missed frame (15 wells);
  without the cross-frame check LNK is lower in {a}/{n} development wells (Wilcoxon p {f['nogap_p']:.1e}).
row 2: false divisions over 15 scripta wells, Trackastra ft greedy {f['ta_false_div']}, ProGCT {f['ours_false_div']};
  DNN 13 videos BC(0) ProGCT {f['bc']:.3f}, Trackastra ft {f['bc_ta']:.3f}.
  Cells: DNN {DIVISION[0]} frames {DIVISION[1]}-{DIVISION[1] + 1}, mother label {DIVISION[2]}; bars = ProGCT's
  mean (p_unrelated, p_same, p_division) of the two daughter edges.
row 3: scripta 12 development wells out of fold (TTA): reliability of the undecided edges (0.02 < p < 0.98), ECE
  {f['ece']:.3f}, and the coverage of the missed candidates (candidate edges the solver left out that join the same
  cell; links without a candidate edge are not counted) ranked by MI among the candidates left out; 5% checked:
  false links among the links made {f['cov_false']:.1f}%, missed candidates among those left out {f['cov_missed']:.1f}%.
"""
    style.save(fig, 'fig0_teaser', legend)


if __name__ == '__main__':
    main()
