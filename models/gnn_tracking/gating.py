"""
gating.py
=========
The adaptive gating radius of graph.py, as a function of the detection positions alone (no ground truth, no torch).

graph.Video.radius calls gating_radius; audit_gating.py calls it too, so the audit measures exactly the radius the
tracker uses. Kept free of torch / torch_geometric so the audit runs without the training environment.
"""
import numpy as np
from scipy.spatial import cKDTree

# --- adaptive gating radius (no ground truth needed; constants calibrated in September on the dnn benchmark) ---
SAFETY = 2.1             # r = SAFETY x the 99% quantile of the frame-to-frame step ...
R_MIN = 10.0             # ... at least this many pixels ...
SPACING_CAP = 1.5        # ... and at most SPACING_CAP x the median distance between neighbouring cells
WINDOW = 8               # frame pairs around t that the estimate uses
R_FALLBACK = 40.0        # fewer than 5 usable cells in the window


def gating_radius(frames, pos, t):
    """Gating radius for links leaving frame t, estimated from positions alone.

    frames: sorted frame numbers of the video; pos[f]: [n_f, 2] (x, y) of frame f's cells.

    Step size: in each consecutive frame pair of a window around t, take the cells that are each other's
    nearest neighbour (almost always the same cell) and collect their distances; q99 = the 99% quantile.
    Density: the median distance from a cell to its nearest neighbour in the same frame.
    r = clip(2.1 x q99, at least 10 px, at most 1.5 x spacing). The cap keeps the candidate graph sparse in
    dense wells: beyond 1.5 neighbour spacings a link would have to jump over other cells.

    Returns (r, details); details = dict(step_q99, n_steps, spacing, bound) where bound names the term that set r:
    'q99' (SAFETY x q99), 'floor' (R_MIN), 'cap' (SPACING_CAP x spacing) or 'fallback' (R_FALLBACK)."""
    i = frames.index(t)
    win = frames[max(0, i - WINDOW // 2): min(len(frames), i + WINDOW // 2 + 2)]
    steps, spacing = [], []
    for a, b in zip(win[:-1], win[1:]):
        pa, pb = pos[a], pos[b]
        d_ab, nn_ab = cKDTree(pb).query(pa, k=1)          # for every cell of a: nearest cell of b
        _, nn_ba = cKDTree(pa).query(pb, k=1)             # and the other way round
        steps += [d_ab[k] for k, j in enumerate(nn_ab) if nn_ba[j] == k]
    for a in win:
        if len(pos[a]) >= 2:
            d, _ = cKDTree(pos[a]).query(pos[a], k=2)     # k=2: [itself, nearest other cell]
            spacing.append(d[:, 1])
    if len(steps) < 5:
        return R_FALLBACK, dict(step_q99=None, n_steps=len(steps), spacing=None, bound='fallback')
    q99 = float(np.quantile(steps, 0.99))
    r = max(SAFETY * q99, R_MIN)
    spacing = float(np.median(np.concatenate(spacing))) if spacing else float("inf")
    bound = 'cap' if SPACING_CAP * spacing < r else ('q99' if SAFETY * q99 >= R_MIN else 'floor')
    return float(min(r, SPACING_CAP * spacing)), dict(step_q99=q99, n_steps=len(steps), spacing=spacing, bound=bound)
