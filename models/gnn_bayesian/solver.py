"""
solver.py
=========
Round-1 assignment with divisions decided on the network's division probability (numpy only).

A frame-t cell is a division candidate if its two most likely daughters both have p_division >= div_thresh and
that pair is more likely than its best continuation (p_div(a) * p_div(b) > max p_same). Such a cell gets two rows
in the assignment (as gnn_tracking.solver.match_division does), both scored with -log p_division and accepted
above div_thresh; every other cell has one row scored with -log p_same and accepted above link_thresh. The matrix
is gnn_tracking.solver.match's, with a per-row threshold.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6


def assign(cost, row, col, n_rows, n_cols, row_thresh, col_thresh):
    """Min-cost assignment with 'stay unmatched' options: edge e is taken only if cost[e] < -log(row_thresh[row])
    when the column's half is -log(col_thresh) / 2. -> [(row, col), ...]."""
    u_col = -np.log(col_thresh) / 2.0
    u_row = -np.log(np.asarray(row_thresh, float)) - u_col
    A = np.full((n_rows + n_cols, n_cols + n_rows), BIG)
    A[row, col] = cost
    A[np.arange(n_rows), n_cols + np.arange(n_rows)] = u_row
    A[n_rows + np.arange(n_cols), np.arange(n_cols)] = u_col
    A[n_rows:, n_cols:] = 0.0
    r, c = linear_sum_assignment(A)
    return [(int(i), int(j)) for i, j in zip(r, c) if i < n_rows and j < n_cols and A[i, j] < BIG]


def dividing_rows(p_same, p_div, row, n_rows, div_thresh):
    out = []
    for r in range(n_rows):
        e = row == r
        if e.sum() < 2:
            continue
        top = np.sort(p_div[e])[::-1]
        if top[1] >= div_thresh and top[0] * top[1] > p_same[e].max():
            out.append(r)
    return out


def match_with_divisions(p_same, p_div, row, col, n_rows, n_cols, link_thresh, div_thresh):
    """-> links [(row, col), ...]; a dividing row appears twice."""
    row, col = np.asarray(row), np.asarray(col)
    p_same, p_div = np.clip(p_same, 1e-9, 1.0), np.clip(p_div, 1e-9, 1.0)
    div = dividing_rows(p_same, p_div, row, n_rows, div_thresh)
    is_div = np.isin(row, div)
    cost = np.where(is_div, -np.log(p_div), -np.log(p_same))
    copy_of = {r: n_rows + k for k, r in enumerate(div)}
    row2 = np.concatenate([row, np.array([copy_of[int(r)] for r in row[is_div]], dtype=int)])
    col2 = np.concatenate([col, col[is_div]])
    cost2 = np.concatenate([cost, cost[is_div]])
    thresh = np.full(n_rows + len(div), link_thresh, float)
    thresh[div] = div_thresh
    thresh[n_rows:] = div_thresh
    links = assign(cost2, row2, col2, n_rows + len(div), n_cols, thresh, link_thresh)
    owner = {v: k for k, v in copy_of.items()}
    return [(owner.get(r, r), c) for r, c in links]
