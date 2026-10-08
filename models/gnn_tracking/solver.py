"""
solver.py
=========
Step 3: edge probabilities -> a one-to-one matching in which every cell may also stay unmatched.

The model scores each candidate edge on its own, so two frame-t cells can both be 90% sure about the same target.
The Hungarian algorithm picks the set of links with the lowest total cost, cost = -log(p), such that every cell is
used at most once.

A plain n_a x n_b cost matrix would force min(n_a, n_b) links: a cell whose true partner is missing would be
handed someone else's partner, and the damage spreads. So the matrix is extended (n_a = 2 rows, n_b = 3 columns):

                  b0    b1    b2  |  a0 alone  a1 alone
        a0      [ c00   c01   c02 |    u         BIG   ]
        a1      [ c10   c11   c12 |    BIG       u     ]
        --------------------------+---------------------
        b0 alone[ u     BIG   BIG |    0         0     ]
        b1 alone[ BIG   u     BIG |    0         0     ]
        b2 alone[ BIG   BIG   u   |    0         0     ]

u = -log(thresh) / 2 per cell. Linking a-b costs -log(p); leaving both alone costs 2u = -log(thresh). So a link is
made only if p > thresh and it is part of the globally cheapest arrangement. With thresh = 0.5, u = 0.35.
Pairs that are not candidate edges cost BIG and are never returned.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6


def match(prob, row, col, n_rows, n_cols, thresh):
    """prob[e] = probability of the edge row[e] -> col[e]. Returns the chosen links [(row, col), ...]."""
    u = -np.log(thresh) / 2.0
    A = np.full((n_rows + n_cols, n_cols + n_rows), BIG)
    A[row, col] = -np.log(np.clip(prob, 1e-9, 1.0))
    A[np.arange(n_rows), n_cols + np.arange(n_rows)] = u
    A[n_rows + np.arange(n_cols), np.arange(n_cols)] = u
    A[n_rows:, n_cols:] = 0.0
    r, c = linear_sum_assignment(A)
    return [(int(i), int(j)) for i, j in zip(r, c) if i < n_rows and j < n_cols and A[i, j] < BIG]


def match_division(prob, row, col, n_rows, n_cols, thresh, div_thresh=0.5):
    """Like match(), but a row may take two columns (a cell dividing into two daughters).

    A row with at least two candidates of probability >= div_thresh gets a second copy of itself in the matrix (same
    costs). The solver may then give the original and the copy one column each. Every row, copies included, keeps its
    own 'stay unmatched' option, so the second slot is only used if that second link is itself better than no link.
    Same logic as gnn_tracking/assignment_division.hungarian_match_division.
    Returns the links [(row, col), ...]; a row that divides appears twice."""
    row, col = np.asarray(row), np.asarray(col)
    strong = np.bincount(row[prob >= div_thresh], minlength=n_rows)
    twice = np.where(strong >= 2)[0]                                  # rows that may divide
    copy_of = {int(r): n_rows + k for k, r in enumerate(twice)}
    sel = np.isin(row, twice)
    row2 = np.concatenate([row, np.array([copy_of[int(r)] for r in row[sel]], dtype=int)])
    links = match(np.concatenate([prob, prob[sel]]), row2, np.concatenate([col, col[sel]]), n_rows + len(twice), n_cols, thresh)
    owner = {v: k for k, v in copy_of.items()}
    return [(owner.get(r, r), c) for r, c in links]


if __name__ == "__main__":
    # a0 and a1 both prefer b0; a1's only other option (b1) is weak.
    row, col, prob = np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([0.90, 0.95, 0.30])
    print("links:", match(prob, row, col, 2, 2, 0.5), " <- a1 takes b0; a0 stays unmatched instead of being forced onto b1")
    # a0 is close to both b0 and b1 (a division); a1 continues to b2.
    row, col, prob = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 2]), np.array([0.90, 0.85, 0.20, 0.95])
    print("division:", sorted(match_division(prob, row, col, 2, 3, 0.5)), " <- a0 gets b0 and b1, a1 gets b2")
