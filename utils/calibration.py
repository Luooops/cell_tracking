"""
calibration.py
==============
Is a predicted probability p what it says, and do uncertain predictions point at the errors?

Calibration (y in {0, 1}, p in [0, 1]):
    ECE          sum over bins of (bin share) x |mean p - fraction of positives|; equal-mass bins (each bin holds
                 the same number of predictions; robust when most p sit near 0 or 1) and, for reference, equal-width
    NLL          mean -log p(y)          Brier   mean (p - y)^2          (both need no binning)
    reliability  per equal-mass bin: n, p range, mean p, fraction of positives (the reliability curve)

Error coverage ("check the k% most uncertain items first: what share of all errors do we find?"):
    the items are ranked by an uncertainty score (largest first, ties in random order); the curve gives the share of
    errors found in the top k% for k = 0..100. A random check finds k% of the errors (the diagonal); area = 0.5.

Uncertainty scores of an ensemble with member probabilities q [n, M]:
    closeness    min(p, 1 - p) of the mean p (0.5 = most uncertain)
    mutual_info  H(mean q) - mean H(q_m): the members' disagreement (epistemic part of the total entropy, nats)
"""
import numpy as np

EPS = 1e-7
COVERAGE_POINTS = np.linspace(0, 100, 101)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -50, 50)))


def entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def mutual_information(q):
    """q [n, M] member probabilities -> [n] H(mean) - mean H(member), in nats (>= 0)."""
    q = np.asarray(q, dtype=np.float64)
    return np.maximum(entropy(q.mean(1)) - entropy(q).mean(1), 0.0)


def reliability(p, y, bins=15):
    """Equal-mass bins -> list of dict(n, p_lo, p_hi, mean_p, frac_pos). Bin edges are the quantiles of p and an
    item goes by its value, so equal p always share a bin (a heavy tie, e.g. many p = 0.9999, makes that bin larger
    and merges empty ones away)."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not len(p):
        return []
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1])
    idx = np.searchsorted(edges, p, side='right')
    return [dict(n=int((idx == b).sum()), p_lo=float(p[idx == b].min()), p_hi=float(p[idx == b].max()),
                 mean_p=float(p[idx == b].mean()), frac_pos=float(y[idx == b].mean()))
            for b in range(bins) if (idx == b).any()]


def ece(p, y, bins=15, equal_mass=True):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not len(p):
        return None
    if equal_mass:
        return float(sum(b['n'] * abs(b['mean_p'] - b['frac_pos']) for b in reliability(p, y, bins)) / len(p))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).sum() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()) / len(p))


def nll(p, y):
    p, y = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS), np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(p) else None


def brier(p, y):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(((p - y) ** 2).mean()) if len(p) else None


def calibration_report(p, y, bins=15):
    """All calibration numbers of one population (p = predicted probability of y = 1)."""
    return dict(n=int(len(p)), positives=int(np.sum(y)), ece=ece(p, y, bins), ece_equal_width=ece(p, y, bins, False),
                nll=nll(p, y), brier=brier(p, y), mean_p=float(np.mean(p)) if len(p) else None,
                frac_pos=float(np.mean(y)) if len(p) else None)


def coverage_curve(score, error, seed=0):
    """score [n] (larger = more uncertain), error [n] bool -> share (%) of all errors in the top k% for
    k = COVERAGE_POINTS, or None when there is no error. Ties are broken in a fixed random order."""
    score, error = np.asarray(score, dtype=np.float64), np.asarray(error, dtype=bool)
    if not error.any():
        return None
    order = np.lexsort((np.random.default_rng(seed).random(len(score)), -score))
    found = np.concatenate([[0], np.cumsum(error[order])]) / error.sum() * 100
    return found[np.round(COVERAGE_POINTS / 100 * len(score)).astype(int)]


def coverage_area(curve):
    """Area under a coverage curve on [0, 1] x [0, 1] (random check = 0.5, perfect ranking -> 1)."""
    return None if curve is None else float(np.trapezoid(curve / 100, COVERAGE_POINTS / 100))
