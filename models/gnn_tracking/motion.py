"""
motion.py
=========
Stage 1 (Bayesian motion): a Kalman filter over each track's past detections gives the GNN a predicted position
together with its uncertainty, in place of the one-step velocity (train.motion = "kalman").

Two linear-Gaussian motion models, per axis (x and y independent, same parameters), one step = one frame:
    cv   constant velocity: state (position, velocity); the velocity changes by white-noise acceleration q,
         Q = q [[1/3, 1/2], [1/2, 1]]; a track starts at its first detection with velocity 0 +- sqrt(v0)
    rw   random walk: state (position); Q = q
Both observe the position with noise variance r. A missed frame is a prediction without an update, so the
uncertainty grows over gaps, and a track seen once has a wide prediction.

fit(): maximum likelihood of (q, r[, v0]) on GT tracks (likelihood of the filter's innovations, Nelder-Mead on log
parameters) for both models; the model with the lower BIC is kept -- Bayesian model selection in its large-sample
form. A fold fits on its training wells only and stores the result in its checkpoints (train.py).

Tracks are passed as Z [n, T, 2]: one track per row, NaN where it was not observed; every function runs all rows at
once.
"""
import numpy as np

MODELS = ('cv', 'rw')
CLIP = 10.0                # residuals in units of the predicted standard deviation are clipped to +-CLIP


def _matrices(model, q):
    if model == 'cv':
        return np.array([[1.0, 1.0], [0.0, 1.0]]), q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    return np.array([[1.0]]), np.array([[q]])


def run_filter(Z, params):
    """Filter every row of Z [n, T, 2] up to its last column. -> (mean [n, d, 2], cov [n, d, d], seen [n],
    log likelihood of all innovations, number of innovations); rows never observed keep seen = 0."""
    F, Q = _matrices(params['model'], params['q'])
    r, d = params['r'], F.shape[0]
    n, T, _ = Z.shape
    m, P = np.zeros((n, d, 2)), np.zeros((n, d, d))
    seen, started = np.zeros(n, int), np.zeros(n, bool)
    P0 = np.diag([r, params.get('v0', 0.0)][:d])
    loglik, innovations = 0.0, 0
    for t in range(T):
        z = Z[:, t]
        obs = ~np.isnan(z[:, 0])
        if started.any():
            m[started] = np.einsum('ij,njk->nik', F, m[started])
            P[started] = F @ P[started] @ F.T + Q
        new = obs & ~started
        upd = obs & started
        if upd.any():
            S = P[upd, 0, 0] + r
            K = P[upd, :, 0] / S[:, None]                                  # [k, d]
            innov = z[upd] - m[upd, 0]                                     # [k, 2]
            loglik += float(-0.5 * (2 * np.log(2 * np.pi * S) + (innov ** 2).sum(1) / S).sum())
            innovations += int(upd.sum())
            m[upd] += K[:, :, None] * innov[:, None, :]
            P[upd] -= K[:, :, None] * P[upd, 0][:, None, :]
            seen[upd] += 1
        if new.any():
            m[new] = 0.0
            m[new, 0] = z[new]
            P[new] = P0
            started |= new
            seen[new] = 1
    return m, P, seen, loglik, innovations


def predict(mean, cov, g, params):
    """Position predicted g frames ahead -> (position [n, 2], variance of the observed position [n])."""
    F, Q = _matrices(params['model'], params['q'])
    m, P = mean.copy(), cov.copy()
    for _ in range(g):
        m = np.einsum('ij,njk->nik', F, m)
        P = F @ P @ F.T + Q
    return m[:, 0], P[:, 0, 0] + params['r']


def fit(Z):
    """Maximum likelihood of both models on the tracks Z and BIC model selection -> parameters for run_filter,
    with the fit of each model under 'fits'."""
    from scipy.optimize import minimize
    steps = np.diff(Z, axis=1).reshape(-1, 2)
    s2 = float(np.nanvar(steps)) or 1.0                              # one-step displacement variance (start values)
    fits = {}
    for model in MODELS:
        names = ('q', 'r', 'v0') if model == 'cv' else ('q', 'r')
        start = np.log([s2 * 0.1, s2 * 0.1, s2][:len(names)])

        def nll(theta):
            params = dict(model=model, **dict(zip(names, np.exp(theta))))
            return -run_filter(Z, params)[3]
        best = minimize(nll, start, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-3, maxiter=400))
        params = dict(model=model, **{k: float(v) for k, v in zip(names, np.exp(best.x))})
        _, _, _, loglik, innovations = run_filter(Z, params)
        n_obs = 2 * innovations
        fits[model] = dict(params, loglik=float(loglik), bic=float(len(names) * np.log(max(n_obs, 1)) - 2 * loglik),
                           n_obs=int(n_obs))
    chosen = min(fits, key=lambda k: fits[k]['bic'])
    out = {k: v for k, v in fits[chosen].items() if k in ('model', 'q', 'r', 'v0')}
    out.setdefault('v0', 0.0)
    out['fits'] = fits
    return out


def histories(tracks, length):
    """[(frame index, x, y), ...] per track, frame index < length -> Z [len(tracks), length, 2]."""
    Z = np.full((len(tracks), length, 2), np.nan)
    for row, track in enumerate(tracks):
        for f, x, y in track:
            if f < length:
                Z[row, f] = (x, y)
    return Z


def state(tracks, length, params):
    """Filter state at frame index length - 1 of each track -> dict(mean, cov, seen, params)."""
    m, P, seen, _, _ = run_filter(histories(tracks, length), params)
    return dict(mean=m, cov=P, seen=seen, params=params)


def features(motion, g, p_src, p_dst, src, dst, r):
    """Edge features [C, 5] and node velocity features [n_t, 3] of the frame-t cells from their filter state:
    the target's offset from the predicted position in predicted standard deviations (dx, dy, length, clipped),
    1 if the source has more than one detection, log(predicted sd / r); node: the filter's velocity / r and the
    same indicator."""
    params = motion['params']
    pred, var = predict(motion['mean'], motion['cov'], g, params)
    sd = np.sqrt(var)
    res = np.clip((p_dst[dst] - pred[src]) / sd[src, None], -CLIP, CLIP)
    has = (motion['seen'] > 1).astype(float)
    edge = np.stack([res[:, 0], res[:, 1], np.minimum(np.hypot(res[:, 0], res[:, 1]), CLIP), has[src],
                     np.log(sd[src] / r)], axis=1)
    velocity = motion['mean'][:, 1] if params['model'] == 'cv' else np.zeros((len(sd), 2))
    node = np.concatenate([velocity / r, has[:, None]], axis=1)
    return edge.astype(np.float32), node.astype(np.float32)
