"""
edge_record.py
==============
Stage 0.5: keep the tracker's "answer sheet" -- every candidate edge it scored during real inference.

track_video / track_video_division call the recorder once per solver call (round 1: every adjacent frame pair;
round 2: every gap-2 pair that has track ends and starts). One row per candidate edge of that graph:

    round        1 or 2
    src_frame    frame t;  g = the gap (dst frame = t + g)
    src_label    mask label (instance id) of the frame-t detection; dst_label likewise in frame t + g
    distance     centroid distance in px;  gate = r(t) x sqrt(g), the candidate radius of that graph
    logits       [E, M] logit of every ensemble member as the tracker used it (after prior_correction, if on; the
                 expected logit of an aleatoric member);  p = mean of their sigmoids (what the solver used)
    log_var      [E, M] each member's s = log sigma^2, only for aleatoric ensembles (model.py)
    in_solver    True if the edge was offered to the solver (round 1: all; round 2: track end -> track start only)
    chosen       True if the solver linked it

No ground truth is read here: labels are mask instance ids only, and the GT relation of each edge is attached
afterwards from utils/lineage.py. Kept free of torch so the records can be read anywhere.

    rec = EdgeRecorder(video); track_video(models, video, recorder=rec); rec.save(path)
    edges = load_edges(path)          # dict of numpy arrays, one entry per column
"""
import numpy as np

COLUMNS = ('round', 'src_frame', 'g', 'src_label', 'dst_label', 'distance', 'gate', 'p', 'in_solver', 'chosen')


class EdgeRecorder:
    def __init__(self, video):
        self.video = video
        self.parts = []

    def __call__(self, round_, t, g, src, dst, p, logits, in_solver, chosen, log_var=None):
        """src / dst: local indices in frames t / t + g (one per candidate edge); p [E]; logits [E, M];
        in_solver: bool [E]; chosen: iterable of (src local, dst local) pairs the solver linked;
        log_var [E, M] or None."""
        v = self.video
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        labels_t = v.df.loc[v.rows[t], 'track_id'].to_numpy()
        labels_n = v.df.loc[v.rows[t + g], 'track_id'].to_numpy()
        picked = set(map(tuple, chosen))
        n = len(src)
        self.parts.append(dict(
            round=np.full(n, round_, np.int8), src_frame=np.full(n, t, np.int32), g=np.full(n, g, np.int8),
            src_label=labels_t[src].astype(np.int64), dst_label=labels_n[dst].astype(np.int64),
            distance=np.hypot(*(v.pos[t + g][dst] - v.pos[t][src]).T).astype(np.float32).reshape(n),
            gate=np.full(n, v.radius(t) * np.sqrt(g), np.float32), p=np.asarray(p, np.float32).reshape(n),
            in_solver=np.asarray(in_solver, bool).reshape(n),
            chosen=np.array([(int(i), int(j)) in picked for i, j in zip(src, dst)], bool).reshape(n),
            logits=np.asarray(logits, np.float32).reshape(n, -1)))
        if log_var is not None:
            self.parts[-1]['log_var'] = np.asarray(log_var, np.float32).reshape(n, -1)

    def arrays(self):
        if not self.parts:
            return {k: np.zeros(0) for k in COLUMNS} | {'logits': np.zeros((0, 0), np.float32)}
        extra = ('log_var',) if all('log_var' in part for part in self.parts) else ()
        return {k: np.concatenate([part[k] for part in self.parts]) for k in COLUMNS + ('logits',) + extra}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())


def load_edges(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
