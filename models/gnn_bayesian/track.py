"""
track.py
========
The two-round tracker of gnn_tracking (track.py / track_division.py: same rounds, bookkeeping and CTC lineage),
scored by the 3-way network: continuations on p_same, divisions (optional) on p_division (solver.py). Every edge is
recorded with p_same (as 'p' and the members' SAME logits, so gnn_tracking.evaluate_answers reads it), p_div and
the ensemble's mutual information. A binary ensemble (model n_classes=2, the ablation) divides as gnn_division:
solver.match_division on p_same.

views: mirrored copies of the video (data.load(..., flip), same detections in the same order); every frame pair is
then scored on each of them too, with the velocities of the same links, and the scores are averaged (test-time
augmentation, model.edge_scores). ablate (cue analysis): 'appearance' shuffles the cell images within each frame of
every graph (appearance carries no information, its distribution is kept); 'velocity' builds every graph without
velocities (as for a cell seen for the first time); position, size and competition stay. cues=True keeps the real
tracking and records, per edge, the probabilities the same graph gets under each ablation (p_noapp, p_novel).
"""
import numpy as np

from models.gnn_tracking.edge_record import EdgeRecorder
from models.gnn_tracking.graph import pair_graph
from models.gnn_tracking.solver import match, match_division
from models.gnn_bayesian.model import edge_scores
from models.gnn_bayesian.solver import match_with_divisions

LINK_THRESH, GAP_THRESH, DIV_THRESH = 0.4, 0.4, 0.5      # from the costs of LNK (0.4) and BC (0.5), not tuned
ABLATIONS = ('appearance', 'velocity')


class Recorder(EdgeRecorder):
    """gnn_tracking's EdgeRecorder + p_div and mi per edge (and p_noapp / p_novel of the cue analysis)."""
    EXTRA = ('p_div', 'mi', 'p_noapp', 'p_novel')

    def __call__(self, round_, t, g, src, dst, scores, in_solver, chosen):
        super().__call__(round_, t, g, src, dst, scores['p_same'], scores['logits'], in_solver, chosen)
        for k in self.EXTRA:
            if k in scores:
                self.parts[-1][k] = np.asarray(scores[k], np.float32)

    def arrays(self):
        out = super().arrays()
        if self.parts:
            for k in self.EXTRA:
                if k in self.parts[0]:
                    out[k] = np.concatenate([part[k] for part in self.parts])
        return out


def _shuffled_images(gr, seed):
    """The graph with its cell images permuted within frame t and within frame t + g."""
    import torch
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(gr.n_t), gr.n_t + rng.permutation(gr.n_n)])
    out = gr.clone()
    out.crops = gr.crops[torch.as_tensor(order)]
    return out


def _order(reference, view):
    """Indices putting the view's candidate edges into the reference's order (same detections, same edges)."""
    ref, other = reference.candidate_edge_index.numpy(), view.candidate_edge_index.numpy()
    where = {(int(a), int(b)): k for k, (a, b) in enumerate(other.T)}
    return np.array([where[(int(a), int(b))] for a, b in ref.T])


def track_video(models, video, target, divisions=False, gap_round=True, link_thresh=LINK_THRESH,
                gap_thresh=GAP_THRESH, div_thresh=DIV_THRESH, recorder=None, prior='none', views=(), ablate=None,
                cues=False):
    """-> (assign, parent) as gnn_tracking.track_division (parent stays empty without divisions). target: the
    dataset of the video (used only by prior 'division'); prior: model.PRIORS; views / ablate / cues: see above."""
    binary = models[0].n_classes == 2
    frames = video.frames
    f0 = frames[0]
    assert frames == list(range(f0, f0 + len(frames))), "frames must be consecutive"
    assign, members, parent, divided, next_id = {}, {}, {}, set(), 0
    for i in range(len(video.rows[f0])):
        assign[(0, i)] = next_id; members[next_id] = [(0, i)]; next_id += 1

    def velocities(fi, vid=video):
        out = {}
        if ablate == 'velocity':
            return out
        for i in range(len(vid.rows[fi + f0])):
            prev = [k for k in members[assign[(fi, i)]] if k[0] == fi - 1]
            if prev:
                out[i] = tuple(vid.pos[fi + f0][i] - vid.pos[fi - 1 + f0][prev[0][1]])
        return out

    def score(t, g):
        """-> (reference graph, scores): every view of frames (t, t + g), ablated as asked, averaged."""
        fi = t - f0
        gr = pair_graph(video, t, g, velocities(fi))
        seed = t * 10 + g
        prep = (lambda x: _shuffled_images(x, seed)) if ablate == 'appearance' else (lambda x: x)
        graphs = [(prep(gr), None)]
        for vid in views:
            other = pair_graph(vid, t, g, velocities(fi, vid))
            graphs.append((prep(other), _order(gr, other)))
        s = edge_scores(models, graphs, g, target, prior)
        if cues:
            s['p_noapp'] = edge_scores(models, [(_shuffled_images(gr, seed), None)], g, target, prior)['p_same']
            s['p_novel'] = edge_scores(models, [(pair_graph(video, t, g, {}), None)], g, target, prior)['p_same']
        return gr, s

    for t in frames[:-1]:                                             # round 1: adjacent frames
        fi = t - f0
        gr, s = score(t, 1)
        src, dst = gr.candidate_edge_index.numpy()
        dst = dst - gr.n_t
        if divisions and binary:
            links = match_division(s['p_same'], src, dst, gr.n_t, gr.n_n, link_thresh, div_thresh)
        elif divisions:
            links = match_with_divisions(s['p_same'], s['p_div'], src, dst, gr.n_t, gr.n_n, link_thresh, div_thresh)
        else:
            links = match(s['p_same'], src, dst, gr.n_t, gr.n_n, link_thresh)
        if recorder is not None:
            recorder(1, t, 1, src, dst, s, np.ones(len(src), bool), links)
        targets = {}
        for i, j in links:
            targets.setdefault(i, []).append(j)
        matched = set()
        for i, js in targets.items():
            tid = assign[(fi, i)]
            if len(js) == 1:
                assign[(fi + 1, js[0])] = tid; members[tid].append((fi + 1, js[0]))
            else:                                                     # division: the mother's track ends here
                divided.add(tid)
                for j in js:
                    assign[(fi + 1, j)] = next_id; members[next_id] = [(fi + 1, j)]; parent[next_id] = tid; next_id += 1
            matched.update(js)
        for j in range(gr.n_n):
            if j not in matched:
                assign[(fi + 1, j)] = next_id; members[next_id] = [(fi + 1, j)]; next_id += 1
    if not gap_round:
        return assign, parent

    for t in frames[:-2]:                                             # round 2: bridge one missed frame (no divisions)
        fi, fj = t - f0, t + 2 - f0
        ends = [tid for tid, ks in members.items() if ks[-1][0] == fi and tid not in divided]
        starts = [tid for tid, ks in members.items() if ks[0][0] == fj and tid not in parent]
        if not ends or not starts:
            continue
        gr, s = score(t, 2)
        src, dst = gr.candidate_edge_index.numpy()
        dst = dst - gr.n_t
        row = {members[tid][-1][1]: r for r, tid in enumerate(ends)}
        col = {members[tid][0][1]: c for c, tid in enumerate(starts)}
        sel = [e for e in range(len(src)) if src[e] in row and dst[e] in col]
        if not sel:
            if recorder is not None:
                recorder(2, t, 2, src, dst, s, np.zeros(len(src), bool), [])
            continue
        links = match(s['p_same'][sel], np.array([row[src[e]] for e in sel]), np.array([col[dst[e]] for e in sel]),
                      len(ends), len(starts), gap_thresh)
        if recorder is not None:
            recorder(2, t, 2, src, dst, s, np.isin(np.arange(len(src)), sel),
                     [(members[ends[r]][-1][1], members[starts[c]][0][1]) for r, c in links])
        for r, c in links:
            a, b = ends[r], starts[c]
            for k in members[b]:
                assign[k] = a
            members[a] = members[a] + members.pop(b)
            if b in divided: divided.discard(b); divided.add(a)       # the merged track inherits b's division
            for d, m in parent.items():
                if m == b: parent[d] = a
    return assign, parent

