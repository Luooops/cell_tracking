"""
track_division.py
=================
track.py with cell division: a cell of frame t may be linked to two cells of frame t+1.

Only the solver call of round 1 changes (solver.match_division). Bookkeeping follows the CTC convention:
    one link    the cell of t+1 inherits the track id (as before)
    two links   the mother's track ends at t; each daughter starts a new track; parent[daughter track] = mother track
Round 2 (bridging one missed frame) never touches a division: a track that ended by dividing is not an 'end', and
a daughter track is not a 'start'.

The same weights as track.py are used. They were trained on the lab data, which has no lineage annotation, so a
mother -> daughter edge was never shown to the model as a positive. In practice this means:
  - on the lab wells this file is worse than track.py (it reports divisions that are not there);
  - on a dataset where divisions are frequent and clear (CTC Fluo-N2DL-HeLa, tried without any retraining) it found
    54-59% of the divisions at 86-90% precision and doubled the share of perfect tracks.
For serious use, the model should be retrained with division labels; the solver here is ready for that.

Returns (assign, parent): assign as in track.py, parent = {daughter track id: mother track id}.
"""
import numpy as np

from models.gnn_tracking.graph import pair_graph
from models.gnn_tracking.model import edge_probabilities
from models.gnn_tracking.solver import match, match_division

LINK_THRESH, GAP_THRESH, DIV_THRESH = 0.5, 0.9, 0.5


def track_video_division(models, video, gap_round=True, div_thresh=DIV_THRESH):
    frames = video.frames
    f0 = frames[0]
    assert frames == list(range(f0, f0 + len(frames))), "frames must be consecutive"
    assign, members, parent, divided, next_id = {}, {}, {}, set(), 0
    for i in range(len(video.rows[f0])):
        assign[(0, i)] = next_id; members[next_id] = [(0, i)]; next_id += 1

    def velocities(fi):
        out = {}
        for i in range(len(video.rows[fi + f0])):
            prev = [k for k in members[assign[(fi, i)]] if k[0] == fi - 1]
            if prev:
                out[i] = tuple(video.pos[fi + f0][i] - video.pos[fi - 1 + f0][prev[0][1]])
        return out

    for t in frames[:-1]:                                             # round 1: adjacent frames, divisions allowed
        fi = t - f0
        gr = pair_graph(video, t, 1, velocities(fi))
        p = edge_probabilities(models, gr)
        src, dst = gr.candidate_edge_index.numpy()
        targets = {}
        for i, j in match_division(p, src, dst - gr.n_t, gr.n_t, gr.n_n, LINK_THRESH, div_thresh):
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
        gr = pair_graph(video, t, 2, velocities(fi))
        p = edge_probabilities(models, gr)
        src, dst = gr.candidate_edge_index.numpy(); dst = dst - gr.n_t
        row = {members[tid][-1][1]: r for r, tid in enumerate(ends)}
        col = {members[tid][0][1]: c for c, tid in enumerate(starts)}
        sel = [e for e in range(len(src)) if src[e] in row and dst[e] in col]
        if not sel:
            continue
        for r, c in match(p[sel], np.array([row[src[e]] for e in sel]), np.array([col[dst[e]] for e in sel]),
                          len(ends), len(starts), GAP_THRESH):
            a, b = ends[r], starts[c]
            for k in members[b]:
                assign[k] = a
            members[a] = members[a] + members.pop(b)
            if b in divided: divided.discard(b); divided.add(a)       # the merged track inherits b's division
            for d, m in parent.items():
                if m == b: parent[d] = a
    return assign, parent
