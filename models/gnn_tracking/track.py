"""
track.py
========
Step 4: inference. Detections of one video -> a track id for every detection.

Round 1 (adjacent frames), frame by frame:
    build the graph (t, t+1) -> edge probabilities -> solver with threshold 0.5.
    A matched cell of t+1 inherits the track id; an unmatched one starts a new track.
    It has to be sequential: a frame-t cell's velocity comes from the link it received one step earlier.

Round 2 (one missed frame), for every t:
    tracks that end at t and tracks that start at t+2 are the only ones that can still be joined.
    The graph (t, t+2) is built from all cells of both frames (the others are context for message passing), the
    same model scores it, and the solver runs on the end -> start edges only, with the stricter threshold 0.9:
    a wrong bridge merges two cells into one track, which costs more than a track left in two pieces.
    The later track takes over the id of the earlier one.

There is no third round (two missed frames): it was tested and lowered perfect tracks by ~0.9 points.

The result is `assign`: {(frame index, local index): track id}, frame index = position of the frame in the video,
local index = position of the cell among that frame's rows in the detection table.

    python track.py r16c02 [out.csv]     # track one well with the shipped weights, optionally save the table
"""
import os, sys, glob
import numpy as np

from utils.dataset import HERE, load_well
from models.gnn_tracking.graph import Video, pair_graph
from models.gnn_tracking.model import load_model, edge_probabilities
from models.gnn_tracking.solver import match

LINK_THRESH = 0.5        # round 1. Fixed a priori ("more likely the same cell than not"), not tuned on the data
GAP_THRESH = 0.9         # round 2


def track_video(models, video, gap_round=True):
    frames = video.frames
    f0 = frames[0]
    assert frames == list(range(f0, f0 + len(frames))), "frames must be consecutive"
    assign, members, next_id = {}, {}, 0          # members[track id] = [(frame index, local index), ...] in time order
    for i in range(len(video.rows[f0])):          # every cell of the first frame starts a track
        assign[(0, i)] = next_id; members[next_id] = [(0, i)]; next_id += 1

    def velocities(fi):
        """{local index: displacement over the last step} for the frame-fi cells whose track was also seen at fi-1."""
        out = {}
        for i in range(len(video.rows[fi + f0])):
            prev = [k for k in members[assign[(fi, i)]] if k[0] == fi - 1]
            if prev:
                out[i] = tuple(video.pos[fi + f0][i] - video.pos[fi - 1 + f0][prev[0][1]])
        return out

    # ---------------- round 1: adjacent frames ----------------
    for t in frames[:-1]:
        fi = t - f0
        gr = pair_graph(video, t, 1, velocities(fi))
        p = edge_probabilities(models, gr)
        src, dst = gr.candidate_edge_index.numpy()
        matched = set()
        for i, j in match(p, src, dst - gr.n_t, gr.n_t, gr.n_n, LINK_THRESH):
            tid = assign[(fi, i)]
            assign[(fi + 1, j)] = tid; members[tid].append((fi + 1, j)); matched.add(j)
        for j in range(gr.n_n):
            if j not in matched:
                assign[(fi + 1, j)] = next_id; members[next_id] = [(fi + 1, j)]; next_id += 1
    if not gap_round:
        return assign

    # ---------------- round 2: bridge one missed frame ----------------
    for t in frames[:-2]:
        fi, fj = t - f0, t + 2 - f0
        ends = [tid for tid, ks in members.items() if ks[-1][0] == fi]
        starts = [tid for tid, ks in members.items() if ks[0][0] == fj]
        if not ends or not starts:
            continue
        gr = pair_graph(video, t, 2, velocities(fi))
        p = edge_probabilities(models, gr)
        src, dst = gr.candidate_edge_index.numpy(); dst = dst - gr.n_t
        row = {members[tid][-1][1]: r for r, tid in enumerate(ends)}       # local index in frame t   -> solver row
        col = {members[tid][0][1]: c for c, tid in enumerate(starts)}      # local index in frame t+2 -> solver column
        sel = [e for e in range(len(src)) if src[e] in row and dst[e] in col]
        if not sel:
            continue
        for r, c in match(p[sel], np.array([row[src[e]] for e in sel]), np.array([col[dst[e]] for e in sel]),
                          len(ends), len(starts), GAP_THRESH):
            a, b = ends[r], starts[c]
            for k in members[b]:
                assign[k] = a
            members[a] = members[a] + members.pop(b)
    return assign


def load_ensemble(folder=os.path.join(HERE, "weights")):
    """The shipped model is 3 networks that differ only in the random seed; their probabilities are averaged."""
    paths = sorted(glob.glob(os.path.join(folder, "*.pt")))
    assert paths, f"no weights in {folder}"
    return [load_model(p) for p in paths]


if __name__ == "__main__":
    well = sys.argv[1] if len(sys.argv) > 1 else "r16c02"
    df, crops = load_well(well)
    video = Video(df, crops)
    assign = track_video(load_ensemble(), video)
    df = df.copy()
    df["pred_track"] = [assign[(fi, i)] for fi, t in enumerate(video.frames) for i in range(len(video.rows[t]))]
    print(f"{well}: {len(df)} detections in {len(video.frames)} frames -> {df.pred_track.nunique()} tracks")
    if len(sys.argv) > 2:
        df.to_csv(sys.argv[2], index=False); print("saved", sys.argv[2])
