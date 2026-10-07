"""
graph.py
========
Step 1: detections -> one PyTorch-Geometric graph per frame pair (t, t+g).

g is the gap: g = 1 is the ordinary pair of adjacent frames; g = 2 pairs frame t with frame t+2 and is what lets the
tracker re-join a cell that the segmentation missed in frame t+1. The same builder and the same model serve both.

What one graph holds (N = n_t + n_n nodes: the n_t cells of frame t get node ids 0 .. n_t-1, the n_n cells of
frame t+g get ids n_t .. N-1):

    geom_x                [N, 8]         node features that are not the image
                                           0-3  x, y, area, mean intensity     (z-scored with this video's own mean/std)
                                           4    frame flag: 0 = frame t, 1 = frame t+g
                                           5-6  velocity of the cell, / r      (frame-t cells only; 0 for frame t+g)
                                           7    1 if the velocity is known
    crops                 [N, 1, 64, 64] the image of every cell: its own pixels only, 64 x 64 (see data.py); the
                                         appearance network inside the model turns it into a 32-dim unit vector
    candidate_edge_index  [2, C]         the links being decided: frame-t cell -> frame-(t+g) cell, closer than r
    candidate_edge_attr   [C, 13]        0-3   dx, dy, distance (all / r), log area ratio
                                           4-7   competition: distance relative to the source's / the target's closest
                                                 candidate, log(1 + number of candidates) of the source / the target
                                           8-11  velocity residual: (target - predicted position) / r as dx, dy,
                                                 length; and 1 if the source has a velocity
                                           12    gap feature (g - 1) / 2: 0 for adjacent frames, 0.5 for g = 2
    message_edge_index    [2, M]         the edges information flows along: the candidate edges in both directions
                                         plus, inside each frame, every cell's 4 nearest neighbours
    message_edge_attr     [M, 13]        same 13 columns; columns 4-11 are filled only for the forward candidate
                                         edges (the first C rows) and are 0 for the reverse and same-frame edges
    msg_fwd               [M]            True for those forward candidate rows (rows 0 .. C-1 of one graph). The model
                                         writes its appearance-similarity features into exactly these rows; the mask is
                                         needed because 'the first C rows' is no longer true once graphs are batched
    candidate_edge_label  [C]            1 if both cells carry the same GT track id       (training / scoring only)
    candidate_edge_unknown[C]            True if that label cannot be trusted             (left out of the loss)

The two edge sets are deliberately separate: message edges are a wide graph that only carries context, candidate
edges are the narrow set that gets a probability.

    python graph.py [well] [frame]      # prints the real tensor shapes of one graph
"""
import sys
import numpy as np, torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from data import UNLABELLED_OFFSET, load_well

KNN = 4                  # same-frame neighbours per cell in the message graph
# --- adaptive gating radius (no ground truth needed; constants calibrated in September on the dnn benchmark) ---
SAFETY = 2.1             # r = SAFETY x the 99% quantile of the frame-to-frame step ...
R_MIN = 10.0             # ... at least this many pixels ...
SPACING_CAP = 1.5        # ... and at most SPACING_CAP x the median distance between neighbouring cells
WINDOW = 8               # frame pairs around t that the estimate uses
R_FALLBACK = 40.0        # fewer than 5 usable cells in the window


class Video:
    """One well: the detection table, the crops, and per-frame lookups that every graph needs.

        rows[t]   df row numbers of frame t's cells (also the row numbers into `crops`)       [n_t]
        pos[t]    their (x, y)                                                                [n_t, 2]

    The position of a cell inside rows[t] is its local index; that is the number the solver and the track
    bookkeeping use. Node id inside a pair graph = local index (frame t) or local index + n_t (frame t+g)."""

    def __init__(self, df, crops):
        self.df, self.crops = df, crops
        self.frames = sorted(df.frame.unique())
        self.rows = {t: df.index[df.frame == t].to_numpy() for t in self.frames}
        self.pos = {t: df.loc[self.rows[t], ["x", "y"]].to_numpy() for t in self.frames}
        # z-score statistics from this video only: wells differ in brightness and cell size, and a per-video
        # normalisation needs no ground truth, so it is available at deployment too.
        self.norm = {c: (df[c].mean(), df[c].std() + 1e-6) for c in ("x", "y", "area", "mean_intensity")}
        self._radius = {}

    def radius(self, t):
        """Gating radius for links leaving frame t, estimated from positions alone.

        Step size: in each consecutive frame pair of a window around t, take the cells that are each other's
        nearest neighbour (almost always the same cell) and collect their distances; q99 = the 99% quantile.
        Density: the median distance from a cell to its nearest neighbour in the same frame.
        r = clip(2.1 x q99, at least 10 px, at most 1.5 x spacing). The cap keeps the candidate graph sparse in
        dense wells: beyond 1.5 neighbour spacings a link would have to jump over other cells."""
        if t not in self._radius:
            i = self.frames.index(t)
            win = self.frames[max(0, i - WINDOW // 2): min(len(self.frames), i + WINDOW // 2 + 2)]
            steps, spacing = [], []
            for a, b in zip(win[:-1], win[1:]):
                pa, pb = self.pos[a], self.pos[b]
                d_ab, nn_ab = cKDTree(pb).query(pa, k=1)          # for every cell of a: nearest cell of b
                _, nn_ba = cKDTree(pa).query(pb, k=1)             # and the other way round
                steps += [d_ab[k] for k, j in enumerate(nn_ab) if nn_ba[j] == k]
            for a in win:
                if len(self.pos[a]) >= 2:
                    d, _ = cKDTree(self.pos[a]).query(self.pos[a], k=2)   # k=2: [itself, nearest other cell]
                    spacing.append(d[:, 1])
            if len(steps) < 5:
                self._radius[t] = R_FALLBACK
            else:
                r = max(SAFETY * float(np.quantile(steps, 0.99)), R_MIN)
                spacing = float(np.median(np.concatenate(spacing))) if spacing else float("inf")
                self._radius[t] = float(min(r, SPACING_CAP * spacing))
        return self._radius[t]


def _node_block(sub, norm, frame_flag):
    """[n, 5] = z-scored x, y, area, intensity + the frame flag."""
    cols = [(sub[c].to_numpy() - norm[c][0]) / norm[c][1] for c in ("x", "y", "area", "mean_intensity")]
    return np.stack(cols + [np.full(len(sub), float(frame_flag))], axis=1).astype(np.float32)


def _geometry(p_src, p_dst, a_src, a_dst, r):
    """[E, 4] = dx, dy, distance (in units of r) and log area ratio of a block of edges src -> dst."""
    dx, dy = (p_dst[:, 0] - p_src[:, 0]) / r, (p_dst[:, 1] - p_src[:, 1]) / r
    return np.stack([dx, dy, np.sqrt(dx ** 2 + dy ** 2), np.log((a_dst + 1) / (a_src + 1))], axis=1).astype(np.float32)


def _knn(pos, k=KNN):
    """[n * k, 2] pairs (i, j): j is one of the k nearest cells of i in the same frame."""
    if len(pos) <= 1:
        return np.empty((0, 2), int)
    _, idx = cKDTree(pos).query(pos, k=min(k + 1, len(pos)))      # column 0 is the cell itself
    return np.array([(i, j) for i in range(len(pos)) for j in idx[i][1:]], int)


def pair_graph(video, t, g, velocity):
    """Graph of frames (t, t+g).

    velocity: {local index of a frame-t cell: (vx, vy)} = its displacement over the last frame step (t-1 -> t), in
    pixels, for the cells whose previous position is known. Training takes it from the GT track, inference from
    the links the tracker has made so far. Over a gap of g frames the cell is expected to move g x that far."""
    df = video.df
    r = video.radius(t) * np.sqrt(g)       # a random walk spreads with sqrt(time): 98.6% of the true 2- and
                                           # 3-frame links are inside r * sqrt(g) (measured on the dev wells)
    sub_t, sub_n = df.loc[video.rows[t]], df.loc[video.rows[t + g]]
    n_t, n_n = len(sub_t), len(sub_n)
    pt, pn = video.pos[t], video.pos[t + g]
    at, an = sub_t.area.to_numpy(), sub_n.area.to_numpy()

    # ---- candidate edges: every (frame-t cell, frame-(t+g) cell) pair closer than r ----
    dist = np.sqrt(((pt[:, None, :] - pn[None, :, :]) ** 2).sum(-1))      # [n_t, n_n]
    src, dst = np.where(dist < r)                                          # local indices, C of each
    C = len(src)
    cand_geom = _geometry(pt[src], pn[dst], at[src], an[dst], r)           # [C, 4]

    # competition features: is this the source's (the target's) best option, and how crowded is the choice?
    d = np.hypot(*(pn[dst] - pt[src]).T) / r
    best_src, best_dst = np.full(n_t, np.inf), np.full(n_n, np.inf)
    np.minimum.at(best_src, src, d)        # best_src[i] = distance of cell i's closest candidate
    np.minimum.at(best_dst, dst, d)
    n_src, n_dst = np.bincount(src, minlength=n_t), np.bincount(dst, minlength=n_n)
    comp = np.stack([d / (best_src[src] + 1e-3), d / (best_dst[dst] + 1e-3),
                     np.log1p(n_src[src]), np.log1p(n_dst[dst])], axis=1)   # [C, 4]

    # velocity features: where would the source be if it kept moving as it did in the last step?
    v, has_v = np.zeros((n_t, 2)), np.zeros(n_t)
    for i, vi in velocity.items():
        v[i] = (g * vi[0], g * vi[1]); has_v[i] = 1.0
    res = (pn[dst] - (pt[src] + v[src])) / r
    vel_edge = np.stack([res[:, 0], res[:, 1], np.hypot(res[:, 0], res[:, 1]), has_v[src]], axis=1)   # [C, 4]
    vel_node = np.zeros((n_t + n_n, 3), np.float32)
    vel_node[:n_t, 0:2] = v / r; vel_node[:n_t, 2] = has_v

    extra = np.concatenate([comp, vel_edge], axis=1).astype(np.float32)    # [C, 8]
    gap = (g - 1) / 2.0

    # ---- message edges: (a) candidates t -> t+g, (b) the same edges reversed, (c) kNN in frame t, (d) kNN in t+g ----
    idx = [np.stack([src, dst + n_t]), np.stack([dst + n_t, src])]
    geom = [cand_geom, _geometry(pn[dst], pt[src], an[dst], at[src], r)]
    kt, kn = _knn(pt), _knn(pn)
    if len(kt):
        idx.append(np.stack([kt[:, 0], kt[:, 1]]))
        geom.append(_geometry(pt[kt[:, 0]], pt[kt[:, 1]], at[kt[:, 0]], at[kt[:, 1]], r))
    if len(kn):
        idx.append(np.stack([kn[:, 0] + n_t, kn[:, 1] + n_t]))
        geom.append(_geometry(pn[kn[:, 0]], pn[kn[:, 1]], an[kn[:, 0]], an[kn[:, 1]], r))
    msg_index, msg_geom = np.concatenate(idx, axis=1), np.concatenate(geom, axis=0)
    M = msg_index.shape[1]
    msg_extra = np.zeros((M, extra.shape[1]), np.float32)
    msg_extra[:C] = extra                  # block (a) comes first, so rows 0 .. C-1 are the forward candidates

    # ---- labels ----
    tid_t, tid_n = sub_t.track_id.to_numpy(), sub_n.track_id.to_numpy()
    same = tid_t[src] == tid_n[dst]
    # A detection without identity has a unique id, so every edge touching it looks like "different cell". That
    # is only certain if the other end is a labelled track that also has its own labelled detection in the
    # unlabelled cell's frame. Otherwise (both ends unlabelled, or the labelled track is missing in that frame)
    # the edge may be a true link, and it is excluded from the loss instead of being taught as a negative.
    unl_t, unl_n = tid_t >= UNLABELLED_OFFSET, tid_n >= UNLABELLED_OFFSET
    src_seen_later = np.isin(tid_t[src], tid_n[~unl_n])
    dst_seen_before = np.isin(tid_n[dst], tid_t[~unl_t])
    us, ud = unl_t[src], unl_n[dst]
    unknown = (us & ud) | (ud & ~us & ~src_seen_later) | (us & ~ud & ~dst_seen_before)

    f32 = lambda a: torch.from_numpy(a)
    gr = Data()
    gr.geom_x = torch.cat([f32(np.concatenate([_node_block(sub_t, video.norm, 0), _node_block(sub_n, video.norm, 1)])),
                           f32(vel_node)], dim=1)                                                    # [N, 8]
    gr.crops = f32(np.concatenate([video.crops[video.rows[t]], video.crops[video.rows[t + g]]])).unsqueeze(1)   # float16 [N, 1, 64, 64]
    gr.msg_fwd = torch.zeros(M, dtype=torch.bool); gr.msg_fwd[:C] = True
    gr.candidate_edge_index = f32(np.stack([src, dst + n_t])).long()
    gr.candidate_edge_attr = torch.cat([f32(cand_geom), f32(extra), torch.full((C, 1), gap)], dim=1)   # [C, 13]
    gr.message_edge_index = f32(msg_index).long()
    gr.message_edge_attr = torch.cat([f32(msg_geom), f32(msg_extra), torch.full((M, 1), gap)], dim=1)  # [M, 13]
    gr.candidate_edge_label = f32(same.astype(np.float32))
    gr.candidate_edge_unknown = f32(unknown)
    gr.n_t, gr.n_n = n_t, n_n
    gr.num_nodes = n_t + n_n               # PyG cannot infer it (there is no `x`), and batching needs it
    return gr


if __name__ == "__main__":
    well = sys.argv[1] if len(sys.argv) > 1 else "r16c02"
    video = Video(*load_well(well))
    t = int(sys.argv[2]) if len(sys.argv) > 2 else video.frames[5]
    for g in (1, 2):
        gr = pair_graph(video, t, g, {})
        y, unk = gr.candidate_edge_label, gr.candidate_edge_unknown
        print(f"\n{well}, frames ({t}, {t + g}): r = {video.radius(t) * np.sqrt(g):.1f} px, n_t = {gr.n_t}, n_n = {gr.n_n}")
        for k in ("geom_x", "crops", "candidate_edge_index", "candidate_edge_attr", "message_edge_index", "message_edge_attr"):
            print(f"  {k:22s} {tuple(gr[k].shape)}")
        print(f"  candidate edges: {int(y[~unk].sum())} true, {int((1 - y[~unk]).sum())} false, {int(unk.sum())} unknown")
        print(f"  message edges  : {2 * gr.candidate_edge_index.shape[1]} candidate (both directions) + "
              f"{gr.message_edge_index.shape[1] - 2 * gr.candidate_edge_index.shape[1]} same-frame kNN")
