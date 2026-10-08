"""
train.py
========
Training. One model, one run, two loss terms.

Training set: for every training well and each of 3 versions (original, mirrored left-right,
mirrored up-down):
    gap 1   every frame pair (t, t+1)
    gap 2   (t, t+2) for even t          the two larger gaps teach the model what "the same cell, later" looks
    gap 3   (t, t+3) for even t          like; inference only uses gaps 1 and 2, but dropping gap 3 from training
                                         cost 1.25 points
Every well has 30 frames -> 29 + 14 + 14 = 57 graphs per version, 12 wells x 3 versions x 57 = 2,052 graphs.

Loss 1, edge classification (unchanged): binary cross-entropy on the edge logits, one vote per candidate edge,
positives weighted by pos_weight = (number of negative edges) / (number of positive edges). Edges whose label is
unknown (see graph.py) are left out.

Loss 2, contrastive (new), on the appearance similarities alone, before any message passing:
    for a cell i of frame t with candidates j1, j2, j3 in frame t+g, softmax(sim(i, j) / 0.1) over the three must put
    its mass on the true one; and the same from the target's side (which of the cells that could link to j is it).
    loss2 = - mean log softmax_i(true edge) - mean log softmax_j(true edge)            (InfoNCE in both directions)
This is the same question the tracker has to answer, asked of the appearance network alone. The negatives are the
cell's real competitors (its spatial neighbours), which is what makes them useful.
total = loss1 + AUX * loss2, AUX = 1 (set a priori, not tuned).

Velocity (teacher forcing): a frame-t cell gets the displacement from its GT position one frame earlier. For 20%
of the cells it is withheld at random, so the model also learns to judge cells without a known velocity -- at
inference that is every new track and every cell just after a missed frame.

Adam, lr 1e-3, 40 epochs, 16 graphs per batch, no schedule, no weight decay. ~25 min per seed on 12 wells (RTX 4070
SUPER), 2-3 GB of GPU memory, ~10 GB of RAM (all training graphs with their crops are kept in memory).

    python train.py                         # 3 seeds on the 12 development wells -> weights/ (existing files are kept)
    python train.py --out my_weights        # the same into another folder (use this to retrain from scratch)
    python train.py --fold 0                # one seed on folds 1+2, scored on fold 0 (16 min; a quick check of this file)
    python train.py --cv                    # full 3-fold cross-validation with 3 seeds (9 trainings, about 2.5 hours)
"""
import os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax

from utils.dataset import HERE, DEV_WELLS, FOLDS, UNLABELLED_OFFSET, load_well
from models.gnn_tracking.graph import Video, pair_graph
from models.gnn_tracking.model import TrackerGNN, DEVICE, TAU

EPOCHS, BATCH, LR, AUX = 40, 16, 1e-3, 1.0
MAX_GAP = 3              # gaps used in training (inference bridges only gap 2)
VELOCITY_DROP = 0.2


def gt_velocities(video, t, rng):
    """{local index in frame t: (vx, vy)} from the GT position of the same track in frame t-1."""
    if t - 1 not in video.rows:
        return {}
    df = video.df
    prev = {tid: k for k, tid in enumerate(df.loc[video.rows[t - 1], "track_id"].to_numpy()) if tid < UNLABELLED_OFFSET}
    out = {}
    for i, tid in enumerate(df.loc[video.rows[t], "track_id"].to_numpy()):
        if tid in prev and rng.random() >= VELOCITY_DROP:
            out[i] = tuple(video.pos[t][i] - video.pos[t - 1][prev[tid]])
    return out


def training_graphs(wells, rng):
    graphs = []
    for w in wells:
        for flip in (None, "h", "v"):
            video = Video(*load_well(w, flip))
            for g in range(1, MAX_GAP + 1):
                for t in video.frames:
                    if t + g in video.rows and (g == 1 or t % 2 == 0):
                        graphs.append(pair_graph(video, t, g, gt_velocities(video, t, rng)))
    return graphs


def contrastive_loss(sim, src, dst, positive):
    """InfoNCE over each source's candidates and over each target's candidates (the arguments are the known edges)."""
    s = sim / TAU
    return -torch.log(softmax(s, src)[positive] + 1e-9).mean() - torch.log(softmax(s, dst)[positive] + 1e-9).mean()


def train(wells, seed=0, log=None):
    graphs = training_graphs(wells, np.random.default_rng(seed))          # kept on the CPU; a batch moves to the GPU
    y = torch.cat([g.candidate_edge_label[~g.candidate_edge_unknown] for g in graphs])
    pos_weight = torch.tensor([float((1 - y).sum() / y.sum())], device=DEVICE)
    torch.manual_seed(seed)
    model = TrackerGNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for b in DataLoader(graphs, batch_size=BATCH, shuffle=True):       # 16 graphs glued into one big graph
            b = b.to(DEVICE)
            known = ~b.candidate_edge_unknown
            label = b.candidate_edge_label[known]
            logit = model(b)
            loss = F.binary_cross_entropy_with_logits(logit[known], label, pos_weight=pos_weight)
            src, dst = b.candidate_edge_index
            loss = loss + AUX * contrastive_loss(model.sim[known], src[known], dst[known], label > 0.5)
            opt.zero_grad(); loss.backward(); opt.step()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()       # batches differ in size and fragment the cache; on Windows a full GPU
                                           # silently spills into system RAM and training becomes ~10x slower
        if log and ep in (0, 9, 19, 29, EPOCHS - 1):
            print(f"  {log}: epoch {ep + 1} / {EPOCHS} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return model.eval(), float(pos_weight)


def save(model, pos_weight, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "pos_weight": pos_weight}, path)


def cross_validate(folds=(0, 1, 2), seeds=(0, 1, 2)):
    """Every development well is scored by models that never saw it. Prints per-well and mean scores."""
    from models.gnn_tracking.track import track_video
    from utils.metrics import gt_lookup, score
    res = {}
    for k in folds:
        models = [train([w for w in DEV_WELLS if w not in FOLDS[k]], seed, log=f"fold {k} seed {seed}")[0] for seed in seeds]
        for w in FOLDS[k]:
            video = Video(*load_well(w))
            res[w] = score(track_video(models, video), gt_lookup(video))
            print(f"fold {k} {w}: perfect {res[w]['perfect']:.1f}  id switches {res[w]['id_switches']}", flush=True)
    print(f"mean over {len(res)} wells: perfect {np.mean([r['perfect'] for r in res.values()]):.2f}  "
          f"id switches {np.mean([r['id_switches'] for r in res.values()]):.1f}  purity {np.mean([r['purity'] for r in res.values()]):.3f}")
    return res


if __name__ == "__main__":
    if "--cv" in sys.argv:
        cross_validate()
    elif "--fold" in sys.argv:
        cross_validate(folds=(int(sys.argv[sys.argv.index("--fold") + 1]),), seeds=(0,))
    else:
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "gnn_tracking", "legacy", "train", "checkpoints"))
        for seed in (0, 1, 2):
            path = os.path.join(HERE, out, f"model_seed{seed}.pt")
            if os.path.exists(path):
                print(path, "exists, skipped"); continue
            model, pw = train(DEV_WELLS, seed, log=f"seed {seed}")
            save(model, pw, path)
            print("seed", seed, "done, pos_weight", round(pw, 2), flush=True)
