"""
train.py
========
Training of the 3-way network on graphs of several datasets. Graphs, augmentation (3 mirror versions), gaps 1-3,
teacher-forced velocities, optimiser (Adam, constant LR), batch size and number of epochs are gnn_tracking/train.py's.
cosine=True decays the learning rate to 0 along a cosine instead; tested (f002): lower training loss, but the
posterior became under-confident on the uncertain edges of held-out wells (more missed links) -> not used. The loss:
    cross-entropy over UNRELATED / SAME / DIVISION for every edge with a known label (no class weights: the
    probabilities stay posteriors under the training mix, used as they are)
  + AUX x the original contrastive appearance loss on the SAME edges.
Every model keeps the class counts of its training edges per (dataset, gap) as its priors.
active_bias=True (Chang, Learned-Miller & McCallum 2017, prediction-variance weighting): every known edge keeps the
running mean / variance of its predicted class probabilities over the epochs (from the training forward passes, no
extra pass); from epoch AB_WARMUP on, its cross-entropy is weighted by w = u + mean(u), u = sqrt(sum of the class
variances). The weight uses no label, so the posterior the loss aims at is unchanged; edges the network is always
sure of (often GT errors when it is surely 'wrong') keep a low weight. Per batch w is scaled to mean 1 with no
weight above AB_CAP. Off by default.
stable=True (for training mixes with dense videos, e.g. DynamicNuclearNet, where an earlier run diverged to NaN):
the total gradient norm is clipped at CLIP, and a step whose loss or gradient norm is not finite is skipped (no update,
no Adam state change). Off by default, so the models trained so far are reproduced exactly. Every epoch logs the
median / max gradient norm (before clipping) and the skipped steps.
lr: the constant learning rate (default gnn_tracking's LR = 1e-3). 5e-4 was piloted for the scripta + RAW264.7 mix,
where the training loss rose 5-6% over the last epochs at 1e-3 (scripta + HeLa: at most 1.6%).
max_gap: the largest frame gap g of the training graphs (default MAX_GAP = 3); max_gap=1 is the ablation 'adjacent
frames only' (the gap feature is then always 0 in training, and round 2 scores g = 2 graphs the network never saw).
n_classes=2 is the binary ablation 'linked or not': SAME and DIVISION are both positive (a daughter is a link of its
mother), so gnn_division's solver can divide on two strong links; same data, augmentation and schedule.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models.gnn_tracking.graph import Video, pair_graph
from models.gnn_tracking.model import DEVICE
from models.gnn_tracking.train import AUX, BATCH, EPOCHS, LR, MAX_GAP, contrastive_loss, gt_velocities
from models.gnn_bayesian.data import SAME, UNRELATED, edge_classes, load, split
from models.gnn_bayesian.model import TrackerGNN3, class_priors

AB_WARMUP, AB_CAP = 10, 3.0                 # fixed before the experiment (not tuned)
CLIP = 5.0                                  # stable=True; fixed before the experiment: ~2x the median gradient norm of
                                            # the RAW264.7-only run (2.0-2.8), so only the outlier steps are cut


def training_graphs(specs, cache, rng, max_gap=MAX_GAP):
    graphs, counts, n_edges = [], {}, 0
    for spec in specs:
        dataset, _ = split(spec)
        for flip in (None, 'h', 'v'):
            video = Video(*load(spec, cache, flip))
            for g in range(1, max_gap + 1):
                for t in video.frames:
                    if t + g in video.rows and (g == 1 or t % 2 == 0):
                        gr = pair_graph(video, t, g, gt_velocities(video, t, rng))
                        gr.edge_class = edge_classes(video, gr, t, g)
                        gr.edge_uid = torch.arange(n_edges, n_edges + len(gr.edge_class))
                        n_edges += len(gr.edge_class)
                        known = ~gr.candidate_edge_unknown.bool()
                        c = np.bincount(gr.edge_class[known].numpy(), minlength=3)
                        counts[(dataset, g)] = counts.get((dataset, g), np.zeros(3, int)) + c
                        graphs.append(gr)
    return graphs, counts


class PredictionHistory:
    """Running mean / variance (Welford) of every training edge's predicted class probabilities over epochs."""

    def __init__(self, n_edges, n_classes):
        self.n = torch.zeros(n_edges, device=DEVICE)
        self.mean = torch.zeros(n_edges, n_classes, device=DEVICE)
        self.m2 = torch.zeros(n_edges, n_classes, device=DEVICE)

    def add(self, uid, q):
        self.n[uid] += 1
        n = self.n[uid].unsqueeze(1)
        delta = q - self.mean[uid]
        self.mean[uid] += delta / n
        self.m2[uid] += delta * (q - self.mean[uid])

    def spread(self, uid):
        """u = sqrt(sum of the class variances) per edge (0 before two epochs)."""
        n = self.n[uid].clamp(min=2).unsqueeze(1)
        return torch.sqrt((self.m2[uid] / (n - 1)).sum(1))


def active_bias_weights(u, eps, cap=AB_CAP):
    """w = u + eps scaled to mean 1 with every weight <= cap (clip and rescale until both hold)."""
    w = u + eps
    for _ in range(50):
        w = (w / w.mean()).clamp(max=cap)
        if float(w.mean()) > 1 - 1e-4:
            break
    return w


def train(specs, seed, cache, log=None, n_classes=3, active_bias=False, cosine=False, stable=False, lr=LR,
          max_gap=MAX_GAP):
    if not 1 <= max_gap <= MAX_GAP:
        raise ValueError(f'max_gap must lie in 1..{MAX_GAP}')
    if log:
        print(f'  {log}: building training graphs for {len(specs)} videos (gaps 1..{max_gap})', flush=True)
    graphs, counts = training_graphs(specs, cache, np.random.default_rng(seed), max_gap)
    torch.manual_seed(seed)
    model = TrackerGNN3(n_classes).to(DEVICE)
    merge = (lambda c: c) if n_classes == 3 else (lambda c: np.array([c[0], c[1] + c[2]]))
    model.priors = class_priors({k: merge(v).tolist() for k, v in counts.items()})
    if log:
        div = {d: int(sum(c[2] for (dd, g), c in counts.items() if dd == d)) for d in {d for d, _ in counts}}
        edges = {d: int(sum(c.sum() for (dd, g), c in counts.items() if dd == d)) for d in div}
        print(f'  {log}: {len(graphs)} graphs, known edges per dataset {edges}, division edges per dataset {div}',
              flush=True)
    history = PredictionHistory(sum(len(g.edge_class) for g in graphs), n_classes) if active_bias else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if cosine else None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        loss_sum, n_batches, norms, skipped = torch.zeros((), device=DEVICE), 0, [], 0
        weighting = active_bias and ep >= AB_WARMUP
        if weighting:
            seen = history.n >= 2
            eps = torch.sqrt((history.m2[seen] / (history.n[seen].unsqueeze(1) - 1)).sum(1)).mean()
            w_max, w_capped = 0.0, 0
        for b in DataLoader(graphs, batch_size=BATCH, shuffle=True):
            b = b.to(DEVICE)
            known = ~b.candidate_edge_unknown.bool()
            label = b.edge_class[known] if n_classes == 3 else (b.edge_class[known] != UNRELATED).long()
            logits = model(b)
            if active_bias:
                uid = b.edge_uid[known]
                if weighting:
                    w = active_bias_weights(history.spread(uid), eps)
                    w_max, w_capped = max(w_max, float(w.max())), w_capped + int((w >= AB_CAP * 0.999).sum())
                    loss = (F.cross_entropy(logits[known], label, reduction='none') * w).mean()
                else:
                    loss = F.cross_entropy(logits[known], label)
                history.add(uid, torch.softmax(logits[known].detach(), dim=1))
            else:
                loss = F.cross_entropy(logits[known], label)
            src, dst = b.candidate_edge_index
            same = b.edge_class[known] == SAME                       # appearance loss on continuations only
            loss = loss + AUX * contrastive_loss(model.sim[known], src[known], dst[known], same)
            opt.zero_grad(); loss.backward()
            if stable:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            else:                                                    # only measured: the gradients stay as they are
                norm = torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm()
            norms.append(norm.detach())
            if stable and not (torch.isfinite(loss) and torch.isfinite(norm)):
                opt.zero_grad(); skipped += 1
                continue
            opt.step()
            loss_sum += loss.detach(); n_batches += 1
        if schedule is not None:
            schedule.step()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        if log:
            elapsed = (time.time() - t0) / 60
            g = torch.stack(norms).float().cpu()
            print(f'  {log}: epoch {ep + 1} / {EPOCHS}  loss {float(loss_sum) / max(n_batches, 1):.4f}  '
                  f'grad norm median {float(g.median()):.2f} max {float(g.max()):.2f}  skipped {skipped}  '
                  f'({elapsed:.1f} min, ~{elapsed / (ep + 1) * (EPOCHS - ep - 1):.1f} min left)'
                  + (f'  active bias: eps {float(eps):.4f}, max w {w_max:.2f}, capped {w_capped}' if weighting else ''),
                  flush=True)
    return model.eval()
