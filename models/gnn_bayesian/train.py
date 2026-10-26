"""
train.py
========
Training of the 3-way network on graphs of several datasets. Graphs, augmentation (3 mirror versions), gaps 1-3,
teacher-forced velocities, optimiser and schedule are gnn_tracking/train.py's. The loss:
    cross-entropy over UNRELATED / SAME / DIVISION for every edge with a known label (no class weights: the
    probabilities stay posteriors under the training mix; model.py exchanges the prior per dataset)
  + AUX x the original contrastive appearance loss on the SAME edges.
Every model keeps the class counts of its training edges per (dataset, gap) as its priors.
active_bias=True (Chang, Learned-Miller & McCallum 2017, prediction-variance weighting): every known edge keeps the
running mean / variance of its predicted class probabilities over the epochs (from the training forward passes, no
extra pass); from epoch AB_WARMUP on, its cross-entropy is weighted by w = u + mean(u), u = sqrt(sum of the class
variances). The weight uses no label, so the posterior the loss aims at is unchanged; edges the network is always
sure of (often GT errors when it is surely 'wrong') keep a low weight. Per batch w is scaled to mean 1 with no
weight above AB_CAP. Off by default.
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


def training_graphs(specs, cache, rng):
    graphs, counts, n_edges = [], {}, 0
    for spec in specs:
        dataset, _ = split(spec)
        for flip in (None, 'h', 'v'):
            video = Video(*load(spec, cache, flip))
            for g in range(1, MAX_GAP + 1):
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


def train(specs, seed, cache, log=None, n_classes=3, active_bias=False):
    if log:
        print(f'  {log}: building training graphs for {len(specs)} videos', flush=True)
    graphs, counts = training_graphs(specs, cache, np.random.default_rng(seed))
    torch.manual_seed(seed)
    model = TrackerGNN3(n_classes).to(DEVICE)
    merge = (lambda c: c) if n_classes == 3 else (lambda c: np.array([c[0], c[1] + c[2]]))
    model.priors = class_priors({k: merge(v).tolist() for k, v in counts.items()})
    if log:
        div = {d: int(sum(c[2] for (dd, g), c in counts.items() if dd == d)) for d in {d for d, _ in counts}}
        print(f'  {log}: {len(graphs)} graphs, division edges per dataset {div}', flush=True)
    history = PredictionHistory(sum(len(g.edge_class) for g in graphs), n_classes) if active_bias else None
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        loss_sum, n_batches = torch.zeros((), device=DEVICE), 0
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
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.detach(); n_batches += 1
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        if log:
            elapsed = (time.time() - t0) / 60
            print(f'  {log}: epoch {ep + 1} / {EPOCHS}  loss {float(loss_sum) / n_batches:.4f}  '
                  f'({elapsed:.1f} min, ~{elapsed / (ep + 1) * (EPOCHS - ep - 1):.1f} min left)'
                  + (f'  active bias: eps {float(eps):.4f}, max w {w_max:.2f}, capped {w_capped}' if weighting else ''),
                  flush=True)
    return model.eval()
