"""
train.py
========
Training of the 3-way network on graphs of several datasets. Graphs, augmentation (3 mirror versions), gaps 1-3,
teacher-forced velocities, optimiser and schedule are gnn_tracking/train.py's. The loss:
    cross-entropy over UNRELATED / SAME / DIVISION for every edge with a known label (no class weights: the
    probabilities stay posteriors under the training mix; model.py exchanges the prior per dataset)
  + AUX x the original contrastive appearance loss on the SAME edges.
Every model keeps the class counts of its training edges per (dataset, gap) as its priors.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models.gnn_tracking.graph import Video, pair_graph
from models.gnn_tracking.model import DEVICE
from models.gnn_tracking.train import AUX, BATCH, EPOCHS, LR, MAX_GAP, contrastive_loss, gt_velocities
from models.gnn_bayesian.data import SAME, edge_classes, load, split
from models.gnn_bayesian.model import TrackerGNN3, class_priors


def training_graphs(specs, cache, rng):
    graphs, counts = [], {}
    for spec in specs:
        dataset, _ = split(spec)
        for flip in (None, 'h', 'v'):
            video = Video(*load(spec, cache, flip))
            for g in range(1, MAX_GAP + 1):
                for t in video.frames:
                    if t + g in video.rows and (g == 1 or t % 2 == 0):
                        gr = pair_graph(video, t, g, gt_velocities(video, t, rng))
                        gr.edge_class = edge_classes(video, gr, t, g)
                        known = ~gr.candidate_edge_unknown.bool()
                        c = np.bincount(gr.edge_class[known].numpy(), minlength=3)
                        counts[(dataset, g)] = counts.get((dataset, g), np.zeros(3, int)) + c
                        graphs.append(gr)
    return graphs, counts


def train(specs, seed, cache, log=None):
    if log:
        print(f'  {log}: building training graphs for {len(specs)} videos', flush=True)
    graphs, counts = training_graphs(specs, cache, np.random.default_rng(seed))
    torch.manual_seed(seed)
    model = TrackerGNN3().to(DEVICE)
    model.priors = class_priors({k: v.tolist() for k, v in counts.items()})
    if log:
        div = {d: int(sum(c[2] for (dd, g), c in counts.items() if dd == d)) for d in {d for d, _ in counts}}
        print(f'  {log}: {len(graphs)} graphs, division edges per dataset {div}', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        loss_sum, n_batches = torch.zeros((), device=DEVICE), 0
        for b in DataLoader(graphs, batch_size=BATCH, shuffle=True):
            b = b.to(DEVICE)
            known = ~b.candidate_edge_unknown.bool()
            label = b.edge_class[known]
            logits = model(b)
            loss = F.cross_entropy(logits[known], label)
            src, dst = b.candidate_edge_index
            loss = loss + AUX * contrastive_loss(model.sim[known], src[known], dst[known], label == SAME)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.detach(); n_batches += 1
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        if log:
            elapsed = (time.time() - t0) / 60
            print(f'  {log}: epoch {ep + 1} / {EPOCHS}  loss {float(loss_sum) / n_batches:.4f}  '
                  f'({elapsed:.1f} min, ~{elapsed / (ep + 1) * (EPOCHS - ep - 1):.1f} min left)', flush=True)
    return model.eval()
