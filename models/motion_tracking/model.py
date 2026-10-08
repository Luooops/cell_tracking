"""Experimental 2-D detection association; independent of the Trackastra package.

One window per call. coords[:, :] = (time, x, y); features are standardized
using training-set statistics. Scores[i, j] connect source i to target j.
Dense attention is deliberate: this prototype has O(N**2) memory, not sparse attention.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ContextBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.spatial = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * width, width))

    def forward(self, x, spatial_mask, temporal_mask):
        z = self.norms[0](x)
        x = x + self.spatial(z, z, z, attn_mask=spatial_mask, need_weights=False)[0]
        z = self.norms[1](x)
        x = x + self.temporal(z, z, z, attn_mask=temporal_mask, need_weights=False)[0]
        return x + self.mlp(self.norms[2](x))


class MotionAssociation(nn.Module):
    def __init__(self, feature_dim=7, width=128, heads=4, layers=3, max_gap=3,
                 distance=60.0, dropout=0.1, use_motion=True, use_pair_features=True):
        super().__init__()
        if (feature_dim < 1 or width < 1 or heads < 1 or width % heads or
                layers < 1 or max_gap < 1 or not math.isfinite(distance) or distance <= 0 or
                not 0 <= dropout < 1):
            raise ValueError('invalid architecture or candidate-gate settings')
        self.config = dict(feature_dim=feature_dim, width=width, heads=heads, layers=layers,
            max_gap=max_gap, distance=distance, dropout=dropout, use_motion=use_motion,
            use_pair_features=use_pair_features)
        self.frequencies = nn.Linear(3, 16, bias=False)
        self.input = nn.Sequential(nn.Linear(feature_dim + 32, width), nn.LayerNorm(width))
        self.blocks = nn.ModuleList(ContextBlock(width, heads, dropout) for _ in range(layers))
        self.source = nn.Linear(width, width)
        self.target = nn.Linear(width, width)
        self.pair = nn.Sequential(nn.Linear(feature_dim + 4, width), nn.GELU(), nn.Linear(width, 1))
        self.motion = nn.Linear(width, 3)  # velocity x/y and isotropic standard deviation
        self.birth = nn.Linear(width, 1)  # no observed predecessor within this window

    def forward(self, coords, features):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError('coords must have shape (N, 3): time, x, y')
        if features.shape != (len(coords), self.config['feature_dim']):
            raise ValueError('features must have shape (N, feature_dim)')
        if not torch.isfinite(coords).all() or not torch.isfinite(features).all():
            raise ValueError('coordinates/features must be finite')
        if not torch.equal(coords[:, 0], coords[:, 0].round()):
            raise ValueError('time indices must be integers')
        n = len(coords)
        if n == 0:
            empty = coords.new_empty((0, 0))
            return dict(logits=empty, valid=empty.bool(), probabilities=empty,
                        null_logits=coords.new_empty(0), null_probabilities=coords.new_empty(0),
                        velocity=coords.new_empty((0, 2)), sigma=coords.new_empty(0))
        dt = coords[None, :, 0] - coords[:, None, 0]
        delta = (coords[None, :, 1:] - coords[:, None, 1:]) / self.config['distance']
        dist = delta.norm(dim=-1)
        valid = (dt >= 1) & (dt <= self.config['max_gap']) & (dist <= dt.clamp_min(1))
        local = coords.clone()
        local[:, 0] = (local[:, 0] - local[:, 0].min()) / self.config['max_gap']
        local[:, 1:] = (local[:, 1:] - local[:, 1:].mean(dim=0)) / self.config['distance']
        phase = self.frequencies(local)
        x = self.input(torch.cat([features, phase.sin(), phase.cos()], dim=-1)).unsqueeze(0)
        diagonal = torch.eye(n, dtype=torch.bool, device=coords.device)
        spatial = (dt == 0) & (dist <= 1)
        temporal = (dt != 0) & (dt.abs() <= self.config['max_gap']) & (dist <= dt.abs().clamp_min(1))
        for block in self.blocks:
            x = block(x, ~(spatial | diagonal), ~(temporal | diagonal))
        x = x.squeeze(0)
        logits = self.source(x) @ self.target(x).T / math.sqrt(x.shape[-1])
        if self.config['use_pair_features']:
            geometry = torch.cat([delta, (dt / self.config['max_gap']).unsqueeze(-1),
                                  dist.unsqueeze(-1), features[None, :, :] - features[:, None, :]], dim=-1)
            logits = logits + self.pair(geometry).squeeze(-1)
        prediction = self.motion(x)
        velocity = prediction[:, :2].tanh()  # displacement / distance per frame
        sigma = F.softplus(prediction[:, 2]) + 0.05
        if self.config['use_motion']:
            uncertainty = sigma[:, None] * dt.clamp_min(1).sqrt()
            residual = delta - velocity[:, None, :] * dt.unsqueeze(-1)
            logits = logits - 0.5 * (residual / uncertainty.unsqueeze(-1)).square().sum(-1)
            logits = logits - 2 * uncertainty.log()
        logits = logits.masked_fill(~valid, -torch.inf)
        null_logits = self.birth(x).squeeze(-1)
        # All candidate time gaps compete for ONE observed predecessor, plus null.
        distribution = torch.cat([logits, null_logits.unsqueeze(0)], dim=0).softmax(dim=0)
        return dict(logits=logits, valid=valid, probabilities=distribution[:-1],
                    null_logits=null_logits, null_probabilities=distribution[-1],
                    velocity=velocity * self.config['distance'], sigma=sigma * self.config['distance'])


def association_loss(output, coords, parents, motion_weight=0.1):
    """parents[j]: verified predecessor index, -1 verified null, -2 unknown/ignored.

    A positive label must name the nearest detected predecessor, not any ancestor.
    -1 means verified absence in the candidate window, not merely missing XML ID.
    """
    n = len(coords)
    if parents.shape != (n,) or parents.dtype != torch.long:
        raise ValueError('parents must be an int64 vector with N entries')
    if ((parents < -2) | (parents >= n)).any() or motion_weight < 0:
        raise ValueError('invalid parent index or motion weight')
    supervised = parents != -2
    if not supervised.any():
        raise ValueError('window has no verified supervision')
    targets = parents[supervised].clone()
    children = torch.arange(n, device=coords.device)[supervised]
    positive = targets >= 0
    if positive.any() and not output['valid'][targets[positive], children[positive]].all():
        raise ValueError('verified link is outside candidate gate')
    targets[targets == -1] = n
    scores = torch.cat([output['logits'], output['null_logits'].unsqueeze(0)], dim=0).T
    association = F.cross_entropy(scores[supervised], targets)
    motion = association.new_zeros(())
    if positive.any() and motion_weight:
        source, target = parents[children[positive]], children[positive]
        dt = coords[target, 0] - coords[source, 0]
        displacement = coords[target, 1:] - coords[source, 1:]
        residual = displacement - output['velocity'][source] * dt[:, None]
        sigma = output['sigma'][source] * dt.sqrt()
        motion = (0.5 * (residual / sigma[:, None]).square().sum(-1) + 2 * sigma.log()).mean()
    return dict(loss=association + motion_weight * motion, association=association, motion=motion)


def decode_tracks(output, coords, threshold=0.5):
    """Maximum-weight path cover without division; allows cross-gap edges.

    Each source/target is used at most once; forward-only edges ensure acyclicity.
    No synthetic detections are inserted in missing frames.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not 0 < threshold < 1:
        raise ValueError('threshold must lie strictly between 0 and 1')
    n = len(coords)
    if not n:
        return [], []
    p = output['probabilities'].detach().cpu().numpy()
    null = output['null_probabilities'].detach().cpu().numpy()
    valid = output['valid'].detach().cpu().numpy() & (p >= threshold)
    log_odds = np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(null[None, :], 1e-30))
    weights = np.where(valid & (log_odds > 0), log_odds, -1e6)
    source, target = linear_sum_assignment(-np.concatenate([weights, np.zeros((n, n))], axis=1))
    edges = [(int(i), int(j)) for i, j in zip(source, target) if j < n and weights[i, j] > 0]
    parent = {j: i for i, j in edges}
    ids, next_id = [0] * n, 1
    for j in sorted(range(n), key=lambda i: (float(coords[i, 0]), i)):
        if j in parent:
            ids[j] = ids[parent[j]]
        else:
            ids[j], next_id = next_id, next_id + 1
    return edges, ids
