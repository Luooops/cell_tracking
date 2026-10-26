"""
model.py
========
gnn_tracking's TrackerGNN with a 3-way last layer: for every candidate edge the logits of UNRELATED, SAME (the same
cell) and DIVISION (mother -> daughter). Everything before the last layer is the original network.

Bayesian helpers:
  - the network is trained with plain cross-entropy (no class weights), so its softmax is the posterior under the
    training mix; it is used as it is (prior 'none'): one model, one cost-derived decision rule for every dataset,
    nothing that depends on knowing where a video comes from. The network partly recognises its data by itself.
    prior='division' (analysis only, labelled as using dataset knowledge) exchanges the division prior of the mix for
    the target dataset's: logit_division + log pi_D - log pi_mix; it is the upper bound of what a dataset's division
    rate adds. Every checkpoint keeps the class frequencies of its training edges (per gap, mix and per dataset).
  - ensemble and views: probabilities are the mean of the softmax over every member and every view of the graph
    (views: the video mirrored as in training, scored edge by edge -- a Bayesian average over nuisance orientations,
    i.e. test-time augmentation); their disagreement is the mutual information H(mean) - mean H(member).
  - binary ablation (n_classes=2): 'linked or not' (SAME + DIVISION positive); its p_same is p(link).
"""
import numpy as np
import torch
import torch.nn as nn

from models.gnn_tracking.model import DEVICE, HIDDEN, TrackerGNN

CLASSES = ('unrelated', 'same', 'division')


class TrackerGNN3(TrackerGNN):
    def __init__(self, n_classes=3):
        super().__init__()
        self.edge_mlp[-1] = nn.Linear(HIDDEN, n_classes)          # forward() then returns [C, n_classes]
        self.n_classes = n_classes                                # 2: CLASSES[:2] with division counted as unrelated
        self.priors = None


def class_priors(counts):
    """{(dataset, gap): [3 counts]} -> {'mix': {gap: [3 probs]}, 'datasets': {dataset: {gap: [3 probs]}}}, each
    smoothed with half a count per class (a class never seen keeps a small but non-zero probability)."""
    norm = lambda c: (np.asarray(c, float) + 0.5) / (np.sum(c) + 0.5 * len(c))
    mix, per = {}, {}
    for (dataset, gap), c in counts.items():
        mix[gap] = mix.get(gap, np.zeros(len(c))) + np.asarray(c, float)
        per.setdefault(dataset, {})[gap] = norm(c).tolist()
    return dict(mix={g: norm(c).tolist() for g, c in mix.items()}, datasets=per)


def save(model, path):
    torch.save({'state_dict': model.state_dict(), 'priors': model.priors, 'classes': CLASSES[:model.n_classes],
                'n_classes': model.n_classes}, path)


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = TrackerGNN3(ck.get('n_classes', 3)).to(DEVICE)
    model.load_state_dict(ck['state_dict'])
    model.priors = ck['priors']
    return model.eval()


PRIORS = ('none', 'division')     # none: the network's own posterior; division: target's division prior (analysis)


def _shift(model, gap, target, prior='none'):
    """Added to the logits: zero for 'none'; for 'division' log pi_target - log pi_mix in the division entry only
    (zero if the target was not in training)."""
    pri = model.priors
    shift = np.zeros(model.n_classes)
    if prior == 'division' and pri is not None and target in pri['datasets'] and model.n_classes > 2:
        g = min(gap, max(pri['mix']))
        shift[2] = np.log(pri['datasets'][target][g][2]) - np.log(pri['mix'][g][2])
    return shift


@torch.no_grad()
def edge_scores(models, views, gap, target, prior='none'):
    """views: [(graph, order)] -- the same frame pair seen several ways (the first is the reference; order [C] puts
    a view's candidate edges into the reference's order, None for the reference itself). -> dict(p_same [C],
    p_div [C], logits [C, M x V] (each member's logit of SAME vs the rest, in every view), mi [C]) numpy: the mean
    softmax over every member and view, prior exchanged as `prior` says. A binary ensemble has p_div = 0."""
    q = []
    for gr, order in views:
        gd = gr.to(DEVICE)
        for m in models:
            z = m(gd) + torch.as_tensor(_shift(m, gap, target, prior), dtype=torch.float32, device=DEVICE)
            out = torch.softmax(z, dim=1)
            q.append(out if order is None else out[torch.as_tensor(order, device=DEVICE)])
        gr.cpu()
    q = torch.stack(q).cpu().double().numpy()                                 # [M x V, C, n_classes]
    p = q.mean(0)
    entropy = lambda x: -(x * np.log(np.clip(x, 1e-12, 1))).sum(-1)
    mi = np.maximum(entropy(p) - entropy(q).mean(0), 0.0)
    same = np.clip(q[:, :, 1], 1e-7, 1 - 1e-7)
    return dict(p_same=p[:, 1], p_div=p[:, 2] if p.shape[1] > 2 else np.zeros(len(p)),
                logits=np.log(same / (1 - same)).T, mi=mi)
