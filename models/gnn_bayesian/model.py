"""
model.py
========
gnn_tracking's TrackerGNN with a 3-way last layer: for every candidate edge the logits of UNRELATED, SAME (the same
cell) and DIVISION (mother -> daughter). Everything before the last layer is the original network.

Bayesian helpers:
  - class priors: every checkpoint keeps the class frequencies of its training edges per gap, for the whole training
    mix and for every training dataset. A network trained on the mix scores with the mix's prior; for a video of
    dataset D the prior is exchanged for D's own (Bayes' rule: logit + log pi_D - log pi_mix). A dataset without
    divisions (scripta) thus gets a division prior near 0 and the network keeps its continuation behaviour there.
  - ensemble: probabilities are the mean of the members' softmax; their disagreement is the mutual information
    H(mean) - mean H(member), the 'where am I unsure' signal recorded with every edge.
"""
import numpy as np
import torch
import torch.nn as nn

from models.gnn_tracking.model import DEVICE, HIDDEN, TrackerGNN

CLASSES = ('unrelated', 'same', 'division')


class TrackerGNN3(TrackerGNN):
    def __init__(self):
        super().__init__()
        self.edge_mlp[-1] = nn.Linear(HIDDEN, len(CLASSES))       # forward() then returns [C, 3]
        self.priors = None


def class_priors(counts):
    """{(dataset, gap): [3 counts]} -> {'mix': {gap: [3 probs]}, 'datasets': {dataset: {gap: [3 probs]}}}, each
    smoothed with half a count per class (a class never seen keeps a small but non-zero probability)."""
    norm = lambda c: (np.asarray(c, float) + 0.5) / (np.sum(c) + 1.5)
    mix, per = {}, {}
    for (dataset, gap), c in counts.items():
        mix[gap] = mix.get(gap, np.zeros(3)) + np.asarray(c, float)
        per.setdefault(dataset, {})[gap] = norm(c).tolist()
    return dict(mix={g: norm(c).tolist() for g, c in mix.items()}, datasets=per)


def save(model, path):
    torch.save({'state_dict': model.state_dict(), 'priors': model.priors, 'classes': CLASSES}, path)


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = TrackerGNN3().to(DEVICE)
    model.load_state_dict(ck['state_dict'])
    model.priors = ck['priors']
    return model.eval()


def _shift(model, gap, target):
    """log pi_target - log pi_mix for this gap (zero if the target dataset was not in training)."""
    pri = model.priors
    if pri is None or target not in pri['datasets']:
        return np.zeros(3)
    g = min(gap, max(pri['mix']))
    return np.log(pri['datasets'][target][g]) - np.log(pri['mix'][g])


@torch.no_grad()
def edge_scores(models, gr, gap, target):
    """-> dict(p_same [C], p_div [C], logits [C, M] (each member's logit of SAME vs the rest), mi [C]) numpy,
    with every member's prior exchanged for the target dataset's."""
    gd = gr.to(DEVICE)
    q = []
    for m in models:
        z = m(gd) + torch.as_tensor(_shift(m, gap, target), dtype=torch.float32, device=DEVICE)
        q.append(torch.softmax(z, dim=1))
    gr.cpu()
    q = torch.stack(q).cpu().double().numpy()                                 # [M, C, 3]
    p = q.mean(0)
    entropy = lambda x: -(x * np.log(np.clip(x, 1e-12, 1))).sum(-1)
    mi = np.maximum(entropy(p) - entropy(q).mean(0), 0.0)
    same = np.clip(q[:, :, 1], 1e-7, 1 - 1e-7)
    return dict(p_same=p[:, 1], p_div=p[:, 2], logits=np.log(same / (1 - same)).T, mi=mi)
