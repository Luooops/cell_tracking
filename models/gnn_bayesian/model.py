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
    The retrack experiment showed the network partly recognises its dataset, so exchanging the same : unrelated
    prior counts it twice; prior='division' exchanges only the division class.
  - EM (Saerens, Latinne & Decaestecker 2002): em_prior() estimates one test video's class prior from the
    network's own posteriors, without labels. The reference is the target dataset's training prior (what the network
    already outputs, by the retrack result), or the mix for a dataset not in training.
  - binary ablation (n_classes=2): 'linked or not' (SAME + DIVISION positive); its p_same is p(link).
  - ensemble: probabilities are the mean of the members' softmax; their disagreement is the mutual information
    H(mean) - mean H(member), the 'where am I unsure' signal recorded with every edge.
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


PRIORS = ('full', 'division', 'none', 'em')    # the target's prior for all classes / only division / none;
                                                # em: the video's own prior estimated by em_prior()


def reference_prior(model, gap, target):
    """The prior the network's raw output stands for on the target: its training prior, else the mix's."""
    pri = model.priors
    g = min(gap, max(pri['mix']))
    return np.asarray(pri['datasets'][target][g] if target in pri['datasets'] else pri['mix'][g])


def _shift(model, gap, target, prior='full', video_prior=None):
    """Added to the logits. full: log pi_target - log pi_mix (zero if the target was not in training); division:
    only its division entry (same : unrelated stays the network's own); none: zero; em: log video_prior[gap] -
    log reference_prior."""
    pri = model.priors
    if prior == 'em':
        return np.log(video_prior[min(gap, max(video_prior))]) - np.log(reference_prior(model, gap, target))
    if prior == 'none' or pri is None or target not in pri['datasets']:
        return np.zeros(model.n_classes)
    g = min(gap, max(pri['mix']))
    shift = np.log(pri['datasets'][target][g]) - np.log(pri['mix'][g])
    if prior == 'division':
        shift[:2] = 0
    return shift


def em_prior(q, reference, iters=1000, tol=1e-9):
    """Saerens et al. (2002): q [E, K] posteriors under the reference prior -> the prior of these edges that is
    consistent with them (fixed point of pi = mean_e normalise(q_e * pi / reference)). No labels used."""
    q, reference = np.asarray(q, float), np.asarray(reference, float)
    pi = reference.copy()
    for _ in range(iters):
        w = q * (pi / reference)
        new = (w / w.sum(1, keepdims=True)).mean(0)
        new = np.clip(new, 1e-6, None); new /= new.sum()
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


@torch.no_grad()
def edge_scores(models, gr, gap, target, prior='full', video_prior=None):
    """-> dict(p_same [C], p_div [C], logits [C, M] (each member's logit of SAME vs the rest), mi [C]) numpy,
    with every member's prior exchanged as prior / video_prior say (_shift). A binary ensemble has p_div = 0."""
    gd = gr.to(DEVICE)
    q = []
    for m in models:
        z = m(gd) + torch.as_tensor(_shift(m, gap, target, prior, video_prior), dtype=torch.float32, device=DEVICE)
        q.append(torch.softmax(z, dim=1))
    gr.cpu()
    q = torch.stack(q).cpu().double().numpy()                                 # [M, C, n_classes]
    p = q.mean(0)
    entropy = lambda x: -(x * np.log(np.clip(x, 1e-12, 1))).sum(-1)
    mi = np.maximum(entropy(p) - entropy(q).mean(0), 0.0)
    same = np.clip(q[:, :, 1], 1e-7, 1 - 1e-7)
    return dict(p_same=p[:, 1], p_div=p[:, 2] if p.shape[1] > 2 else np.zeros(len(p)), logits=np.log(same / (1 - same)).T, mi=mi)
