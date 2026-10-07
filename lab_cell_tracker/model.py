"""
model.py
========
Step 2: the network. One graph in, one logit per candidate edge out ("are these two detections the same cell?").

    crops  [N, 1, 64, 64] --AppearanceNet--> e [N, 32], unit length          (what the cell looks like)
    for a candidate edge i -> j:   sim = e_i . e_j   (cosine similarity, -1 .. 1)
                                   + softmax of sim / 0.1 over i's candidates, + over j's candidates
                                     ("is j the one that looks most like i among i's options", and the reverse)
    node input  = [geom_x (8), e (32)]            --Linear + ReLU--> h [N, 64]
    edge input  = [13 geometric features, 3 appearance features]   = 16
    h --2 x attention message passing over the message edges--> h [N, 64]
    cat(h[i], h[j], edge input) [64 + 64 + 16 = 144] --MLP--> 1 logit

Why the appearance network has its own training objective (train.py): an earlier version fed a small crop encoder
that was trained only through the edge loss, and it contributed nothing -- setting its output to zero did not change
the score. The same small encoder, trained instead with a contrastive loss that asks which candidate is the same
cell, picks the right candidate 92% of the time from appearance alone. The training signal was the problem, not
the size of the network.

An appearance embedding trained by metric learning and fed to a tracking GNN, the cosine similarity as an edge
feature, masked crops: Ben-Haim & Riklin-Raviv, "Graph Neural Network for Cell Tracking in Microscopy Videos",
ECCV 2022. Here it is trained jointly with the tracker, on the tracker's own candidate sets.

`python model.py` prints the parameter table.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

DEVICE = torch.device(os.environ.get("LAB_TRACKER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))   # set LAB_TRACKER_DEVICE=cpu to force the CPU
F_NODE, F_GEOM_EDGE, F_APP_EDGE, F_VISUAL, HIDDEN, HEADS, LAYERS = 8, 13, 3, 32, 64, 4, 2
F_EDGE = F_GEOM_EDGE + F_APP_EDGE
TAU = 0.1                 # temperature of the softmax over candidates (features here, and the contrastive loss in train.py)


class AppearanceNet(nn.Module):
    """[n, 1, 64, 64] crop -> [n, 32]. Three conv blocks (64 -> 32 -> 16 -> 8 px), the average over the 8 x 8 map, a
    linear layer. 25,600 parameters; a ten times larger net gave the same tracking result."""

    def __init__(self, dim=F_VISUAL):
        super().__init__()
        def block(a, c):
            return [nn.Conv2d(a, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, dim)

    def forward(self, x):
        return F.normalize(self.fc(self.conv(x).flatten(1)), dim=1)


class AttentionLayer(MessagePassing):
    """One round of message passing with multi-head query/key/value attention on the sparse message graph.

    For a message edge j -> i (PyG convention: x_j = sender, x_i = receiver):
        query  = Wq  h_i                       what the receiver is looking for
        key    = Wk [h_j, edge features]       what the sender (and the relation to it) offers
        value  = Wv [h_j, edge features]       what gets sent
        score  = query . key / sqrt(16)        per head: 64 dims = 4 heads x 16
        alpha  = softmax of the scores over all edges that arrive at the same receiver i
        message= alpha * value
    The receiver sums its incoming messages (aggr="add": the alphas already sum to 1, so this is a weighted
    average) and updates itself with an MLP on [its own h, the summed message]."""

    def __init__(self, dim=HIDDEN, f_edge=F_EDGE, heads=HEADS):
        super().__init__(aggr="add")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim + f_edge, dim)
        self.v_proj = nn.Linear(dim + f_edge, dim)
        self.upd_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, h, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=h, edge_attr=edge_attr)        # [N, 64]
        return self.upd_mlp(torch.cat([h, agg], dim=-1))

    def message(self, x_i, x_j, index, size_i, edge_attr):
        kv_in = torch.cat([x_j, edge_attr], dim=-1)                       # [M, 80]
        q = self.q_proj(x_i).view(-1, self.heads, self.head_dim)          # [M, 4, 16]
        k = self.k_proj(kv_in).view(-1, self.heads, self.head_dim)
        v = self.v_proj(kv_in).view(-1, self.heads, self.head_dim)
        score = (q * k).sum(-1) / self.head_dim ** 0.5                    # [M, 4]
        alpha = softmax(score, index, num_nodes=size_i)                   # normalised per receiver, per head
        return (alpha.unsqueeze(-1) * v).reshape(-1, self.dim)            # [M, 64]


class TrackerGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = AppearanceNet()
        self.enc = nn.Linear(F_NODE + F_VISUAL, HIDDEN)
        self.layers = nn.ModuleList([AttentionLayer() for _ in range(LAYERS)])
        self.edge_mlp = nn.Sequential(nn.Linear(2 * HIDDEN + F_EDGE, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def _embed_chunk(self, x):
        with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):       # half precision for the convolutions
            return self.embed(x.float()).float()

    def appearance(self, crops, chunk=1536):
        """Embeddings of all cells of a batch. A training batch of 16 graphs holds > 10,000 crops; keeping every
        activation for the backward pass would need ~19 GB. So the crops go through the net in chunks with gradient
        checkpointing: a chunk's activations are dropped after the forward pass and recomputed when its gradient is
        needed. Same network and gradients, ~1.5x the compute, 2-3 GB."""
        parts = range(0, len(crops), chunk)
        if self.training:
            e = torch.cat([checkpoint(self._embed_chunk, crops[i:i + chunk], use_reentrant=False) for i in parts])
        else:
            e = torch.cat([self._embed_chunk(crops[i:i + chunk]) for i in parts])
        return F.normalize(e, dim=1)

    def forward(self, g):
        e = self.appearance(g.crops)                                                             # [N, 32]
        src, dst = g.candidate_edge_index                                                        # [C] each
        sim = (e[src] * e[dst]).sum(1)                                                           # [C] cosine similarity
        self.sim = sim                                                                           # train.py reads it for the contrastive loss
        app = torch.stack([sim, softmax(sim / TAU, src), softmax(sim / TAU, dst)], dim=1)        # [C, 3]
        cand_attr = torch.cat([g.candidate_edge_attr, app], dim=1)                               # [C, 16]
        msg_attr = torch.cat([g.message_edge_attr, torch.zeros(g.message_edge_attr.shape[0], F_APP_EDGE, device=sim.device)], dim=1)
        msg_attr[g.msg_fwd, -F_APP_EDGE:] = app            # forward candidate rows only, like the competition / velocity features
        h = torch.relu(self.enc(torch.cat([g.geom_x, e], dim=1)))                                # [N, 64]
        for layer in self.layers:
            h = layer(h, g.message_edge_index, msg_attr)
        return self.edge_mlp(torch.cat([h[src], h[dst], cand_attr], dim=-1)).squeeze(-1)         # [C] logits


def load_model(path):
    model = TrackerGNN().to(DEVICE)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    return model.eval()


@torch.no_grad()
def edge_probabilities(models, gr):
    """Probability of every candidate edge = the mean over the ensemble members' sigmoid outputs. [C] numpy."""
    gd = gr.to(DEVICE)
    p = torch.stack([torch.sigmoid(m(gd)) for m in models]).mean(0).cpu().numpy()
    gr.cpu()
    return p


if __name__ == "__main__":
    m = TrackerGNN()
    for name, part in (("appearance net", m.embed), ("node encoder", m.enc), ("message passing", m.layers), ("edge MLP", m.edge_mlp)):
        print(f"{name:16s} {sum(p.numel() for p in part.parameters()):6d}")
    print(f"{'total':16s} {sum(p.numel() for p in m.parameters()):6d}")
