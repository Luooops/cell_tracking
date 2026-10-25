"""
gnn_bayesian
============
The GNN tracker (models/gnn_tracking, unchanged and imported from there) with cell division and Bayesian helpers,
trained on several CTC datasets at once (e.g. scripta_data_refine + Fluo-N2DL-HeLa).

    data.py     detections of 'dataset/well' from any registered dataset, with GT cell, mother cell and the first /
                last frame of every cell (edge classes need them)
    model.py    the same network with a 3-way edge output (unrelated / same cell / division); per-dataset class
                priors and the ensemble's probabilities and disagreement (mutual information)
    solver.py   assignment with divisions decided on the division probability
    track.py    the two-round tracker of gnn_tracking with those probabilities (divisions optional)
    train.py    training on graphs of several datasets: cross-entropy over the 3 classes + the contrastive loss
    run.py      cross-validation on scripta (+ extra datasets in training) and leave-one-sequence-out on HeLa

What changes against gnn_tracking: only the last layer (1 -> 3 outputs) and the loss; network, graphs, features,
gating, matching and the two rounds are the original ones.
"""
