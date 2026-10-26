"""
gnn_bayesian
============
The GNN tracker (models/gnn_tracking, unchanged and imported from there) with cell division and Bayesian helpers,
trained on several CTC datasets at once (scripta_data_refine2 + Fluo-N2DL-HeLa). One model for every dataset: the
network's own posterior, cost-derived thresholds, nothing at inference depends on the dataset.

    data.py     detections of 'dataset/well' from any registered dataset, with GT cell, mother cell and the first /
                last frame of every cell (edge classes need them)
    model.py    the same network with a 3-way edge output (unrelated / same cell / division); the probabilities
                of an ensemble over several views of a graph and their disagreement (mutual information)
    solver.py   assignment with divisions decided on the division probability
    track.py    the two-round tracker of gnn_tracking with those probabilities (divisions optional), test-time
                mirrored views, cue ablations (appearance / velocity) for the cue analysis
    confidence.py  how sure the tracker is of each output link and track, expected numbers of errors
    train.py    training on graphs of several datasets: cross-entropy over the 3 classes + the contrastive loss
                (optional Active Bias weighting, binary ablation)
    run.py      cross-validation on scripta (+ extra datasets in training), leave-one-sequence-out on HeLa, and
                the final run on the test wells

What changes against gnn_tracking: only the last layer (1 -> 3 outputs) and the loss; network, graphs, features,
gating, matching and the two rounds are the original ones.
"""
