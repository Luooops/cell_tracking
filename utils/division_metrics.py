"""
division_metrics.py
===================
Predicted divisions against the ground-truth divisions of utils/lineage.py.

A predicted division = a predicted mother track with exactly two daughter tracks (lineage.json of gnn_division:
{daughter track id: mother track id}); its frame is the mother track's last frame. It is correct (true positive)
if the mother's last detection belongs to the GT mother cell, the two daughters' first detections belong to that
mother's two GT daughter cells, and the frames differ by at most `tolerance` (1 frame, as in CTC BC(1)). Each GT
division is matched at most once. A prediction that touches a detection without GT identity (label >= 50000 or not
in man_track.txt) cannot be checked and is counted as unverifiable, neither right nor wrong.

    false_positives_per_1000_detections   the false-division rate (on data without divisions every verifiable
                                          prediction is a false positive)
"""
from collections import defaultdict

import numpy as np


def predicted_divisions(tracks, lineage_json):
    """tracks: rows (frame, instance label, predicted track id); lineage_json: {daughter: mother} (ids as strings or
    ints). -> [dict(frame, mother=(frame, label), daughters=[(frame, label), (frame, label)])] for mothers with
    exactly two daughters; mothers with another number of daughters are returned separately as malformed."""
    first, last = {}, {}
    for frame, label, track in tracks:
        frame, label, track = int(frame), int(label), int(track)
        if track not in first or frame < first[track][0]:
            first[track] = (frame, label)
        if track not in last or frame > last[track][0]:
            last[track] = (frame, label)
    kids = defaultdict(list)
    for daughter, mother in lineage_json.items():
        kids[int(mother)].append(int(daughter))
    out, malformed = [], []
    for mother, daughters in sorted(kids.items()):
        if len(daughters) != 2 or mother not in last or any(d not in first for d in daughters):
            malformed.append(dict(mother=mother, daughters=daughters))
            continue
        out.append(dict(frame=last[mother][0], mother=last[mother], daughters=[first[d] for d in sorted(daughters)]))
    return out, malformed


def gt_divisions(lineage):
    """[dict(frame, mother_cell, daughter_cells)] for every observed GT parent with exactly two observed children."""
    out = []
    for parent, kids in lineage.children.items():
        if len(kids) == 2 and lineage.known(parent) and lineage.observed.get(parent) and \
                all(lineage.known(k) and lineage.observed.get(k) for k in kids):
            out.append(dict(frame=lineage.observed[parent][-1], mother_cell=lineage.cell_of[parent],
                            daughter_cells=frozenset(lineage.cell_of[k] for k in kids)))
    return out


def division_scores(predicted, lineage, n_detections, tolerance=1):
    truth = gt_divisions(lineage)
    unmatched = list(range(len(truth)))
    tp = unverifiable = 0
    for pred in predicted:
        labels = [pred['mother'][1]] + [d[1] for d in pred['daughters']]
        if not all(lineage.known(l) and l in lineage.cell_of for l in labels):
            unverifiable += 1
            continue
        mother, daughters = lineage.cell_of[labels[0]], frozenset(lineage.cell_of[l] for l in labels[1:])
        hit = next((i for i in unmatched if truth[i]['mother_cell'] == mother and truth[i]['daughter_cells'] == daughters
                    and abs(truth[i]['frame'] - pred['frame']) <= tolerance), None)
        if hit is not None:
            unmatched.remove(hit)
            tp += 1
    verifiable = len(predicted) - unverifiable
    fp = verifiable - tp
    precision = tp / verifiable if verifiable else None
    recall = tp / len(truth) if truth else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if truth and verifiable else None)
    return dict(gt_divisions=len(truth), predicted=len(predicted), unverifiable=unverifiable, true_positives=tp,
                false_positives=fp, false_negatives=len(truth) - tp, precision=precision, recall=recall, f1=f1,
                tolerance_frames=tolerance, detections=int(n_detections),
                false_positives_per_1000_detections=1000.0 * fp / n_detections if n_detections else None)


def pooled(scores):
    """Sum the counts of several division_scores reports and recompute the rates."""
    keys = ('gt_divisions', 'predicted', 'unverifiable', 'true_positives', 'false_positives', 'false_negatives',
            'detections')
    total = {k: int(np.sum([s[k] for s in scores])) for k in keys}
    verifiable = total['predicted'] - total['unverifiable']
    p = total['true_positives'] / verifiable if verifiable else None
    r = total['true_positives'] / total['gt_divisions'] if total['gt_divisions'] else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if total['gt_divisions'] and verifiable else None)
    return dict(total, precision=p, recall=r, f1=f1,
                false_positives_per_1000_detections=1000.0 * total['false_positives'] / total['detections']
                if total['detections'] else None)
