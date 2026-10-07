"""
metrics.py
==========
Trajectory-level scores against the ground-truth track ids.

    perfect tracks   a GT track counts if (a) all its detections carry one predicted id and (b) that predicted
                     track contains no detection of any other GT track. One wrong link anywhere fails it.
    id switches      over all GT tracks: how often the predicted id changes from one detection of the track to
                     the next.
    purity           per predicted track, the share of its (labelled) detections that come from its most common
                     GT track; averaged over predicted tracks.

A frame in which the segmentation missed the cell is not counted against the track by itself (there is nothing to
link there), but the pieces before and after must still carry the same predicted id, otherwise (a) fails.
Detections without a GT identity are ignored: they are neither a track to recover nor a foreign member.
"""
from collections import Counter, defaultdict
import numpy as np

from data import UNLABELLED_OFFSET


def gt_lookup(video):
    """{(frame index, local index): GT track id} for the detections that have one."""
    out = {}
    for fi, t in enumerate(video.frames):
        for i, tid in enumerate(video.df.loc[video.rows[t], "track_id"].to_numpy()):
            if tid < UNLABELLED_OFFSET:
                out[(fi, i)] = int(tid)
    return out


def score(assign, gt):
    by_gt = defaultdict(list)                     # GT track -> its detections in time order
    for key in sorted(gt):
        by_gt[gt[key]].append(key)
    content = defaultdict(list)                   # predicted track -> GT ids of its labelled detections
    for key, pid in assign.items():
        if key in gt:
            content[pid].append(gt[key])
    perfect = switches = 0
    for gt_id, keys in by_gt.items():
        pids = [assign[k] for k in keys]
        switches += sum(a != b for a, b in zip(pids[:-1], pids[1:]))
        perfect += len(set(pids)) == 1 and set(content[pids[0]]) == {gt_id}
    purity = [Counter(v).most_common(1)[0][1] / len(v) for v in content.values()]
    return dict(perfect=100.0 * perfect / len(by_gt), id_switches=int(switches), purity=float(np.mean(purity)),
                n_gt_tracks=len(by_gt), n_pred_tracks=len(content))
