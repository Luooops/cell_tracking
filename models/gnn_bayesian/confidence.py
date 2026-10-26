"""
confidence.py
=============
How sure the tracker is of what it output, from the calibrated edge posteriors it recorded (no ground truth).

    links.csv              every link it made: frames, labels, kind (continuation / division), p (the posterior of
                           that link: p_same, or p_division for a division), mi (ensemble disagreement) and
                           uncertain (LO <= p <= HI)
    tracks_confidence.csv  every output track: frames, number of links, p_all_correct = product of its links' p (the
                           probability that none of them is wrong, links taken as independent), its weakest link and
                           the frame it leads into
    summary (metrics.json 'confidence')
                           expected false links = sum over the links made of (1 - p); expected missed links = sum of
                           p_same over the solver's edges left out; links made, uncertain links, tracks below 0.5

With calibrated posteriors the expected counts estimate the real ones (on the scripta OOF run of b001, refine2:
915 expected vs 812 actual false links, 753 vs 754 missed). Links of round 2 (bridging a missed frame) count as well.
"""
import numpy as np
import pandas as pd

LO, HI = 0.1, 0.9


def summarize(edges, tracks):
    """edges: Recorder.arrays(); tracks: instance_tracks.csv (frame_index, instance_id, track_id).
    -> (links DataFrame, tracks DataFrame, summary dict)."""
    track_of = {(int(f), int(i)): int(t) for f, i, t in zip(tracks.frame_index, tracks.instance_id, tracks.track_id)}
    chosen = edges['chosen']
    src_t = [track_of.get((int(f), int(l))) for f, l in zip(edges['src_frame'][chosen], edges['src_label'][chosen])]
    dst_f = edges['src_frame'][chosen] + edges['g'][chosen]
    dst_t = [track_of.get((int(f), int(l))) for f, l in zip(dst_f, edges['dst_label'][chosen])]
    division = np.array([a != b for a, b in zip(src_t, dst_t)], bool)
    p_same, p_div = edges['p'][chosen].astype(float), edges['p_div'][chosen].astype(float)
    has_div = bool(np.any(edges['p_div'] > 0))                    # a binary ensemble divides on p_same (p(link))
    p = np.where(division & has_div, p_div, p_same)
    links = pd.DataFrame(dict(
        round=edges['round'][chosen], src_frame=edges['src_frame'][chosen], dst_frame=dst_f,
        src_label=edges['src_label'][chosen], dst_label=edges['dst_label'][chosen], src_track=src_t, dst_track=dst_t,
        kind=np.where(division, 'division', 'continuation'), p=p.round(5), mi=edges['mi'][chosen].round(5),
        uncertain=(p >= LO) & (p <= HI)))
    rows = []
    by_track = links.groupby('dst_track')
    for t, d in tracks.groupby('track_id'):
        mine = by_track.get_group(t) if t in by_track.groups else links.iloc[:0]
        weakest = mine.loc[mine.p.idxmin()] if len(mine) else None
        rows.append(dict(track_id=int(t), first_frame=int(d.frame_index.min()), last_frame=int(d.frame_index.max()),
                         detections=len(d), links=len(mine), p_all_correct=float(np.prod(mine.p)) if len(mine) else 1.0,
                         weakest_p=float(weakest.p) if weakest is not None else None,
                         weakest_into_frame=int(weakest.dst_frame) if weakest is not None else None,
                         uncertain_links=int(mine.uncertain.sum())))
    track_table = pd.DataFrame(rows)
    left_out = edges['in_solver'] & ~chosen
    summary = dict(links=len(links), uncertain_links=int(links.uncertain.sum()),
                   expected_false_links=round(float((1 - p).sum()), 2),
                   expected_missed_links=round(float(edges['p'][left_out].astype(float).sum()), 2),
                   tracks=len(track_table), tracks_below_half=int((track_table.p_all_correct < 0.5).sum()),
                   band=[LO, HI])
    return links, track_table, summary
