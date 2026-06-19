"""
mitosis_tracker.py

Online (during-tracking) cell-division handling, implemented as a subclass of
SimpleCellTrackerV3.

It overrides update(): after the normal per-frame matching runs, it detects
divisions and fixes the lineage, covering the two cases the diagnostic found:

  BRIDGED: an existing track matched a detection whose area is about half its
    previous area, and a new track appeared right next to it whose area makes up
    the rest (child1_area + child2_area ~= parent_area). The parent had
    "absorbed" one daughter.
    Fix: split it -> finish the parent at the previous frame, turn the
    just-matched detection into a NEW child track, and link both daughters.

  CLEAN: an existing track failed to match this frame (went lost), and
    two new tracks appeared next to its last position summing to its area.
    Fix: finish the parent now and link the two new tracks as its children.

Lineage is stored in  self.parent_of = {child_track_id: parent_track_id}.

The detection is deliberately conservative so it does not disturb normal, non-dividing tracks.
"""

import itertools

from tracking import SimpleCellTrackerV3, euclidean_distance


class MitosisTracker(SimpleCellTrackerV3):
    def __init__(
        self,
        *args,
        div_min_drop: float = 1.5,   # parent_area / matched_area >= this signals a possible split
        div_max_dist: float = 30.0,  # children must be within this many px of the split point
        div_min_sum: float = 0.6,    # (child1+child2)/parent must be within ...
        div_max_sum: float = 1.6,    # ... [div_min_sum, div_max_sum]  (mass conservation)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.div_min_drop = div_min_drop
        self.div_max_dist = div_max_dist
        self.div_min_sum = div_min_sum
        self.div_max_sum = div_max_sum
        self.parent_of = {}   # child_track_id -> parent_track_id

    def update(self, detections):
        if not detections:
            super().update(detections)
            return

        frame = detections[0]["frame"]

        # snapshot each active track's pre-frame reference area and last position, before the base tracker mutates them.
        pre = {}
        for tid, trk in self.active_tracks.items():
            ref = self._get_reference_state(trk)
            last = trk["history"][-1]
            pre[tid] = {"area": ref["area"], "y": last["y"], "x": last["x"]} 

        active_before = set(self.active_tracks.keys())
        next_id_before = self.next_track_id

        # run the normal matching
        super().update(detections)

        # tracks the base matcher created this frame = candidate daughters
        new_ids = [tid for tid in self.active_tracks if tid >= next_id_before]
        used_new = set()

        # Case 1: bridged
        for tid in list(active_before):
            if tid not in self.active_tracks:
                continue
            trk = self.active_tracks[tid]
            # must have matched this frame (ignore lingering track) and have a previous detection too
            if trk["last_frame"] != frame or len(trk["history"]) < 2: 
                continue

            matched = trk["history"][-1]          # the daughter the parent absorbed
            prev_area = pre[tid]["area"]           # the parent's area before the match
            if matched["area"] <= 0:
                continue
            if prev_area / matched["area"] < self.div_min_drop:
                continue                           # area didn't reduce by the desired ratio -> not a division

            # find the best NEW sibling that completes the parent's area
            best, best_cost = None, None
            for qid in new_ids:
                if qid in used_new:
                    continue
                q = self.active_tracks[qid]["history"][-1]
                dist = euclidean_distance(matched["y"], matched["x"], q["y"], q["x"])
                if dist > self.div_max_dist:
                    continue
                ratio = (matched["area"] + q["area"]) / max(prev_area, 1e-6)
                if not (self.div_min_sum <= ratio <= self.div_max_sum):
                    continue
                cost = abs(ratio - 1.0) + dist / self.div_max_dist
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, qid
            if best is None:
                continue

            # split: parent ends at the previous frame; matched detection becomes child1
            daughter1 = trk["history"].pop()       # remove the absorbed daughter
            self._finish_track(tid)                # the parent (tid) now ends cleanly
            self._create_track(daughter1)          # child1 = a fresh track
            child1 = self.next_track_id - 1
            used_new.add(best)                     # child2 (best) = the new sibling
            self.parent_of[child1] = tid
            self.parent_of[best] = tid

        # Case 2: clean division
        for tid in list(active_before):
            if tid not in self.active_tracks:
                continue
            trk = self.active_tracks[tid]
            # must have been present last frame and not matched this frame (lost now)
            if trk["last_frame"] != frame - 1:
                continue

            py, px, parea = pre[tid]["y"], pre[tid]["x"], pre[tid]["area"]
            cand = []
            for qid in new_ids:
                if qid in used_new:
                    continue
                q = self.active_tracks[qid]["history"][-1]
                dist = euclidean_distance(py, px, q["y"], q["x"])
                if dist <= self.div_max_dist:
                    cand.append((qid, q, dist))
            if len(cand) < 2:
                continue

            # pick the best pair of new tracks summing to the parent's area
            best, best_cost = None, None
            for (i1, d1, dd1), (i2, d2, dd2) in itertools.combinations(cand, 2):
                ratio = (d1["area"] + d2["area"]) / max(parea, 1e-6)
                if not (self.div_min_sum <= ratio <= self.div_max_sum):
                    continue
                cost = abs(ratio - 1.0) + (dd1 + dd2) / (2 * self.div_max_dist)
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, (i1, i2)
            if best is None:
                continue

            c1, c2 = best
            used_new.add(c1)
            used_new.add(c2)
            self._finish_track(tid)                # parent ends now
            self.parent_of[c1] = tid
            self.parent_of[c2] = tid
