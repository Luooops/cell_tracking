import unittest
from track_eval.evaluate import match_positions, track_metrics, centroid
import numpy as np


class EvaluationTests(unittest.TestCase):
    def test_gated_assignment_maximizes_valid_matches(self):
        matches, ambiguity, _ = match_positions([(0, 0), (3, 0)], [(1, 0), (-2, 0)], 2.1)
        self.assertEqual({i: p[0] for i, p in matches.items()}, {0: 1, 1: 0})
        self.assertTrue(ambiguity.all())

    def test_empty_and_far(self):
        self.assertEqual(match_positions([(0, 0)], [], 1)[0], {})
        self.assertEqual(match_positions([(0, 0)], [(5, 5)], 1)[0], {})

    def test_missing_and_ambiguous_do_not_bridge_clean_pairs(self):
        rows = [dict(sequence="s", gt_track_id="8", image_name=str(t), time_index=t,
                     pred_track_id=pid, prediction_available=True, ambiguous=amb)
                for t, pid, amb in [(1, "23", False), (2, "23", False), (3, "", False), (4, "41", False), (5, "42", True)]]
        metrics, events = track_metrics(rows)
        self.assertEqual(metrics[0]["matched_fragments"], 2)
        self.assertEqual(metrics[0]["id_changes_between_matched_observations"], 2)
        self.assertEqual(metrics[0]["unambiguous_pairs"], 1)
        self.assertEqual(metrics[0]["unambiguous_id_changes"], 0)
        self.assertEqual(len(events), 4)

    def test_polygon_centroid(self):
        np.testing.assert_allclose(centroid(np.array([(0, 0), (4, 0), (4, 2), (0, 2)])), [2, 1])

    def test_conflicting_gt_identity_excluded(self):
        metrics, events = track_metrics([dict(sequence="s", gt_track_id="1", gt_id_conflict=True)])
        self.assertEqual(metrics, [])
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
