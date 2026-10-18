"""utils/division_metrics.py: predicted divisions from lineage.json matched to GT divisions (synthetic lineage)."""
import unittest

from utils.division_metrics import division_scores, gt_divisions, pooled, predicted_divisions
from utils.lineage import Lineage


def lineage_with_division():
    """GT: mother 1 (frames 0-2) divides into 2 and 3 (frames 3-4); track 4 runs through frames 0-4."""
    table = {1: (0, 2, 0), 2: (3, 4, 1), 3: (3, 4, 1), 4: (0, 4, 0)}
    obs = [(f, 1) for f in range(3)] + [(f, l) for f in (3, 4) for l in (2, 3)] + [(f, 4) for f in range(5)]
    return Lineage(table, obs)


def tracks(daughters=(2, 3), mother_label=1):
    """Predicted tracks: 10 = the mother (frames 0-2), 11 / 12 = the daughters, 13 = track 4."""
    rows = [(f, mother_label, 10) for f in range(3)] + [(f, 4, 13) for f in range(5)]
    rows += [(f, daughters[0], 11) for f in (3, 4)] + [(f, daughters[1], 12) for f in (3, 4)]
    return rows


class TestPredictedDivisions(unittest.TestCase):
    def test_mother_with_two_daughters_is_one_division(self):
        divisions, malformed = predicted_divisions(tracks(), {'11': 10, '12': 10})
        self.assertEqual(divisions, [dict(frame=2, mother=(2, 1), daughters=[(3, 2), (3, 3)])])
        self.assertEqual(malformed, [])

    def test_single_daughter_is_malformed(self):
        divisions, malformed = predicted_divisions(tracks(), {'11': 10})
        self.assertEqual((divisions, len(malformed)), ([], 1))


class TestDivisionScores(unittest.TestCase):
    def test_correct_division_is_true_positive(self):
        lineage = lineage_with_division()
        self.assertEqual(len(gt_divisions(lineage)), 1)
        s = division_scores(predicted_divisions(tracks(), {'11': 10, '12': 10})[0], lineage, 15)
        self.assertEqual((s['true_positives'], s['precision'], s['recall'], s['f1']), (1, 1.0, 1.0, 1.0))
        self.assertEqual(s['false_positives_per_1000_detections'], 0.0)

    def test_wrong_daughter_is_false_positive(self):
        lineage = Lineage({1: (0, 2, 0), 2: (3, 4, 1), 3: (3, 4, 1), 4: (0, 4, 0), 5: (3, 4, 0)},
                          [(f, 1) for f in range(3)] + [(f, l) for f in (3, 4) for l in (2, 3, 5)]
                          + [(f, 4) for f in range(5)])
        s = division_scores(predicted_divisions(tracks(daughters=(2, 5)), {'11': 10, '12': 10})[0], lineage, 20)
        self.assertEqual((s['true_positives'], s['false_positives'], s['recall'], s['precision']), (0, 1, 0.0, 0.0))
        self.assertEqual(s['false_positives_per_1000_detections'], 50.0)

    def test_unknown_identity_is_unverifiable(self):
        lineage = lineage_with_division()
        s = division_scores(predicted_divisions(tracks(mother_label=50001), {'11': 10, '12': 10})[0], lineage, 15)
        self.assertEqual((s['unverifiable'], s['false_positives'], s['precision']), (1, 0, None))

    def test_no_prediction_on_data_without_divisions(self):
        lineage = Lineage({4: (0, 4, 0)}, [(f, 4) for f in range(5)])
        s = division_scores([], lineage, 5)
        self.assertEqual((s['precision'], s['recall'], s['false_positives_per_1000_detections']), (None, None, 0.0))

    def test_pooled_sums_counts(self):
        lineage = lineage_with_division()
        hit = division_scores(predicted_divisions(tracks(), {'11': 10, '12': 10})[0], lineage, 15)
        miss = division_scores([], lineage, 15)
        total = pooled([hit, miss])
        self.assertEqual((total['gt_divisions'], total['true_positives'], total['recall'], total['precision']),
                         (2, 1, 0.5, 1.0))


if __name__ == '__main__':
    unittest.main()
