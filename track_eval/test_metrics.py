import unittest
from track_eval.evaluate import track_metrics
from track_eval.metrics import evaluate_recovery


def row(t, pid, gt='A', ambiguous=False):
    return dict(sequence='s',gt_track_id=gt,time_index=t,image_name=str(t),pred_track_id=pid,
                ambiguous=ambiguous,prediction_available=True,gt_id_conflict=False)


class RecoveryTests(unittest.TestCase):
    def evaluate(self, rows):
        tracks,_=track_metrics(rows)
        metrics,*_=evaluate_recovery(rows,tracks)
        return {m['metric']:m for m in metrics},tracks

    def test_complete_and_missing(self):
        m,t=self.evaluate([row(1,'1'),row(2,'1'),row(1,'2','B'),row(2,'','B')])
        self.assertEqual(m['complete_gt_track_recovery']['value'],.5)
        self.assertEqual(m['conditional_identity_retention']['value'],1)
        self.assertEqual(m['gt_link_recovery_conservative']['value'],.5)

    def test_mixing_blocks_complete(self):
        m,t=self.evaluate([row(1,'1'),row(2,'1'),row(3,'1','B'),row(4,'1','B')])
        self.assertEqual(m['complete_gt_track_recovery']['value'],0)
        self.assertTrue(all(r['prediction_identity_mixed'] for r in t))

    def test_ambiguity_stays_in_denominator(self):
        m,t=self.evaluate([row(1,'1'),row(2,'1',ambiguous=True)])
        self.assertEqual(m['complete_gt_track_recovery']['value'],0)
        self.assertEqual(t[0]['recovery_status'],'needs_review')
        self.assertIsNone(m['conditional_identity_retention']['value'])

    def test_sparse_single_empty(self):
        m,t=self.evaluate([row(1,'1'),row(3,'1'),row(1,'2','B')])
        self.assertEqual(m['complete_gt_track_recovery']['denominator'],1)
        self.assertEqual(m['complete_gt_track_recovery']['value'],1)
        self.assertIsNone(m['gt_link_recovery_conservative']['value'])
        self.assertIsNone(self.evaluate([])[0]['complete_gt_track_recovery']['value'])


if __name__=='__main__': unittest.main()
