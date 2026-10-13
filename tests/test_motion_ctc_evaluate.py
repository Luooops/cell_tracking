"""Motion CTC evaluation: partial labels, gap identity, mixing and saved-result CLI."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from utils.ctc import write_csv
from utils.ctc_evaluate import evaluate_well, main


class MotionCTCEvaluationTests(unittest.TestCase):
    def make_data(self, root):
        images=root/'data/well'
        masks=root/'data/well_GT/TRA'
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        rows=[]
        for t in range(4):
            mask=np.zeros((64,64),np.uint16)
            if t!=1:
                mask[35:40,10:15]=1
                rows.append(dict(frame_index=t,instance_id=1,track_id=103 if t==3 else 101))
            mask[35:40,40:45]=2
            rows.append(dict(frame_index=t,instance_id=2,track_id=202))
            mask[50:54,25:29]=50000+t
            rows.append(dict(frame_index=t,instance_id=50000+t,track_id=101 if t==1 else 999))
            tifffile.imwrite(images/f't{t:03d}.tif',mask.astype(np.float32))
            tifffile.imwrite(masks/f'man_track{t:03d}.tif',mask)
        (masks/'man_track.txt').write_text('1 0 3 0\n2 0 3 0\n'+''.join(f'{50000+t} {t} {t} 0\n' for t in range(4)))
        pred=root/'pred/well'
        pred.mkdir(parents=True)
        write_csv(pred/'instance_tracks.csv',rows,['frame_index','instance_id','track_id'])
        return pred,rows

    def test_gap_gt_ids_and_prediction_ids_need_not_equal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pred,rows=self.make_data(root)
            report=evaluate_well(root/'data','well',pred,root/'evaluation')
            self.assertEqual(report['perfect'],50)
            self.assertEqual(report['id_switches'],1)
            self.assertEqual(report['unknown_observations'],4)
            self.assertEqual(report['mixed_pred_tracks'],0)
            self.assertEqual(report['identity_link_recall_by_dt']['2']['recall'],1)
            self.assertEqual(report['gt_observations'],7)
            self.assertEqual(report['observation_coverage'],1)
            self.assertFalse(report['official_ctc_metrics'])
            self.assertEqual(len(list((root/'evaluation/visualizations').glob('*_comparison.png'))),4)
            self.assertEqual(main(['--data-root',str(root/'data'),'--tracking-output',str(root/'pred'),
                '--output-root',str(root/'rescore'),'--visualize','none']),0)
            saved=json.loads((root/'rescore/well/metrics.json').read_text())
            self.assertEqual(saved,report)
            for suffix in ('.pdf','.svg','.png','_legend.txt','_source_data.csv'):
                self.assertTrue((root/f'rescore/figures/metrics_summary{suffix}').is_file())

    def test_compare_figure_from_saved_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.make_data(root)
            for name in ('one','two'):
                self.assertEqual(main(['--data-root',str(root/'data'),'--tracking-output',str(root/'pred'),
                    '--output-root',str(root/name),'--visualize','none','--no-plots']),0)
            self.assertFalse((root/'one/figures').exists())
            self.assertEqual(main(['--output-root',str(root/'cmp'),'--compare',str(root/'one'),str(root/'two'),
                '--labels','A','B']),0)
            self.assertTrue((root/'cmp/figures/metrics_comparison.pdf').is_file())
            data=(root/'cmp/figures/metrics_comparison_source_data.csv').read_text(encoding='utf-8')
            self.assertIn('Perfect tracks',data)
            self.assertIn('A',data)
            with self.assertRaises(SystemExit):
                main(['--output-root',str(root/'cmp2'),'--compare',str(root/'one'),'--labels','A','B'])

    def test_missing_prediction_reduces_coverage_and_track_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pred,rows=self.make_data(root)
            rows=[r for r in rows if not (r['frame_index']==2 and r['instance_id']==1)]
            write_csv(pred/'instance_tracks.csv',rows,['frame_index','instance_id','track_id'])
            report=evaluate_well(root/'data','well',pred,root/'evaluation',False)
            self.assertEqual(report['status'],'partial')
            self.assertAlmostEqual(report['observation_coverage'],6/7)
            self.assertEqual(report['perfect'],50)
            self.assertEqual(report['identity_link_recall_by_dt']['2']['recall'],0)

    def test_mixing_and_duplicate_mapping_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pred,rows=self.make_data(root)
            for row in rows:
                if row['frame_index']==1 and row['instance_id']==2:
                    row['track_id']=101
                if row['frame_index']==1 and row['instance_id']>=50000:
                    row['track_id']=999
            write_csv(pred/'instance_tracks.csv',rows,['frame_index','instance_id','track_id'])
            report=evaluate_well(root/'data','well',pred,root/'evaluation',False)
            self.assertEqual(report['mixed_pred_tracks'],1)
            self.assertEqual(report['perfect'],0)
            write_csv(pred/'instance_tracks.csv',rows+[rows[0]],['frame_index','instance_id','track_id'])
            with self.assertRaisesRegex(ValueError,'duplicate prediction'):
                evaluate_well(root/'data','well',pred,root/'bad',False)

    def test_gt_table_is_required_and_unknown_only_is_not_perfect(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            pred,rows=self.make_data(root)
            table=root/'data/well_GT/TRA/man_track.txt'
            table.write_text('1 0 2 0\n2 0 3 0\n')
            with self.assertRaisesRegex(ValueError,'outside man_track.txt'):
                evaluate_well(root/'data','well',pred,root/'bad',False)
            for path in (root/'data/well_GT/TRA').glob('man_track*.tif'):
                mask=tifffile.imread(path)
                mask[(mask>0)&(mask<50000)]=0
                tifffile.imwrite(path,mask)
            table.write_text(''.join(f'{50000+t} {t} {t} 0\n' for t in range(4)))
            write_csv(pred/'instance_tracks.csv',[r for r in rows if r['instance_id']>=50000],
                ['frame_index','instance_id','track_id'])
            report=evaluate_well(root/'data','well',pred,root/'unknown',False)
            self.assertIsNone(report['perfect'])
            self.assertIsNone(report['purity'])
            self.assertEqual(report['n_gt_tracks'],0)
            self.assertEqual(report['status'],'no_labelled_detections')


if __name__=='__main__':
    unittest.main()
