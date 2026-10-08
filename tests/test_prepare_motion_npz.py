"""Verify temporal boundaries, unknown identities, gate exclusions and NPZ compatibility."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tifffile

from utils.prepare_motion_npz import main, previous_detections, window_parents, windows


class MotionPreparationTests(unittest.TestCase):
    def test_parents_gaps_and_window_boundary(self):
        coords = np.array([[0,0,0],[1,1,0],[3,3,0],[4,4,0]], dtype=np.float32)
        labels = np.ones(4, dtype=np.int64)
        previous = previous_detections(coords, labels)
        parents, _ = window_parents(coords, labels, previous, np.arange(4), 0, 3, 60)
        np.testing.assert_array_equal(parents, [-2,0,1,2])
        parents, _ = window_parents(coords, labels, previous, np.array([2,3]), 3, 3, 60)
        np.testing.assert_array_equal(parents, [-2,0])

    def test_unknown_and_out_of_gate_are_not_null(self):
        coords = np.array([[0,0,0],[1,1,0],[2,2,0],[3,1000,0],[4,500,0]], dtype=np.float32)
        labels = np.array([1,50000,1,1,2])
        parents, reasons = window_parents(coords, labels, previous_detections(coords,labels), np.arange(5),0,3,60)
        self.assertEqual(parents[1],-2)
        self.assertEqual(parents[2],-2)
        self.assertEqual(parents[3],-2)
        self.assertEqual(parents[4],-1)
        self.assertEqual(reasons['ambiguous_unknown_predecessor'],1)
        self.assertEqual(reasons['positive_outside_distance_gate'],1)

    def test_whole_frame_adaptation_and_overflow(self):
        coords = np.array([[t,i,0] for t in range(6) for i in range(4)],dtype=np.float32)
        result = list(windows(coords,list(range(6)),6,3,9))
        self.assertEqual({int(coords[i,0]) for _,_,selected in result for i in selected},set(range(6)))
        self.assertTrue(all(len(selected)==8 for _,_,selected in result))
        with self.assertRaisesRegex(ValueError,'increase --max-tokens'):
            list(windows(coords,list(range(6)),6,3,7))

    def test_ctc_to_original_training_contract(self):
        import torch
        from models.motion_tracking.train import load_windows
        from models.motion_tracking.model import MotionAssociation, association_loss
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for well in ['well_a','well_b']:
                images=root/'data'/well
                masks=root/'data'/f'{well}_GT'/'TRA'
                images.mkdir(parents=True)
                masks.mkdir(parents=True)
                (masks/'man_track.txt').write_text('1 0 4 0\n')
                for t in range(5):
                    mask=np.zeros((16,16),np.uint16)
                    mask[4:7,3+t:6+t]=1
                    tifffile.imwrite(images/f't{t:03d}.tif',mask*100)
                    tifffile.imwrite(masks/f'man_track{t:03d}.tif',mask)
            args=['--data-root',str(root/'data'),'--output-root',str(root/'npz'),
                  '--train-wells','well_a','--val-wells','well_b']
            self.assertEqual(main(args),0)
            train,val=load_windows(root/'npz/train'),load_windows(root/'npz/val')
            self.assertNotEqual(train[0]['dataset_id'],val[0]['dataset_id'])
            sample=train[0]
            coords=torch.from_numpy(sample['coords'])
            feats=torch.from_numpy(sample['features'])
            parents=torch.from_numpy(sample['parents'])
            model=MotionAssociation(width=16,layers=1)
            loss=association_loss(model(coords,feats),coords,parents)['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            with self.assertRaises(SystemExit): main(args)


if __name__ == '__main__':
    unittest.main()
