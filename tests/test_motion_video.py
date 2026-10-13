"""Full-video contracts and conservative augmented association supervision."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import tifffile

from models.motion_tracking.augmentation import drop_detections
from models.motion_tracking.inference import track_video, video_windows, decode_global
from models.motion_tracking.model import MotionAssociation, association_loss
from test import main as test_main


class DistanceScorer(torch.nn.Module):
    """Deterministic geometry-only oracle for testing fusion, not model accuracy."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = dict(max_gap=3)

    def forward(self, coords, features):
        dt = coords[None,:,0] - coords[:,None,0]
        distance = (coords[None,:,1:] - coords[:,None,1:]).norm(dim=-1)
        valid = (dt >= 1) & (dt <= 3) & (distance <= 60*dt)
        return dict(valid=valid, logits=(18-4*dt-2*distance).masked_fill(~valid, -torch.inf),
                    null_logits=coords.new_zeros(len(coords)))


class MotionVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_30_frames_and_two_missing_detections(self):
        coords = np.array([[t, base+t*.2, 0] for t in range(30) for base in (0,40)
                           if not (base == 0 and t in (10,11))], np.float32)
        edges, ids, candidates, count = track_video(DistanceScorer(), coords, np.zeros((len(coords),7),np.float32),
            list(range(30)), np.zeros(7), np.ones(7))
        self.assertEqual(len(set(ids)),2)
        for base in (0,40):
            self.assertEqual(len({ids[i] for i,c in enumerate(coords) if abs(c[1]-(base+c[0]*.2))<.01}),1)
        self.assertTrue(any(coords[j,0]-coords[i,0] == 3 for i,j in edges))
        self.assertTrue(any(c['windows']>1 for c in candidates))
        self.assertGreater(count,1)
        self.assertEqual(len({i for i,j in edges}),len(edges))
        self.assertEqual(len({j for i,j in edges}),len(edges))

    def test_dense_window_keeps_candidate_horizon_or_fails(self):
        coords=np.array([[t,i,0] for t in range(9) for i in range(4)],np.float32)
        windows=list(video_windows(coords,list(range(9)),6,3,16))
        for target in range(9):
            self.assertTrue(any(start<=max(0,target-3) and end>=target for start,end,_ in windows))
        with self.assertRaisesRegex(ValueError,'Full candidate horizon'):
            list(video_windows(coords,list(range(9)),6,3,15))
        with self.assertRaisesRegex(ValueError,'at least'):
            list(video_windows(coords,list(range(9)),3,3,1024))

    def test_sparse_assignment_resolves_competition(self):
        coords=np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]],np.float32)
        candidates=[dict(source=i,target=j,log_odds=w,probability=.8)
                    for i,j,w in [(0,2,10),(0,3,9),(1,2,9),(1,3,1)]]
        edges,ids=decode_global(coords,candidates,.5)
        self.assertEqual(set(edges),{(0,3),(1,2)})

    def test_dropout_relinks_and_does_not_invent_birth(self):
        coords=torch.tensor([[0.,0,0],[1,1,0],[2,2,0],[3,3,0]])
        parents=torch.tensor([-2,0,1,2])
        c,f,p=drop_detections(coords,torch.zeros(4,7),parents,torch.tensor([True,False,False,True]),3,60)
        self.assertEqual(p.tolist(),[-2,0])
        c,f,p=drop_detections(coords,torch.zeros(4,7),parents,torch.tensor([True,False,False,True]),2,60)
        # No supervision remains: augmentation is safely rolled back.
        self.assertEqual(len(c),4)
        parents=torch.tensor([-1,0,1,2])
        c,f,p=drop_detections(coords,torch.zeros(4,7),parents,torch.tensor([False,True,True,True]),3,60)
        self.assertEqual(p.tolist(),[-2,0,1])

    def test_successor_loss_and_unknown_boundary(self):
        coords=torch.tensor([[0.,0,0],[1,1,0],[2,2,0]])
        model=MotionAssociation(width=16,layers=1,use_successor=True)
        output=model(coords,torch.zeros(3,7))
        loss=association_loss(output,coords,torch.tensor([-2,0,1]))
        self.assertTrue(torch.isfinite(loss['loss']))
        self.assertGreater(float(loss['successor'].detach()),0)
        loss['loss'].backward()
        self.assertIsNotNone(model.end.weight.grad)
        no_links=association_loss(output,coords,torch.tensor([-1,-2,-2]))
        self.assertEqual(float(no_links['successor']),0)

    def test_pair_mlp_only_evaluates_valid_edges(self):
        model=MotionAssociation(width=16,layers=1,use_motion=False,dropout=0).eval()
        coords=torch.tensor([[0.,0,0],[0,1000,0],[1,1,0]])
        features=torch.randn(3,7)
        seen={}
        handles=[model.source.register_forward_hook(lambda m,a,o:seen.update(source=o)),
                 model.target.register_forward_hook(lambda m,a,o:seen.update(target=o)),
                 model.pair.register_forward_pre_hook(lambda m,a:seen.update(pairs=len(a[0])))]
        output=model(coords,features)
        for handle in handles:
            handle.remove()
        delta=(coords[None,:,1:]-coords[:,None,1:])/60
        dt=coords[None,:,0]-coords[:,None,0]
        geometry=torch.cat([delta,(dt/3)[:,:,None],delta.norm(dim=-1)[:,:,None],
                            features[None]-features[:,None]],dim=-1)
        expected=seen['source']@seen['target'].T/4+model.pair(geometry).squeeze(-1)
        self.assertTrue(torch.allclose(output['logits'][output['valid']],expected[output['valid']],atol=1e-6))
        self.assertEqual(seen['pairs'],int(output['valid'].sum()))
        self.assertLess(seen['pairs'],len(coords)**2)

    def test_ctc_entrypoint_exports_full_video_without_identity_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            images=root/'data'/'well'
            masks=root/'data'/'well_GT'/'TRA'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            for t in range(30):
                mask=np.zeros((16,16),np.uint16)
                if t not in (10,11):
                    mask[3:6,3:6]=t+1  # deliberately changing identity labels
                tifffile.imwrite(images/f't{t:03d}.tif',mask.astype(np.float32)*10)
                tifffile.imwrite(masks/f'man_track{t:03d}.tif',mask)
            (masks/'man_track.txt').write_text(''.join(f'{t+1} {t} {t} 0\n' for t in range(30) if t not in (10,11)))
            checkpoint=dict(config={},model=DistanceScorer().state_dict(),feature_mean=torch.zeros(7),feature_std=torch.ones(7))
            weights=root/'weights.pt'
            torch.save(checkpoint,weights)
            with patch('models.motion_tracking.inference.module.MotionAssociation',side_effect=lambda **kwargs:DistanceScorer()):
                result=test_main(['--model','motion_tracking','--version','video','--data-root',str(root/'data'),
                    '--output-root',str(root/'out'),'--wells','well','--weights',str(weights),'--device','cpu'])
            self.assertEqual(result,0)
            dest=root/'out/data/motion_tracking/video/test/well'
            with (dest/'instance_tracks.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len({r['track_id'] for r in rows}),1)
            self.assertEqual(len(rows),28)
            self.assertEqual(len(list((dest/'visualizations').glob('*.png'))),30)
            summary=json.loads((dest/'summary.json').read_text())
            self.assertEqual(summary['n_frames'],30)
            self.assertEqual(summary['gap_edges'],1)
            evaluation=dest/'evaluation'
            self.assertEqual(len(list((evaluation/'visualizations').glob('*_comparison.png'))),30)
            metrics=json.loads((evaluation/'metrics.json').read_text())
            self.assertEqual(metrics['mixed_pred_tracks'],1)
            self.assertEqual(metrics['perfect'],0)
            self.assertFalse(metrics['official_ctc_metrics'])



if __name__=='__main__':
    unittest.main()
