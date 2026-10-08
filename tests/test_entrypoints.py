"""Small real I/O checks for the unified entry points, without downloading models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
import tifffile
from train import main as train_main
from test import main as test_main
from utils.config import ROOT


class EntrypointTests(unittest.TestCase):
    def test_ctc_classical_variants_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root/'data'/'well'
            masks = root/'data'/'well_GT'/'TRA'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            for i in range(3):
                a = np.zeros((16,16), np.uint16)
                a[4:8,4+i:8+i] = 1
                tifffile.imwrite(images/f't{i:03}.tif', a)
                tifffile.imwrite(masks/f'man_track{i:03}.tif', a)
            for model in ['classical_tracking','classical_legacy','mitosis']:
                args=['--model',model,'--version','test','--data-root',str(root/'data'),
                      '--output-root',str(root/'out'),'--wells','well']
                self.assertEqual(test_main(args),0)
                metrics=json.loads((root/'out'/model/'test'/'test'/'metrics.json').read_text())
                self.assertEqual(metrics['well']['perfect'],100)
                with self.assertRaises(SystemExit): test_main(args)
            import networkx as nx
            graph=nx.DiGraph()
            for i in range(3):
                graph.add_node(i,time=i,label=1)
                if i: graph.add_edge(i-1,i)
            fake=Mock()
            fake.track.return_value=(graph,None)
            fake.device='cpu'
            package=SimpleNamespace(__version__='test')
            api=SimpleNamespace(Trackastra=SimpleNamespace(from_pretrained=lambda *args,**kwargs:fake))
            with patch.dict('sys.modules',{'trackastra':package,'trackastra.model':api}):
                self.assertEqual(test_main(['--model','trackastra','--version','test','--data-root',str(root/'data'),
                    '--output-root',str(root/'out'),'--wells','well']),0)
            metrics=json.loads((root/'out'/'trackastra'/'test'/'test'/'metrics.json').read_text())
            self.assertEqual(metrics['well']['perfect'],100)
            with self.assertRaisesRegex(ValueError,'no training'):
                train_main(['--model','trackastra','--version','unused','--output-root',str(root/'out')])

    def test_motion_training_to_window_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for split in ['train','val']:
                (root/split).mkdir()
                np.savez(root/split/'window.npz', coords=np.array([[0,1,1],[1,2,1]],np.float32),
                         features=np.ones((2,7),np.float32), parents=np.array([-1,0]), dataset_id=split)
            args=['--model','motion_tracking','--version','one','--output-root',str(root/'out'),'--device','cpu']
            self.assertEqual(train_main(args+['--train-dir',str(root/'train'),'--val-dir',str(root/'val'),
                '--set','epochs=1','--set','width=16','--set','layers=1']),0)
            weights=root/'out'/'motion_tracking'/'one'/'train'/'checkpoints'/'best.pt'
            self.assertEqual(test_main(args+['--windows-dir',str(root/'val'),'--weights',str(weights)]),0)
            self.assertTrue((root/'out'/'motion_tracking'/'one'/'test'/'window'/'instance_tracks.csv').is_file())


if __name__ == '__main__':
    unittest.main()
