"""Motion training and inference through the unified CLI entry points."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from train import main as train_main
from test import main as test_main


class MotionEntrypointTests(unittest.TestCase):
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
