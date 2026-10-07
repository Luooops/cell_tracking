import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from motion_tracking.model import MotionAssociation, association_loss, decode_tracks
from motion_tracking.train import main as train_main


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2)
        self.coords = torch.tensor([[0., 0, 0], [0, 40, 0], [2, 4, 0], [2, 44, 0]])
        self.features = torch.randn(4, 7)
        self.parents = torch.tensor([-1, -1, 0, 1])
        self.model = MotionAssociation(width=32, layers=1, dropout=0)

    def test_probabilities_and_gradients(self):
        out = self.model(self.coords, self.features)
        self.assertTrue(torch.allclose(out['probabilities'].sum(0) + out['null_probabilities'], torch.ones(4)))
        self.assertTrue((out['probabilities'][~out['valid']] == 0).all())
        self.assertTrue(out['valid'][0, 2])  # frame 1 has no observations: cross-gap candidate
        self.assertTrue((out['null_probabilities'][:2] == 1).all())
        loss = association_loss(out, self.coords, self.parents)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_translation_and_permutation(self):
        self.model.eval()
        original = self.model(self.coords, self.features)
        shifted = self.model(self.coords + torch.tensor([7., 100, -30]), self.features)
        self.assertTrue(torch.allclose(original['probabilities'], shifted['probabilities'], atol=1e-6))
        order = torch.tensor([3, 1, 0, 2])
        permuted = self.model(self.coords[order], self.features[order])
        self.assertTrue(torch.allclose(original['probabilities'][order][:, order],
                                       permuted['probabilities'], atol=1e-6))

    def test_empty_and_isolated(self):
        out = self.model(torch.empty(0, 3), torch.empty(0, 7))
        self.assertEqual(decode_tracks(out, torch.empty(0, 3)), ([], []))
        out = self.model(self.coords[:1], self.features[:1])
        self.assertEqual(out['null_probabilities'].item(), 1)

    def test_unknown_supervision_and_bad_gate(self):
        out = self.model(self.coords, self.features)
        with self.assertRaisesRegex(ValueError, 'no verified'):
            association_loss(out, self.coords, torch.full((4,), -2))
        with self.assertRaisesRegex(ValueError, 'outside candidate'):
            association_loss(out, self.coords, torch.tensor([2, -2, -2, -2]))
        parents = torch.tensor([-2, -2, 0, -2])
        loss = association_loss(out, self.coords, parents, motion_weight=0)['loss']
        expected = -torch.log(out['probabilities'][0, 2])
        self.assertTrue(torch.allclose(loss, expected))

    def test_decoder_gap_and_competing_links(self):
        p = torch.zeros(4, 4)
        p[0, 2], p[1, 2], p[0, 3], p[1, 3] = .8, .1, .2, .7
        valid = p > 0
        edges, ids = decode_tracks(dict(probabilities=p, valid=valid,
                                        null_probabilities=1-p.sum(0)), self.coords)
        self.assertEqual(set(edges), {(0, 2), (1, 3)})
        self.assertEqual(ids[0], ids[2])
        self.assertEqual(ids[1], ids[3])
        self.assertNotEqual(ids[0], ids[1])

    def test_small_window_can_overfit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.003)
        initial = float(association_loss(self.model(self.coords, self.features), self.coords,
                                        self.parents)['association'].detach())
        for _ in range(60):
            optimizer.zero_grad()
            loss = association_loss(self.model(self.coords, self.features), self.coords, self.parents)['loss']
            loss.backward()
            optimizer.step()
        out = self.model(self.coords, self.features)
        final = float(association_loss(out, self.coords, self.parents)['association'].detach())
        self.assertLess(final, initial * .25)
        self.assertEqual(set(decode_tracks(out, self.coords)[0]), {(0, 2), (1, 3)})

    def test_training_checkpoint_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('train', 'val'):
                (root / name).mkdir()
                np.savez(root / name / 'sample.npz', coords=self.coords.numpy(),
                         features=self.features.numpy(), parents=self.parents.numpy(), dataset_id=name)
            args = ['--train-dir', str(root / 'train'), '--val-dir', str(root / 'val'),
                    '--output', str(root / 'run'), '--epochs', '2', '--width', '32',
                    '--layers', '1', '--device', 'cpu']
            self.assertEqual(train_main(args), 0)
            checkpoint = torch.load(root / 'run' / 'best.pt', weights_only=True)
            loaded = MotionAssociation(**checkpoint['config'])
            loaded.load_state_dict(checkpoint['model'])
            self.assertEqual(loaded.config, checkpoint['config'])
            np.savez(root / 'val' / 'sample.npz', coords=self.coords.numpy(),
                     features=self.features.numpy(), parents=self.parents.numpy(), dataset_id='train')
            args[args.index('--output') + 1] = str(root / 'leaky')
            with self.assertRaisesRegex(ValueError, 'IDs overlap'):
                train_main(args)
            self.assertFalse((root / 'leaky').exists())


if __name__ == '__main__':
    unittest.main()
