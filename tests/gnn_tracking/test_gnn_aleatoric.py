"""models/gnn_tracking model.py / train.py with learned edge noise (aleatoric=True).

Needs torch_geometric (skipped where it is not installed, e.g. on machines that do not run the GNN)."""
import importlib.util
import unittest

import numpy as np

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None


@unittest.skipUnless(HAS_PYG, 'torch_geometric is not installed')
class TestAleatoric(unittest.TestCase):
    def test_loss_falls_back_to_plain_cross_entropy_without_noise(self):
        import torch
        import torch.nn.functional as F
        from models.gnn_tracking.train import aleatoric_loss
        torch.manual_seed(0)
        mu, label = torch.randn(500), (torch.rand(500) > 0.5).float()
        w = torch.tensor([3.0])
        plain = F.binary_cross_entropy_with_logits(mu, label, pos_weight=w)
        self.assertAlmostEqual(float(aleatoric_loss(mu, torch.full((500,), -10.0), label, w)), float(plain), places=3)

    def test_noise_lowers_the_loss_of_a_wrong_confident_edge_only(self):
        import torch
        from models.gnn_tracking.train import aleatoric_loss
        torch.manual_seed(0)
        one = torch.ones(1)
        wrong, right = torch.tensor([6.0]), torch.tensor([6.0])           # confident "same cell"
        quiet, noisy = torch.tensor([-10.0]), torch.tensor([3.0])
        self.assertLess(float(aleatoric_loss(wrong, noisy, 0 * one, one, 2000)), float(aleatoric_loss(wrong, quiet, 0 * one, one, 2000)))
        self.assertGreater(float(aleatoric_loss(right, noisy, one, one, 2000)), float(aleatoric_loss(right, quiet, one, one, 2000)))

    def test_expected_logit_pulls_towards_zero_and_keeps_the_sign(self):
        import torch
        from models.gnn_tracking.model import expected_logit
        mu = torch.tensor([3.0, -3.0, 3.0])
        out = expected_logit(mu, torch.tensor([-10.0, 2.0, 2.0]))
        self.assertAlmostEqual(float(out[0]), 3.0, places=3)
        self.assertTrue(0 < float(out[2]) < 3.0 and -3.0 < float(out[1]) < 0)

    def test_aleatoric_ensemble_records_noise_and_uses_the_expected_logit(self):
        import torch
        from models.gnn_tracking.edge_record import EdgeRecorder
        from models.gnn_tracking.model import DEVICE, TrackerGNN
        from models.gnn_tracking.track import track_video
        from tests.gnn_tracking.test_gnn_track_recording import synthetic_video
        torch.manual_seed(0)
        models = [TrackerGNN(aleatoric=True).to(DEVICE).eval() for _ in range(2)]
        video = synthetic_video()
        rec = EdgeRecorder(video)
        track_video(models, video, recorder=rec)
        e = rec.arrays()
        self.assertEqual(e['log_var'].shape, e['logits'].shape)
        np.testing.assert_allclose(e['p'], (1 / (1 + np.exp(-e['logits'].astype(np.float64)))).mean(1), atol=1e-5)

    def test_checkpoint_round_trip_keeps_the_noise_head(self):
        import os
        import tempfile
        import torch
        from models.gnn_tracking.model import DEVICE, TrackerGNN, load_model
        from models.gnn_tracking.train import save
        model = TrackerGNN(aleatoric=True).to(DEVICE)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'm.pt')
            save(model, 1.0, path)
            loaded = load_model(path)
        self.assertTrue(loaded.aleatoric)
        self.assertEqual(loaded.edge_mlp[-1].out_features, 2)


if __name__ == '__main__':
    unittest.main()
