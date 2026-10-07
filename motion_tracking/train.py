"""Train the experimental model on explicitly supervised NPZ windows.

Required keys: coords (N,3), features (N,F), parents (N,), dataset_id (scalar).
parents = predecessor index / -1 verified null / -2 unknown. No automatic XML
conversion: partial annotations need audited predecessor labels first.
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_tracking.model import MotionAssociation, association_loss
import numpy as np
import torch


def load_windows(folder):
    samples = []
    for path in sorted(folder.rglob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            coords = np.asarray(data['coords'], dtype=np.float32)
            features = np.asarray(data['features'], dtype=np.float32)
            raw_parents = np.asarray(data['parents'])
            raw_id = np.asarray(data['dataset_id'])
            if (coords.ndim != 2 or coords.shape[1] != 3 or features.ndim != 2 or
                    len(features) != len(coords) or raw_parents.shape != (len(coords),) or
                    not np.issubdtype(raw_parents.dtype, np.integer) or raw_id.ndim != 0):
                raise ValueError(f'invalid window schema: {path}')
            dataset_id = str(raw_id.item()).strip()
            if not dataset_id or not len(coords) or not (raw_parents != -2).any():
                raise ValueError(f'empty ID/window/supervision: {path}')
            if not np.isfinite(coords).all() or not np.isfinite(features).all():
                raise ValueError(f'nonfinite window: {path}')
            samples.append(dict(coords=coords, features=features,
                                parents=raw_parents.astype(np.int64), dataset_id=dataset_id))
    if not samples:
        raise ValueError(f'no NPZ training windows in {folder}')
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--max-gap', type=int, default=3)
    parser.add_argument('--distance', type=float, default=60)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-motion', action='store_true')
    parser.add_argument('--no-pair-features', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.max_tokens < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, token limit and learning rate must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty; use a new experiment directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val = load_windows(args.train_dir), load_windows(args.val_dir)
    if any(len(s['coords']) > args.max_tokens for s in train + val):
        raise ValueError('window exceeds --max-tokens; crop spatially without splitting track labels')
    train_ids, val_ids = [{s['dataset_id'] for s in samples} for samples in (train, val)]
    if train_ids & val_ids:
        raise ValueError(f'experiment IDs overlap across train/validation: {train_ids & val_ids}')
    features = np.concatenate([s['features'] for s in train], axis=0)
    if any(s['features'].shape[1] != features.shape[1] for s in train + val):
        raise ValueError('feature dimensions differ between windows')
    mean, std = features.mean(0), features.std(0).clip(min=1e-6)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = MotionAssociation(feature_dim=features.shape[1], width=args.width, layers=args.layers,
        max_gap=args.max_gap, distance=args.distance, use_motion=not args.no_motion,
        use_pair_features=not args.no_pair_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tensors = []
    for samples in (train, val):
        tensors.append([(torch.as_tensor(s['coords'], device=device),
                         torch.as_tensor((s['features'] - mean) / std, device=device),
                         torch.as_tensor(s['parents'], device=device)) for s in samples])
    # Validate gates/labels BEFORE creating outputs or taking an optimizer step.
    model.eval()
    with torch.no_grad():
        for split in tensors:
            for coords, feats, parents in split:
                association_loss(model(coords, feats), coords, parents,
                                 motion_weight=0 if args.no_motion else 0.1)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], float('inf')
    for epoch in range(1, args.epochs + 1):
        results = []
        for index, split in enumerate(tensors):
            model.train(index == 0)
            order = list(range(len(split)))
            if index == 0:
                random.shuffle(order)
            total, count = 0.0, 0
            with torch.set_grad_enabled(index == 0):
                for i in order:
                    coords, feats, parents = split[i]
                    loss = association_loss(model(coords, feats), coords, parents,
                        motion_weight=0 if args.no_motion else 0.1)['loss']
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite training loss')
                    if index == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    verified = int((parents != -2).sum())
                    total += float(loss.detach()) * verified
                    count += verified
            results.append(total / count)
        history.append(dict(epoch=epoch, train_loss=results[0], val_loss=results[1]))
        print(json.dumps(history[-1]), flush=True)
        if results[1] < best:
            best = results[1]
            torch.save(dict(config=model.config, model=model.state_dict(),
                feature_mean=torch.from_numpy(mean), feature_std=torch.from_numpy(std),
                epoch=epoch, val_loss=best, train_ids=sorted(train_ids), val_ids=sorted(val_ids)),
                args.output / 'best.pt')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
