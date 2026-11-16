"""
finetune.py
===========
Fine-tune Caliban (torch-tracking) from its released weights, with every training setting at torch-tracking's own
default (training.TrainingConfig: RAdam, learning rate 1e-3 with reduce-on-plateau, 50 epochs, batch 8, gradient
clip 1, weighted cross-entropy with class weights [1, 10, 100], bfloat16, best epoch by validation loss, no early
stopping). torch-tracking has no fine-tuning entry, so this is its run_training with one addition: the released
state dict is loaded into the freshly built model before the trainer is made. Only execution settings can be set
here: --seed (recorded, for reproducibility) and --num-workers (default: torch-tracking's 4; use 0 on Windows, where
every dataloader worker holds its own copy of the training arrays; workers change data loading only). Batch 8 with
the scripta movies (hundreds of cell slots each, prepare_zarr.py) does not fit a 16 GB card; it runs on a large GPU.

    python models/caliban/finetune.py --data-dir development/stage10_caliban/zarr_final \
        --checkpoint-dir development/stage10_caliban/finetune --seed 0

Writes torch-tracking's run folder (config.json, best_model.pt, per-epoch checkpoints, training_history.json) under
--checkpoint-dir, plus finetune.json (released weights and their hash, versions).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RELEASED = 'torch-tracking_2026-11-09.pt'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, required=True, help='Folder with train_proc.zarr and val_proc.zarr')
    parser.add_argument('--checkpoint-dir', type=Path, required=True, help='New directory')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-workers', type=int, help="Default: torch-tracking's TrainingConfig default")
    args = parser.parse_args(argv)
    if args.checkpoint_dir.exists():
        parser.error(f'{args.checkpoint_dir} exists; choose a new --checkpoint-dir')

    import torch
    import torch_tracking
    from huggingface_hub import hf_hub_download
    from torch_tracking.loader import create_trk_dataloaders
    from torch_tracking.training import Trainer, TrainingConfig, build_model, resolve_seed, seed_everything
    from torch_tracking.utils import DISTANCE_THRESHOLD

    released = Path(hf_hub_download(repo_id='vanvalenlab/torch-tracking', filename=RELEASED,
                                    local_dir=Path.home() / '.deepcell/models'))
    args.checkpoint_dir.mkdir(parents=True)
    execution = {} if args.num_workers is None else dict(num_workers=args.num_workers)
    config = TrainingConfig(data_dir=str(args.data_dir), checkpoint_dir=str(args.checkpoint_dir / 'runs'),
                            log_dir=str(args.checkpoint_dir / 'logs'), seed=args.seed, **execution)
    clone = Path(torch_tracking.__file__).resolve().parents[1]
    commit = subprocess.run(['git', '-C', str(clone), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout
    (args.checkpoint_dir / 'finetune.json').write_text(json.dumps(dict(
        released_weights=str(released), released_sha256=sha256(released), torch_tracking_commit=commit.strip(),
        torch=torch.__version__, config={k: str(v) for k, v in vars(config).items()}), indent=2), encoding='utf-8')

    # torch-tracking's run_training, with the released weights loaded before the trainer is built
    config.seed = resolve_seed(config.seed)
    seed_everything(config.seed)
    model = build_model(config)
    state = torch.load(released, map_location='cpu')['model_state_dict']
    model.load_state_dict(state)
    train_loader, val_loader, _ = create_trk_dataloaders(
        train_path=args.data_dir / 'train_proc.zarr', val_path=args.data_dir / 'val_proc.zarr',
        batch_size=config.batch_size, distance_threshold=DISTANCE_THRESHOLD, num_workers=config.num_workers,
        truncate_dataset=config.truncate_dataset)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    trainer.train()
    print('saved', args.checkpoint_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
