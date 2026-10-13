"""Dispatch to existing training routines. No training for fixed/pretrained trackers."""
import os
from utils.config import ctc_sequences, options, save_config


def main(argv=None):
    a = options('train', argv)
    if a.model not in {'gnn_tracking', 'motion_tracking'}:
        raise ValueError(f'{a.model} has no training entry (Trackastra fine-tuning is not included)')
    if a.model == 'motion_tracking':
        if not a.train_dir or not a.val_dir:
            raise ValueError('Motion requires --train-dir and --val-dir with NPZ windows. Generate lab CTC windows with python -m models.motion_tracking.prepare_npz --output-root <new directory>.')
        from models.motion_tracking.train import main as train
        args = ['--train-dir', *map(str, a.train_dir), '--val-dir', *map(str, a.val_dir),
                '--output', str(a.destination / 'checkpoints'), '--device', a.device]
        for key, value in a.params.items():
            if isinstance(value, bool):
                if value:
                    args.append('--' + key.replace('_', '-'))
            else:
                args.extend(['--' + key.replace('_', '-'), str(value)])
        save_config(a)
        return train(args)
    os.environ['LAB_TRACKER_DATA'] = str(a.data_root)
    os.environ['LAB_TRACKER_CACHE'] = str(a.destination / 'cache')
    if a.device != 'auto':
        os.environ['LAB_TRACKER_DEVICE'] = a.device
    from models.gnn_tracking import train as trainer
    from utils import dataset
    dataset.LAB_ROOT = str(a.data_root)
    dataset.CACHE = str(a.destination / 'cache')
    # Well splits come from the selected dataset; the original GNN constants only describe scripta_data.
    ds = a.dataset_spec
    trainer.FOLDS = ds['folds']
    trainer.DEV_WELLS = ([w for fold in ds['folds'] for w in fold] or ds['train_wells'] + ds['val_wells']
                         or [w for w in ctc_sequences(a.data_root) if w not in ds['held_out']])
    if a.cv and not ds['folds']:
        raise ValueError(f'--cv requires "folds" for dataset {a.dataset} in the config')
    if not (a.wells or trainer.DEV_WELLS):
        raise ValueError(f'No training wells for dataset {a.dataset}; pass --wells or declare a split in the config')
    trainer.EPOCHS = a.params['epochs']
    trainer.BATCH = a.params['batch']
    trainer.LR = a.params['lr']
    save_config(a)
    if a.cv:
        trainer.cross_validate(folds=tuple(range(len(ds['folds']))), seeds=tuple(a.params['seeds']))
    else:
        for seed in a.params['seeds']:
            model, pos_weight = trainer.train(a.wells or trainer.DEV_WELLS, seed, log=f'seed {seed}')
            trainer.save(model, pos_weight, str(a.destination / 'checkpoints' / f'model_seed{seed}.pt'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
