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
        if a.gpus:
            raise ValueError('--gpus is GNN only; run separate motion experiments with CUDA_VISIBLE_DEVICES instead')
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
    if a.gpus and a.device == 'cpu':
        raise ValueError('--gpus cannot be combined with --device cpu')
    ds = a.dataset_spec
    # Well splits come from the selected dataset; the original GNN constants only describe scripta_data.
    dev_wells = ([w for fold in ds['folds'] for w in fold] or ds['train_wells'] + ds['val_wells']
                 or [w for w in ctc_sequences(a.data_root) if w not in ds['held_out']])
    if a.cv and not ds['folds']:
        raise ValueError(f'--cv requires "folds" for dataset {a.dataset} in the config')
    if not (a.wells or dev_wells):
        raise ValueError(f'No training wells for dataset {a.dataset}; pass --wells or declare a split in the config')
    settings = dict(data_root=str(a.data_root), cache=str(a.destination / 'cache'), folds=ds['folds'],
                    dev_wells=dev_wells, epochs=a.params['epochs'], batch=a.params['batch'], lr=a.params['lr'])
    seeds = list(a.params['seeds'])
    if a.cv:
        jobs = [('fold', k, seeds) for k in range(len(ds['folds']))]
    else:
        jobs = [('seed', seed, a.wells or dev_wells, str(a.destination / 'checkpoints' / f'model_seed{seed}.pt'))
                for seed in seeds]
    save_config(a)
    if not a.gpus:
        if a.device != 'auto':
            os.environ['LAB_TRACKER_DEVICE'] = a.device
        results = _run_gnn_jobs(settings, jobs)
    else:
        results = _run_gnn_parallel(settings, jobs, a.gpus, a.wells or dev_wells)
    if a.cv:
        import numpy as np
        res = {w: r for part in results for w, r in part.items()}
        print(f"all folds, mean over {len(res)} wells: perfect {np.mean([r['perfect'] for r in res.values()]):.2f}  "
              f"id switches {np.mean([r['id_switches'] for r in res.values()]):.1f}  "
              f"purity {np.mean([r['purity'] for r in res.values()]):.3f}", flush=True)
    return 0


def _run_gnn_jobs(settings, jobs):
    """Configure the original GNN trainer for this process and run its jobs in order."""
    os.environ['LAB_TRACKER_DATA'] = settings['data_root']
    os.environ['LAB_TRACKER_CACHE'] = settings['cache']
    from models.gnn_tracking import train as trainer
    from utils import dataset
    dataset.LAB_ROOT = settings['data_root']
    dataset.CACHE = settings['cache']
    trainer.FOLDS = settings['folds']
    trainer.DEV_WELLS = settings['dev_wells']
    trainer.EPOCHS, trainer.BATCH, trainer.LR = settings['epochs'], settings['batch'], settings['lr']
    results = []
    for job in jobs:
        if job[0] == 'fold':
            results.append(trainer.cross_validate(folds=(job[1],), seeds=tuple(job[2])))
        else:
            _, seed, wells, path = job
            model, pos_weight = trainer.train(wells, seed, log=f'seed {seed}')
            trainer.save(model, pos_weight, path)
            results.append(None)
    return results


class _Prefixed:
    """Line-prefixing stdout so interleaved worker logs show their GPU."""
    def __init__(self, stream, prefix):
        self.stream, self.prefix, self.at_line_start = stream, prefix, True

    def write(self, text):
        for line in text.splitlines(keepends=True):
            if self.at_line_start:
                self.stream.write(self.prefix)
            self.stream.write(line)
            self.at_line_start = line.endswith('\n')
        return len(text)

    def flush(self):
        self.stream.flush()


def _gnn_worker(gpu, worker, settings, jobs, queue):
    # Runs in a fresh (spawned) interpreter: pin the GPU before torch is imported.
    import sys
    sys.stdout = _Prefixed(sys.stdout, f'[GPU {gpu} worker {worker}] ')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['LAB_TRACKER_DEVICE'] = 'cuda'
    try:
        queue.put((gpu, _run_gnn_jobs(settings, jobs)))
    except BaseException as exc:
        queue.put((gpu, exc))              # never leave the parent waiting on the queue
        raise


def _run_gnn_parallel(settings, jobs, gpus, wells):
    """Independent seeds / folds on separate GPUs; each job trains exactly as in the serial run."""
    import multiprocessing as mp
    from utils import dataset
    # Build the well cache once up front so the workers only read it (no concurrent writes of the same file).
    dataset.LAB_ROOT = settings['data_root']
    dataset.CACHE = settings['cache']
    cv = any(job[0] == 'fold' for job in jobs)
    for well in sorted(settings['dev_wells'] if cv else set(wells)):
        dataset.load_well(well)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    # One worker per entry; repeating an id (--gpus 0 0 1 1) runs several workers on that GPU.
    workers = [ctx.Process(target=_gnn_worker, args=(gpu, i, settings, jobs[i::len(gpus)], queue))
               for i, gpu in enumerate(gpus) if jobs[i::len(gpus)]]
    for worker in workers:
        worker.start()
    results, errors = [], []
    for _ in workers:
        gpu, out = queue.get()
        if isinstance(out, BaseException):
            errors.append(f'GPU {gpu}: {out!r}')
        else:
            results.extend(out)
    for worker in workers:
        worker.join()
    if errors or any(w.exitcode for w in workers):
        raise RuntimeError('GNN training worker(s) failed: ' + ('; '.join(errors) or 'nonzero exit code'))
    return results

if __name__ == '__main__':
    raise SystemExit(main())
