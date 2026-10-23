"""Dispatch to existing training routines. No training for fixed/pretrained trackers."""
import os
from utils.config import ctc_sequences, options, read_config, save_config
from utils.run_log import run_logged, tee_process


def main(argv=None):
    a = options('train', argv)
    return run_logged(a.log_path, train_model, a, argv=argv)


def train_model(a):
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
    if a.params.get('cv_outputs') and not a.cv:
        raise ValueError('cv_outputs=true only applies to --cv runs')
    if a.cv and not ds['folds']:
        raise ValueError(f'--cv requires "folds" for dataset {a.dataset} in the config')
    if not (a.wells or dev_wells):
        raise ValueError(f'No training wells for dataset {a.dataset}; pass --wells or declare a split in the config')
    settings = dict(data_root=str(a.data_root), cache=str(a.destination / 'cache'), folds=ds['folds'],
                    dev_wells=dev_wells, epochs=a.params['epochs'], batch=a.params['batch'], lr=a.params['lr'],
                    pos_weight=a.params.get('pos_weight', 'auto'), log_path=str(a.log_path))
    if settings['pos_weight'] != 'auto' and not (isinstance(settings['pos_weight'], (int, float)) and settings['pos_weight'] > 0):
        raise ValueError('pos_weight must be "auto" or a positive number')
    if a.cv and a.params.get('cv_outputs'):
        # Keep the fold models and track every held-out well with them (both trackers, test settings of
        # config.json, edges recorded): the out-of-fold answer sheets that calibration is fitted on.
        cfg, _ = read_config(a.config)
        settings.update(destination=str(a.destination), oof_params={
            m: {k: v for k, v in cfg['models'][m].get('test', {}).items() if k != 'record_edges'}
            for m in ('gnn_tracking', 'gnn_division')})
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
        path = _write_cv_metrics(a.destination / 'cv_metrics.json', res, ds['folds'], seeds)
        print(f'cross-validation metrics saved to {path}', flush=True)
    return 0


def _write_cv_metrics(path, res, folds, seeds):
    """Per-well CV reports with their fold, per-fold means, and the overall mean / std over folds."""
    import json
    import numpy as np
    keys = ('perfect', 'id_switches', 'purity')

    def mean(reports):
        return {k: float(np.mean(v)) if (v := [r[k] for r in reports if r.get(k) is not None]) else None
                for k in keys}

    fold_of = {w: k for k, fold in enumerate(folds) for w in fold}
    per_fold = [dict(fold=k, wells=fold, **mean([res[w] for w in fold if w in res])) for k, fold in enumerate(folds)]
    overall = dict(n_wells=len(res), **mean(res.values()))
    for k in keys:
        values = [f[k] for f in per_fold if f[k] is not None]
        overall[f'{k}_std_over_folds'] = float(np.std(values)) if len(values) > 1 else None
    report = dict(seeds=seeds, ensemble='per fold: mean edge probability of the seeds trained on the other folds',
                  metric_scope='project identity metrics (perfect / id_switches / purity), not official CTC TRA/IDF1',
                  overall=overall, folds=per_fold,
                  wells={w: dict(fold=fold_of.get(w), **r) for w, r in sorted(res.items())})
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return path


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
    trainer.POS_WEIGHT = settings.get('pos_weight', 'auto')
    results = []
    for job in jobs:
        if job[0] == 'fold':
            save_dir = os.path.join(settings['destination'], 'checkpoints') if settings.get('destination') else None
            results.append(trainer.cross_validate(folds=(job[1],), seeds=tuple(job[2]), save_dir=save_dir))
            if save_dir:
                _write_oof(settings, job[1], job[2], save_dir)
        else:
            _, seed, wells, path = job
            model, pos_weight = trainer.train(wells, seed, log=f'seed {seed}')
            trainer.save(model, pos_weight, path)
            results.append(None)
    return results


def _write_oof(settings, k, seeds, save_dir):
    """Track the held-out wells of fold k with that fold's saved models, as test.py would (gnn_tracking and
    gnn_division, edges recorded) -> <destination>/oof/<tracker>/<well>/."""
    import json
    from pathlib import Path
    from models.gnn_tracking.edge_record import EdgeRecorder
    from models.gnn_tracking.graph import Video
    from models.gnn_tracking.model import load_model
    from models.gnn_tracking.track import track_video
    from models.gnn_tracking.track_division import track_video_division
    from utils.dataset import load_well
    from utils.runners import export_gnn_well
    models = [load_model(os.path.join(save_dir, f'fold{k}_seed{seed}.pt')) for seed in seeds]
    for well in settings['folds'][k]:
        video = Video(*load_well(well))
        for name, params in settings['oof_params'].items():
            recorder, parent = EdgeRecorder(video), None
            if name == 'gnn_division':
                assign, parent = track_video_division(models, video, recorder=recorder, **params)
            else:
                assign = track_video(models, video, recorder=recorder, **params)
            folder = Path(settings['destination']) / 'oof' / name
            folder.mkdir(parents=True, exist_ok=True)
            report = export_gnn_well(folder / well, video, assign, parent, recorder)
            (folder / well / 'metrics.json').write_text(json.dumps(dict(fold=k, seeds=list(seeds), **report), indent=2),
                                                         encoding='utf-8')
            print(f"fold {k} {well} out-of-fold {name}: perfect {report['perfect']:.1f}  "
                  f"id switches {report['id_switches']}", flush=True)


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


def _gnn_worker(gpu, device, worker, settings, jobs, queue):
    # Runs in a fresh (spawned) interpreter: pin the GPU before torch is imported.
    import sys
    tee_process(settings.get('log_path'))
    sys.stdout = _Prefixed(sys.stdout, f'[GPU {gpu} worker {worker}] ')
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.environ['LAB_TRACKER_DEVICE'] = 'cuda'
    try:
        queue.put((gpu, _run_gnn_jobs(settings, jobs)))
    except BaseException as exc:
        queue.put((gpu, exc))              # never leave the parent waiting on the queue
        raise


def _run_gnn_parallel(settings, jobs, gpus, wells):
    """Independent seeds / folds on separate GPUs; each job trains exactly as in the serial run."""
    import multiprocessing as mp
    import torch
    from utils import dataset
    # --gpus are indices into the GPUs this process can see (0..n-1). Under Slurm/CUDA_VISIBLE_DEVICES they are
    # mapped back to the allocated devices, so a worker never leaves the job's allocation.
    available = torch.cuda.device_count()
    visible = [d.strip() for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
    bad = [g for g in gpus if not g.isdigit() or int(g) >= available]
    if bad:
        raise ValueError(f'--gpus {" ".join(bad)} not available: this process sees {available} GPU(s) '
                         f'(use 0..{available - 1})' + (f'; CUDA_VISIBLE_DEVICES={",".join(visible)}' if visible else ''))
    device_of = {g: visible[int(g)] if visible else g for g in gpus}
    print('GPU mapping: ' + ', '.join(f'--gpus {g} -> device {device_of[g]}' for g in dict.fromkeys(gpus)), flush=True)
    # Build the well cache once up front so the workers only read it (no concurrent writes of the same file).
    dataset.LAB_ROOT = settings['data_root']
    dataset.CACHE = settings['cache']
    cv = any(job[0] == 'fold' for job in jobs)
    for well in sorted(settings['dev_wells'] if cv else set(wells)):
        dataset.load_well(well)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    # One worker per entry; repeating an id (--gpus 0 0 1 1) runs several workers on that GPU.
    workers = [ctx.Process(target=_gnn_worker, args=(gpu, device_of[gpu], i, settings, jobs[i::len(gpus)], queue))
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
