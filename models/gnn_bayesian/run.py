"""
run.py
======
Experiments of gnn_bayesian (training on server GPUs; evaluation with gnn_tracking's evaluate_answers / evaluate_ctc).

    cv    3-fold cross-validation on the scripta dataset, every fold trained on its 8 training wells PLUS every
          sequence of the extra datasets; the held-out scripta wells are tracked with gnn_tracking (no divisions) and
          gnn_division (divisions on p_division), scored with scripta's own priors -> <out>/oof/<tracker>/<well>/.
          Compare with the scripta-only baseline (cv_v003, thresholds 0.4) to check scripta does not get worse.
    hela  leave-one-sequence-out on an extra dataset: trained on every scripta development well + the other
          sequences of that dataset, the held-out sequence tracked with that dataset's priors
          -> <out>/<dataset>/<tracker>/<sequence>/.

    cache fill <out>/cache with every video used (run it once before starting several processes on one root).

--retrack none|division|full (cv / hela) trains nothing: it loads the ensembles in <out>/checkpoints and tracks the
same held-out videos again with that prior exchange (model.PRIORS: every class / only division / none) into
<out>/retrack_<prior>/... (oof/<tracker>/<well>/ or <dataset>/<tracker>/<sequence>/); existing folders are refused.

--gpus 0 1 fills the cache, then splits the folds (cv) or held-out sequences (hela) round-robin over the GPUs, one
child process per GPU (each with its own log). Several processes may also share one --output-root by hand
(--folds / --held-out); a fold or held-out sequence whose checkpoints exist is refused.

    python -m models.gnn_bayesian.run cv --gpus 0 1 --output-root development/stage2_bayesian/cv_b001
    python -m models.gnn_bayesian.run hela --gpus 0 1 --output-root development/stage2_bayesian/hela_b001
    python -m models.gnn_bayesian.run cv --retrack none --output-root development/stage2_bayesian/cv_b001
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from utils.config import ctc_sequences, read_config, resolve_dataset
from utils.run_log import log_path, log_root, run_logged

TRACKERS = {'gnn_tracking': False, 'gnn_division': True}


def _track_and_export(models, spec, target, cache, folder_of, extra_report, prior='full'):
    from models.gnn_tracking.graph import Video
    from models.gnn_bayesian.data import load, split
    from models.gnn_bayesian.track import Recorder, track_video
    from utils.runners import export_gnn_well
    video = Video(*load(spec, cache))
    _, well = split(spec)
    taken = [folder_of(name) / well for name in TRACKERS if (folder_of(name) / well).exists()]
    if taken:
        raise FileExistsError(f'already tracked: {taken}')
    for name, divisions in TRACKERS.items():
        recorder = Recorder(video)
        assign, parent = track_video(models, video, target, divisions=divisions, recorder=recorder, prior=prior)
        folder = folder_of(name)
        folder.mkdir(parents=True, exist_ok=True)
        report = export_gnn_well(folder / well, video, assign, parent if divisions else None, recorder)
        (folder / well / 'metrics.json').write_text(json.dumps(dict(**extra_report, prior=prior, **report), indent=2),
                                                   encoding='utf-8')
        print(f"{spec} {name}: perfect {report['perfect']:.1f}  id switches {report['id_switches']}"
              + (f"  divisions {len(parent)}" if divisions else ''), flush=True)


def _train_ensemble(specs, seeds, cache, paths, label):
    from models.gnn_bayesian.model import save
    from models.gnn_bayesian.train import train
    models = []
    for seed, path in zip(seeds, paths):
        model = train(specs, seed, cache, log=f'{label} seed {seed}')
        path.parent.mkdir(parents=True, exist_ok=True)
        save(model, path)
        models.append(model)
    return models


def _ensemble(args, specs, cache, paths, label):
    """Train the ensemble (refusing existing checkpoints) or, with --retrack, load it."""
    if args.retrack:
        from models.gnn_bayesian.model import load_model
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f'{label}: no checkpoints {missing}')
        return [load_model(p) for p in paths]
    if any(p.exists() for p in paths):
        raise FileExistsError(f'{label} already has checkpoints in {paths[0].parent}')
    return _train_ensemble(specs, args.seeds, cache, paths, label)


def _fill_cache(args, cache, extra):
    from models.gnn_bayesian.data import load
    for spec in [f'{args.scripta}/{w}' for f in args.scripta_spec['folds'] for w in f] + extra:
        load(spec, cache)


def _split_over_gpus(args):
    """One child process per GPU, each given its share of the folds / held-out sequences; waits for all."""
    key, items = ('--folds', args.folds) if args.command == 'cv' else ('--held-out', args.held_out)
    common = [args.command, '--scripta', args.scripta, '--extra', *args.extra, '--seeds', *map(str, args.seeds),
              '--device', args.device, '--output-root', str(args.output_root)] + (['--retrack', args.retrack] if args.retrack else [])
    children = []
    for i, gpu in enumerate(args.gpus):
        share = items[i::len(args.gpus)]
        if share:
            cmd = [sys.executable, '-m', 'models.gnn_bayesian.run', *common, key, *map(str, share)]
            print(f'GPU {gpu}: {key} {" ".join(map(str, share))}', flush=True)
            children.append((gpu, subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})))
    failed = [gpu for gpu, child in children if child.wait() != 0]
    if failed:
        raise RuntimeError(f'child processes on GPUs {failed} failed (see their logs)')


def run(args):
    if args.device != 'auto':
        os.environ['LAB_TRACKER_DEVICE'] = args.device
    out, cache = args.output_root, args.output_root / 'cache'
    out.mkdir(parents=True, exist_ok=True)
    dest, prior = (out / f'retrack_{args.retrack}', args.retrack) if args.retrack else (out, 'full')
    scripta = args.scripta_spec
    extra = [f'{d}/{s}' for d in args.extra for s in ctc_sequences(args.extra_roots[d])]
    config = out / 'config.json'
    if not config.exists():
        config.write_text(json.dumps(dict(
            model='gnn_bayesian', scripta=args.scripta, data_root=str(scripta['root']), extra=args.extra, extra_videos=extra,
            seeds=args.seeds, folds=scripta['folds'], created=datetime.datetime.now().isoformat(timespec='seconds'),
            scope='3-way edges (unrelated / same / division) trained on scripta + extra datasets; per-dataset class priors'),
            indent=2), encoding='utf-8')
    if args.command == 'cache':
        _fill_cache(args, cache, extra)
    elif args.gpus:
        _fill_cache(args, cache, extra)
        _split_over_gpus(args)
    elif args.command == 'cv':
        for k in args.folds:
            paths = [out / 'checkpoints' / f'fold{k}_seed{s}.pt' for s in args.seeds]
            fold = scripta['folds'][k]
            specs = [f'{args.scripta}/{w}' for f in scripta['folds'] for w in f if w not in fold] + extra
            models = _ensemble(args, specs, cache, paths, f'fold {k}')
            for w in fold:
                _track_and_export(models, f'{args.scripta}/{w}', args.scripta, cache,
                                  lambda name: dest / 'oof' / name, dict(fold=k, seeds=args.seeds), prior)
    else:
        dataset = args.extra[0]
        for held in args.held_out:
            paths = [out / 'checkpoints' / f'{dataset}_{held}_seed{s}.pt' for s in args.seeds]
            specs = ([f'{args.scripta}/{w}' for f in scripta['folds'] for w in f]
                     + [s for s in extra if s != f'{dataset}/{held}'])
            models = _ensemble(args, specs, cache, paths, f'{dataset} without {held}')
            _track_and_export(models, f'{dataset}/{held}', dataset, cache,
                              lambda name: dest / dataset / name, dict(held_out=held, seeds=args.seeds), prior)
    print('saved', dest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['cache', 'cv', 'hela'])
    parser.add_argument('--scripta', default='scripta_data_refine', help='Dataset with folds (cross-validated)')
    parser.add_argument('--extra', nargs='+', default=['Fluo-N2DL-HeLa'], help='Datasets added to every training')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--held-out', nargs='+', default=['01', '02'], help='hela: sequences of the first extra dataset')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--gpus', nargs='+', type=int, help='Split the folds / held-out sequences over these GPUs')
    parser.add_argument('--retrack', choices=['none', 'division', 'full'],
                        help='Load the checkpoints and track again with this prior exchange (no training)')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg, base = read_config()
    args.scripta_spec = resolve_dataset(cfg, base, args.scripta, None)
    if not args.scripta_spec['folds']:
        parser.error(f'{args.scripta} has no folds in config.json')
    args.extra_roots = {d: resolve_dataset(cfg, base, d, None)['root'] for d in args.extra}
    if args.command == 'hela':
        missing = sorted(set(args.held_out) - set(ctc_sequences(args.extra_roots[args.extra[0]])))
        if missing:
            parser.error(f'{args.extra[0]} has no sequences {missing}')
    stage = {'cache': 'cache', 'cv': 'cv_folds' + ''.join(map(str, args.folds)), 'hela': 'hela_held' + '_'.join(args.held_out)}[args.command]
    if args.retrack and args.command != 'cache':
        stage += f'_retrack_{args.retrack}'
    if args.gpus and args.command != 'cache':
        stage += '_gpus' + ''.join(map(str, args.gpus))                 # the children log under their own stages
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_bayesian',
                   f'{args.output_root.resolve().name}_{stage}')
    return run_logged(log, run, args, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
