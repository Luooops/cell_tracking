"""
run.py
======
Experiments of gnn_bayesian (training on server GPUs; evaluation with gnn_tracking's evaluate_answers / evaluate_ctc).
One model for every dataset: 3-way edges trained on scripta (default the main GT scripta_data_refine2) + the extra
datasets (default Fluo-N2DL-HeLa; any CTC dataset under data/ can be added with --extra), the network's own posterior
(prior 'none'), cost-derived thresholds; nothing at inference depends on the dataset. 3 ensemble members (seeds 0-2;
5 members were tested in f003: no better, the posterior of the uncertain edges under-confident -> not used).

    cv      3-fold cross-validation on the scripta development wells: each fold trained on its 8 training wells PLUS
            every extra sequence; the held-out wells -> <out>/oof/<tracker>/<well>/          (checkpoints fold<k>_)
    loso    --dataset D: leave one sequence of extra dataset D out: trained on every scripta development well + every
            other extra sequence -> <out>/<D>/<tracker>/<sequence>/                          (checkpoints <D>_<seq>_)
    unseen  --dataset D: D never seen in training (scripta development wells + the other extra datasets); every
            sequence of D tracked -> <out>/unseen_<D>/<tracker>/<sequence>/                  (checkpoints unseen_<D>_)
    final   every scripta development well + every extra sequence; the scripta test wells, once
            -> <out>/test/<tracker>/<well>/                                                  (checkpoints final_)
    cache   fill <out>/cache with every video used (run it once before starting several processes on one root).

Trackers: gnn_tracking (no divisions) and gnn_division (divisions on p_division). Every tracked video gets
instance_tracks.csv, lineage.json (divisions), edges.npz, links.csv and tracks_confidence.csv (confidence.py) and
metrics.json (identity scores + 'confidence': expected false / missed links).

Tracking options (after training, or with --retrack):
    --tta                   also score every frame pair on the video mirrored horizontally and vertically and average
                            (test-time augmentation; part of the final version)
    --prior division        the dataset's division prior (analysis only: uses knowledge of the dataset); default none
    --ablate appearance|velocity   cue analysis: track with the cell images shuffled within each frame / without
                            velocities;  --cues  track normally and record per edge the probabilities without each cue
--retrack trains nothing: it loads the checkpoints and tracks again into <out>/retrack_<prior>[_tta][_no<cue>][_cues]/.

Training options: --classes 2 (binary ablation 'linked or not'); --active-bias (weight edges by their prediction
variance over the epochs; tested, not significant); --cosine-lr (cosine learning-rate decay; tested in f002, made the
posterior under-confident on uncertain edges -> not used; the default is the constant learning rate).

--gpus 0 1 fills the cache, trains the missing members one per job ((fold or sequence, seed) pairs, round-robin
over the GPUs, one child process per GPU, each with its own log), then tracks everything in this process. A member
whose checkpoint exists is not trained again; tracked folders that exist are refused.

    python -m models.gnn_bayesian.run cv --tta --gpus 0 1 --output-root development/stage3_final/cv_f001
    python -m models.gnn_bayesian.run loso --dataset Fluo-N2DL-HeLa --tta --gpus 0 1 --output-root development/stage3_final/loso_hela_f001
    python -m models.gnn_bayesian.run final --tta --gpus 0 1 --output-root development/stage3_final/final
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


def _tag(args):
    return '_'.join([args.prior] + (['tta'] if args.tta else []) + ([f'no{args.ablate}'] if args.ablate else [])
                    + (['cues'] if args.cues else []))


def units(args, extra):
    """-> [(unit id, checkpoint prefix, training specs, [(spec to track, target dataset, folder of a tracker)])]."""
    scripta, out = args.scripta, args.dest
    dev = [f'{scripta}/{w}' for w in args.dev_wells]
    if args.command == 'cv':
        out_of = lambda name: out / 'oof' / name
        return [(str(k), f'fold{k}', [s for s in dev if s.split('/')[1] not in fold] + extra,
                 [(f'{scripta}/{w}', scripta, out_of) for w in fold])
                for k, fold in enumerate(args.scripta_spec['folds']) if k in args.folds]
    if args.command == 'loso':
        d = args.dataset
        return [(seq, f'{d}_{seq}', dev + [s for s in extra if s != f'{d}/{seq}'],
                 [(f'{d}/{seq}', d, lambda name, d=d: out / d / name)]) for seq in args.held_out]
    if args.command == 'unseen':
        d = args.dataset
        return [('unseen', f'unseen_{d}', dev + [s for s in extra if not s.startswith(f'{d}/')],
                 [(f'{d}/{seq}', d, lambda name, d=d: out / f'unseen_{d}' / name)
                  for seq in ctc_sequences(args.extra_roots[d])])]
    return [('final', 'final', dev + extra,
             [(f'{scripta}/{w}', scripta, lambda name: out / 'test' / name) for w in args.test_wells])]


def _track_and_export(args, models, spec, target, cache, folder_of, extra_report):
    import pandas as pd
    from models.gnn_tracking.graph import Video
    from models.gnn_bayesian.confidence import summarize
    from models.gnn_bayesian.data import load, split
    from models.gnn_bayesian.track import Recorder, track_video
    from utils.runners import export_gnn_well
    video = Video(*load(spec, cache))
    views = [Video(*load(spec, cache, flip)) for flip in ('h', 'v')] if args.tta else []
    _, well = split(spec)
    taken = [folder_of(name) / well for name in TRACKERS if (folder_of(name) / well).exists()]
    if taken:
        raise FileExistsError(f'already tracked: {taken}')
    for name, divisions in TRACKERS.items():
        recorder = Recorder(video)
        assign, parent = track_video(models, video, target, divisions=divisions, recorder=recorder, prior=args.prior,
                                     views=views, ablate=args.ablate, cues=args.cues)
        folder = folder_of(name)
        folder.mkdir(parents=True, exist_ok=True)
        report = export_gnn_well(folder / well, video, assign, parent if divisions else None, recorder)
        links, tracks, confidence = summarize(recorder.arrays(), pd.read_csv(folder / well / 'instance_tracks.csv'))
        links.to_csv(folder / well / 'links.csv', index=False)
        tracks.to_csv(folder / well / 'tracks_confidence.csv', index=False)
        (folder / well / 'metrics.json').write_text(json.dumps(dict(
            **extra_report, prior=args.prior, tta=args.tta, ablate=args.ablate, cues=args.cues, **report,
            confidence=confidence), indent=2), encoding='utf-8')
        print(f"{spec} {name}: perfect {report['perfect']:.1f}  id switches {report['id_switches']}"
              + (f"  divisions {len(parent)}" if divisions else '')
              + f"  expected false / missed links {confidence['expected_false_links']:.0f} / "
                f"{confidence['expected_missed_links']:.0f}", flush=True)


def _checkpoint(args, prefix, seed):
    return args.output_root / 'checkpoints' / f'{prefix}_seed{seed}.pt'


def _train_member(args, specs, cache, prefix, seed):
    from models.gnn_bayesian.model import save
    from models.gnn_bayesian.train import train
    path = _checkpoint(args, prefix, seed)
    if path.exists():
        raise FileExistsError(f'{path} exists')
    model = train(specs, seed, cache, log=f'{prefix} seed {seed}', n_classes=args.classes,
                  active_bias=args.active_bias, cosine=args.cosine_lr)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, path)
    return model


def _ensemble(args, specs, cache, prefix):
    """Every member of this unit: loaded if its checkpoint exists (trained by a GPU job or an earlier call),
    trained here otherwise -- except with --retrack, which only loads."""
    from models.gnn_bayesian.model import load_model
    models = []
    for seed in args.seeds:
        path = _checkpoint(args, prefix, seed)
        if path.exists():
            models.append(load_model(path))
        elif args.retrack:
            raise FileNotFoundError(f'{prefix}: no checkpoint {path}')
        else:
            models.append(_train_member(args, specs, cache, prefix, seed))
    return models


def _fill_cache(cache, specs):
    from models.gnn_bayesian.data import load
    for spec in sorted(set(specs)):
        load(spec, cache)


def _train_on_gpus(args, todo):
    """todo: [(unit id, prefix, seed)] members still to train -> one child per GPU, each training its share."""
    common = [args.command, '--scripta', args.scripta, '--extra', *args.extra, '--device', args.device,
              '--classes', str(args.classes), '--output-root', str(args.output_root)] \
        + (['--dataset', args.dataset] if args.dataset else []) + (['--active-bias'] if args.active_bias else []) \
        + (['--cosine-lr'] if args.cosine_lr else [])
    children = []
    for i, gpu in enumerate(args.gpus):
        share = todo[i::len(args.gpus)]
        if share:
            jobs = [f'{unit}:{seed}' for unit, _, seed in share]
            cmd = [sys.executable, '-m', 'models.gnn_bayesian.run', *common, '--jobs', *jobs]
            print(f'GPU {gpu}: {" ".join(jobs)}', flush=True)
            children.append((gpu, subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})))
    failed = [gpu for gpu, child in children if child.wait() != 0]
    if failed:
        raise RuntimeError(f'training on GPUs {failed} failed (see their logs)')


def run(args):
    if args.device != 'auto':
        os.environ['LAB_TRACKER_DEVICE'] = args.device
    out, cache = args.output_root, args.output_root / 'cache'
    out.mkdir(parents=True, exist_ok=True)
    args.dest = out / f'retrack_{_tag(args)}' if args.retrack else out
    extra = [f'{d}/{s}' for d in args.extra for s in ctc_sequences(args.extra_roots[d])]
    plan = units(args, extra)
    config = out / 'config.json'
    if not config.exists():
        config.write_text(json.dumps(dict(
            model='gnn_bayesian', command=args.command, dataset=args.dataset, scripta=args.scripta,
            data_root=str(args.scripta_spec['root']), extra=args.extra, extra_videos=extra, seeds=args.seeds,
            folds=args.scripta_spec['folds'], test_wells=args.test_wells, classes=args.classes,
            active_bias=args.active_bias, cosine_lr=args.cosine_lr,
            created=datetime.datetime.now().isoformat(timespec='seconds'),
            scope=('3-way edges (unrelated / same / division)' if args.classes == 3 else 'binary edges (linked or not)')
            + ' trained on scripta + extra datasets; tracking options per video in metrics.json'),
            indent=2), encoding='utf-8')
    if args.command == 'cache':
        return _fill_cache(cache, [s for _, _, specs, track in plan for s in specs + [t[0] for t in track]]) or 0
    if args.jobs:                                                  # a GPU child: train the members it was given
        by_id = {unit: (prefix, specs) for unit, prefix, specs, _ in plan}
        for job in args.jobs:
            unit, seed = job.rsplit(':', 1)
            prefix, specs = by_id[unit]
            _train_member(args, specs, cache, prefix, int(seed))
        return 0
    if args.gpus:
        _fill_cache(cache, [s for _, _, specs, track in plan for s in specs + [t[0] for t in track]])
        todo = [(unit, prefix, seed) for unit, prefix, _, _ in plan for seed in args.seeds
                if not _checkpoint(args, prefix, seed).exists()]
        if todo and not args.retrack:
            _train_on_gpus(args, todo)
    for unit, prefix, specs, track in plan:
        models = _ensemble(args, specs, cache, prefix)
        for spec, target, folder_of in track:
            _track_and_export(args, models, spec, target, cache, folder_of,
                              dict(command=args.command, unit=unit, seeds=args.seeds))
    print('saved', args.dest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['cache', 'cv', 'loso', 'unseen', 'final'])
    parser.add_argument('--scripta', default='scripta_data_refine2', help='Dataset with folds and test wells')
    parser.add_argument('--extra', nargs='+', default=['Fluo-N2DL-HeLa'],
                        help='Datasets added to every training')
    parser.add_argument('--dataset', help='loso / unseen: the extra dataset held out')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2], help='cv: folds to run')
    parser.add_argument('--held-out', nargs='+', help='loso: sequences to hold out (default: every sequence)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--gpus', nargs='+', type=int, help='Train the members in parallel, one process per GPU')
    parser.add_argument('--classes', type=int, choices=[2, 3], default=3, help='2: the binary ablation (training)')
    parser.add_argument('--active-bias', action='store_true', help='Weight edges by prediction variance (training)')
    parser.add_argument('--cosine-lr', action='store_true', help='Cosine learning-rate decay (training; not used)')
    parser.add_argument('--prior', choices=['none', 'division'], default='none',
                        help="none: the network's own posterior; division: the dataset's division prior (analysis)")
    parser.add_argument('--tta', action='store_true', help='Also score the mirrored video and average')
    parser.add_argument('--ablate', choices=['appearance', 'velocity'], help='Cue analysis: track without this cue')
    parser.add_argument('--cues', action='store_true', help='Record per edge the scores without each cue')
    parser.add_argument('--retrack', action='store_true', help='Load the checkpoints and track again (no training)')
    parser.add_argument('--jobs', nargs='+', help=argparse.SUPPRESS)          # internal: members a GPU child trains
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg, base = read_config()
    args.scripta_spec = resolve_dataset(cfg, base, args.scripta, None)
    if not args.scripta_spec['folds']:
        parser.error(f'{args.scripta} has no folds in config.json')
    args.dev_wells = [w for f in args.scripta_spec['folds'] for w in f]
    args.test_wells = list(args.scripta_spec['held_out'])
    if set(args.dev_wells) & set(args.test_wells):
        parser.error('test wells appear in the folds')
    try:
        args.extra_roots = {d: resolve_dataset(cfg, base, d, None)['root'] for d in args.extra}
    except FileNotFoundError as exc:
        parser.error(f'{exc} (download it, or leave it out with --extra)')
    if args.command in ('loso', 'unseen'):
        if args.dataset not in args.extra:
            parser.error(f'{args.command} needs --dataset, one of --extra {args.extra}')
        sequences = ctc_sequences(args.extra_roots[args.dataset])
        args.held_out = args.held_out or sequences
        if missing := sorted(set(args.held_out) - set(sequences)):
            parser.error(f'{args.dataset} has no sequences {missing}')
    if args.command == 'final' and not args.test_wells:
        parser.error('final needs test wells in config.json')
    stage = {'cache': 'cache', 'cv': 'cv_folds' + ''.join(map(str, args.folds)), 'loso': f'loso_{args.dataset}',
             'unseen': f'unseen_{args.dataset}', 'final': 'final'}[args.command]
    if args.jobs:
        stage += '_train_' + '_'.join(j.replace(':', 's') for j in args.jobs)
    elif args.retrack and args.command != 'cache':
        stage += f'_retrack_{_tag(args)}'
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_bayesian',
                   f'{args.output_root.resolve().name}_{stage}')
    return run_logged(log, run, args, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
