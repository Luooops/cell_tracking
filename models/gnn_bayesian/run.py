"""
run.py
======
Experiments of gnn_bayesian (training on server GPUs; evaluation with gnn_tracking's evaluate_answers / evaluate_ctc).
One model for every dataset: 3-way edges trained on scripta (default the main GT scripta_data_refine2) + HeLa, the
network's own posterior (prior 'none'), cost-derived thresholds; nothing at inference depends on the dataset.

    cv     3-fold cross-validation on the scripta development wells, every fold trained on its 8 training wells PLUS
           every sequence of the extra datasets; the held-out wells are tracked with gnn_tracking (no divisions) and
           gnn_division (divisions on p_division) -> <out>/oof/<tracker>/<well>/.
    hela   leave-one-sequence-out on the first extra dataset: trained on every scripta development well + the other
           sequences -> <out>/<dataset>/<tracker>/<sequence>/.
    final  trained on every scripta development well + every extra sequence, the scripta test wells tracked once
           -> <out>/test/<tracker>/<well>/ (checkpoints final_seed<s>.pt).
    cache  fill <out>/cache with every video used (run it once before starting several processes on one root).

Every tracked well gets instance_tracks.csv, lineage.json (divisions), edges.npz, links.csv and
tracks_confidence.csv (confidence.py: how sure the tracker is of each link and track) and metrics.json (identity
scores + 'confidence': expected false / missed links).

Tracking options (after training, or with --retrack):
    --prior none|division   none (default): the network's own posterior; division: the target dataset's division
                            prior (analysis only: uses knowledge of the dataset)
    --tta                   also score every frame pair on the video mirrored horizontally and vertically (as in
                            training) and average (test-time augmentation; a Bayesian average over orientations)
    --ablate appearance|velocity   cue analysis: track with the cell images shuffled within each frame / without
                            velocities
    --cues                  track normally and record per edge the probabilities without appearance / velocity
                            (p_noapp, p_novel in edges.npz)
--retrack trains nothing: it loads the ensembles in <out>/checkpoints and tracks the same videos again into
<out>/retrack_<prior>[_tta][_no<cue>][_cues]/...; existing folders are refused.

Training options: --classes 2 (binary ablation 'linked or not', divisions as gnn_division: two strong links);
--active-bias (weight each edge by its prediction variance over the epochs; tested, not significant).

--gpus 0 1 fills the cache, then splits the folds (cv) or held-out sequences (hela) round-robin over the GPUs, one
child process per GPU (each with its own log). A fold or held-out sequence whose checkpoints exist is refused.

    python -m models.gnn_bayesian.run cv --gpus 0 1 --output-root development/stage3_final/cv_f001
    python -m models.gnn_bayesian.run cv --retrack --tta --gpus 0 1 --output-root development/stage3_final/cv_f001
    python -m models.gnn_bayesian.run final --output-root development/stage3_final/final_f001
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


def _train_ensemble(args, specs, cache, paths, label):
    from models.gnn_bayesian.model import save
    from models.gnn_bayesian.train import train
    models = []
    for seed, path in zip(args.seeds, paths):
        model = train(specs, seed, cache, log=f'{label} seed {seed}', n_classes=args.classes,
                      active_bias=args.active_bias)
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
    return _train_ensemble(args, specs, cache, paths, label)


def _fill_cache(args, cache, extra):
    from models.gnn_bayesian.data import load
    for spec in [f'{args.scripta}/{w}' for w in args.dev_wells + (args.test_wells if args.command == 'final' else [])] \
            + extra:
        load(spec, cache)


def _split_over_gpus(args):
    """One child process per GPU, each given its share of the folds / held-out sequences; waits for all."""
    key, items = ('--folds', args.folds) if args.command == 'cv' else ('--held-out', args.held_out)
    common = [args.command, '--scripta', args.scripta, '--extra', *args.extra, '--seeds', *map(str, args.seeds),
              '--device', args.device, '--classes', str(args.classes), '--prior', args.prior,
              '--output-root', str(args.output_root)] + (['--retrack'] if args.retrack else []) \
        + (['--active-bias'] if args.active_bias else []) + (['--tta'] if args.tta else []) \
        + (['--ablate', args.ablate] if args.ablate else []) + (['--cues'] if args.cues else [])
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
    dest = out / f'retrack_{_tag(args)}' if args.retrack else out
    scripta = args.scripta_spec
    extra = [f'{d}/{s}' for d in args.extra for s in ctc_sequences(args.extra_roots[d])]
    config = out / 'config.json'
    if not config.exists():
        config.write_text(json.dumps(dict(
            model='gnn_bayesian', scripta=args.scripta, data_root=str(scripta['root']), extra=args.extra,
            extra_videos=extra, seeds=args.seeds, folds=scripta['folds'], test_wells=args.test_wells,
            classes=args.classes, active_bias=args.active_bias, created=datetime.datetime.now().isoformat(timespec='seconds'),
            scope=('3-way edges (unrelated / same / division)' if args.classes == 3 else 'binary edges (linked or not)')
            + ' trained on scripta + extra datasets; tracking options per well in metrics.json'),
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
            specs = [f'{args.scripta}/{w}' for w in args.dev_wells if w not in fold] + extra
            models = _ensemble(args, specs, cache, paths, f'fold {k}')
            for w in fold:
                _track_and_export(args, models, f'{args.scripta}/{w}', args.scripta, cache,
                                  lambda name: dest / 'oof' / name, dict(fold=k, seeds=args.seeds))
    elif args.command == 'hela':
        dataset = args.extra[0]
        for held in args.held_out:
            paths = [out / 'checkpoints' / f'{dataset}_{held}_seed{s}.pt' for s in args.seeds]
            specs = [f'{args.scripta}/{w}' for w in args.dev_wells] + [s for s in extra if s != f'{dataset}/{held}']
            models = _ensemble(args, specs, cache, paths, f'{dataset} without {held}')
            _track_and_export(args, models, f'{dataset}/{held}', dataset, cache,
                              lambda name: dest / dataset / name, dict(held_out=held, seeds=args.seeds))
    else:
        paths = [out / 'checkpoints' / f'final_seed{s}.pt' for s in args.seeds]
        specs = [f'{args.scripta}/{w}' for w in args.dev_wells] + extra
        models = _ensemble(args, specs, cache, paths, 'final')
        for w in args.test_wells:
            _track_and_export(args, models, f'{args.scripta}/{w}', args.scripta, cache,
                              lambda name: dest / 'test' / name, dict(final=True, seeds=args.seeds))
    print('saved', dest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['cache', 'cv', 'hela', 'final'])
    parser.add_argument('--scripta', default='scripta_data_refine2', help='Dataset with folds and test wells')
    parser.add_argument('--extra', nargs='+', default=['Fluo-N2DL-HeLa'], help='Datasets added to every training')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--held-out', nargs='+', default=['01', '02'], help='hela: sequences of the first extra dataset')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--gpus', nargs='+', type=int, help='cv / hela: split the folds / held-out sequences over GPUs')
    parser.add_argument('--classes', type=int, choices=[2, 3], default=3, help='2: the binary ablation (training)')
    parser.add_argument('--active-bias', action='store_true', help='Weight edges by prediction variance (training)')
    parser.add_argument('--prior', choices=['none', 'division'], default='none',
                        help="none: the network's own posterior; division: the dataset's division prior (analysis)")
    parser.add_argument('--tta', action='store_true', help='Also score the mirrored video and average')
    parser.add_argument('--ablate', choices=['appearance', 'velocity'], help='Cue analysis: track without this cue')
    parser.add_argument('--cues', action='store_true', help='Record per edge the scores without each cue')
    parser.add_argument('--retrack', action='store_true', help='Load the checkpoints and track again (no training)')
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
    if args.command == 'final' and (not args.test_wells or args.gpus):
        parser.error('final needs test wells in config.json and runs as one process (no --gpus)')
    args.extra_roots = {d: resolve_dataset(cfg, base, d, None)['root'] for d in args.extra}
    if args.command == 'hela':
        missing = sorted(set(args.held_out) - set(ctc_sequences(args.extra_roots[args.extra[0]])))
        if missing:
            parser.error(f'{args.extra[0]} has no sequences {missing}')
    stage = {'cache': 'cache', 'cv': 'cv_folds' + ''.join(map(str, args.folds)),
             'hela': 'hela_held' + '_'.join(args.held_out), 'final': 'final'}[args.command]
    if args.retrack and args.command != 'cache':
        stage += f'_retrack_{_tag(args)}'
    if args.gpus and args.command != 'cache':
        stage += '_gpus' + ''.join(map(str, args.gpus))                 # the children log under their own stages
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_bayesian',
                   f'{args.output_root.resolve().name}_{stage}')
    return run_logged(log, run, args, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
