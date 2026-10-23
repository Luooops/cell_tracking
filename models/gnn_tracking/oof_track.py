"""
oof_track.py
============
Stage 1.2: track the held-out wells of a finished cross-validation run again with its saved fold models, under other
inference settings (prior correction, link thresholds), without training. Every well is tracked only by the models
of the fold that held it out, exactly as train.py --cv --set cv_outputs=true wrote the original out-of-fold sheets.

The run folder is the train folder of that CV run (config.json + checkpoints/fold<k>_seed<s>.pt); its config gives
the dataset root, the folds and the seeds. Inference settings start from the test section of the current
config.json for each tracker and are overridden with --set (keys must exist there, as for test.py).

    python -m models.gnn_tracking.oof_track \
        --cv-run development/stage0_evaluation/scripta_data_refine/gnn_tracking/cv_v002/train \
        --set prior_correction=true \
        --output-root development/stage1_calibration/oof_cv_v002_prior

Writes into a new --output-root: oof/<tracker>/<well>/ (instance_tracks.csv, metrics.json, edges.npz) and
config.json (the run, dataset root and settings), so evaluate_answers / evaluate_ctc can score it like the original
out-of-fold sheets.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

from utils.config import read_config
from utils.run_log import log_path, log_root, run_logged

TRACKERS = ('gnn_tracking', 'gnn_division')


def parse_sets(items, allowed):
    out = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'--set needs KEY=JSON, got {item!r}')
        if not any(key in a for a in allowed):
            raise ValueError(f'--set {key}: not a test setting of {", ".join(TRACKERS)} in config.json')
        out[key] = json.loads(value)
    return out


def run(args, run_config, settings):
    os.environ['LAB_TRACKER_DATA'] = args.data_root
    if args.device != 'auto':
        os.environ['LAB_TRACKER_DEVICE'] = args.device
    from models.gnn_tracking.edge_record import EdgeRecorder
    from models.gnn_tracking.graph import Video
    from models.gnn_tracking.model import load_model
    from models.gnn_tracking.track import track_video
    from models.gnn_tracking.track_division import track_video_division
    from utils import dataset
    from utils.runners import export_gnn_well
    cache = args.cv_run / 'cache'
    dataset.LAB_ROOT = args.data_root
    dataset.CACHE = str(cache if cache.is_dir() else args.output_root / 'cache')
    folds, seeds = run_config['dataset_spec']['folds'], list(run_config['params']['seeds'])
    out = args.output_root
    out.mkdir(parents=True)
    (out / 'config.json').write_text(json.dumps(dict(
        cv_run=str(args.cv_run.resolve()), dataset=run_config['dataset'], data_root=args.data_root, folds=folds,
        seeds=seeds, trackers=settings, created=datetime.datetime.now().isoformat(timespec='seconds'),
        scope='out-of-fold tracking with the fold models of the CV run; no training'), indent=2), encoding='utf-8')
    for k, fold in enumerate(folds):
        models = [load_model(str(args.cv_run / 'checkpoints' / f'fold{k}_seed{s}.pt')) for s in seeds]
        if any(settings[t].get('prior_correction') for t in settings):
            print(f"fold {k}: pos_weight {[round(m.pos_weight, 3) for m in models]}", flush=True)
        for well in fold:
            video = Video(*dataset.load_well(well))
            for name, params in settings.items():
                recorder, parent = EdgeRecorder(video), None
                if name == 'gnn_division':
                    assign, parent = track_video_division(models, video, recorder=recorder, **params)
                else:
                    assign = track_video(models, video, recorder=recorder, **params)
                folder = out / 'oof' / name
                folder.mkdir(parents=True, exist_ok=True)
                report = export_gnn_well(folder / well, video, assign, parent, recorder)
                (folder / well / 'metrics.json').write_text(json.dumps(dict(fold=k, seeds=seeds, settings=params, **report),
                                                                       indent=2), encoding='utf-8')
                print(f"fold {k} {well} {name}: perfect {report['perfect']:.1f}  id switches {report['id_switches']}",
                      flush=True)
    print('saved', out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Default: project config.json (test settings of the trackers)')
    parser.add_argument('--cv-run', type=Path, required=True, help='Train folder of a --cv run with checkpoints/fold*_seed*.pt')
    parser.add_argument('--data-root', help='Dataset root (default: the data_root recorded in the run config)')
    parser.add_argument('--trackers', nargs='+', choices=TRACKERS, default=list(TRACKERS))
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='Override a test setting')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for the out-of-fold sheets')
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    run_config_path = args.cv_run / 'config.json'
    if not run_config_path.is_file() or not (args.cv_run / 'checkpoints').is_dir():
        parser.error(f'{args.cv_run} is not the train folder of a CV run (config.json + checkpoints/)')
    run_config = json.loads(run_config_path.read_text(encoding='utf-8'))
    if not run_config.get('cv') or not run_config.get('dataset_spec', {}).get('folds'):
        parser.error(f'{args.cv_run} was not trained with --cv on a dataset with folds')
    tests = {t: {k: v for k, v in cfg['models'][t].get('test', {}).items() if k != 'record_edges'} for t in args.trackers}
    try:
        overrides = parse_sets(args.set, list(tests.values()))
    except ValueError as exc:
        parser.error(str(exc))
    settings = {t: {**p, **{k: v for k, v in overrides.items() if k in p}} for t, p in tests.items()}
    args.data_root = args.data_root or run_config['data_root']
    if not Path(args.data_root).is_dir():
        parser.error(f'dataset root not found: {args.data_root} (pass --data-root)')
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_oof_track_{run_config["dataset"]}')
    return run_logged(log, run, args, run_config, settings, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
