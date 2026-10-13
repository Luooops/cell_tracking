"""JSON defaults and CLI overrides; relative paths are relative to the config file.

Datasets are CTC folders: <root>/<sequence>/t*.tif and <root>/<sequence>_GT/TRA/man_track*.tif.
config['datasets'][name] may declare root and well splits; any other folder under config['data_dir']
can be selected by name with --dataset and its sequences are discovered automatically (no predefined split)."""
import argparse
import json
import shutil
from pathlib import Path
from models import MODEL_MODULES
from utils.run_log import log_path

ROOT = Path(__file__).resolve().parents[1]


def read_config(path=None):
    """-> (config dict, directory that relative config paths resolve against)."""
    path = Path(path or ROOT / 'config.json')
    return json.loads(path.read_text(encoding='utf-8-sig')), path.resolve().parent


def ctc_sequences(root):
    """Names of complete CTC sequences under root (a <name> folder with a matching <name>_GT/TRA)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.endswith('_GT') and (root / f'{p.name}_GT' / 'TRA').is_dir())


def list_datasets(folder):
    """Dataset folders directly under folder (those containing at least one CTC sequence)."""
    folder = Path(folder)
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and ctc_sequences(p)) if folder.is_dir() else []


def resolve_dataset(cfg, base, name=None, root=None, must_exist=True):
    """Select a dataset by name and/or explicit root.

    Returns dict(name, root, test_wells, held_out, train_wells, val_wells, folds). held_out lists only declared
    test wells; test_wells falls back to every discovered sequence when the dataset declares none.
    Without a name, an explicit root is named after its folder; otherwise config['dataset'] is used."""
    if name is None:
        name = Path(root).resolve().name if root is not None else cfg['dataset']
    spec = cfg.get('datasets', {}).get(name, {})
    if root is None:
        root = base / spec['root'] if 'root' in spec else base / cfg.get('data_dir', 'data') / name
    root = Path(root).resolve()
    if must_exist and not root.is_dir():
        known = ', '.join(sorted(set(cfg.get('datasets', {})) | set(list_datasets(base / cfg.get('data_dir', 'data')))))
        raise FileNotFoundError(f'Dataset {name!r} not found at {root}; known datasets: {known or "none"}')
    held_out = list(spec.get('test_wells', []))
    return dict(name=name, root=root, test_wells=held_out or ctc_sequences(root), held_out=held_out,
                train_wells=list(spec.get('train_wells', [])), val_wells=list(spec.get('val_wells', [])),
                folds=[list(f) for f in spec.get('folds', [])])


def options(stage, argv=None):
    p = argparse.ArgumentParser(description=f'Unified {stage} entry; algorithms keep their original implementations.')
    p.add_argument('--model', choices=MODEL_MODULES, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--version', required=True, help='New run name; existing runs are never overwritten')
    p.add_argument('--dataset', help='Dataset name from config.json datasets, or a folder under data_dir (default: config dataset)')
    p.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--wells', nargs='+')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='Override a model parameter, e.g. epochs=1')
    p.add_argument('--overwrite', action='store_true', help='Delete an existing run of this model/version/stage and start it again')
    if stage == 'train':
        p.add_argument('--train-dir', type=Path, nargs='+', help='Motion: existing audited NPZ training windows (one or more roots)')
        p.add_argument('--val-dir', type=Path, nargs='+', help='Motion: existing audited NPZ validation windows (one or more roots)')
        p.add_argument('--cv', action='store_true', help='GNN: original three-fold cross-validation')
        p.add_argument('--gpus', nargs='+', metavar='ID', help='GNN: CUDA device ids; seeds (or CV folds) train in parallel, one process per GPU')
    else:
        p.add_argument('--windows-dir', type=Path, help='Motion: optional independent NPZ window debug mode')
    a = p.parse_args(argv)
    if not a.version or a.version in {'.', '..'} or any(c in a.version for c in '/\\:'):
        p.error('version must be a single directory name')
    cfg, base = read_config(a.config)
    spec = cfg['models'][a.model]
    a.params = dict(spec.get(stage, {}))
    for item in a.set:
        try:
            key, value = item.split('=', 1)
            if key not in a.params:
                p.error(f'Unknown {stage} parameter: {key}; add supported parameters to the model config first')
            a.params[key] = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            p.error('--set requires KEY=JSON')
    try:
        # Motion training and NPZ window debugging read NPZ only; the dataset then just names the output folder.
        needs_data = not (a.model == 'motion_tracking' and (stage == 'train' or a.windows_dir))
        a.dataset_spec = resolve_dataset(cfg, base, a.dataset, a.data_root, must_exist=needs_data)
    except FileNotFoundError as exc:
        p.error(str(exc))
    a.dataset, a.data_root = a.dataset_spec['name'], a.dataset_spec['root']
    a.output_root = (a.output_root or base / cfg['output_root']).resolve()
    a.weights = a.weights or (base / spec['weights'] if 'weights' in spec else None)
    if stage == 'test':
        a.wells = a.wells or a.dataset_spec['test_wells']
    a.destination = a.output_root / a.dataset / a.model / a.version / stage
    a.log_path = log_path(a.output_root, a.model, f'{a.version}_{stage}_{a.dataset}')
    if a.destination.exists() and not a.overwrite:
        p.error(f'Run already exists: {a.destination}; select a new version or pass --overwrite')
    return a


def save_config(a):
    if getattr(a, 'overwrite', False) and a.destination.exists():
        # Only the selected <model>/<version>/<stage> directory is removed, after all arguments were validated.
        print(f'Overwriting existing run: {a.destination}', flush=True)
        shutil.rmtree(a.destination)
    a.destination.mkdir(parents=True, exist_ok=False)
    (a.destination / 'config.json').write_text(json.dumps(vars(a), default=str, indent=2), encoding='utf-8')
