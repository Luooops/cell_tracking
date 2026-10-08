"""JSON defaults and CLI overrides; relative paths are relative to the config file."""
import argparse
import json
from pathlib import Path
from models import MODEL_MODULES

ROOT = Path(__file__).resolve().parents[1]


def options(stage, argv=None):
    p = argparse.ArgumentParser(description=f'Unified {stage} entry; algorithms keep their original implementations.')
    p.add_argument('--model', choices=MODEL_MODULES, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--version', required=True, help='New run name; existing runs are never overwritten')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--wells', nargs='+')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='Override a model parameter, e.g. epochs=1')
    if stage == 'train':
        p.add_argument('--train-dir', type=Path, help='Motion: existing audited NPZ training windows')
        p.add_argument('--val-dir', type=Path, help='Motion: existing audited NPZ validation windows')
        p.add_argument('--cv', action='store_true', help='GNN: original three-fold cross-validation')
    else:
        p.add_argument('--windows-dir', type=Path, help='Motion: existing NPZ windows (no automatic video fusion)')
    a = p.parse_args(argv)
    if not a.version or a.version in {'.', '..'} or any(c in a.version for c in '/\\:'):
        p.error('version must be a single directory name')
    cfg = json.loads(a.config.read_text(encoding='utf-8-sig'))
    base = a.config.resolve().parent
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
    a.data_root = (a.data_root or base / cfg['data_root']).resolve()
    a.output_root = (a.output_root or base / cfg['output_root']).resolve()
    a.weights = a.weights or (base / spec['weights'] if 'weights' in spec else None)
    if stage == 'test':
        a.wells = a.wells or cfg.get('test_wells', [])
    a.destination = a.output_root / a.model / a.version / stage
    if a.destination.exists():
        p.error(f'Run already exists: {a.destination}; select a new version')
    return a


def save_config(a):
    a.destination.mkdir(parents=True, exist_ok=False)
    (a.destination / 'config.json').write_text(json.dumps(vars(a), default=str, indent=2), encoding='utf-8')
