"""
prepare_train.py
================
Training data for fine-tuning Trackastra on the data gnn_bayesian is trained on (phase 3 of the comparison). Writes a
new folder; data/ is only read.

Every sequence used for training -- the scripta development wells (never the test wells) and every sequence of the
extra datasets -- is written as a CTC sequence Trackastra's training script reads:
    <out>/<dataset>/<seq>/t###.tif             hard links (copies where linking fails) of the raw images
    <out>/<dataset>/<seq>_GT/TRA/              utils/ctc_export.canonical_gt: gaps split into single-child segments
                                               (valid CTC, Trackastra follows the parent links across the gap) and
                                               detections without identity (labels >= 50000) REMOVED
Trackastra has no way to leave a detection out of its loss: an unknown detection kept in man_track.txt would be
learnt as a one-frame track linked to nothing. Removing it is the closest to gnn_bayesian, which keeps those
detections as nodes but excludes their edges from the loss. Detections whose identity the GT review resolved (refine)
carry a known label and stay. At inference every detection is kept, as for every other method.

Units, the same as gnn_bayesian's cv / final (run.py), each with Trackastra's own validation set (it picks the best
checkpoint and stops early on the validation loss): the validation wells are the first well of every scripta fold
the unit trains on, taken out of its training set (a fixed rule, not tuned):
    fold<k>   train: scripta development wells outside fold k minus the validation wells + every extra sequence
    final     train: every scripta development well minus the validation wells + every extra sequence
<out>/units/<unit>_train.txt and <unit>_val.txt list the sequence folders (space separated, for --input_train /
--input_val); prepare.json records the sources and what canonical_gt removed.

    python -m models.trackastra.prepare_train --output-root development/stage4_trackastra/data
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from utils.config import ctc_sequences, read_config, resolve_dataset
from utils.ctc import sequence
from utils.ctc_export import canonical_gt
from utils.lineage import read_track_table


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_sequence(root, seq, dest):
    """One CTC sequence -> dest/<seq>/ (raw images linked) + dest/<seq>_GT/TRA/ (canonical GT, unknown removed)."""
    frames = sequence(root, seq)
    (dest / seq).mkdir(parents=True)
    for _, image, _ in frames:
        link(image, dest / seq / image.name)
    table = read_track_table(Path(root) / f'{seq}_GT' / 'TRA' / 'man_track.txt')
    return canonical_gt(table, [(t, m) for t, _, m in frames], dest / f'{seq}_GT')


def units(folds, extra):
    """-> {unit: (train sequence keys, validation sequence keys)}; keys are '<dataset>/<seq>'."""
    out = {}
    for k in [*range(len(folds)), None]:
        training = [f for i, f in enumerate(folds) if i != k]
        val = [f[0] for f in training]
        train = [w for f in training for w in f if w not in val]
        out['final' if k is None else f'fold{k}'] = (train, val)
    return {u: ([f'scripta/{w}' for w in t] + extra, [f'scripta/{w}' for w in v]) for u, (t, v) in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scripta', default='scripta_data_refine2', help='Dataset with folds and test wells')
    parser.add_argument('--extra', nargs='+', default=['Fluo-N2DL-HeLa'], help='Extra datasets (every sequence)')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory')
    args = parser.parse_args(argv)
    cfg, base = read_config()
    spec = resolve_dataset(cfg, base, args.scripta)
    folds = spec['folds']
    if not folds:
        parser.error(f'{args.scripta} has no folds in config.json')
    dev = [w for f in folds for w in f]
    if set(dev) & set(spec['held_out']):
        parser.error('test wells appear in the folds')
    roots = {'scripta': (args.scripta, spec['root'], dev)}
    for name in args.extra:
        root = resolve_dataset(cfg, base, name)['root']
        roots[name] = (name, root, ctc_sequences(root))
    out = args.output_root
    out.mkdir(parents=True)                                   # refuses an existing folder
    record = dict(scripta=args.scripta, extra=args.extra, test_wells_excluded=spec['held_out'], sequences={})
    for key, (name, root, seqs) in roots.items():
        for seq in seqs:
            info = write_sequence(root, seq, out / name)
            record['sequences'][f'{name}/{seq}'] = dict(source=str(Path(root) / seq), **info)
            print(f'{name}/{seq}: {info}', flush=True)
    extra = [f'{roots[d][0]}/{s}' for d in args.extra for s in roots[d][2]]
    plan = units(folds, extra)
    (out / 'units').mkdir()
    folder = {f'scripta/{w}': out / args.scripta / w for w in dev} | {k: out / k for k in extra}
    for unit, (train, val) in plan.items():
        for part, keys in (('train', train), ('val', val)):
            (out / 'units' / f'{unit}_{part}.txt').write_text(' '.join(folder[k].as_posix() for k in keys) + '\n',
                                                              encoding='utf-8')
        print(f'{unit}: train {train}  val {val}', flush=True)
    record['units'] = {u: dict(train=t, val=v) for u, (t, v) in plan.items()}
    (out / 'prepare.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
