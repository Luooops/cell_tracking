"""
prepare_zarr.py
===============
Training data for fine-tuning Caliban (torch-tracking) on the data ProGCT and fine-tuned Trackastra are trained on:
CTC sequences -> the processed zarr torch-tracking's trainer reads (<split>_proc.zarr). Writes a new folder; data/ is
only read.

Each movie goes through torch-tracking's own conversion steps (preprocess.correct_lineages, then get_features at the
movie's declared resolution), so the features are the ones convert_trk_to_zarr would compute; only the input differs
(CTC folders instead of a .trks file). Two adaptations, both needed because the inputs mix datasets:
  * movies of different image sizes (scripta 2160 x 2160, DNN 584 x 600) are converted one at a time and stacked in
    the processed arrays, which hold model-resolution crops and centroids only (no full images), padded with zeros
    to the longest movie (empty trailing frames are padding to the sampler) and to the most labels of any movie of
    the split;
  * detections without identity (labels >= 50000) stay in the masks, so they are cells of the graph, but their
    lineage record has no frames, so torch-tracking's targets never grade them (loader.get_temporal_adjacency pads
    every entry of a cell absent from the lineage frames). This is how ProGCT treats them; fine-tuned Trackastra had
    to drop them instead. They are one-frame detections, so they share label slots across frames (pack_unknown).
    torch-tracking trains on one slot per label of the whole movie and scores every slot pair of every frame pair
    (decoder: batch x 7 frame pairs x N x N), so its memory grows with the square of the labels of a movie, not with
    the cells of a frame. DynamicNuclearNet movies have ~100-260 labels; a scripta well has up to 833 in the train
    split (1,208 in val), and batch 8 then exceeds a 94 GB H100 (the decoder asks for a further 37 GB at ~54 GB in
    use). Sharing brings the train split to 518 slots and changes no graded target, so the training code and all its
    settings stay torch-tracking's own.
Lineage by the project's convention (utils/lineage.py): same label or a single child = the same cell (single
children are merged into their parent's label; gaps stay one track, and targets are only built between consecutive
frames, so a gap is simply not graded); exactly two children = a division at the parent's last frame; three or more
children = unknown (parent and children left ungraded, and a division of an ungraded cell is not graded either).

Split of the fine-tuned Trackastra 'final' unit (development/stage7_raw/trackastra/runs/final/train_config.yaml):
    train  scripta development wells minus the first well of every fold + DynamicNuclearNet train60..train90
    val    r08c13 r15c12 r13c12 (picks the best epoch); the test wells r16c02 r01c20 r07c07 are never read

    python models/caliban/prepare_zarr.py --scripta-root data/scripta_data_refine2 --scripta-mpp 0.149 \
        --dnn-root data/DynamicNuclearNet --dnn-mpp 0.65 --output-root development/stage10_caliban/zarr_final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.ctc import sequence  # noqa: E402

UNKNOWN = 50000
TRAIN_SCRIPTA = ['r11c12', 'r03c12', 'r14c13', 'r12c13', 'r07c02', 'r09c01', 'r16c13', 'r09c09', 'r14c03']
VAL_SCRIPTA = ['r08c13', 'r15c12', 'r13c12']
TEST_SCRIPTA = ['r16c02', 'r01c20', 'r07c07']
TRAIN_DNN = [f'train{i}' for i in range(60, 91)]


def read_table(path):
    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
    return {int(l): (int(b), int(e), int(p)) for l, b, e, p in rows}


def load_movie(root, name):
    frames = sequence(root, name)
    if [t for t, _, _ in frames] != list(range(len(frames))):
        raise ValueError(f'{name}: frames are not 0..T-1')
    X = np.stack([tifffile.imread(p) for _, p, _ in frames]).astype(np.float32)[..., None]
    y = np.stack([tifffile.imread(p) for _, _, p in frames]).astype(np.int32)[..., None]
    return X, y


def merge_continuations(y, table):
    """The project's lineage convention (utils/lineage.py): a parent with exactly one child is the same cell
    continuing, so the child's masks take the parent's label (chains followed to their root); two children are a
    division; three or more are unknown. -> (y relabelled, {label: daughters} of the divisions, labels to leave
    ungraded, counts)."""
    children = {}
    for label, (_, _, parent) in table.items():
        if parent:
            children.setdefault(parent, []).append(label)
    root = {}
    for label in sorted(table):
        r = label
        while table.get(r, (0, 0, 0))[2] and len(children.get(table[r][2], [])) == 1:
            r = table[r][2]
        root[label] = r
    lut = {l: r for l, r in root.items() if r != l}
    if lut:
        flat = y.reshape(-1)
        for old, new in lut.items():
            flat[flat == old] = new
    divisions = {root[p]: sorted(root[c] for c in cs) for p, cs in children.items() if len(cs) == 2}
    many = {p: cs for p, cs in children.items() if len(cs) >= 3}
    ungraded = {root[p] for p in many} | {root[c] for cs in many.values() for c in cs}
    return y, divisions, ungraded, dict(single_child_merged=len(lut), two_children=len(divisions),
                                        many_children=len(many))


def pack_unknown(y):
    """Detections without identity (labels >= UNKNOWN, one frame each) share label slots across frames: in every frame
    they take UNKNOWN, UNKNOWN + 1, ... in their original order. Their slots are never graded (no lineage frames), so
    sharing changes no training target. -> (y, number of unknown detections)."""
    n = 0
    for t in range(y.shape[0]):
        frame = y[t]
        unknown = np.unique(frame[frame >= UNKNOWN])
        for k, label in enumerate(unknown):
            frame[frame == label] = -(k + 1)                      # temporary, avoids clashing with original labels
        frame[frame < 0] = UNKNOWN - 1 - frame[frame < 0]        # -(k+1) -> UNKNOWN + k
        n += unknown.size
    return y, n


def raw_lineage(y, table):
    """torch-tracking lineage record for every label in y (before its sequential relabelling)."""
    y, divisions, ungraded_known, counts = merge_continuations(y, table)
    y, unknown_detections = pack_unknown(y)
    counts = dict(counts, unknown_detections=int(unknown_detections))
    labels = [int(l) for l in np.unique(y) if l]
    present = set(labels)
    lineage = {}
    for label in labels:
        known = label < UNKNOWN and label in table
        daughters = [d for d in divisions.get(label, []) if d in present] if known else []
        if len(daughters) != 2:
            daughters = []
        lineage[label] = dict(label=label, frames=[], parent=None, daughters=daughters,
                              frame_div=None, capped=bool(daughters))
    for label, rec in lineage.items():
        for d in rec['daughters']:
            lineage[d]['parent'] = label
    ungraded = [l for l in labels if not (l < UNKNOWN and l in table) or l in ungraded_known]
    return y, lineage, ungraded, counts


def convert(root, name, mpp, n_cells, correct_lineages, get_features):
    X, y = load_movie(root, name)
    y, lineage, ungraded, counts = raw_lineage(y, read_table(Path(root) / f'{name}_GT' / 'TRA' / 'man_track.txt'))
    Xc, yc, lineages = correct_lineages(X[None], y[None], [lineage])
    lineage = lineages[0]
    order = sorted(int(l) for l in np.unique(y) if l)                 # relabel_sequential keeps the order
    new_id = {old: i + 1 for i, old in enumerate(order)}
    dropped = 0
    for old in ungraded:                                               # in the graph, never graded
        rec = lineage[new_id[old]]
        for d in rec['daughters']:            # a division of a cell whose own links are unknown: not graded;
            lineage[d]['parent'] = None       # its children stay graded as cells that appear
            dropped += 1
        rec['daughters'], rec['capped'], rec['frames'] = [], False, []
    counts = dict(counts, divisions_of_ungraded_dropped=dropped)
    for rec in lineage.values():
        if rec['daughters']:
            rec['frame_div'] = int(rec['frames'][-1]) + 1 if rec['frames'] else None
    features = get_features(Xc, yc, n_cells, mpps=np.asarray([mpp], dtype='float64'))
    info = dict(frames=int(y.shape[0]), shape=list(y.shape[1:3]), mpp=mpp, labels=len(order),
                ungraded=len(ungraded), divisions=sum(1 for r in lineage.values() if r['daughters']), **counts)
    return features, {int(k): v for k, v in lineage.items()}, info


def write_split(path, movies, n_cells, track_length, correct_lineages, get_features, window_chunks):
    import zarr
    T = max(m['frames'] for m in movies)
    store = zarr.open(str(path) + '.partial', mode='w')
    arrays, lineages, infos = {}, [], {}
    for b, m in enumerate(movies):
        features, lineage, info = convert(m['root'], m['name'], m['mpp'], n_cells, correct_lineages, get_features)
        if info['frames'] != m['frames']:
            raise ValueError(f"{m['name']}: frame count changed")
        for k, v in features.items():
            if k not in arrays:
                shape = (len(movies), T) + v.shape[2:]
                arrays[k] = store.create_array(k, shape=shape, dtype=v.dtype, fill_value=0,
                                               chunks=window_chunks(shape, track_length))
            arrays[k][b, :v.shape[1]] = v[0]
        lineages.append(lineage)
        infos[m['name']] = info
        print(f"  {path.name} {b + 1}/{len(movies)} {m['name']}: {info}", flush=True)
    store.create_array('mpp', data=np.asarray([m['mpp'] for m in movies], dtype='float64'))
    store.attrs['track_length'] = int(track_length)
    store.attrs['lineages'] = json.loads(json.dumps(lineages))
    Path(str(path) + '.partial').rename(path)
    return infos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scripta-root', type=Path, required=True)
    parser.add_argument('--scripta-mpp', type=float, required=True, help='um/px of the scripta images')
    parser.add_argument('--dnn-root', type=Path, required=True)
    parser.add_argument('--dnn-mpp', type=float, required=True, help='um/px of DynamicNuclearNet')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory')
    parser.add_argument('--track-length', type=int, default=8, help="torch-tracking's training window")
    args = parser.parse_args(argv)
    if args.output_root.exists():
        parser.error(f'{args.output_root} exists; choose a new --output-root')
    if set(TEST_SCRIPTA) & set(TRAIN_SCRIPTA + VAL_SCRIPTA):
        raise ValueError('test wells in train/val')

    from torch_tracking import preprocess
    from torch_tracking.preprocess import correct_lineages, get_features, window_chunks
    if preprocess.LINEAGES_KEY != 'lineages':
        raise ValueError(f'torch-tracking stores lineages under {preprocess.LINEAGES_KEY!r}')

    splits = {
        'train': [dict(root=args.scripta_root, name=w, mpp=args.scripta_mpp) for w in TRAIN_SCRIPTA]
                 + [dict(root=args.dnn_root, name=v, mpp=args.dnn_mpp) for v in TRAIN_DNN],
        'val': [dict(root=args.scripta_root, name=w, mpp=args.scripta_mpp) for w in VAL_SCRIPTA],
    }
    n_cells = {}                    # slots per movie of a split: its most labels after merging and packing (as
                                    # convert_trk_to_zarr, which sets max_cells per converted split)
    for split, movies in splits.items():
        for m in movies:
            frames = sequence(m['root'], m['name'])
            m['frames'] = len(frames)
            y = np.stack([tifffile.imread(p) for _, _, p in frames]).astype(np.int32)
            y, _, _, _ = merge_continuations(y, read_table(Path(m['root']) / f"{m['name']}_GT" / 'TRA' /
                                                           'man_track.txt'))
            y, _ = pack_unknown(y)
            n_cells[split] = max(n_cells.get(split, 0), int((np.unique(y) > 0).sum()))
    print(f'label slots per movie: {n_cells}', flush=True)
    args.output_root.mkdir(parents=True)
    report = dict(source='torch-tracking preprocess.correct_lineages + get_features per movie', n_cells=n_cells,
                  track_length=args.track_length, scripta_root=str(args.scripta_root.resolve()),
                  dnn_root=str(args.dnn_root.resolve()), scripta_mpp=args.scripta_mpp, dnn_mpp=args.dnn_mpp,
                  test_wells_never_read=TEST_SCRIPTA, splits={})
    for split, movies in splits.items():
        report['splits'][split] = write_split(args.output_root / f'{split}_proc.zarr', movies, n_cells[split],
                                              args.track_length, correct_lineages, get_features, window_chunks)
    (args.output_root / 'prepare.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('saved', args.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
