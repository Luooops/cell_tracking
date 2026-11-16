"""
run_caliban.py
==============
Caliban baseline (Schwartz et al.) through torch-tracking, the official PyTorch port of the Caliban tracking model
(github.com/vanvalenlab/torch-tracking; weights vanvalenlab/torch-tracking on HuggingFace, gated). It tracks the fixed
detections (the GT masks of each frame, as for every other method), so the result can be scored with
models/gnn_tracking/evaluate_ctc.py exactly like the other runs.

Runs in its own conda env `caliban` (Python 3.12, torch + torch_geometric, torch-tracking installed from the clone);
HF_TOKEN must allow the gated weights. Defaults are torch-tracking's own inference settings (division 0.5, birth and
death 0.999, distance threshold 89 px at the model's 0.55 um/px, 8 frames of track history); --mpp is the pixel size
of the data (DynamicNuclearNet: 0.65 um/px, from the Caliban paper's 64 px = 41.6 um).

Fairness on DynamicNuclearNet: the released weights were trained on the DNN train split with the val split for early
stopping, so only test videos are unseen by Caliban; val videos are reported separately.

    python models/caliban/run_caliban.py --data-root data/DynamicNuclearNet \
        --videos test09 test10 test11 --output-root development/stage10_caliban/dnn/caliban

Writes into a new --output-root: <video>/instance_tracks.csv (frame_index, instance_id, track_id),
<video>/lineage.json ({child track: parent track}) and run.json (settings, weights hash, versions, timings).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.ctc import sequence  # noqa: E402


def load(root, video):
    frames = sequence(root, video)
    if [t for t, _, _ in frames] != list(range(len(frames))):
        raise ValueError(f'{video}: frames are not 0..T-1')
    X = np.stack([tifffile.imread(p) for _, p, _ in frames]).astype(np.float32)[..., None]
    y = np.stack([tifffile.imread(p) for _, _, p in frames]).astype(np.int32)[..., None]
    return X, y


def assignments(y, y_tracked):
    """-> rows (frame, GT label, Caliban track) by the track label covering each GT mask; checks the masks agree."""
    rows, mismatched = [], 0
    for t in range(y.shape[0]):
        gt, tr = y[t, ..., 0], y_tracked[t, ..., 0]
        fg = gt > 0
        pairs, counts = np.unique(np.stack([gt[fg], tr[fg]]), axis=1, return_counts=True)
        best = {}
        for (g, k), n in zip(pairs.T, counts):
            if k and n > best.get(int(g), (0, 0))[1]:
                best[int(g)] = (int(k), int(n))
        area = dict(zip(*np.unique(gt[fg], return_counts=True)))
        for g, (k, n) in best.items():
            mismatched += int(area[g] - n)
            rows.append((t, g, k))
        missing = set(int(g) for g in area) - set(best)
        if missing:
            raise ValueError(f'frame {t}: GT labels without a Caliban track: {sorted(missing)[:10]}')
    return rows, mismatched


def lineage_of(lineage):
    """torch-tracking lineage dict -> {child track: parent track} from the daughters lists."""
    out = {}
    for label, rec in lineage.items():
        for d in rec['daughters']:
            if int(d) in out and out[int(d)] != int(label):
                raise ValueError(f'track {d} has two parents')
            out[int(d)] = int(label)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True, help='New directory')
    parser.add_argument('--mpp', type=float, required=True, help='Pixel size of the data in um/px')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--weights', type=Path, help='Checkpoint file; default: the published HuggingFace weights')
    args = parser.parse_args(argv)
    if args.output_root.exists():
        parser.error(f'{args.output_root} exists; choose a new --output-root')

    import torch
    import torch_tracking
    from torch_tracking.tracker import CellTracker

    args.output_root.mkdir(parents=True)
    report = dict(method='Caliban (torch-tracking, official PyTorch port)', data_root=str(args.data_root.resolve()),
                  mpp=args.mpp, device=args.device, videos={}, torch=torch.__version__,
                  torch_tracking_path=str(Path(torch_tracking.__file__).parent))
    clone = Path(torch_tracking.__file__).resolve().parents[1]
    try:
        report['torch_tracking_commit'] = subprocess.run(['git', '-C', str(clone), 'rev-parse', 'HEAD'],
                                                         capture_output=True, text=True).stdout.strip() or None
    except OSError:
        report['torch_tracking_commit'] = None
    for video in args.videos:
        t0 = time.time()
        X, y = load(args.data_root, video)
        tracker = CellTracker(checkpoint_path=str(args.weights) if args.weights else None, device=args.device,
                              mpp=args.mpp, verbose=False)
        settings = dict(division=tracker.division, birth=tracker.birth, death=tracker.death,
                        distance_threshold=tracker.distance_threshold, track_length=tracker.track_length,
                        model_mpp=tracker.model_mpp, checkpoint=str(tracker.checkpoint_path))
        tracker.preprocess_movie(movie=X, annotation=y)
        tracker.track_cells()
        y_tracked = np.asarray(tracker.y_tracked)
        if y_tracked.shape != y.shape:
            raise ValueError(f'{video}: y_tracked {y_tracked.shape} != masks {y.shape}')
        rows, mismatched = assignments(y, y_tracked)
        lineage = lineage_of(tracker.get_lineage_dict())
        out = args.output_root / video
        out.mkdir()
        pd.DataFrame(rows, columns=['frame_index', 'instance_id', 'track_id']).to_csv(out / 'instance_tracks.csv',
                                                                                     index=False)
        (out / 'lineage.json').write_text(json.dumps({str(k): v for k, v in sorted(lineage.items())}),
                                          encoding='utf-8')
        report.setdefault('settings', settings)
        report['videos'][video] = dict(frames=int(y.shape[0]), detections=len(rows),
                                       tracks=len({k for _, _, k in rows}), divisions=len(set(lineage.values())),
                                       mismatched_pixels=mismatched, seconds=round(time.time() - t0, 1))
        print(f'{video}: {report["videos"][video]}', flush=True)
        del tracker
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None
    report['weights_sha256'] = sha256(report['settings']['checkpoint'])
    (args.output_root / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('saved', args.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
