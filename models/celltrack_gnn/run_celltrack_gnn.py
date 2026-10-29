"""
run_celltrack_gnn.py
====================
CellTrack-GNN (Ben-Haim & Riklin-Raviv, ECCV 2022; github.com/talbenha/cell-tracker-gnn) on our detections, through
its official inference scripts, run unchanged in their own environment (`python` parameter):

    preprocess_seq2graph_clean.py   -> <work>/01_CSV/csv/frame_###.csv   (metric-learning features per detection)
    inference_clean.py              -> <work>/01_RES_inference/          (GNN edge scores)
    postprocess_clean.py            -> <work>/01_RES/mask###.tif + res_track.txt   (tracks, CTC result format)

Input per well (written into <work>, data/ is only read):
    <work>/01/t###.tif              hard links (copies where linking fails) of the raw images
    <work>/01_SEG_RES/mask###.tif   COPIES of the detection masks: preprocess removes fragments below min_cell_size and
                                    overwrites these files in place
Weights = one trained model folder: metric_learning/all_params.pth + gnn/checkpoints/<one>.ckpt with gnn/.hydra/
(inference_clean.py reads the data configuration from <checkpoint>/../../.hydra/config.yaml).

Back to our format: every detection gets the result label covering most of its pixels (track_id); a detection the
official clean-up removed, or that lost a result label shared with a larger detection of the same frame, becomes a
one-detection track (counted in summary.json). The parents in res_track.txt (postprocess gives a track starting in
frame t the nearest track that ended in frame t-1) are written to lineage.json, as the method outputs them; the
identity metrics use track_id only, the CTC evaluation also the parents.
"""
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile

SCRIPTS = 'src/inference'


def csv_read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def find_weights(folder):
    """Trained model folder -> (metric-learning parameters, GNN checkpoint)."""
    folder = Path(folder)
    metric = folder / 'metric_learning' / 'all_params.pth'
    checkpoints = sorted((folder / 'gnn' / 'checkpoints').glob('*.ckpt'))
    if not metric.is_file() or len(checkpoints) != 1 or not (folder / 'gnn' / '.hydra' / 'config.yaml').is_file():
        raise FileNotFoundError(f'{folder}: expected metric_learning/all_params.pth, gnn/.hydra/config.yaml and '
                                f'exactly one gnn/checkpoints/*.ckpt')
    return metric.resolve(), checkpoints[0].resolve()


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage(frames, work):
    """frames: [(frame, image path, mask path)] -> the official input layout under the new folder work."""
    (work / '01').mkdir(parents=True)
    (work / '01_SEG_RES').mkdir()
    for frame, image, mask in frames:
        link(image, work / '01' / f't{frame:03d}.tif')
        shutil.copyfile(mask, work / '01_SEG_RES' / f'mask{frame:03d}.tif')   # rewritten by preprocess


def run_official(work, weights, repo, python, min_cell_size, modality):
    metric, checkpoint = weights
    cwd = Path(repo).resolve() / SCRIPTS
    if not cwd.is_dir():
        raise FileNotFoundError(f'{cwd} not found; pass --set repo="<cell-tracker-gnn checkout>"')
    # The scripts import src.* and src_metric_learning.* from the repository root.
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               PYTHONPATH=os.pathsep.join(filter(None, [str(Path(repo).resolve()), os.environ.get('PYTHONPATH')])))
    work = work.resolve()
    for script, *args in (
            ('preprocess_seq2graph_clean.py', '-cs', str(min_cell_size), '-ii', work / '01',
             '-iseg', work / '01_SEG_RES', '-im', metric, '-oc', work / '01_CSV'),
            ('inference_clean.py', '-mp', checkpoint.as_posix(), '-ns', '01', '-oc', work),
            ('postprocess_clean.py', '-modality', modality, '-iseg', work / '01_SEG_RES',
             '-oi', work / '01_RES_inference')):
        command = [python, script, *map(str, args)]
        print('$', ' '.join(command), flush=True)
        subprocess.run(command, cwd=cwd, env=env, check=True)


def read_result(work, frames):
    """-> ({(frame, detection label): result label}, {result label: parent label}, counts)."""
    results = sorted((work / '01_RES').glob('mask*.tif'))
    if len(results) != len(frames):
        raise ValueError(f'{work / "01_RES"}: {len(results)} result masks for {len(frames)} frames')
    track_of, removed, displaced = {}, 0, 0
    for (frame, _, mask_path), result_path in zip(frames, results):
        mask, result = tifffile.imread(mask_path).astype(np.int64), tifffile.imread(result_path).astype(np.int64)
        if mask.shape != result.shape:
            raise ValueError(f'{result_path}: shape {result.shape}, detections {mask.shape}')
        both = (mask > 0) & (result > 0)
        pairs, counts = np.unique(np.stack([mask[both], result[both]]), axis=1, return_counts=True)
        best = {}                                         # detection -> (overlap, result label)
        for (label, res), n in zip(pairs.T, counts):
            if n > best.get(label, (0, 0))[0]:
                best[label] = (n, res)
        owner = {}                                        # result label -> (overlap, detection): largest keeps it
        for label, (n, res) in best.items():
            if n > owner.get(res, (0, 0))[0]:
                owner[res] = (n, label)
        for label in np.unique(mask[mask > 0]):
            label = int(label)
            if label not in best:
                removed += 1
            elif owner[best[label][1]][1] != label:
                displaced += 1
            else:
                track_of[(frame, label)] = int(best[label][1])
    parent = {}
    for line in (work / '01_RES' / 'res_track.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            label, _, _, mother = map(int, line.split())
            if mother > 0:
                parent[label] = mother
    return track_of, parent, dict(removed_by_cleanup=removed, displaced=displaced)


def track_sequence(frames, rows, work, destination, weights, repo, python, min_cell_size, modality):
    """frames: utils.ctc.sequence of the well; rows: its detections (utils.ctc.detections). Writes destination/
    instance_tracks.csv, lineage.json and summary.json; returns rows with track_id."""
    t0 = time.time()
    stage(frames, work)
    run_official(work, weights, repo, python, min_cell_size, modality)
    track_of, parent, counts = read_result(work, frames)
    next_id = max([*track_of.values(), *parent, *parent.values()], default=0) + 1
    for r in rows:
        key = (int(r['frame_index']), int(r['instance_id']))
        if key not in track_of:
            track_of[key], next_id = next_id, next_id + 1
        r['track_id'] = track_of[key]
    used = set(track_of.values())
    lineage = {d: m for d, m in parent.items() if d in used and m in used}
    destination.mkdir()
    columns = list(rows[0]) if rows else ['frame_index', 'instance_id', 'track_id']
    with open(destination / 'instance_tracks.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    (destination / 'lineage.json').write_text(json.dumps(lineage), encoding='utf-8')
    (destination / 'summary.json').write_text(json.dumps(dict(
        tracker='celltrack_gnn', metric_learning=str(weights[0]), checkpoint=str(weights[1]),
        parameters=dict(min_cell_size=min_cell_size, modality=modality), detections=len(rows),
        tracks=len(set(track_of.values())), parents=len(lineage), seconds=round(time.time() - t0, 1), **counts),
        indent=2), encoding='utf-8')
    return rows
