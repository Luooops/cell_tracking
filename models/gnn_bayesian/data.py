"""
data.py
=======
Detections of a 'dataset/well' (any dataset of config.json, or a folder under data_dir), cached per dataset.

Columns added to utils.dataset.read_well's table:
    cell_id      GT cell (utils.dataset.cell_ids: single-child continuations are one cell)
    mother_cell  for a division daughter (a parent with exactly two children): the mother's cell; -1 otherwise
    cell_first, cell_last   first / last frame in which the detection's cell is observed
A candidate edge is a division edge if its target is a daughter of its source's cell, the source is that cell's
last observation and the target the daughter's first (graph classes in edge_classes()).
"""
from pathlib import Path

import numpy as np
import torch

from utils.config import read_config, resolve_dataset
from utils.dataset import cell_ids, read_well
from utils.lineage import Lineage, read_track_table

UNRELATED, SAME, DIVISION = 0, 1, 2
_ROOTS = {}


def dataset_root(name):
    if name not in _ROOTS:
        cfg, base = read_config()
        _ROOTS[name] = resolve_dataset(cfg, base, name, None)['root']
    return _ROOTS[name]


def split(spec):
    dataset, well = spec.split('/', 1)
    return dataset, well


def load(spec, cache, flip=None):
    """'dataset/well' -> (df, crops) as utils.dataset.load_well, with the columns above; flip = None | 'h' | 'v'."""
    dataset, well = split(spec)
    root = dataset_root(dataset)
    path = Path(cache) / dataset / f'{well}.pt'
    if path.exists():
        d = torch.load(path, weights_only=False)
        df, crops, (height, width) = d['df'], d['crops'].numpy(), d['shape']
    else:
        print(f'  caching {spec} (first use: reading frames and masks)', flush=True)
        df, crops, (height, width) = read_well(well, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'df': df, 'crops': torch.from_numpy(crops), 'shape': (height, width)}, path)
    labels = df.track_id.to_numpy()
    df = df.assign(cell_id=cell_ids(well, labels, root))
    table_path = Path(root) / f'{well}_GT' / 'TRA' / 'man_track.txt'
    mother = {}
    if table_path.exists():
        lineage = Lineage(read_track_table(table_path), ())
        for p, kids in lineage.children.items():
            if len(kids) == 2 and lineage.known(p):
                for k in kids:
                    mother[k] = lineage.cell_of.get(p, p)
    df['mother_cell'] = [mother.get(int(l), -1) for l in labels]
    span = df.groupby('cell_id').frame.agg(['min', 'max'])
    df['cell_first'] = span['min'].reindex(df.cell_id).to_numpy()
    df['cell_last'] = span['max'].reindex(df.cell_id).to_numpy()
    df['dataset'] = dataset
    if flip == 'h':
        df = df.assign(x=width - 1 - df.x)
        crops = crops[:, :, ::-1]
    elif flip == 'v':
        df = df.assign(y=height - 1 - df.y)
        crops = crops[:, ::-1, :]
    return df, np.ascontiguousarray(crops)


def edge_classes(video, gr, t, g):
    """[C] long: UNRELATED / SAME / DIVISION of the candidate edges of the graph (t, t + g)."""
    src, dst = gr.candidate_edge_index.numpy()
    dst = dst - gr.n_t
    a, b = video.df.loc[video.rows[t]], video.df.loc[video.rows[t + g]]
    same = gr.candidate_edge_label.numpy() > 0.5
    division = ((b.mother_cell.to_numpy()[dst] == a.cell_id.to_numpy()[src])
                & (a.frame.to_numpy()[src] == a.cell_last.to_numpy()[src])
                & (b.frame.to_numpy()[dst] == b.cell_first.to_numpy()[dst]))
    return torch.from_numpy(np.where(same, SAME, np.where(division, DIVISION, UNRELATED))).long()
