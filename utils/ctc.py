"""Read the existing lab CTC layout and adapt its detections to legacy manifests."""
import csv
import json
from pathlib import Path
import numpy as np
import tifffile


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sequence(root, well):
    root = Path(root).resolve()
    if Path(well).name != well or well in {'.', '..'} or ':' in well or '\\' in well:
        raise ValueError(f'Invalid well: {well}')
    images = sorted((root / well).glob('t*.tif'))
    masks = sorted((root / f'{well}_GT' / 'TRA').glob('man_track*.tif'))
    if not images or not masks:
        raise FileNotFoundError(f'No images/masks for {well} under {root}')
    raw = {int(p.stem[1:]): p for p in images}
    labelled = {int(p.stem[len('man_track'):]): p for p in masks}
    if raw.keys() != labelled.keys():
        raise ValueError(f'Frame/mask mismatch for {well}')
    return [(t, raw[t], labelled[t]) for t in sorted(raw)]


def detections(root, well):
    from models.classical_tracking.tracking_v0 import extract_detections_from_mask
    frames, rows, gt = [], [], {}
    for frame, image_path, mask_path in sequence(root, well):
        mask = tifffile.imread(mask_path)
        with tifffile.TiffFile(image_path) as image:
            shape = image.series[0].shape
        if mask.ndim != 2 or tuple(shape) != mask.shape:
            raise ValueError(f'Expected matching 2-D images/masks: {image_path}')
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f'Expected integer labels: {mask_path}')
        frames.append(dict(sequence_id=well, frame_index=frame, time_index=frame,
            image_name=image_path.name, source_relative_path=image_path.name,
            mask_path=str(mask_path), overlay_path='', height=mask.shape[0], width=mask.shape[1], status='complete'))
        for row in extract_detections_from_mask(mask, frame):
            label = row['label']
            row.update(instance_id=label, frame_index=frame, image_name=image_path.name,
                       mask_path=str(mask_path), track_id=0)
            rows.append(row)
            if label < 50000:
                gt[(frame, label)] = label
    return frames, rows, gt


def manifest(folder, root, well, frames, rows):
    folder.mkdir(parents=True)
    write_csv(folder / 'frames.csv', frames, list(frames[0]))
    columns = list(rows[0]) if rows else ['image_name', 'instance_id', 'track_id', 'frame_index', 'mask_path']
    write_csv(folder / 'instance_tracks.csv', rows, columns)
    (folder / 'summary.json').write_text(json.dumps(dict(status='complete', sequence_id=well,
        input_dir=str(Path(root).resolve() / well), parameters={'min_track_length': 5},
        mask_values='Lab CTC labels; GT identity is used only for evaluation')), encoding='utf-8')
