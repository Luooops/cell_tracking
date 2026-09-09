"""Generate per-frame instance TIFF masks, PNG overlays and a track-ID CSV.

Requires numpy, Pillow and tifffile. Input may be plain XML or ZIP disguised
as XML. Image basenames must match XML names exactly (including channel).
"""

import argparse
import colorsys
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import tifffile


def read_gt(path):
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {'.xml', '.zip'})
        if len(files) != 1:
            raise ValueError(f'Expected one XML/ZIP in {path}, found {len(files)}; pass a file.')
        path = files[0]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist()
                       if not m.is_dir() and m.filename.lower().endswith('.xml')]
            if len(members) != 1:
                raise ValueError('Expected exactly one XML member in archive.')
            with archive.open(members[0]) as source:
                return ET.parse(source).getroot()
    return ET.parse(path).getroot()


def output_name(data_dir):
    parts = data_dir.resolve().parts
    ids = [p for p in parts if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', p)]
    wells = [p for p in parts if re.fullmatch(r'r\d+c\d+', p, re.I)]
    chosen = ids[-1:] + wells[-1:]
    if not chosen or parts[-1] not in chosen:
        chosen.append(parts[-1])
    return '__'.join(chosen)


def read_display(path, width, height):
    if path.suffix.lower() in {'.tif', '.tiff'}:
        data = tifffile.imread(path)
    else:
        with Image.open(path) as image:
            data = np.asarray(image)
    if data.shape not in {(height, width), (height, width, 3), (height, width, 4)}:
        raise ValueError(f'{path}: shape {data.shape} does not match XML {(height, width)}; expected single-frame grayscale/RGB.')
    data = data.astype(np.float32)
    if data.ndim == 3:
        data = data[..., :3]
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError(f'{path}: no finite pixels')
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = finite.min(), finite.max()
    data = np.nan_to_num(data, nan=float(low), posinf=float(high), neginf=float(low))
    data = (np.clip((data - low) / max(float(high - low), 1e-8), 0, 1) * 255).astype(np.uint8)
    return np.repeat(data[..., None], 3, axis=2) if data.ndim == 2 else data


def color_for(key):
    hue = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') / 2**32
    return tuple(round(v * 255) for v in colorsys.hsv_to_rgb(hue, .75, 1))


def run(args):
    root = read_gt(args.gt)
    frames = root.findall('image')
    if not frames:
        raise ValueError('No <image> nodes found; expected per-image polygon annotations.')
    frames.sort(key=lambda frame: int(frame.attrib['id']))
    index = {}
    for path in args.data_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}:
            index.setdefault(path.name, []).append(path)
    jobs = []
    stems = set()
    for frame in frames:
        basename = frame.attrib['name'].replace('\\', '/').split('/')[-1]
        matches = index.get(basename, [])
        if len(matches) != 1:
            raise ValueError(f'{basename}: found {len(matches)} matching source images; expected exactly one in --data-dir.')
        stem = Path(basename).stem
        if stem in stems:
            raise ValueError(f'Duplicate output name: {stem}')
        stems.add(stem)
        jobs.append((frame, matches[0], stem))
    out = args.output_root / output_name(args.data_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output folder is not empty: {out}. Use a different --output-root.')
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    (out / 'overlays').mkdir()
    rows, summary = [], []
    for frame, source, stem in jobs:
        width, height = int(frame.attrib['width']), int(frame.attrib['height'])
        base = read_display(source, width, height)
        labels = np.zeros((height, width), dtype=np.uint32)
        polygons = [p for p in frame.findall('polygon') if p.get('label') == args.label]
        palette = np.zeros((len(polygons) + 1, 3), dtype=np.uint8)
        frame_rows, text_labels = [], []
        overlap = np.zeros((height, width), dtype=bool)
        for instance, polygon in enumerate(polygons, 1):
            vertices = np.array([list(map(float, pair.split(',')))
                                 for pair in polygon.attrib['points'].split(';')])
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
                raise ValueError(f'{stem}: invalid polygon {instance}')
            # Explicit rounding to nearest integer, ties toward positive infinity.
            points = [tuple(map(int, point)) for point in np.floor(vertices + .5)]
            region = Image.new('1', (width, height))
            ImageDraw.Draw(region).polygon(points, fill=1)
            selected = np.asarray(region, dtype=bool)
            overlap |= selected & (labels != 0)
            labels[selected] = instance  # Later polygon in XML owns overlap pixels.
            track = next(((a.text or '').strip() for a in polygon.findall('attribute')
                          if a.get('name') == 'track_id'), '')
            palette[instance] = color_for('track:' + track if track else f"missing:{frame.get('id')}:{instance}")
            ys, xs = np.nonzero(selected)
            if len(xs):
                text_labels.append(((int(xs.mean()), int(ys.mean())), track or f'?{instance}'))
            frame_rows.append(dict(frame_id=frame.get('id'), image_name=source.name,
                                   instance_id=instance, track_id=track,
                                   polygon_pixels=int(selected.sum())))
        areas = np.bincount(labels.ravel(), minlength=len(polygons) + 1)
        for row in frame_rows:
            row['mask_pixels'] = int(areas[row['instance_id']])
        rows.extend(frame_rows)
        mask = labels.astype(np.uint16) if len(polygons) <= 65535 else labels
        tifffile.imwrite(out / 'masks' / f'{stem}_mask.tiff', mask, photometric='minisblack')
        foreground = labels != 0
        base[foreground] = ((1 - args.alpha) * base[foreground] + args.alpha * palette[labels[foreground]]).astype(np.uint8)
        overlay = Image.fromarray(base)
        if not args.no_ids:
            draw = ImageDraw.Draw(overlay)
            for position, text in text_labels:
                draw.text(position, text, fill='white', stroke_width=1, stroke_fill='black')
        overlay.save(out / 'overlays' / f'{stem}_overlay.png')
        summary.append(dict(frame_id=frame.get('id'), image_name=source.name,
                            instances=len(polygons), missing_track_ids=sum(not r['track_id'] for r in frame_rows),
                            overlap_pixels=int(overlap.sum())))
        print(f'{stem}: {len(polygons)} cells, {int(overlap.sum())} overlap pixels', flush=True)
    with (out / 'instance_tracks.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame_id', 'image_name', 'instance_id', 'track_id', 'polygon_pixels', 'mask_pixels'])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(gt=str(args.gt.resolve()), data_dir=str(args.data_dir.resolve()),
                  mask_values='0=background; positive values=frame-local instance IDs; see instance_tracks.csv',
                  rasterization='round vertices with floor(x+0.5), Pillow polygon fill; clip at image bounds',
                  overlap_policy='later polygon in XML wins',
                  overlay='1st-99th percentile display scaling; stable track colors; ?instance for missing track ID',
                  frames=summary)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {len(jobs)} frames to: {out.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', required=True, type=Path, help='GT XML/ZIP file or folder with one XML/ZIP')
    parser.add_argument('--data-dir', required=True, type=Path, help='Original cell image folder (searched recursively)')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--label', default='cell')
    parser.add_argument('--alpha', type=float, default=.35)
    parser.add_argument('--no-ids', action='store_true', help='Hide track-ID text on overlays')
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    if not args.data_dir.is_dir():
        parser.error('--data-dir must be an existing directory')
    run(args)


if __name__ == '__main__':
    main()
