"""Refine existing cell polygons with Cellpose; retain XML identities and layout."""
import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

# Configure MKL before NumPy/SciPy/PyTorch load their native runtimes.
# Sequential MKL avoids a second Intel OpenMP runtime; CUDA is unaffected.
if sys.platform == 'win32':
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import cv2
import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import find_objects
from types import SimpleNamespace


def load_xml(path):
    member = None
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.lower().endswith('.xml')]
            if len(members) != 1:
                raise ValueError(f'Expected one XML member: {path}')
            member = members[0]
            data = z.read(member)
    else:
        data = path.read_bytes()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    return ET.fromstring(data, parser=parser), member


def match_masks(polys, masks, max_distance, min_overlap):
    """Maximize valid one-to-one matches, then minimize overlap/position cost."""
    regions = []
    for label, box in enumerate(find_objects(masks), 1):
        if box is None:
            continue
        yy, xx = np.nonzero(masks[box] == label)
        regions.append(SimpleNamespace(label=label, area=len(yy),
            centroid=(yy.mean()+box[0].start, xx.mean()+box[1].start)))
    n, m = len(polys), len(regions)
    if not n or not m:
        return {}
    lookup = {r.label: j for j, r in enumerate(regions)}
    cost = np.full((n, m), np.inf)
    for i, node in enumerate(polys):
        pts = np.array([list(map(float, p.split(','))) for p in node.attrib['points'].split(';')])
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError('Invalid GT polygon')
        raster = np.zeros(masks.shape, np.uint8)
        cv2.fillPoly(raster, [np.rint(pts).astype(np.int32)], 1)
        rr, cc = np.nonzero(raster)
        if not len(rr):
            continue
        cy, cx = rr.mean(), cc.mean()
        labels, overlaps = np.unique(masks[rr, cc], return_counts=True)
        for label, overlap in zip(labels, overlaps):
            if label == 0:
                continue
            j = lookup[label]
            r = regions[j]
            containment = overlap / min(len(rr), r.area)
            distance = np.hypot(cy-r.centroid[0], cx-r.centroid[1])
            if containment >= min_overlap and distance <= max_distance:
                iou = overlap / (len(rr) + r.area - overlap)
                area_difference = abs(len(rr)-r.area) / (len(rr)+r.area)
                cost[i, j] = distance/max_distance + (1-iou) + area_difference
    penalty = 3 * (n+1)
    matrix = np.concatenate((np.where(np.isfinite(cost), cost, penalty*3),
                             np.full((n, n), penalty)), axis=1)
    rows, cols = linear_sum_assignment(matrix)
    return {int(i): regions[j].label for i, j in zip(rows, cols)
            if j < m and np.isfinite(cost[i, j])}


def outline(masks, label, changes=None):
    binary = (masks == label).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    external = [] if hierarchy is None else [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]
    valid = [c for c in external if len(c) >= 3 and cv2.contourArea(c) > 0]
    if not valid:
        raise ValueError(f'Cellpose instance {label} has no nonzero-area polygon')
    contour = max(valid, key=cv2.contourArea)
    raster = np.zeros_like(binary)
    cv2.fillPoly(raster, [contour], 1)
    added = int(np.count_nonzero((raster != 0) & (binary == 0)))
    dropped = int(np.count_nonzero((binary != 0) & (raster == 0)))
    if changes is not None and (added or dropped):
        changes.append(dict(cellpose_label=int(label), reason='single_outer_polygon',
                            filled_pixels=added, dropped_pixels=dropped,
                            external_regions=len(external)))
    return ';'.join(f'{x},{y}' for x, y in contour[:, 0, :])


def refine_image(image, masks, distance, overlap, geometry_changes=None):
    # Keep original polygon indices in the audit, including non-cell objects.
    original = [(i, p) for i, p in enumerate(image.findall('polygon'), 1)
                if p.get('label') == 'cell']
    ids = {p: next(((a.text or '').strip() for a in p.findall('attribute')
                   if a.get('name') == 'track_id'), '') for _, p in original}
    counts = Counter(ids.values())
    removed, polys = [], []
    indices = dict((p, i) for i, p in original)
    for index, node in original:
        tid = ids[node]
        reason = 'no_track_id' if not tid else 'duplicate_id_in_frame' if counts[tid] > 1 else None
        if reason:
            removed.append(dict(polygon_index=index, track_id=tid, reason=reason))
            image.remove(node)
        else:
            polys.append(node)
    matches = match_masks(polys, masks, distance, overlap)
    for i, node in enumerate(polys):
        if i in matches:
            changes = []
            try:
                node.set('points', outline(masks, matches[i], changes))
            except ValueError:
                removed.append(dict(polygon_index=indices[node], track_id=ids[node],
                                    reason='degenerate_cellpose_outline', cellpose_label=int(matches[i])))
                del matches[i]
            if geometry_changes is not None:
                geometry_changes.extend(dict(polygon_index=indices[node], track_id=ids[node], **c)
                                        for c in changes)
        else:
            removed.append(dict(polygon_index=indices[node], track_id=ids[node],
                                reason='no_cellpose_match'))
    for i, node in enumerate(polys):
        if i not in matches:
            image.remove(node)
    removed.sort(key=lambda r: r['polygon_index'])
    return len(original), len(matches), removed


def output_complete(path, out, images):
    """Recognize legacy outputs too; an XML alone is not a finished export."""
    if not out.is_file():
        return False
    try:
        root, _ = load_xml(out)
        if [im.attrib for im in root.findall('image')] != [im.attrib for im in images]:
            return False
        for raw in path.parent.iterdir():
            if raw.is_file() and raw.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
                copied = out.parent / raw.name
                if not copied.is_file() or copied.stat().st_size != raw.stat().st_size:
                    return False
        return True
    except (OSError, ET.ParseError, ValueError, zipfile.BadZipFile):
        return False


def save_audit(target, audit):
    temporary = target / 'refine_audit.json.tmp'
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target / 'refine_audit.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gt-root', type=Path, default=Path('E:/aws_gt_data'))
    ap.add_argument('--out-root', type=Path, default=Path('E:/aws_gt_data_refine'))
    ap.add_argument('--model', type=Path, default=Path('E:/aws_gt_data/cellpose_mask_gt/model/cp4_20260721_210258'))
    ap.add_argument('--max-distance', type=float, default=60, help='GT/Cellpose centroid distance, pixels')
    ap.add_argument('--min-overlap', type=float, default=0.5, help='Intersection / smaller area')
    ap.add_argument('--cellprob-threshold', type=float, default=0)
    ap.add_argument('--flow-threshold', type=float, default=0.4)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    source, target = args.gt_root.resolve(), args.out_root.resolve()
    if source == target or source in target.parents or target in source.parents:
        ap.error('Input and output trees must be separate')
    if not args.model.is_file() or not np.isfinite(args.max_distance) or args.max_distance <= 0:
        ap.error('Missing model or invalid distance')
    if not 0 < args.min_overlap <= 1:
        ap.error('Invalid overlap')
    parameters = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    audit_path = target / 'refine_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else dict(frames=[])
    if audit_path.exists() and (audit.get('parameters') != parameters or audit.get('model_sha256') != model_hash):
        ap.error('Existing output uses different parameters/model; choose a new --out-root')
    audit.update(status='running', parameters=parameters, model_sha256=model_hash)
    files = sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.xml', '.zip')
                   and 'cellpose_mask_gt' not in p.relative_to(source).parts and p.is_file())
    if not files:
        ap.error('No XML annotations found')
    jobs = []
    for path in files:
        root, member = load_xml(path)
        images = root.findall('image')
        if not images:
            raise ValueError(f'No per-image annotations: {path}')
        for im in images:
            raw = path.parent / im.attrib['name'].replace('\\', '/').split('/')[-1]
            if not raw.is_file():
                raise FileNotFoundError(raw)
        if output_complete(path, target / path.relative_to(source), images):
            print(f'Skipped complete: {path.relative_to(source)}', flush=True)
        else:
            jobs.append((path, root, member, images))
    target.mkdir(parents=True, exist_ok=True)
    if not jobs:
        audit['status'] = 'complete'
        save_audit(target, audit)
        print('All XMLs already complete; no model inference required.')
        return
    from cellpose import models
    import torch
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --cpu explicitly')
    model = models.CellposeModel(gpu=not args.cpu, pretrained_model=str(args.model))
    try:
        for path, root, member, images in jobs:
            # A failed XML is restarted; do not duplicate its partial audit rows.
            audit['frames'] = [r for r in audit['frames'] if r['xml'] != str(path.relative_to(source))]
            out = target / path.relative_to(source)
            out.parent.mkdir(parents=True, exist_ok=True)
            for im in images:
                raw_path = path.parent / im.attrib['name'].replace('\\', '/').split('/')[-1]
                raw = tifffile.imread(raw_path)
                expected = (int(im.attrib['height']), int(im.attrib['width']))
                if raw.shape != expected or not np.isfinite(raw).all():
                    raise ValueError(f'Invalid image dimensions/pixels: {raw_path}')
                masks = np.asarray(model.eval(np.stack([raw.astype(np.float32)]*3, axis=-1),
                    cellprob_threshold=args.cellprob_threshold, flow_threshold=args.flow_threshold)[0])
                if masks.shape != expected or not np.issubdtype(masks.dtype, np.integer) or masks.min() < 0:
                    raise ValueError(f'Invalid segmentation mask: {raw_path}')
                geometry_changes = []
                before, kept, removed = refine_image(im, masks.astype(np.int32), args.max_distance, args.min_overlap, geometry_changes)
                audit['frames'].append(dict(xml=str(path.relative_to(source)), image=im.attrib['name'],
                                            before=before, kept=kept, removed=removed, geometry_changes=geometry_changes))
                print(f'{path.parent.name}/{raw_path.name}: {before} -> {kept}', flush=True)
            data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            if member:
                with zipfile.ZipFile(path) as old, zipfile.ZipFile(out, 'w') as new:
                    for info in old.infolist():
                        new.writestr(info, data if info.filename == member else old.read(info.filename))
            else:
                out.write_bytes(data)
            # Copy all image channels in the corresponding annotation folder.
            for raw_path in path.parent.iterdir():
                if raw_path.is_file() and raw_path.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
                    shutil.copy2(raw_path, out.parent / raw_path.name)
            save_audit(target, audit)
        audit['status'] = 'complete'
    finally:
        save_audit(target, audit)


if __name__ == '__main__':
    main()
