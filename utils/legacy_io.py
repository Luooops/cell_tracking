"""Readers retained for historical evaluation compatibility; no GT generation."""
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from PIL import Image
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
