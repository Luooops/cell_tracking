"""Cellpose helpers shared by folder and single-image workflows."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tifffile
from skimage import io
from scipy.ndimage import gaussian_filter
from main_tracking.mask_area_filter import compute_mask_area_stats, filter_small_instances_by_mean, remove_small_instances


def preprocess(img, p_low=1, p_high=99, bg_sigma=80, black_gamma=1.8):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (p_low, p_high))
    img = (np.clip(img, lo, hi) - lo) / (hi - lo + 1e-8)
    return np.clip(img - gaussian_filter(img, sigma=bg_sigma), 0, 1) ** black_gamma


def read_image(path):
    image = tifffile.imread(path) if path.suffix.lower() in {".tif", ".tiff"} else io.imread(path)
    if image.ndim != 2:
        raise ValueError(f"{path}: expected single-frame 2D grayscale, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: non-finite pixels")
    return image


def segment_image(model, image, *, diameter=25, cellprob_threshold=1.8,
                  flow_threshold=1.0, min_area=None, auto_min_area_fraction=0.25,
                  preprocessing="legacy", **preprocess_kwargs):
    processed = preprocess(image, **preprocess_kwargs) if preprocessing == "legacy" else image
    masks, _, _ = model.eval(processed, diameter=diameter,
                            cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)
    stats = compute_mask_area_stats(masks)
    if min_area is None:
        masks, threshold, _ = filter_small_instances_by_mean(masks, auto_min_area_fraction)
    else:
        threshold = min_area
        masks = remove_small_instances(masks, min_area)
    return masks, int(threshold), stats


def save_mask(path, masks):
    dtype = np.uint16 if masks.max() <= 65535 else np.uint32
    tifffile.imwrite(path, masks.astype(dtype), photometric="minisblack")


def batch_segment_folder(in_dir, out_dir, use_gpu=True, diameter=25,
                         cellprob_threshold=1.8, flow_threshold=1.0,
                         p_low=1, p_high=99, bg_sigma=80, black_gamma=1.8,
                         min_area=0, auto_min_area_fraction=0.25):
    """Compatibility helper. New main uses the grouped manifest workflow."""
    from cellpose import models
    source, destination = Path(in_dir).resolve(), Path(out_dir).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Output must be outside the input directory")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
    if not files:
        raise FileNotFoundError(f"No images in {source}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate image stems; use the grouped main pipeline")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam_v2")
    thresholds, means = {}, []
    for path in files:
        mask, threshold, stats = segment_image(model, read_image(path), diameter=diameter,
            cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold,
            min_area=min_area, auto_min_area_fraction=auto_min_area_fraction,
            p_low=p_low, p_high=p_high, bg_sigma=bg_sigma, black_gamma=black_gamma)
        target = destination / f"{path.stem}_mask.tiff"
        save_mask(target, mask)
        thresholds[target.name] = threshold
        means.append(stats.mean_area)
    return dict(min_area_by_mask_file=thresholds, used_min_areas=list(thresholds.values()),
                mean_used_min_area=float(np.mean(list(thresholds.values()))), mean_cell_area=float(np.mean(means)))
