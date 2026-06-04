from pathlib import Path

import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
from cellpose import models

from mask_area_filter import (
    compute_mask_area_stats,
    filter_small_instances_by_mean,
    remove_small_instances,
)


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(img, p_low=1, p_high=99, bg_sigma=80, black_gamma=1.8):
    img = img.astype(np.float32)

    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)

    bg = gaussian_filter(img, sigma=bg_sigma)
    img = img - bg
    img = np.clip(img, 0, 1)

    # gamma > 1 darkens shadows / background
    img = img ** black_gamma

    return img.astype(np.float32)


# -----------------------------
# Batch inference
# -----------------------------
def batch_segment_folder(
    in_dir: str,
    out_dir: str,
    use_gpu: bool = True,
    diameter: float = 25,
    cellprob_threshold: float = 1.8,
    flow_threshold: float = 1.0,
    p_low: float = 1,
    p_high: float = 99,
    bg_sigma: float = 80,
    black_gamma: float = 1.8,
    min_area: int | None = 0,
    auto_min_area_fraction: float = 0.25,
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # init model once (important for speed)
    model = models.CellposeModel(gpu=use_gpu)

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in exts]

    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {in_dir} with extensions {exts}")

    print(f"[INFO] Found {len(files)} images in: {in_dir}")
    print(f"[INFO] Saving masks to: {out_dir}")

    min_area_by_mask_file = {}
    used_min_areas = []
    mean_areas = []

    for i, img_path in enumerate(files, 1):
        try:
            img = io.imread(str(img_path))

            # If RGB or multi-channel, take first channel for now (minimal and safe)
            # You can customize this if your data has a specific channel layout.
            if img.ndim == 3:
                img = img[..., 0]

            img01 = preprocess(
                img,
                p_low=p_low,
                p_high=p_high,
                bg_sigma=bg_sigma,
                black_gamma=black_gamma,
            )

            masks, flows, styles = model.eval(
                img01,
                diameter=diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
            )

            n_before = int(masks.max())
            stats = compute_mask_area_stats(masks)

            if min_area is None:
                masks, used_min_area, stats = filter_small_instances_by_mean(
                    masks,
                    fraction_of_mean=auto_min_area_fraction,
                )
            else:
                used_min_area = min_area
                masks = remove_small_instances(masks, min_area=used_min_area)

            n_after = int(masks.max())

            # Save as label image (0=background, 1..N=instances)
            # Use uint16 unless you expect >65535 instances (unlikely for nuclei)
            out_path = out_dir / f"{img_path.stem}_mask.tiff"
            io.imsave(str(out_path), masks.astype(np.uint16), check_contrast=False)

            min_area_by_mask_file[out_path.name] = int(used_min_area)
            used_min_areas.append(int(used_min_area))
            mean_areas.append(float(stats.mean_area))

            if i % 10 == 0 or i == 1:
                print(
                    f"[OK] {i}/{len(files)}  {img_path.name}  "
                    f"instances={n_after} removed_small={n_before - n_after}  "
                    f"mean_area={stats.mean_area:.1f} min_area={used_min_area}"
                )

        except Exception as e:
            print(f"[ERROR] {i}/{len(files)}  {img_path}  -> {repr(e)}")

    print("[DONE] Batch segmentation finished.")
    return {
        "min_area_by_mask_file": min_area_by_mask_file,
        "used_min_areas": used_min_areas,
        "mean_areas": mean_areas,
        "mean_used_min_area": float(np.mean(used_min_areas)) if used_min_areas else 0.0,
        "mean_cell_area": float(np.mean(mean_areas)) if mean_areas else 0.0,
    }


if __name__ == "__main__":
    INPUT_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2"
    OUTPUT_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2/masks_out"

    batch_segment_folder(
        in_dir=INPUT_DIR,
        out_dir=OUTPUT_DIR,
        use_gpu=True,
        diameter=25,
        cellprob_threshold=1.8,
        flow_threshold=1.0,
        p_low=1,
        p_high=99,
        bg_sigma=80,
        black_gamma=1.8,
        min_area=None,
        auto_min_area_fraction=0.25,
    )
