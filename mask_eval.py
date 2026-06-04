import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops


# =============================
# Utility functions
# =============================
def normalize_for_display(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """
    Normalize image to [0, 1] for visualization only.
    """
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


def make_overlay(img: np.ndarray, mask: np.ndarray, boundary_color=(1.0, 0.0, 0.0)) -> np.ndarray:
    """
    Create RGB overlay image with mask boundaries on top of the original image.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image.
    mask : np.ndarray
        2D instance mask, where 0 is background and 1..N are object IDs.
    boundary_color : tuple
        RGB color for boundaries.

    Returns
    -------
    overlay : np.ndarray
        RGB image in [0, 1].
    """
    img_vis = normalize_for_display(img)

    # Convert grayscale image to RGB
    overlay = np.stack([img_vis, img_vis, img_vis], axis=-1)

    # Extract mask boundaries
    boundaries = find_boundaries(mask, mode="outer")

    # Paint boundaries
    overlay[boundaries] = boundary_color

    return overlay


def collect_instance_areas(mask: np.ndarray) -> list:
    """
    Extract area of each instance from an instance mask.
    """
    props = regionprops(mask)
    areas = [p.area for p in props]
    return areas


# =============================
# Main evaluation function
# =============================
def evaluate_segmentation(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    image_exts=(".tif", ".tiff"),
    mask_suffix="_mask",
    overlay_sample_limit=None,
    hist_bins=50,
):
    """
    Evaluate segmentation without ground truth.

    Outputs:
    1. Overlay images
    2. Area histogram

    Parameters
    ----------
    image_dir : str
        Directory containing original images.
    mask_dir : str
        Directory containing predicted masks.
    output_dir : str
        Directory to save outputs.
    image_exts : tuple
        Allowed image file extensions.
    mask_suffix : str
        Mask filename suffix. Example:
        image: aaa.tif
        mask : aaa_mask.tif
        then mask_suffix = "_mask"
    overlay_sample_limit : int or None
        If set, only save first N overlay images.
        If None, save all overlays.
    hist_bins : int
        Number of bins for area histogram.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find original images
    image_files = []
    for ext in image_exts:
        image_files.extend(sorted(image_dir.glob(f"*{ext}")))
        image_files.extend(sorted(image_dir.glob(f"*{ext.upper()}")))

    # Remove duplicates if any
    image_files = sorted(list(set(image_files)))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    print(f"[INFO] Found {len(image_files)} original images")

    all_areas = []
    matched_count = 0
    saved_overlay_count = 0

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem

        # Try to find corresponding mask:
        # image: xxx.tif
        # mask : xxx_mask.tif / xxx_mask.tiff
        candidate_masks = [
            mask_dir / f"{stem}{mask_suffix}.tif",
            mask_dir / f"{stem}{mask_suffix}.tiff",
            mask_dir / f"{stem}{mask_suffix}.TIF",
            mask_dir / f"{stem}{mask_suffix}.TIFF",
        ]

        mask_path = None
        for cand in candidate_masks:
            if cand.exists():
                mask_path = cand
                break

        if mask_path is None:
            print(f"[WARN] No matching mask found for: {img_path.name}")
            continue

        try:
            img = io.imread(str(img_path))
            mask = io.imread(str(mask_path))

            # If image is multi-channel, use the first channel for visualization
            if img.ndim == 3:
                img = img[..., 0]

            # If mask is multi-channel by accident, squeeze it
            if mask.ndim > 2:
                mask = np.squeeze(mask)

            if img.shape[:2] != mask.shape[:2]:
                print(f"[WARN] Shape mismatch: {img_path.name} vs {mask_path.name}")
                print(f"       image shape = {img.shape}, mask shape = {mask.shape}")
                continue

            matched_count += 1

            # Collect areas
            areas = collect_instance_areas(mask)
            all_areas.extend(areas)

            # Save overlay
            save_overlay = True
            if overlay_sample_limit is not None and saved_overlay_count >= overlay_sample_limit:
                save_overlay = False

            if save_overlay:
                overlay = make_overlay(img, mask, boundary_color=(1.0, 0.0, 0.0))

                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title(f"{img_path.name} | instances={int(mask.max())}")
                plt.axis("off")
                out_overlay_path = overlay_dir / f"{stem}_overlay.png"
                plt.tight_layout()
                plt.savefig(out_overlay_path, dpi=200, bbox_inches="tight")
                plt.close()

                saved_overlay_count += 1

            if idx % 10 == 0 or idx == 1:
                print(
                    f"[OK] {idx}/{len(image_files)}  "
                    f"{img_path.name}  "
                    f"instances={int(mask.max())}  "
                    f"areas_collected={len(areas)}"
                )

        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {repr(e)}")

    print(f"[INFO] Matched image-mask pairs: {matched_count}")

    if len(all_areas) == 0:
        raise RuntimeError("No instance areas were collected. Please check your image/mask pairing.")

    all_areas = np.array(all_areas, dtype=np.float32)

    # Save histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_areas, bins=hist_bins)
    plt.xlabel("Instance area (pixels)")
    plt.ylabel("Count")
    plt.title(f"Area Histogram | total instances = {len(all_areas)}")
    plt.tight_layout()
    hist_path = output_dir / "area_histogram.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # Save raw area values
    np.save(output_dir / "areas.npy", all_areas)

    # Save a simple text summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Matched image-mask pairs: {matched_count}\n")
        f.write(f"Total instances: {len(all_areas)}\n")
        f.write(f"Min area: {all_areas.min():.2f}\n")
        f.write(f"Max area: {all_areas.max():.2f}\n")
        f.write(f"Mean area: {all_areas.mean():.2f}\n")
        f.write(f"Median area: {np.median(all_areas):.2f}\n")

    print(f"[DONE] Overlay images saved to: {overlay_dir}")
    print(f"[DONE] Area histogram saved to: {hist_path}")
    print(f"[DONE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    IMAGE_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r15c12/ch2"
    MASK_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r15c12/ch2_results/masks_out"
    OUTPUT_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r15c12/ch2_results/eval_results"

    evaluate_segmentation(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        image_exts=(".tif", ".tiff"),
        mask_suffix="_mask",
        overlay_sample_limit=None,
        hist_bins=50,
    )