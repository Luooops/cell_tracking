from pathlib import Path

from segmentation import batch_segment_folder
from tif2png import convert_tiff_to_png
from tracking import run_tracking_pipeline


# =====================================
# Path configuration
# =====================================

# 1. Raw image folder.
#    segmentation.py reads image files from this folder:
#    .tif, .tiff, .png, .jpg, .jpeg
DATASET_ROOT = Path(
    "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/"
    "dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images"
)

WELL_ID = "r15c12"
CHANNEL = "ch2"

RAW_IMAGE_DIR = DATASET_ROOT / WELL_ID / CHANNEL

# All generated outputs are kept outside RAW_IMAGE_DIR so recursive input
# scanning does not reprocess masks, PNG previews, or tracking plots.
OUTPUT_ROOT_DIR = DATASET_ROOT / WELL_ID / f"{CHANNEL}_results"

# 2. Mask output folder.
#    segmentation.py saves label masks here.
#    tracking.py uses this same folder as its input.
MASK_OUTPUT_DIR = OUTPUT_ROOT_DIR / "masks_out"

# 3. Tracking output folder.
#    tracking.py saves tracks.csv, plots, and summary.txt here.
TRACKING_OUTPUT_DIR = OUTPUT_ROOT_DIR / "simple_tracking_results"

# 4. PNG copy of generated TIFF masks.
MASK_PNG_OUTPUT_DIR = OUTPUT_ROOT_DIR / "mask_out_png"


# =====================================
# Segmentation parameters
# =====================================

SEGMENTATION_PARAMS = {
    "use_gpu": True,
    "diameter": 25,
    "cellprob_threshold": 1.8,
    "flow_threshold": 1.0,
    "p_low": 1,
    "p_high": 99,
    "bg_sigma": 80,
    "black_gamma": 1.8,
    # Use None to choose min_area automatically from each frame's mean mask area.
    # Automatic threshold = mean cell mask area * auto_min_area_fraction.
    # Example: 0.25 removes instances smaller than 25% of average cell area.
    "min_area": None,
    "auto_min_area_fraction": 0.25,
}


# =====================================
# Tracking parameters
# =====================================

TRACKING_PARAMS = {
    # This is overwritten in main() with the exact per-mask min_area values
    # computed during segmentation.
    "min_area": 0,
    "auto_min_area_fraction": SEGMENTATION_PARAMS["auto_min_area_fraction"],
    "max_distance": 45.0,
    "max_area_ratio": 1.8,
    "max_shape_ratio": 1.8,
    "max_lost": 3,
    "area_weight": 5.0,
    "shape_weight": 3.0,
    "min_track_length": 5,
    "max_angle_diff_deg": 90.0,
    "max_close_cost": 12.0,
}


def main():
    print("========== Pipeline paths ==========")
    print(f"Raw image input:      {RAW_IMAGE_DIR}")
    print(f"Output root:          {OUTPUT_ROOT_DIR}")
    print(f"Mask output/input:    {MASK_OUTPUT_DIR}")
    print(f"Mask PNG output:      {MASK_PNG_OUTPUT_DIR}")
    print(f"Tracking output:      {TRACKING_OUTPUT_DIR}")
    print("====================================")

    print("\n[STEP 1/2] Segmentation")
    segmentation_result = batch_segment_folder(
        in_dir=str(RAW_IMAGE_DIR),
        out_dir=str(MASK_OUTPUT_DIR),
        **SEGMENTATION_PARAMS,
    )

    print("\n[STEP 2/3] Convert TIFF masks to PNG")
    convert_tiff_to_png(
        src_dir=MASK_OUTPUT_DIR,
        dst_dir=MASK_PNG_OUTPUT_DIR,
    )

    tracking_params = TRACKING_PARAMS.copy()
    tracking_params["min_area"] = segmentation_result["min_area_by_mask_file"]
    print(
        "[INFO] Passing segmentation min_area values to tracking | "
        f"frames={len(segmentation_result['used_min_areas'])} "
        f"mean_min_area={segmentation_result['mean_used_min_area']:.1f} "
        f"mean_cell_area={segmentation_result['mean_cell_area']:.1f}"
    )

    print("\n[STEP 3/3] Tracking")
    run_tracking_pipeline(
        mask_dir=str(MASK_OUTPUT_DIR),
        output_dir=str(TRACKING_OUTPUT_DIR),
        **tracking_params,
    )

    print("\n[DONE] Full segmentation + tracking pipeline finished.")


if __name__ == "__main__":
    main()
