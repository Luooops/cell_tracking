import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle


def sort_key_by_time(path: Path):
    """
    Extract time index from filenames like:
    r01c12f01p01-ch02t01.tiff
    r01c12f01p01-ch02t01_mask.tiff
    """
    m = re.search(r"t(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem


def load_image_files(image_dir: str):
    image_dir = Path(image_dir)
    files = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))
    files = sorted(files, key=sort_key_by_time)
    if len(files) == 0:
        raise FileNotFoundError(f"No tif/tiff images found in: {image_dir}")
    return files


def normalize_for_display(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """
    Normalize grayscale image to [0, 1] for visualization.
    """
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


def build_track_color_map(track_ids):
    """
    Assign a fixed RGB color to each track_id.
    """
    track_ids = sorted(track_ids)
    cmap = cm.get_cmap("tab20", len(track_ids) if len(track_ids) > 0 else 1)

    color_map = {}
    for i, tid in enumerate(track_ids):
        color_map[tid] = cmap(i)[:3]
    return color_map


def plot_tracks_on_frames(
    image_dir: str,
    csv_path: str,
    output_dir: str,
    draw_bbox: bool = False,
    draw_id: bool = True,
    draw_trail: bool = True,
    trail_length: int | None = None,
    linewidth: float = 1.5,
    marker_size: float = 18,
    alpha_image: float = 1.0,
    p_low: float = 1,
    p_high: float = 99,
    min_track_length: int = 1,
):
    """
    Visualize tracking results from CSV on each frame.

    Parameters
    ----------
    image_dir : str
        Directory containing original tif/tiff images.
    csv_path : str
        Path to tracks.csv.
    output_dir : str
        Directory to save visualization frames.
    draw_bbox : bool
        Whether to draw bounding boxes.
    draw_id : bool
        Whether to draw track IDs next to centroids.
    draw_trail : bool
        Whether to draw trajectory trails up to current frame.
    trail_length : int or None
        Number of previous points to show for each track.
        None means show full history up to current frame.
    linewidth : float
        Line width for trails and bbox.
    marker_size : float
        Marker size for current centroid.
    alpha_image : float
        Alpha for background image.
    p_low, p_high : float
        Percentile normalization for image display.
    min_track_length : int
        Only visualize tracks with at least this many rows in CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = load_image_files(image_dir)
    df = pd.read_csv(csv_path)

    required_cols = {"track_id", "frame", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Filter short tracks if requested
    track_lengths = df.groupby("track_id").size()
    valid_track_ids = track_lengths[track_lengths >= min_track_length].index.tolist()
    df = df[df["track_id"].isin(valid_track_ids)].copy()

    # Build fixed colors per track
    color_map = build_track_color_map(df["track_id"].unique())

    # Group all rows by frame for quick lookup
    frame_groups = {int(k): v.copy() for k, v in df.groupby("frame")}

    # Pre-sort each track history
    track_histories = {}
    for tid, g in df.groupby("track_id"):
        track_histories[int(tid)] = g.sort_values("frame").copy()

    print(f"[INFO] Found {len(image_files)} image frames")
    print(f"[INFO] Found {len(valid_track_ids)} tracks to visualize")
    print(f"[INFO] Saving overlay frames to: {output_dir}")

    for frame_idx, img_path in enumerate(image_files):
        img = tifffile.imread(str(img_path))
        if img.ndim > 2:
            img = np.squeeze(img)
        img_vis = normalize_for_display(img, p_low=p_low, p_high=p_high)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_vis, cmap="gray", alpha=alpha_image)
        ax.set_title(f"{img_path.name} | frame={frame_idx}")
        ax.axis("off")

        frame_df = frame_groups.get(frame_idx, None)

        if frame_df is not None and len(frame_df) > 0:
            for _, row in frame_df.iterrows():
                tid = int(row["track_id"])
                color = color_map[tid]
                x = float(row["x"])
                y = float(row["y"])

                # Draw current centroid
                ax.scatter(x, y, s=marker_size, c=[color])

                # Draw track ID
                if draw_id:
                    ax.text(
                        x + 3,
                        y + 3,
                        str(tid),
                        color=color,
                        fontsize=7,
                        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
                    )

                # Draw bbox if columns exist
                if draw_bbox:
                    bbox_cols = {
                        "bbox_min_row",
                        "bbox_min_col",
                        "bbox_max_row",
                        "bbox_max_col",
                    }
                    if bbox_cols.issubset(frame_df.columns):
                        min_row = int(row["bbox_min_row"])
                        min_col = int(row["bbox_min_col"])
                        max_row = int(row["bbox_max_row"])
                        max_col = int(row["bbox_max_col"])
                        rect = Rectangle(
                            (min_col, min_row),
                            max_col - min_col,
                            max_row - min_row,
                            fill=False,
                            edgecolor=color,
                            linewidth=linewidth,
                        )
                        ax.add_patch(rect)

                # Draw trajectory trail up to current frame
                if draw_trail:
                    hist = track_histories[tid]
                    hist = hist[hist["frame"] <= frame_idx]

                    if trail_length is not None:
                        hist = hist.tail(trail_length)

                    if len(hist) >= 2:
                        ax.plot(
                            hist["x"].values,
                            hist["y"].values,
                            color=color,
                            linewidth=linewidth,
                        )

        out_path = output_dir / f"{img_path.stem}_tracks.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or frame_idx == len(image_files) - 1:
            print(f"[OK] Saved {frame_idx + 1}/{len(image_files)}: {out_path.name}")

    print("[DONE] Visualization finished.")


if __name__ == "__main__":
    IMAGE_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r15c12/ch2"
    CSV_PATH = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r15c12/ch2_results/simple_tracking_results/tracks.csv"
    OUTPUT_DIR = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r15c12/ch2_results/simple_tracking_results/tracks_overlay"

    plot_tracks_on_frames(
        image_dir=IMAGE_DIR,
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        draw_bbox=False,
        draw_id=True,
        draw_trail=True,
        trail_length=10,     # None = full history up to current frame
        linewidth=1.5,
        marker_size=18,
        alpha_image=1.0,
        p_low=1,
        p_high=99,
        min_track_length=5,
    )