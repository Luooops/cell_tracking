from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskAreaStats:
    count: int
    mean_area: float
    median_area: float
    min_area: float
    max_area: float
    p10_area: float
    p25_area: float


def instance_areas(mask: np.ndarray) -> np.ndarray:
    """
    Return pixel areas for all non-background labels in a label mask.
    """
    labels = mask.astype(np.int64, copy=False)
    counts = np.bincount(labels.ravel())
    if counts.size <= 1:
        return np.array([], dtype=np.int64)
    return counts[1:]


def compute_mask_area_stats(mask: np.ndarray) -> MaskAreaStats:
    """
    Compute area statistics for cell instances in a label mask.
    """
    areas = instance_areas(mask)
    areas = areas[areas > 0]

    if areas.size == 0:
        return MaskAreaStats(
            count=0,
            mean_area=0.0,
            median_area=0.0,
            min_area=0.0,
            max_area=0.0,
            p10_area=0.0,
            p25_area=0.0,
        )

    return MaskAreaStats(
        count=int(areas.size),
        mean_area=float(np.mean(areas)),
        median_area=float(np.median(areas)),
        min_area=float(np.min(areas)),
        max_area=float(np.max(areas)),
        p10_area=float(np.percentile(areas, 10)),
        p25_area=float(np.percentile(areas, 25)),
    )


def choose_min_area_from_mean(mask: np.ndarray, fraction_of_mean: float = 0.25) -> int:
    """
    Choose an automatic area threshold from the mean cell mask area.

    Example: fraction_of_mean=0.25 removes instances smaller than 25% of the
    average instance area in the current frame.
    """
    stats = compute_mask_area_stats(mask)
    if stats.count == 0:
        return 0
    return max(1, int(round(stats.mean_area * fraction_of_mean)))


def remove_small_instances(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """
    Remove labeled instances smaller than min_area pixels and relabel to 1..N.
    """
    mask = mask.astype(np.int32, copy=False)
    if min_area <= 0:
        return mask.copy()

    clean_mask = np.zeros_like(mask, dtype=np.int32)
    new_label = 1

    for label, area in enumerate(instance_areas(mask), start=1):
        if area >= min_area:
            clean_mask[mask == label] = new_label
            new_label += 1

    return clean_mask


def filter_small_instances_by_mean(
    mask: np.ndarray,
    fraction_of_mean: float = 0.25,
) -> tuple[np.ndarray, int, MaskAreaStats]:
    """
    Remove instances much smaller than the current frame's average area.
    """
    stats = compute_mask_area_stats(mask)
    min_area = 0
    if stats.count > 0:
        min_area = max(1, int(round(stats.mean_area * fraction_of_mean)))

    return remove_small_instances(mask, min_area=min_area), min_area, stats
