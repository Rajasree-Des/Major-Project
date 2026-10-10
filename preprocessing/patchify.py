"""Spatially aligned thermal/RGB patch extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatchRecord:
    """One aligned thermal/RGB patch pair."""

    patch_id: int
    row: int
    col: int
    thermal: np.ndarray
    rgb: np.ndarray
    invalid_fraction: float


@dataclass
class PatchExtractionResult:
    """Summary of patch extraction."""

    patches: list[PatchRecord]
    skipped_count: int
    patch_size: int
    stride: int
    max_invalid_percent: float


def _patch_invalid_fraction(valid_mask: np.ndarray, row: int, col: int, size: int) -> float:
    window = valid_mask[row : row + size, col : col + size]
    invalid = 1.0 - (np.count_nonzero(window) / window.size)
    return float(invalid)


def extract_patches(
    thermal: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    patch_size: int = 256,
    stride: int = 128,
    max_invalid_percent: float = 10.0,
) -> PatchExtractionResult:
    """Extract corresponding thermal and RGB patches from the same grid."""
    if thermal.shape[:2] != rgb.shape[:2] or thermal.shape[:2] != valid_mask.shape:
        raise ValueError("Thermal, RGB, and valid_mask must share the same height and width.")

    height, width = thermal.shape[:2]
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")

    patches: list[PatchRecord] = []
    skipped = 0
    patch_id = 0
    max_invalid_fraction = max_invalid_percent / 100.0

    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            invalid_fraction = _patch_invalid_fraction(valid_mask, row, col, patch_size)
            if invalid_fraction > max_invalid_fraction:
                skipped += 1
                continue

            thermal_patch = thermal[row : row + patch_size, col : col + patch_size]
            rgb_patch = rgb[row : row + patch_size, col : col + patch_size]

            if not np.any(np.isfinite(thermal_patch)) or not np.any(np.isfinite(rgb_patch)):
                skipped += 1
                continue

            patch_id += 1
            patches.append(
                PatchRecord(
                    patch_id=patch_id,
                    row=row,
                    col=col,
                    thermal=thermal_patch.astype(np.float32, copy=False),
                    rgb=rgb_patch.astype(np.float32, copy=False),
                    invalid_fraction=invalid_fraction,
                )
            )

    logger.info(
        "Patch extraction: %d patches kept, %d skipped (size=%d stride=%d max_invalid=%.1f%%).",
        len(patches),
        skipped,
        patch_size,
        stride,
        max_invalid_percent,
    )
    return PatchExtractionResult(
        patches=patches,
        skipped_count=skipped,
        patch_size=patch_size,
        stride=stride,
        max_invalid_percent=max_invalid_percent,
    )
