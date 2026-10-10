"""Valid-pixel masks for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.constants import RGB_BANDS, THERMAL_BAND


@dataclass
class MaskReport:
    """Summary of valid vs invalid pixels for a scene."""

    total_pixels: int
    valid_pixels: int
    invalid_pixels: int
    valid_fraction: float
    per_band_invalid_counts: dict[str, int]


def band_invalid_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return True where pixels are invalid (nodata or non-finite)."""
    invalid = ~np.isfinite(data)
    if nodata is not None:
        invalid |= data == nodata
    return invalid


def build_per_band_valid_masks(
    dn_bands: dict[str, np.ndarray],
    nodata_values: dict[str, float | None],
) -> dict[str, np.ndarray]:
    """Build per-band valid masks (True = valid pixel)."""
    masks: dict[str, np.ndarray] = {}
    for name, array in dn_bands.items():
        nodata = nodata_values.get(name)
        masks[name] = ~band_invalid_mask(array, nodata)
    return masks


def combine_valid_mask(per_band_valid: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels valid in all required bands."""
    names = [THERMAL_BAND, *RGB_BANDS]
    combined = np.ones(per_band_valid[names[0]].shape, dtype=bool)
    for name in names:
        combined &= per_band_valid[name]
    return combined


def summarize_mask(valid_mask: np.ndarray, per_band_valid: dict[str, np.ndarray]) -> MaskReport:
    """Compute mask statistics for logging and metadata."""
    total = int(valid_mask.size)
    valid = int(np.count_nonzero(valid_mask))
    invalid = total - valid
    per_band_invalid = {
        name: int(np.count_nonzero(~mask)) for name, mask in per_band_valid.items()
    }
    return MaskReport(
        total_pixels=total,
        valid_pixels=valid,
        invalid_pixels=invalid,
        valid_fraction=valid / total if total else 0.0,
        per_band_invalid_counts=per_band_invalid,
    )
