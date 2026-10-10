"""ML-oriented normalization with reversible metadata."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChannelNormalization:
    """Per-channel z-score parameters."""

    mean: float
    std: float
    method: str = "z_score_valid_pixels"


@dataclass
class NormalizationParams:
    """Normalization applied after radiometric conversion."""

    thermal: ChannelNormalization
    rgb: list[ChannelNormalization]
    note: str = (
        "Z-score computed on scene-level valid pixels (finite, combined mask). "
        "Inverse: value_phys = value_norm * std + mean"
    )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": "z_score_valid_pixels",
            "note": self.note,
            "thermal": asdict(self.thermal),
            "rgb": [asdict(ch) for ch in self.rgb],
        }


def _zscore_params(values: np.ndarray) -> ChannelNormalization:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot normalize: no valid pixels for statistics.")
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std == 0.0:
        std = 1.0
    return ChannelNormalization(mean=mean, std=std)


def normalize_thermal(
    thermal_kelvin: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, ChannelNormalization]:
    """Z-score normalize thermal Kelvin using valid pixels only."""
    params = _zscore_params(thermal_kelvin[valid_mask])
    out = np.full(thermal_kelvin.shape, np.nan, dtype=np.float32)
    valid = valid_mask & np.isfinite(thermal_kelvin)
    out[valid] = ((thermal_kelvin[valid] - params.mean) / params.std).astype(np.float32)
    return out, params


def normalize_rgb(
    rgb_reflectance: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, list[ChannelNormalization]]:
    """Z-score normalize each RGB channel using valid pixels only."""
    params_list: list[ChannelNormalization] = []
    out = np.full(rgb_reflectance.shape, np.nan, dtype=np.float32)
    for channel in range(rgb_reflectance.shape[-1]):
        channel_data = rgb_reflectance[..., channel]
        params = _zscore_params(channel_data[valid_mask])
        params_list.append(params)
        valid = valid_mask & np.isfinite(channel_data)
        out[..., channel][valid] = (
            (channel_data[valid] - params.mean) / params.std
        ).astype(np.float32)
    return out, params_list


def normalize_scene_arrays(
    thermal_kelvin: np.ndarray,
    rgb_reflectance: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, NormalizationParams]:
    """Normalize thermal and RGB stacks for model training."""
    thermal_norm, thermal_params = normalize_thermal(thermal_kelvin, valid_mask)
    rgb_norm, rgb_params = normalize_rgb(rgb_reflectance, valid_mask)
    params = NormalizationParams(thermal=thermal_params, rgb=rgb_params)
    logger.info(
        "Normalization complete (thermal mean=%.4f std=%.4f).",
        thermal_params.mean,
        thermal_params.std,
    )
    return thermal_norm, rgb_norm, params
