"""Landsat Collection 2 Level-2 radiometric scaling (USGS)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from preprocessing.constants import RGB_BANDS, THERMAL_BAND

logger = logging.getLogger(__name__)

# USGS Landsat Collection 2 Level-2 Surface Reflectance
SR_REFLECTANCE_MULT = 0.0000275
SR_REFLECTANCE_ADD = -0.2

# USGS Landsat Collection 2 Level-2 Surface Temperature (Kelvin)
ST_TEMPERATURE_MULT = 0.00341802
ST_TEMPERATURE_ADD_K = 149.0


@dataclass(frozen=True)
class RadiometricScaling:
    """Documentation of scale/offset applied to a band group."""

    band_group: str
    formula: str
    mult: float
    add: float
    output_units: str
    output_dtype: str = "float32"


THERMAL_SCALING = RadiometricScaling(
    band_group=THERMAL_BAND,
    formula="temperature_k = dn * ST_TEMPERATURE_MULT + ST_TEMPERATURE_ADD_K",
    mult=ST_TEMPERATURE_MULT,
    add=ST_TEMPERATURE_ADD_K,
    output_units="kelvin",
)

SR_SCALING = RadiometricScaling(
    band_group="SR_B2/SR_B3/SR_B4",
    formula="reflectance = dn * SR_REFLECTANCE_MULT + SR_REFLECTANCE_ADD",
    mult=SR_REFLECTANCE_MULT,
    add=SR_REFLECTANCE_ADD,
    output_units="unitless_surface_reflectance",
)


def convert_st_b10_to_kelvin(dn: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convert ST_B10 DNs to Kelvin; invalid pixels become NaN."""
    out = np.full(dn.shape, np.nan, dtype=np.float32)
    valid = valid_mask & np.isfinite(dn)
    out[valid] = (
        dn[valid].astype(np.float64) * ST_TEMPERATURE_MULT + ST_TEMPERATURE_ADD_K
    ).astype(np.float32)
    return out


def convert_sr_to_reflectance(dn: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convert SR DNs to reflectance; invalid pixels become NaN."""
    out = np.full(dn.shape, np.nan, dtype=np.float32)
    valid = valid_mask & np.isfinite(dn)
    out[valid] = (
        dn[valid].astype(np.float64) * SR_REFLECTANCE_MULT + SR_REFLECTANCE_ADD
    ).astype(np.float32)
    return out


def radiometric_convert_scene(
    dn_bands: dict[str, np.ndarray],
    per_band_valid: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, RadiometricScaling]]:
    """Apply Collection 2 L2 scaling to thermal and RGB stacks."""
    if THERMAL_BAND not in dn_bands:
        raise ValueError(f"Missing thermal band {THERMAL_BAND}")

    thermal = convert_st_b10_to_kelvin(dn_bands[THERMAL_BAND], per_band_valid[THERMAL_BAND])

    rgb_channels = []
    for name in RGB_BANDS:
        if name not in dn_bands:
            raise ValueError(f"Missing RGB band {name}")
        rgb_channels.append(convert_sr_to_reflectance(dn_bands[name], per_band_valid[name]))

    rgb = np.stack(rgb_channels, axis=-1).astype(np.float32)
    scaling: dict[str, RadiometricScaling] = {THERMAL_BAND: THERMAL_SCALING, "rgb": SR_SCALING}
    logger.info("Radiometric conversion complete (thermal: K, RGB: reflectance).")
    return thermal, rgb, scaling
