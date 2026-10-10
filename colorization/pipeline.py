"""Apply colorization without mutating thermal Kelvin arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from colorization.methods import method_rgb_guided, method_rgb_overlay, method_standard_colormap
from colorization.types import GuidedColorizationConfig, OverlayConfig


@dataclass(frozen=True)
class ColorizationInputs:
    thermal_kelvin: np.ndarray
    rgb_reflectance: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.thermal_kelvin.shape != self.valid_mask.shape:
            raise ValueError("thermal and valid_mask shape mismatch")
        if self.rgb_reflectance.shape[:2] != self.thermal_kelvin.shape:
            raise ValueError("rgb and thermal spatial mismatch")


def apply_all_methods(
    inputs: ColorizationInputs,
    *,
    guided: GuidedColorizationConfig | None = None,
    overlay: OverlayConfig | None = None,
) -> dict[str, Any]:
    """Return RGB and RGBA per method; thermal array is never modified."""
    thermal = np.array(inputs.thermal_kelvin, copy=True)
    rgb_refl = np.array(inputs.rgb_reflectance, copy=True)
    valid = inputs.valid_mask.astype(bool, copy=False)

    gconf = guided or GuidedColorizationConfig()
    oconf = overlay or OverlayConfig()

    before_thermal = thermal.copy()
    colormap_rgb, colormap_rgba = method_standard_colormap(thermal, valid, gconf.colormap)
    guided_rgb, guided_rgba = method_rgb_guided(thermal, rgb_refl, valid, gconf)
    overlay_rgb, overlay_rgba = method_rgb_overlay(thermal, rgb_refl, valid, oconf)

    if not np.allclose(before_thermal, thermal, equal_nan=True, rtol=0, atol=0):
        raise RuntimeError("colorization mutated thermal kelvin array")

    return {
        "standard_colormap": {"rgb": colormap_rgb, "rgba": colormap_rgba},
        "rgb_guided": {"rgb": guided_rgb, "rgba": guided_rgba},
        "rgb_overlay": {"rgb": overlay_rgb, "rgba": overlay_rgba},
        "thermal_kelvin_unchanged": np.allclose(
            before_thermal, inputs.thermal_kelvin, equal_nan=True, rtol=0, atol=0
        ),
    }
