"""Verify thermal/RGB patch alignment and document data conventions."""

from __future__ import annotations

from typing import Any

import numpy as np

from dataset_loader.metadata import ProcessedSceneMetadata


def audit_scene_inputs(
    meta: ProcessedSceneMetadata,
    thermal_hw: np.ndarray,
    rgb_hwc: np.ndarray,
) -> dict[str, Any]:
    """Return alignment and representation audit; raises if shapes mismatch."""
    if thermal_hw.shape[0] != thermal_hw.shape[1]:
        raise ValueError(f"thermal must be square, got {thermal_hw.shape}")
    if rgb_hwc.shape[:2] != thermal_hw.shape:
        raise ValueError(
            f"thermal/RGB spatial mismatch: thermal {thermal_hw.shape}, rgb {rgb_hwc.shape}"
        )
    if rgb_hwc.shape[2] != 3:
        raise ValueError(f"RGB must be H×W×3, got {rgb_hwc.shape}")

    valid = np.isfinite(thermal_hw) & np.all(np.isfinite(rgb_hwc), axis=-1)
    return {
        "scene_id": meta.scene_id,
        "patch_size": meta.patch_size,
        "spatial_alignment": "thermal and RGB exported on the same 30 m Landsat grid per patch (H×W match)",
        "thermal_resolution_note": (
            "Patch thermal arrays are the 30 m HR reference grid used for synthetic SR; "
            "SR outputs are presented on this same grid. RGB (SR_B2/B3/B4) is native 30 m reflectance, "
            "not independent 15 m thermal information."
        ),
        "thermal_representation": meta.thermal_representation,
        "rgb_representation": meta.rgb_representation,
        "thermal_physical_units": "kelvin (after inverse z-score)",
        "rgb_physical_units": "unitless_surface_reflectance (after inverse z-score)",
        "rgb_display_note": (
            "Reflectance is clipped/scaled to [0,1] for visualization using a fixed reflectance ceiling; "
            "not display-ready sRGB out of the box."
        ),
        "valid_pixel_fraction": float(valid.mean()),
        "valid_pixels": int(valid.sum()),
        "thermal_mean_k": float(meta.thermal_norm.mean),
        "thermal_std_k": float(meta.thermal_norm.std),
    }
