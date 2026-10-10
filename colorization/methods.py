"""Three deterministic colorization methods (visualization only)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from colorization._boxfilter import guided_filter
from colorization.types import ColormapConfig, GuidedColorizationConfig, OverlayConfig


def _apply_invalid_rgba(rgb: np.ndarray, valid: np.ndarray, invalid_rgba: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    safe = np.nan_to_num(rgb, nan=0.0)
    out[..., :3] = np.clip(safe * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    inv = ~valid
    out[inv] = np.array(invalid_rgba, dtype=np.uint8)
    return out


def kelvin_to_colormap_rgb(
    thermal_k: np.ndarray,
    valid: np.ndarray,
    config: ColormapConfig,
) -> np.ndarray:
    """Map Kelvin to RGB in [0,1] using fixed limits (no per-image scaling)."""
    norm = np.clip((thermal_k - config.vmin_kelvin) / (config.vmax_kelvin - config.vmin_kelvin), 0.0, 1.0)
    cmap = plt.get_cmap(config.cmap_name)
    rgb = cmap(norm)[..., :3].astype(np.float64)
    rgb[~valid] = 0.0
    return rgb


def reflectance_to_display_rgb(
    rgb_refl: np.ndarray,
    valid: np.ndarray,
    display_max: float,
) -> np.ndarray:
    """Fixed linear scale to [0,1] for visualization (not physical sRGB)."""
    disp = np.clip(rgb_refl / display_max, 0.0, 1.0)
    disp[~valid] = 0.0
    return disp


def method_standard_colormap(
    thermal_k: np.ndarray,
    valid: np.ndarray,
    config: ColormapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """A: standard thermal colormap with fixed Kelvin limits."""
    rgb = kelvin_to_colormap_rgb(thermal_k, valid, config)
    rgba = _apply_invalid_rgba(rgb, valid, config.invalid_color)
    return rgb, rgba


def method_rgb_guided(
    thermal_k: np.ndarray,
    rgb_refl: np.ndarray,
    valid: np.ndarray,
    config: GuidedColorizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """B: colormap base + conservative RGB luminance-guided smoothing."""
    base = kelvin_to_colormap_rgb(thermal_k, valid, config.colormap)
    gray = reflectance_to_display_rgb(rgb_refl, valid, config.rgb_reflectance_display_max)
    guide = (0.2989 * gray[..., 0] + 0.5870 * gray[..., 1] + 0.1140 * gray[..., 2]).astype(np.float64)

    gy, gx = np.gradient(guide)
    edge_rgb = np.hypot(gx, gy)
    ty, tx = np.gradient(thermal_k.astype(np.float64))
    edge_t = np.hypot(tx, ty)
    edge_rgb = edge_rgb / (np.nanmax(edge_rgb[valid]) + 1e-8)
    edge_t = edge_t / (np.nanmax(edge_t[valid]) + 1e-8)
    # Only allow RGB edges where thermal is relatively smooth (avoid inventing structure).
    w = np.clip(config.edge_mix * edge_rgb * (1.0 - edge_t), 0.0, config.edge_mix)

    refined = np.empty_like(base)
    for c in range(3):
        smoothed = guided_filter(guide, base[..., c], config.radius, config.eps)
        refined[..., c] = (1.0 - w) * base[..., c] + w * smoothed
    refined = np.nan_to_num(refined, nan=0.0)
    refined[~valid] = 0.0
    rgba = _apply_invalid_rgba(refined, valid, config.colormap.invalid_color)
    return refined, rgba


def method_rgb_overlay(
    thermal_k: np.ndarray,
    rgb_refl: np.ndarray,
    valid: np.ndarray,
    config: OverlayConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """C: configurable blend of thermal colormap and RGB context (baseline)."""
    thermal_rgb = kelvin_to_colormap_rgb(thermal_k, valid, config.colormap)
    context = reflectance_to_display_rgb(rgb_refl, valid, config.rgb_reflectance_display_max)
    a = float(config.thermal_weight)
    blend = a * thermal_rgb + (1.0 - a) * context
    blend = np.clip(blend, 0.0, 1.0)
    blend[~valid] = 0.0
    rgba = _apply_invalid_rgba(blend, valid, config.colormap.invalid_color)
    return blend, rgba
