"""Configuration for deterministic colorization methods."""

from __future__ import annotations

from dataclasses import dataclass, field

COLORIZATION_VERSION = "guided_colorization_v1"


@dataclass(frozen=True)
class ColormapConfig:
    """Fixed Kelvin limits for comparable figures across patches/scenes."""

    vmin_kelvin: float = 295.0
    vmax_kelvin: float = 320.0
    cmap_name: str = "inferno"
    invalid_color: tuple[int, int, int, int] = (200, 200, 200, 0)  # RGBA transparent neutral


@dataclass(frozen=True)
class GuidedColorizationConfig:
    """Conservative RGB-guided refinement of colormap RGB (visualization aid only)."""

    colormap: ColormapConfig = field(default_factory=ColormapConfig)
    radius: int = 8
    eps: float = 0.02
    edge_mix: float = 0.35
    rgb_reflectance_display_max: float = 0.25


@dataclass(frozen=True)
class OverlayConfig:
    """Blend thermal colormap with RGB context."""

    colormap: ColormapConfig = field(default_factory=ColormapConfig)
    thermal_weight: float = 0.55
    rgb_reflectance_display_max: float = 0.25
