"""RGB-guided thermal colorization (visualization only; does not modify thermal SR)."""

from colorization.pipeline import ColorizationInputs, apply_all_methods
from colorization.types import (
    COLORIZATION_VERSION,
    ColormapConfig,
    GuidedColorizationConfig,
    OverlayConfig,
)

__all__ = [
    "COLORIZATION_VERSION",
    "ColormapConfig",
    "GuidedColorizationConfig",
    "OverlayConfig",
    "ColorizationInputs",
    "apply_all_methods",
]
