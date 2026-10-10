"""Frozen scientific parameters (must match completed evaluation runs)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from colorization.types import (
    COLORIZATION_VERSION,
    ColormapConfig,
    GuidedColorizationConfig,
    OverlayConfig,
)
from sr_data.metadata import DegradationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed_qa"
METHOD_SPEC = PROJECT_ROOT / "outputs" / "evaluation" / "physics_refinement_v1" / "METHOD.txt"
VALIDATION_SCENE_ID = "LC08_L2SP_144053_20250225_02_T1"


@dataclass(frozen=True)
class FrozenExperimentConfig:
    processed_root: Path
    degradation: DegradationConfig
    refinement_beta: float
    colormap: ColormapConfig
    guided: GuidedColorizationConfig
    overlay: OverlayConfig
    swinir_variant: str = "lightweight_x2"
    pretrained_checkpoint: str = "auto"
    eval_seed: int = 42
    pipeline_version: str = "integrated_pipeline_v1"
    colorization_version: str = COLORIZATION_VERSION

    def require_beta(self, value: float) -> None:
        if value != self.refinement_beta:
            raise ValueError(
                f"refinement beta must be frozen at {self.refinement_beta}, got {value}"
            )


def load_frozen_config(
    *,
    processed_root: Path | None = None,
    method_spec: Path | None = None,
) -> FrozenExperimentConfig:
    """Load beta from METHOD.txt; other params match physics_refinement_v1 / colorization v1."""
    mpath = (method_spec or METHOD_SPEC).resolve()
    if not mpath.is_file():
        raise FileNotFoundError(f"Missing frozen method spec: {mpath}")
    beta = float(json.loads(mpath.read_text(encoding="utf-8"))["beta"])

    colormap = ColormapConfig(vmin_kelvin=295.0, vmax_kelvin=320.0, cmap_name="inferno")
    guided = GuidedColorizationConfig(colormap=colormap, radius=8, eps=0.02, edge_mix=0.35)
    overlay = OverlayConfig(colormap=colormap, thermal_weight=0.55)

    return FrozenExperimentConfig(
        processed_root=(processed_root or DEFAULT_PROCESSED_ROOT).resolve(),
        degradation=DegradationConfig(scale=2, base_seed=42),
        refinement_beta=beta,
        colormap=colormap,
        guided=guided,
        overlay=overlay,
    )
