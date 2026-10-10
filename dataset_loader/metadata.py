"""Load preprocessing metadata.json for processed scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChannelNormStats:
    mean: float
    std: float
    method: str


@dataclass(frozen=True)
class ProcessedSceneMetadata:
    """Subset of fields required by the PyTorch dataset layer."""

    scene_id: str
    scene_dir: Path
    patch_size: int
    patches_generated: int
    patches_skipped: int
    normalization_method: str
    thermal_norm: ChannelNormStats
    rgb_norm: tuple[ChannelNormStats, ChannelNormStats, ChannelNormStats]
    thermal_representation: str
    rgb_representation: str
    raw: dict[str, Any]

    @property
    def data_already_normalized(self) -> bool:
        return self.normalization_method == "z_score_valid_pixels"


def load_scene_metadata(scene_dir: Path) -> ProcessedSceneMetadata:
    """Load and parse ``metadata.json`` from a processed scene directory."""
    scene_dir = scene_dir.resolve()
    meta_path = scene_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json in {scene_dir}")

    with meta_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    norm = raw["normalization"]
    thermal = norm["thermal"]
    rgb_list = norm["rgb"]

    return ProcessedSceneMetadata(
        scene_id=str(raw["scene_id"]),
        scene_dir=scene_dir,
        patch_size=int(raw["patch_size"]),
        patches_generated=int(raw["patches_generated"]),
        patches_skipped=int(raw["patches_skipped"]),
        normalization_method=str(norm["method"]),
        thermal_norm=ChannelNormStats(
            mean=float(thermal["mean"]),
            std=float(thermal["std"]),
            method=str(thermal.get("method", norm["method"])),
        ),
        rgb_norm=tuple(
            ChannelNormStats(
                mean=float(ch["mean"]),
                std=float(ch["std"]),
                method=str(ch.get("method", norm["method"])),
            )
            for ch in rgb_list
        ),
        thermal_representation=str(raw["thermal_data_representation"]["exported"]),
        rgb_representation=str(raw["rgb_data_representation"]["exported"]),
        raw=raw,
    )


def discover_scene_directories(processed_root: Path) -> list[Path]:
    """Return processed scene directories under ``processed_root``.

    If ``processed_root`` itself contains ``metadata.json``, it is treated as a
    single-scene directory.
    """
    processed_root = processed_root.resolve()
    if (processed_root / "metadata.json").is_file():
        return [processed_root]

    scenes: list[Path] = []
    for child in sorted(processed_root.iterdir()):
        if child.is_dir() and (child / "metadata.json").is_file():
            scenes.append(child)
    return scenes
