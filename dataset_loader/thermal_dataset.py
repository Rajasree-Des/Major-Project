"""PyTorch Dataset for preprocessed thermal/RGB patch pairs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loader.metadata import (
    ProcessedSceneMetadata,
    discover_scene_directories,
    load_scene_metadata,
)

logger = logging.getLogger(__name__)

PATCH_FILE_PATTERN = re.compile(r"^patch_(\d{6})\.npy$")


class DatasetStructureError(Exception):
    """Raised when processed files are missing or inconsistent."""


class PatchValidationError(Exception):
    """Raised when a patch fails unexpected validation checks."""


@dataclass(frozen=True)
class PatchReference:
    """Index entry for one thermal/RGB patch pair."""

    scene_id: str
    scene_dir: Path
    patch_id: int
    patch_key: str
    thermal_path: Path
    rgb_path: Path


def parse_patch_filename(name: str) -> str | None:
    match = PATCH_FILE_PATTERN.match(name)
    return match.group(1) if match else None


def index_scene_patches(scene_dir: Path, metadata: ProcessedSceneMetadata) -> list[PatchReference]:
    """Build matched patch references for one processed scene."""
    thermal_dir = scene_dir / "thermal"
    rgb_dir = scene_dir / "rgb"
    if not thermal_dir.is_dir() or not rgb_dir.is_dir():
        raise DatasetStructureError(f"Missing thermal/ or rgb/ under {scene_dir}")

    thermal_files = {parse_patch_filename(p.name): p for p in thermal_dir.glob("patch_*.npy")}
    rgb_files = {parse_patch_filename(p.name): p for p in rgb_dir.glob("patch_*.npy")}
    thermal_files.pop(None, None)
    rgb_files.pop(None, None)

    thermal_keys = set(thermal_files)
    rgb_keys = set(rgb_files)
    missing_rgb = sorted(thermal_keys - rgb_keys)
    missing_thermal = sorted(rgb_keys - thermal_keys)
    if missing_rgb or missing_thermal:
        raise DatasetStructureError(
            f"Patch ID mismatch in {scene_dir}: "
            f"missing_rgb={len(missing_rgb)}, missing_thermal={len(missing_thermal)}"
        )

    refs: list[PatchReference] = []
    for key in sorted(thermal_keys):
        patch_id = int(key)
        refs.append(
            PatchReference(
                scene_id=metadata.scene_id,
                scene_dir=scene_dir,
                patch_id=patch_id,
                patch_key=key,
                thermal_path=thermal_files[key],
                rgb_path=rgb_files[key],
            )
        )

    if metadata.patches_generated != len(refs):
        logger.warning(
            "metadata patches_generated=%d but found %d files for %s",
            metadata.patches_generated,
            len(refs),
            metadata.scene_id,
        )
    return refs


def build_combined_valid_mask(thermal_hw: np.ndarray, rgb_hwc: np.ndarray) -> np.ndarray:
    """True where thermal and all RGB channels are finite (H, W)."""
    thermal_ok = np.isfinite(thermal_hw)
    rgb_ok = np.all(np.isfinite(rgb_hwc), axis=-1)
    return thermal_ok & rgb_ok


def validate_patch_arrays(
    thermal: np.ndarray,
    rgb: np.ndarray,
    *,
    expected_hw: int | None = None,
) -> None:
    """Validate dtypes/shapes; expected NaNs are allowed."""
    if thermal.ndim != 2:
        raise PatchValidationError(f"Thermal must be H×W, got shape {thermal.shape}")
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise PatchValidationError(f"RGB must be H×W×3, got shape {rgb.shape}")
    if thermal.shape[:2] != rgb.shape[:2]:
        raise PatchValidationError(
            f"Spatial mismatch thermal={thermal.shape} rgb={rgb.shape}"
        )
    if expected_hw is not None and thermal.shape != (expected_hw, expected_hw):
        raise PatchValidationError(
            f"Expected thermal shape ({expected_hw}, {expected_hw}), got {thermal.shape}"
        )
    if thermal.dtype != np.float32:
        raise PatchValidationError(f"Thermal dtype must be float32, got {thermal.dtype}")
    if rgb.dtype != np.float32:
        raise PatchValidationError(f"RGB dtype must be float32, got {rgb.dtype}")

    if np.isinf(thermal).any() or np.isinf(rgb).any():
        raise PatchValidationError("Unexpected Inf in patch arrays")

    if not np.any(np.isfinite(thermal)) or not np.any(np.isfinite(rgb)):
        raise PatchValidationError("Patch has no finite thermal or RGB values")


class ThermalSatelliteDataset(Dataset):
    """Dataset of normalized thermal/RGB patch pairs from ``dataset/processed/``."""

    def __init__(
        self,
        processed_root: Path,
        *,
        scene_ids: Sequence[str] | None = None,
        patch_indices: Sequence[int] | None = None,
        return_mask: bool = True,
    ) -> None:
        """
        Args:
            processed_root: ``dataset/processed`` or a single scene directory.
            scene_ids: Optional filter of scene folder names.
            patch_indices: Optional subset of dataset indices (for splits).
            return_mask: Include ``valid_mask`` in each sample.
        """
        self.processed_root = processed_root.resolve()
        self.return_mask = return_mask

        scene_dirs = discover_scene_directories(self.processed_root)
        if scene_ids is not None:
            allowed = set(scene_ids)
            scene_dirs = [d for d in scene_dirs if d.name in allowed or self._scene_id(d) in allowed]

        if not scene_dirs:
            raise DatasetStructureError(f"No processed scenes found under {self.processed_root}")

        self.scene_metadata: dict[str, ProcessedSceneMetadata] = {}
        self.patch_refs: list[PatchReference] = []

        for scene_dir in scene_dirs:
            meta = load_scene_metadata(scene_dir)
            self.scene_metadata[meta.scene_id] = meta
            self.patch_refs.extend(index_scene_patches(scene_dir, meta))

        if not self.patch_refs:
            raise DatasetStructureError("No patch pairs indexed.")

        if patch_indices is not None:
            self.patch_refs = [self.patch_refs[i] for i in patch_indices]

        logger.info(
            "Indexed %d patch pairs from %d scene(s).",
            len(self.patch_refs),
            len(self.scene_metadata),
        )

    @staticmethod
    def _scene_id(scene_dir: Path) -> str:
        return load_scene_metadata(scene_dir).scene_id

    def __len__(self) -> int:
        return len(self.patch_refs)

    @property
    def metadata_by_scene(self) -> dict[str, ProcessedSceneMetadata]:
        return self.scene_metadata

    def get_scene_metadata(self, scene_id: str) -> ProcessedSceneMetadata:
        if scene_id not in self.scene_metadata:
            raise KeyError(f"Unknown scene_id: {scene_id}")
        return self.scene_metadata[scene_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.patch_refs[index]
        meta = self.scene_metadata[ref.scene_id]

        thermal = np.load(ref.thermal_path)
        rgb = np.load(ref.rgb_path)

        validate_patch_arrays(thermal, rgb, expected_hw=meta.patch_size)

        valid_mask = build_combined_valid_mask(thermal, rgb)

        thermal_t = torch.from_numpy(thermal).unsqueeze(0)  # (1, H, W)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # (3, H, W)
        mask_t = torch.from_numpy(valid_mask.astype(np.bool_)).unsqueeze(0)  # (1, H, W)

        sample: dict[str, Any] = {
            "thermal": thermal_t.to(dtype=torch.float32),
            "rgb": rgb_t.to(dtype=torch.float32),
            "patch_id": ref.patch_key,
            "patch_index": ref.patch_id,
            "scene_id": ref.scene_id,
        }
        if self.return_mask:
            sample["valid_mask"] = mask_t
        return sample
