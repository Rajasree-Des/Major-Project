"""Resolve patch IDs to dataset indices (thermal + SR must refer to the same patch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sr_data.metadata import DegradationConfig


@dataclass(frozen=True)
class PatchRef:
    scene_id: str
    patch_key: str
    thermal_dataset_index: int
    sr_dataset_index: int


def normalize_patch_key(patch_id: str) -> str:
    s = str(patch_id).strip()
    if s.isdigit():
        return f"{int(s):06d}"
    return s


def resolve_patch(
    processed_root: Path,
    scene_id: str,
    patch_id: str,
    *,
    degradation: DegradationConfig,
) -> PatchRef:
    from dataset_loader.thermal_dataset import ThermalSatelliteDataset

    patch_key = normalize_patch_key(patch_id)
    scene_ds = ThermalSatelliteDataset(processed_root, scene_ids=[scene_id])
    thermal_idx = next(
        (i for i, ref in enumerate(scene_ds.patch_refs) if ref.patch_key == patch_key),
        None,
    )
    if thermal_idx is None:
        raise FileNotFoundError(f"patch {patch_key} not found in scene {scene_id} thermal index")

    from sr_data.dataset import ThermalSRDataset

    sr_ds = ThermalSRDataset(processed_root, degradation=degradation, scene_ids=[scene_id])
    sr_idx = next(
        (
            i
            for i, ref in enumerate(sr_ds.base_dataset.patch_refs)
            if ref.scene_id == scene_id and ref.patch_key == patch_key
        ),
        None,
    )
    if sr_idx is None:
        raise FileNotFoundError(f"patch {patch_key} not found in SR dataset for {scene_id}")

    return PatchRef(scene_id, patch_key, thermal_idx, sr_idx)
