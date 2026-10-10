"""Produce refined thermal SR on the HR patch grid (no HR pixels in output path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dataset_loader.physical import denormalize_from_scene_metadata
from dataset_loader.thermal_dataset import ThermalSatelliteDataset
from sr_data.consistency_refinement import ConsistencyRefinementConfig, refine_hr_normalized
from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import DegradationConfig
from training.train_swinir import prepare_lr


def predict_refined_thermal_kelvin(
    processed_root: Path,
    scene_id: str,
    patch_index: int,
    model: torch.nn.Module,
    *,
    degradation: DegradationConfig,
    beta: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run bicubic path omitted; returns refined SwinIR HR in Kelvin + aligned RGB reflectance."""
    base_ds = ThermalSatelliteDataset(processed_root, scene_ids=[scene_id], patch_indices=[patch_index])
    sr_ds = ThermalSRDataset(
        processed_root,
        degradation=degradation,
        scene_ids=[scene_id],
        patch_indices=[patch_index],
    )
    base = base_ds[0]
    sr = sr_ds[0]
    meta = base_ds.get_scene_metadata(scene_id)

    if base["thermal"].shape[-2:] != sr["hr"].shape[-2:]:
        raise ValueError("base thermal HR grid != SR HR grid")

    lr = prepare_lr(sr["lr"].unsqueeze(0))
    with torch.no_grad():
        swinir_z = model(lr)[0, 0].cpu().numpy()
    conf = ConsistencyRefinementConfig(beta=beta, degradation=degradation)
    refined_z = refine_hr_normalized(
        swinir_z,
        sr["lr"][0].numpy(),
        sr["lr_valid_mask"][0].numpy().astype(bool),
        meta.thermal_norm,
        conf,
    )
    thermal_k, rgb_refl_hwc = denormalize_from_scene_metadata(
        refined_z,
        base["rgb"].permute(1, 2, 0).numpy(),
        meta,
    )
    valid = base["valid_mask"][0].numpy().astype(bool)
    return {
        "thermal_kelvin": np.asarray(thermal_k, dtype=np.float64),
        "rgb_reflectance": np.asarray(rgb_refl_hwc, dtype=np.float64),
        "valid_mask": valid,
        "patch_id": base["patch_id"],
        "scene_id": scene_id,
        "metadata": meta,
    }
