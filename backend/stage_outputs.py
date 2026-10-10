"""Save stage comparison PNGs for Processing Studio (single-patch UI runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from colorization.methods import kelvin_to_colormap_rgb, method_standard_colormap
from colorization.types import ColormapConfig
from pipeline.frozen_config import FrozenExperimentConfig
from pipeline.patch_refs import resolve_patch
from sr_data.consistency_refinement import ConsistencyRefinementConfig, refine_hr_normalized
from sr_data.dataset import ThermalSRDataset
from training.metrics import bicubic_upsample, denormalize_kelvin
from training.train_swinir import prepare_lr


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path)


def _z_to_kelvin(z: torch.Tensor, valid: torch.Tensor, stats) -> np.ndarray:
    k = denormalize_kelvin(z.double(), stats).numpy()
    mask = valid.numpy().astype(bool)
    out = np.full(z.shape, np.nan, dtype=np.float64)
    out[mask] = k[mask]
    return out


def _kelvin_colormap_png(kelvin: np.ndarray, valid: np.ndarray, cmap_cfg: ColormapConfig) -> np.ndarray:
    _, rgba = method_standard_colormap(kelvin, valid, cmap_cfg)
    return rgba


def render_stage_pngs(
    config: FrozenExperimentConfig,
    model: torch.nn.Module,
    scene_id: str,
    patch_id: str,
    out_dir: Path,
) -> dict[str, str]:
    """Run forward pass once; write bicubic/swinir/refined/LR preview PNGs under out_dir/stages/."""
    pref = resolve_patch(config.processed_root, scene_id, patch_id, degradation=config.degradation)
    sr_ds = ThermalSRDataset(
        config.processed_root,
        degradation=config.degradation,
        scene_ids=[scene_id],
        patch_indices=[pref.sr_dataset_index],
    )
    sample = sr_ds[0]
    meta = sr_ds.base_dataset.get_scene_metadata(scene_id)
    stats = meta.thermal_norm
    lr_raw = sample["lr"].unsqueeze(0)
    lr = prepare_lr(lr_raw)
    hr_valid = sample["hr_valid_mask"][0] & torch.isfinite(sample["hr"][0])
    valid_np = hr_valid.numpy().astype(bool)

    cmap_cfg = config.colormap
    stages_dir = out_dir / "stages"
    rel: dict[str, str] = {}

    with torch.no_grad():
        bic_z = bicubic_upsample(lr, config.degradation.scale)[0, 0]
        swin_z = model(lr)[0, 0]
    conf = ConsistencyRefinementConfig(beta=config.refinement_beta, degradation=config.degradation)
    refined_z = refine_hr_normalized(
        swin_z.cpu().numpy(),
        sample["lr"][0].numpy(),
        sample["lr_valid_mask"][0].numpy().astype(bool),
        stats,
        conf,
    )
    refined_t = torch.from_numpy(refined_z)

    lr_valid = sample["lr_valid_mask"][0].numpy().astype(bool)
    lr_k = denormalize_kelvin(sample["lr"][0].double(), stats).numpy()
    lr_up = np.repeat(np.repeat(lr_k, config.degradation.scale, axis=0), config.degradation.scale, axis=1)
    lr_up = lr_up[: valid_np.shape[0], : valid_np.shape[1]]
    lr_valid_up = np.repeat(np.repeat(lr_valid, config.degradation.scale, axis=0), config.degradation.scale, axis=1)
    lr_valid_up = lr_valid_up[: valid_np.shape[0], : valid_np.shape[1]]

    for key, z_tensor in [
        ("bicubic", bic_z),
        ("swinir", swin_z),
        ("refined", refined_t),
    ]:
        kelvin = _z_to_kelvin(z_tensor, hr_valid, stats)
        rgba = _kelvin_colormap_png(kelvin, valid_np, cmap_cfg)
        fname = f"{key}.png"
        _save_rgba(stages_dir / fname, rgba)
        rel[key] = f"stages/{fname}"

    lr_rgb = kelvin_to_colormap_rgb(
        np.where(lr_valid_up, lr_up, np.nan),
        lr_valid_up & np.isfinite(lr_up),
        cmap_cfg,
    )
    lr_rgba = np.zeros((*lr_rgb.shape[:2], 4), dtype=np.uint8)
    lr_rgba[..., :3] = (np.clip(lr_rgb, 0, 1) * 255).astype(np.uint8)
    lr_rgba[..., 3] = np.where(lr_valid_up, 255, 0).astype(np.uint8)
    _save_rgba(stages_dir / "input_lr.png", lr_rgba)
    rel["input_lr"] = "stages/input_lr.png"

    return rel


def run_forward_for_metrics(
    config: FrozenExperimentConfig,
    model: torch.nn.Module,
    scene_id: str,
    patch_id: str,
) -> dict[str, Any]:
    """Delegate to inference_helpers after forward (metrics only)."""
    from backend.inference_helpers import compute_sr_comparison

    return compute_sr_comparison(config, model, scene_id, patch_id)
