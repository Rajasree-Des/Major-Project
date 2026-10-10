"""On-demand patch preview PNGs (cached under outputs/ui_cache)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

from backend.config import UI_CACHE_ROOT
from backend.scenes import patch_metadata
from dataset_loader.physical import denormalize_from_scene_metadata
from pipeline.frozen_config import load_frozen_config

VMIN_K, VMAX_K, CMAP = 295.0, 320.0, "inferno"


def _cache_path(scene_id: str, patch_id: str, kind: str) -> Path:
    safe_scene = scene_id.replace("/", "_")
    return UI_CACHE_ROOT / safe_scene / f"patch_{patch_id}_{kind}.png"


def ensure_preview(scene_id: str, patch_id: str, kind: str) -> Path:
    """kind: thermal_colormap | rgb"""
    from dataset_loader.thermal_dataset import ThermalSatelliteDataset
    from pipeline.patch_refs import resolve_patch

    path = _cache_path(scene_id, patch_id, kind)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_frozen_config()
    pref = resolve_patch(cfg.processed_root, scene_id, patch_id, degradation=cfg.degradation)
    ds = ThermalSatelliteDataset(cfg.processed_root, scene_ids=[scene_id], patch_indices=[pref.thermal_dataset_index])
    sample = ds[0]
    meta = ds.get_scene_metadata(scene_id)
    valid = sample["valid_mask"][0].numpy().astype(bool)
    if kind == "rgb":
        rgb = sample["rgb"].permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
        Image.fromarray(rgba).save(path)
        return path
    if kind == "thermal_colormap":
        z = sample["thermal"][0].numpy()
        kelvin, _ = denormalize_from_scene_metadata(z, sample["rgb"].permute(1, 2, 0).numpy(), meta)
        kelvin = np.where(valid, kelvin, np.nan)
        norm = Normalize(vmin=VMIN_K, vmax=VMAX_K)
        cmap_obj = matplotlib.colormaps.get_cmap(CMAP)
        colored = cmap_obj(norm(np.nan_to_num(kelvin, nan=VMIN_K)))[..., :3]
        alpha = np.where(valid, 1.0, 0.0)
        rgba = np.dstack([colored, alpha])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgba)
        ax.axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        return path
    raise ValueError(f"Unknown preview kind: {kind}")
