"""Scene and patch metadata from real QA dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import PROCESSED_ROOT, PROCESSING_MANIFEST, SPLIT_PATH
from pipeline.frozen_config import load_frozen_config
from pipeline.patch_refs import normalize_patch_key


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_scenes() -> list[dict[str, Any]]:
    if not PROCESSING_MANIFEST.is_file():
        return []
    manifest = _load_json(PROCESSING_MANIFEST)
    split = _load_json(SPLIT_PATH) if SPLIT_PATH.is_file() else {}
    split_scenes = split.get("scenes", {})
    out: list[dict[str, Any]] = []
    for scene_id, entry in manifest.get("scenes", {}).items():
        meta_path = PROCESSED_ROOT / scene_id / "metadata.json"
        region = split_scenes.get(scene_id, {})
        out.append(
            {
                "scene_id": scene_id,
                "patches_after_qa": entry.get("patches_after_qa"),
                "thermal_norm": entry.get("thermal_norm"),
                "valid_fraction_after_qa": entry.get("valid_fraction_after_qa"),
                "processed_at": entry.get("processed_at"),
                "split_role": region.get("role"),
                "region": region.get("region"),
                "sensor": region.get("sensor"),
                "acquisition_date": region.get("acquisition_date"),
                "wrs_path_row": region.get("wrs_path_row"),
                "metadata_available": meta_path.is_file(),
            }
        )
    out.sort(key=lambda s: s["scene_id"])
    return out


def list_patch_ids(scene_id: str, *, limit: int | None = None, offset: int = 0) -> dict[str, Any]:
    from dataset_loader.thermal_dataset import ThermalSatelliteDataset

    if not (PROCESSED_ROOT / scene_id).is_dir():
        raise FileNotFoundError(f"Scene not found: {scene_id}")
    ds = ThermalSatelliteDataset(PROCESSED_ROOT, scene_ids=[scene_id])
    keys = [ref.patch_key for ref in ds.patch_refs]
    total = len(keys)
    slice_keys = keys[offset : offset + limit if limit is not None else None]
    return {"scene_id": scene_id, "total": total, "offset": offset, "patch_ids": slice_keys}


DEMO_SCENE_ID = "LC08_L2SP_144053_20250225_02_T1"
DEMO_PATCH_ID = "000096"


def demo_patch_available() -> bool:
    from pipeline.patch_refs import resolve_patch

    try:
        cfg = load_frozen_config()
        resolve_patch(cfg.processed_root, DEMO_SCENE_ID, DEMO_PATCH_ID, degradation=cfg.degradation)
        return True
    except (FileNotFoundError, ValueError):
        return False


def patch_metadata(scene_id: str, patch_id: str) -> dict[str, Any]:
    from dataset_loader.thermal_dataset import ThermalSatelliteDataset
    from pipeline.patch_refs import resolve_patch

    cfg = load_frozen_config()
    pref = resolve_patch(PROCESSED_ROOT, scene_id, patch_id, degradation=cfg.degradation)
    ds = ThermalSatelliteDataset(PROCESSED_ROOT, scene_ids=[scene_id], patch_indices=[pref.thermal_dataset_index])
    sample = ds[0]
    meta = ds.get_scene_metadata(scene_id)
    valid = sample["valid_mask"][0].numpy()
    thermal = sample["thermal"][0].numpy()
    import numpy as np

    valid_count = int(valid.astype(bool).sum())
    h, w = thermal.shape[-2], thermal.shape[-1]
    return {
        "scene_id": scene_id,
        "patch_id": pref.patch_key,
        "thermal_dataset_index": pref.thermal_dataset_index,
        "sr_dataset_index": pref.sr_dataset_index,
        "height": h,
        "width": w,
        "valid_pixels": valid_count,
        "valid_fraction": valid_count / (h * w) if h * w else 0.0,
        "normalization": {
            "thermal": {
                "mean": meta.thermal_norm.mean,
                "std": meta.thermal_norm.std,
                "method": meta.thermal_norm.method,
            },
        },
        "preview_note": "Explorer previews are derived from prepared 30 m reference patches (not pipeline inference).",
    }
