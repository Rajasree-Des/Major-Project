"""Shared fixtures: a tiny synthetic processed root with two scenes and different statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PATCH = 16
STRIDE = 8
SCENES = {
    "LC09_L2SP_000001_20240101_02_T1": {"mean": 300.0, "std": 2.0},
    "LC08_L2SP_000002_20240102_02_T1": {"mean": 320.0, "std": 5.0},
}


def write_scene(root: Path, scene_id: str, mean: float, std: float, *, seed: int,
                empty_patch_ids: tuple[int, ...] = (), all_empty: bool = False) -> None:
    d = root / scene_id
    (d / "thermal").mkdir(parents=True)
    (d / "rgb").mkdir()
    rng = np.random.default_rng(seed)
    coords = []
    pid = 0
    for row in range(0, 32 - PATCH + 1, STRIDE):
        for col in range(0, 32 - PATCH + 1, STRIDE):
            pid += 1
            t = rng.normal(0, 1, (PATCH, PATCH)).astype(np.float32)
            t[:2, :2] = np.nan
            rgb = rng.normal(0, 1, (PATCH, PATCH, 3)).astype(np.float32)
            rgb[:, :, 0][~np.isfinite(t)] = np.nan
            if all_empty or pid in empty_patch_ids:
                # Thermal and RGB each partly finite (loader accepts the patch) but never at the same pixel.
                t[:, : PATCH // 2] = np.nan
                rgb[:, PATCH // 2:, 0] = np.nan
            np.save(d / "thermal" / f"patch_{pid:06d}.npy", t)
            np.save(d / "rgb" / f"patch_{pid:06d}.npy", rgb)
            coords.append({"patch_id": pid, "row": row, "col": col, "invalid_fraction": 0.0})
    meta = {
        "scene_id": scene_id, "patch_size": PATCH, "stride": STRIDE,
        "patches_generated": pid, "patches_skipped": 0, "patch_coordinates": coords,
        "normalization": {"method": "z_score_valid_pixels",
                          "thermal": {"mean": mean, "std": std, "method": "z_score_valid_pixels"},
                          "rgb": [{"mean": 0.1, "std": 0.05}] * 3},
        "thermal_data_representation": {"exported": "z_score_normalized_kelvin"},
        "rgb_data_representation": {"exported": "z_score_normalized_reflectance"},
    }
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def two_scene_root(tmp_path: Path) -> Path:
    root = tmp_path / "processed_qa"
    for i, (sid, s) in enumerate(SCENES.items()):
        write_scene(root, sid, s["mean"], s["std"], seed=i, empty_patch_ids=(2,) if i == 0 else ())
    return root
