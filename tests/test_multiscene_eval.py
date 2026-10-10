"""Multi-scene evaluation: per-scene Kelvin stats, masks, empty-valid handling, determinism."""

from __future__ import annotations

import json
import math

import pytest
import torch

from conftest import PATCH, SCENES, write_scene
from evaluation.evaluate_multiscene import evaluate_role, select_patch_indices
from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import DegradationConfig
from training.metrics import bicubic_upsample

A, B = list(SCENES)
DELTA = 0.1


def make_methods(hr_lookup=None):
    bicubic = ("bicubic", "Bicubic", lambda x: bicubic_upsample(x, 2))
    offset = ("offset", "Bicubic + 0.1 z", lambda x: bicubic_upsample(x, 2) + DELTA)
    return [bicubic, offset]


def run(root, scene_ids, selection="all", methods=None):
    return evaluate_role(root, scene_ids, methods or make_methods(), degradation=DegradationConfig(scale=2, base_seed=42),
                         selection=selection, batch_size=3, log_every=0)


def test_kelvin_metrics_use_each_scenes_own_std(two_scene_root):
    """An oracle returning HR + delta (z) must have a Kelvin error of delta * that scene's std."""
    ds = ThermalSRDataset(two_scene_root, degradation=DegradationConfig(scale=2, base_seed=42))
    hr_by_lr = {}
    for i in range(len(ds)):
        s = ds[i]
        hr_by_lr[torch.nan_to_num(s["lr"]).sum().item()] = s["hr"]

    def oracle(x):
        return torch.stack([torch.nan_to_num(hr_by_lr[x[i].sum().item()]) + DELTA for i in range(x.shape[0])])

    methods = [("bicubic", "Bicubic", lambda x: bicubic_upsample(x, 2)), ("oracle", "HR + delta", oracle)]
    r = run(two_scene_root, [A, B], methods=methods)
    for s in r["per_scene"]:
        m = s["metrics"]["oracle"]
        assert m["mae_kelvin"] == pytest.approx(DELTA * SCENES[s["scene_id"]]["std"], rel=1e-5)
        assert m["mae_z"] == pytest.approx(DELTA, rel=1e-5)


def test_masks_identical_across_methods_and_nan_hr_excluded(two_scene_root):
    r = run(two_scene_root, [A, B])
    assert r["sanity_checks"]["passed"]
    assert r["sanity_checks"]["identical_valid_pixels_all_methods"]
    for s in r["per_scene"]:
        assert s["valid_pixels"] < s["patches_scored"] * PATCH * PATCH
        assert all(math.isfinite(v) for v in s["metrics"]["bicubic"].values() if isinstance(v, float))


def test_zero_valid_patch_excluded_and_counted(two_scene_root):
    r = run(two_scene_root, [A, B])
    a = next(s for s in r["per_scene"] if s["scene_id"] == A)
    b = next(s for s in r["per_scene"] if s["scene_id"] == B)
    assert a["patches_excluded_zero_valid"] == 1 and a["patches_scored"] == a["patches_total"] - 1
    assert b["patches_excluded_zero_valid"] == 0
    excluded = [row for row in r["_rows"] if row.get("excluded")]
    assert [(row["scene_id"], row["patch_id"]) for row in excluded] == [(A, "000002")]


def test_scene_without_valid_pixels_gets_null_metrics(tmp_path):
    root = tmp_path / "root"
    write_scene(root, A, 300.0, 2.0, seed=0)
    write_scene(root, B, 320.0, 5.0, seed=1, all_empty=True)
    r = run(root, [A, B])
    b = next(s for s in r["per_scene"] if s["scene_id"] == B)
    assert b["metrics"] is None and b["valid_pixels"] == 0 and "reason" in b
    agg = r["aggregate"]
    assert agg["scenes_with_metrics"] == 1 and agg["scenes_without_metrics"] == 1
    assert math.isfinite(agg["bicubic"]["rmse_kelvin"]["macro_mean_over_scenes"])
    assert math.isfinite(agg["bicubic"]["pooled_rmse_kelvin"])


def test_non_overlapping_selection(two_scene_root):
    ds = ThermalSRDataset(two_scene_root, degradation=DegradationConfig(scale=2, base_seed=42))
    idx = select_patch_indices(ds, "non_overlapping")
    coords = {}
    for sid, meta in ds.base_dataset.metadata_by_scene.items():
        for c in meta.raw["patch_coordinates"]:
            coords[(sid, c["patch_id"])] = (c["row"], c["col"])
    picked = [coords[(ds.base_dataset.patch_refs[i].scene_id, ds.base_dataset.patch_refs[i].patch_id)] for i in idx]
    assert len(idx) == 8
    assert all(r % PATCH == 0 and c % PATCH == 0 for r, c in picked)
    assert select_patch_indices(ds, "all") == list(range(len(ds)))


def test_evaluation_is_deterministic(two_scene_root):
    def strip(r):
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        for s in r["per_scene"]:
            for m in (s["metrics"] or {}).values():
                m.pop("forward_s", None)
        return json.dumps(r, sort_keys=True)

    assert strip(run(two_scene_root, [A, B])) == strip(run(two_scene_root, [A, B]))


def test_role_rejects_missing_scene(two_scene_root):
    with pytest.raises(ValueError, match="found"):
        run(two_scene_root, [A, "LC09_L2SP_999999_20240101_02_T1"])


def test_offset_method_worse_than_bicubic_everywhere(two_scene_root):
    r = run(two_scene_root, [A, B])
    agg = r["aggregate"]["offset"]
    assert agg["scenes_better_than_baseline_rmse_kelvin"] == 0
    for s in r["per_scene"]:
        assert s["delta_vs_bicubic"]["offset"]["rmse_kelvin"] > 0
