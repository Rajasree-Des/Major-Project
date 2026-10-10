"""Scene-level split integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import SCENES
from dataset_loader.split import (
    SceneLevelSplit,
    SceneSplitError,
    development_patch_split,
    list_processed_scene_ids,
    load_scene_split_manifest,
    require_single_scene_for_patch_split,
    save_scene_split_manifest,
    scene_ids_for_role,
    validate_scene_split,
)
from dataset_loader.thermal_dataset import ThermalSatelliteDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
A, B = list(SCENES)


def test_valid_split_accepted():
    validate_scene_split(SceneLevelSplit(train=(A,), validation=(), test=(B,)), [A, B])


@pytest.mark.parametrize("split, match", [
    (SceneLevelSplit(train=(A,), validation=(A,), test=(B,)), "both train and validation"),
    (SceneLevelSplit(train=(A, A), validation=(), test=(B,)), "duplicate"),
    (SceneLevelSplit(train=(A, "LC09_UNKNOWN"), validation=(), test=(B,)), "unknown"),
    (SceneLevelSplit(train=(A,), validation=(), test=()), "test role is empty"),
    (SceneLevelSplit(train=(A,), validation=(), test=("LC09_OTHER",)), "unassigned"),
])
def test_invalid_splits_rejected(split, match):
    with pytest.raises(SceneSplitError, match=match):
        validate_scene_split(split, [A, B])


def test_manifest_round_trip(tmp_path):
    split = SceneLevelSplit(train=(A,), validation=(), test=(B,))
    path = tmp_path / "split.json"
    save_scene_split_manifest(path, split, seed=None, available_scene_ids=[A, B], extra={"rationale": "x"})
    assert json.loads(path.read_text())["seed"] is None
    loaded = load_scene_split_manifest(path, [A, B])
    assert (loaded.train, loaded.validation, loaded.test) == ((A,), (), (B,))
    assert scene_ids_for_role(loaded, "test") == (B,)
    with pytest.raises(SceneSplitError):
        scene_ids_for_role(loaded, "holdout")


def test_save_refuses_invalid_split(tmp_path):
    with pytest.raises(SceneSplitError):
        save_scene_split_manifest(tmp_path / "s.json", SceneLevelSplit(train=(A,), validation=(), test=(A,)),
                                  seed=None, available_scene_ids=[A, B])
    assert not (tmp_path / "s.json").exists()


def test_role_datasets_contain_only_role_scenes(two_scene_root):
    split = SceneLevelSplit(train=(A,), validation=(), test=(B,))
    for role in ("train", "test"):
        ids = scene_ids_for_role(split, role)
        ds = ThermalSatelliteDataset(two_scene_root, scene_ids=list(ids))
        assert {ref.scene_id for ref in ds.patch_refs} == set(ids)
        assert len(ds) == 9


def test_dev_patch_split_mixes_scenes_and_guard_raises(two_scene_root):
    ds = ThermalSatelliteDataset(two_scene_root)
    scene_of = [r.scene_id for r in ds.patch_refs]
    split = development_patch_split(len(scene_of), seed=42)
    train_scenes = {scene_of[i] for i in split.train_indices}
    held_scenes = {scene_of[i] for i in split.validation_indices + split.test_indices}
    assert train_scenes & held_scenes, "expected the patch split to leak scenes across partitions"
    with pytest.raises(ValueError, match="mix 2 scenes"):
        require_single_scene_for_patch_split(scene_of)
    require_single_scene_for_patch_split([A, A])


def test_frozen_project_manifest_is_valid():
    manifest = PROJECT_ROOT / "dataset" / "splits" / "scene_split_qa_v1.json"
    root = PROJECT_ROOT / "dataset" / "processed_qa"
    if not manifest.exists() or not root.exists():
        pytest.skip("project split manifest or qa_v1 root not present")
    split = load_scene_split_manifest(manifest, list_processed_scene_ids(root))
    assert len(split.train) == 4 and len(split.validation) == 1 and len(split.test) == 2
    data = json.loads(manifest.read_text())
    assert all(data["scenes"][s]["role"] == r for r in ("train", "validation", "test")
               for s in scene_ids_for_role(split, r))
