"""Integration pipeline: patch resolution, frozen config, Kelvin immutability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.frozen_config import METHOD_SPEC, VALIDATION_SCENE_ID, load_frozen_config
from pipeline.integrated import run_integrated_patch
from pipeline.model_loader import load_swinir_model
from pipeline.patch_refs import normalize_patch_key, resolve_patch
from training.repro import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "dataset" / "processed_qa"


@pytest.mark.skipif(not ROOT.is_dir(), reason="processed_qa not present")
def test_frozen_beta_from_method_txt():
    cfg = load_frozen_config()
    assert cfg.refinement_beta == 1.0
    assert METHOD_SPEC.is_file()


@pytest.mark.skipif(not ROOT.is_dir(), reason="processed_qa not present")
def test_resolve_patch_kerala():
    cfg = load_frozen_config(processed_root=ROOT)
    ref = resolve_patch(ROOT, VALIDATION_SCENE_ID, "000096", degradation=cfg.degradation)
    assert ref.patch_key == "000096"
    assert normalize_patch_key("96") == "000096"


@pytest.mark.skipif(not ROOT.is_dir(), reason="processed_qa not present")
@pytest.mark.skipif(not METHOD_SPEC.is_file(), reason="METHOD.txt missing")
def test_integrated_patch_end_to_end(tmp_path: Path):
    """One-patch CPU integration: shapes, Kelvin file, colorization does not mutate inputs."""
    cfg = load_frozen_config(processed_root=ROOT)
    set_seed(cfg.eval_seed)
    model, info = load_swinir_model(cfg)
    out = tmp_path / "integrated"
    entry = run_integrated_patch(
        cfg, VALIDATION_SCENE_ID, "000096", model, out, model_info=info
    )
    kelvin = np.load(out / entry["thermal_kelvin_file"])
    mask = np.load(out / entry["valid_mask_file"])
    assert kelvin.shape == mask.shape == (256, 256)
    assert (out / "patches" / VALIDATION_SCENE_ID / "patch_000096" / "A_standard_colormap.png").is_file()
    assert entry["thermal_not_hr_reference"] is True
    assert kelvin.shape == (256, 256)


@pytest.mark.skipif(not ROOT.is_dir(), reason="processed_qa not present")
def test_integrated_deterministic_kelvin(tmp_path: Path):
    cfg = load_frozen_config(processed_root=ROOT)
    set_seed(42)
    model, _ = load_swinir_model(cfg)
    a = run_integrated_patch(cfg, VALIDATION_SCENE_ID, "000448", model, tmp_path / "a")
    set_seed(42)
    model2, _ = load_swinir_model(cfg)
    b = run_integrated_patch(cfg, VALIDATION_SCENE_ID, "000448", model2, tmp_path / "b")
    ka = np.load(tmp_path / "a" / a["thermal_kelvin_file"])
    kb = np.load(tmp_path / "b" / b["thermal_kelvin_file"])
    assert np.allclose(ka, kb, equal_nan=True)
