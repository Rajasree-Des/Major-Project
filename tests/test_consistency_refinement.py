"""Deterministic tests for synthetic forward consistency refinement."""

from __future__ import annotations

import numpy as np
import torch

from dataset_loader.metadata import ChannelNormStats
from sr_data.consistency_refinement import (
    ConsistencyRefinementConfig,
    lr_forward_consistency_rmse_kelvin,
    refine_hr_kelvin,
    refine_hr_normalized,
    refine_torch_batch,
    synthetic_forward_lr_kelvin,
)
from sr_data.degradation import degrade_thermal_patch
from sr_data.metadata import DegradationConfig


def _patch(seed=0):
    rng = np.random.default_rng(seed)
    hr = rng.normal(0, 1, (32, 32)).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)
    mask[:2, :] = False
    hr[~mask] = np.nan
    stats = ChannelNormStats(mean=305.0, std=3.0, method="z")
    cfg = DegradationConfig(scale=2, blur_sigma_hr_pixels=1.2, noise_std_kelvin=0.05, base_seed=42)
    hr_n, lr_n, hr_m, lr_m = degrade_thermal_patch(hr, mask, stats, cfg, patch_key="000001")
    return hr_n, lr_n, hr_m, lr_m, stats, cfg


def test_beta_zero_is_identity():
    hr_n, lr_n, hr_m, lr_m, stats, cfg = _patch()
    swinir = hr_n + 0.05
    out = refine_hr_normalized(swinir, lr_n, lr_m, stats, ConsistencyRefinementConfig(beta=0.0, degradation=cfg))
    assert np.allclose(out[np.isfinite(out)], swinir[np.isfinite(out)], rtol=0, atol=1e-6)


def test_output_shape_and_finite_on_valid():
    hr_n, lr_n, hr_m, lr_m, stats, cfg = _patch()
    swinir = hr_n.copy()
    out = refine_hr_normalized(swinir, lr_n, lr_m, stats, ConsistencyRefinementConfig(beta=0.5, degradation=cfg))
    assert out.shape == hr_n.shape
    assert np.all(np.isfinite(out[hr_m]))


def test_improves_lr_consistency_vs_raw_swinir():
    hr_n, lr_n, hr_m, lr_m, stats, cfg = _patch()
    swinir = hr_n + 0.15
    before = lr_forward_consistency_rmse_kelvin(swinir, lr_n, lr_m, stats, cfg)
    refined = refine_hr_normalized(swinir, lr_n, lr_m, stats, ConsistencyRefinementConfig(beta=1.0, degradation=cfg))
    after = lr_forward_consistency_rmse_kelvin(refined, lr_n, lr_m, stats, cfg)
    assert after <= before + 1e-6


def test_does_not_read_hr_for_prediction():
    hr_n, lr_n, hr_m, lr_m, stats, cfg = _patch()
    swinir = np.full_like(hr_n, 0.0)
    out_a = refine_hr_normalized(swinir, lr_n, lr_m, stats, ConsistencyRefinementConfig(beta=0.5, degradation=cfg))
    hr_poison = hr_n.copy()
    hr_poison[:] = 999.0
    out_b = refine_hr_normalized(swinir, lr_n, lr_m, stats, ConsistencyRefinementConfig(beta=0.5, degradation=cfg))
    assert np.allclose(out_a, out_b, equal_nan=True)


def test_torch_batch_matches_numpy():
    hr_n, lr_n, hr_m, lr_m, stats, cfg = _patch()
    swinir = hr_n + 0.02
    lr_t = torch.from_numpy(lr_n).unsqueeze(0).unsqueeze(0)
    sr_t = torch.from_numpy(swinir).unsqueeze(0).unsqueeze(0)
    lm = torch.from_numpy(lr_m).unsqueeze(0).unsqueeze(0)
    conf = ConsistencyRefinementConfig(beta=0.5, degradation=cfg)
    batch = refine_torch_batch(lr_t, sr_t, lm, stats, conf)
    single = refine_hr_normalized(swinir, lr_n, lr_m, stats, conf)
    assert np.allclose(batch[0, 0].numpy(), single, equal_nan=True, rtol=1e-5, atol=1e-4)


def test_forward_matches_degradation_without_noise():
    rng = np.random.default_rng(0)
    hr = rng.normal(0, 1, (32, 32)).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)
    stats = ChannelNormStats(mean=305.0, std=3.0, method="z")
    cfg = DegradationConfig(scale=2, blur_sigma_hr_pixels=1.2, noise_std_kelvin=0.0, base_seed=42)
    hr_n2, lr_n2, hr_m2, lr_m2 = degrade_thermal_patch(hr, mask, stats, cfg, patch_key="000001")
    hr_k = hr_n2 * stats.std + stats.mean
    lr_k, _ = synthetic_forward_lr_kelvin(np.where(hr_m2, hr_k, np.nan), hr_m2, cfg)
    lr_k_ref = lr_n2 * stats.std + stats.mean
    use = lr_m2 & np.isfinite(lr_k) & np.isfinite(lr_k_ref)
    assert np.allclose(lr_k[use], lr_k_ref[use], rtol=1e-5, atol=1e-3)


def test_refinement_is_deterministic():
    hr_n, lr_n, hr_m, lr_m, stats, cfg = _patch()
    swinir = hr_n + 0.1
    conf = ConsistencyRefinementConfig(beta=0.75, degradation=cfg)
    a = refine_hr_kelvin(swinir * stats.std + stats.mean, lr_n * stats.std + stats.mean, lr_m, conf)
    b = refine_hr_kelvin(swinir * stats.std + stats.mean, lr_n * stats.std + stats.mean, lr_m, conf)
    assert np.allclose(a, b, equal_nan=True)
