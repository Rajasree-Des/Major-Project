"""Kelvin conversion, masking, and aggregation for multiscene metrics."""

from __future__ import annotations

import math

import pytest
import torch

from dataset_loader.metadata import ChannelNormStats
from evaluation.metric_core import (
    MethodSums,
    SceneSums,
    accumulate_method_on_patch,
    aggregate,
    finalize_method,
    finalize_scene,
    kelvin_mae_from_z,
)
from training.metrics import denormalize_kelvin


def test_kelvin_uses_scene_specific_std():
    stats = ChannelNormStats(mean=300.0, std=4.0, method="z_score_valid_pixels")
    hr = torch.tensor([[0.0, 1.0], [0.5, -0.5]])
    pred = hr + 0.1
    valid = torch.ones_like(hr, dtype=torch.bool)
    m = MethodSums()
    accumulate_method_on_patch(m, pred, hr, valid, stats)
    fin = finalize_method(m)
    assert fin["mae_z"] == pytest.approx(0.1, rel=1e-6)
    assert fin["mae_kelvin"] == pytest.approx(kelvin_mae_from_z(0.1, 4.0), rel=1e-6)
    p_k = denormalize_kelvin(pred[valid].double(), stats)
    h_k = denormalize_kelvin(hr[valid].double(), stats)
    assert fin["mae_kelvin"] == pytest.approx(float((p_k - h_k).abs().mean()), rel=1e-6)


def test_second_scene_different_std_changes_kelvin_not_z():
    s1 = ChannelNormStats(mean=300.0, std=2.0, method="z")
    s2 = ChannelNormStats(mean=320.0, std=8.0, method="z")
    hr = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    pred = hr + 0.2
    valid = torch.ones_like(hr, dtype=torch.bool)
    m1, m2 = MethodSums(), MethodSums()
    accumulate_method_on_patch(m1, pred, hr, valid, s1)
    accumulate_method_on_patch(m2, pred, hr, valid, s2)
    assert finalize_method(m1)["mae_z"] == finalize_method(m2)["mae_z"]
    assert finalize_method(m2)["mae_kelvin"] == pytest.approx(4 * finalize_method(m1)["mae_kelvin"], rel=1e-6)


def test_nan_hr_pixels_excluded_from_accumulation():
    stats = ChannelNormStats(mean=300.0, std=3.0, method="z")
    hr = torch.tensor([[0.0, float("nan")], [1.0, 0.5]])
    pred = hr.clone()
    pred[torch.isfinite(hr)] += 0.1
    valid = torch.isfinite(hr)
    m = MethodSums()
    accumulate_method_on_patch(m, pred, hr, valid, stats)
    assert m.n_valid == 3


def test_aggregate_macro_vs_pooled():
    sums_a = SceneSums("A", 300.0, 2.0, {"m": MethodSums(n_valid=100, sum_abs_k=10.0, sum_sq_k=20.0,
                                                         sum_abs=5.0, sum_sq=8.0, psnr=[30.0], ssim=[0.9])},
                       valid_pixels=100, patches_scored=1)
    sums_b = SceneSums("B", 310.0, 4.0, {"m": MethodSums(n_valid=400, sum_abs_k=40.0, sum_sq_k=80.0,
                                                         sum_abs=10.0, sum_sq=16.0, psnr=[32.0], ssim=[0.8])},
                       valid_pixels=400, patches_scored=1)
    scenes = [finalize_scene(sums_a, "m"), finalize_scene(sums_b, "m")]
    agg = aggregate(scenes, [sums_a, sums_b], ["m"], "m")
    assert agg["m"]["pooled_mae_kelvin"] == pytest.approx(50.0 / 500, rel=1e-9)
    assert agg["m"]["pooled_rmse_kelvin"] == pytest.approx(math.sqrt(100.0 / 500), rel=1e-9)
    macro = agg["m"]["rmse_kelvin"]["macro_mean_over_scenes"]
    assert macro == pytest.approx((math.sqrt(20 / 100) + math.sqrt(80 / 400)) / 2, rel=1e-9)

