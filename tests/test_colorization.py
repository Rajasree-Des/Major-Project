"""Colorization integrity: no thermal mutation, masks, determinism, shape."""

from __future__ import annotations

import numpy as np

from colorization.methods import method_rgb_guided, method_rgb_overlay, method_standard_colormap
from colorization.pipeline import ColorizationInputs, apply_all_methods
from colorization.types import ColormapConfig, GuidedColorizationConfig, OverlayConfig


def _sample():
    rng = np.random.default_rng(0)
    h = 32
    thermal = rng.normal(300, 5, (h, h))
    rgb = rng.normal(0.05, 0.02, (h, h, 3))
    valid = np.ones((h, h), dtype=bool)
    valid[:3, :] = False
    thermal[~valid] = np.nan
    rgb[~valid] = np.nan
    return thermal, rgb, valid


def test_thermal_array_not_mutated():
    thermal, rgb, valid = _sample()
    inputs = ColorizationInputs(thermal, rgb, valid)
    before = inputs.thermal_kelvin.copy()
    apply_all_methods(inputs)
    assert np.allclose(before, inputs.thermal_kelvin, equal_nan=True, rtol=0, atol=0)


def test_output_shapes_and_determinism():
    thermal, rgb, valid = _sample()
    cfg = ColormapConfig(vmin_kelvin=290, vmax_kelvin=310)
    a1, _ = method_standard_colormap(thermal, valid, cfg)
    a2, _ = method_standard_colormap(thermal, valid, cfg)
    assert a1.shape == (32, 32, 3)
    assert np.array_equal(a1, a2)


def test_invalid_pixels_transparent():
    thermal, rgb, valid = _sample()
    _, rgba = method_standard_colormap(thermal, valid, ColormapConfig())
    assert rgba[~valid, 3].max() == 0


def test_guided_and_overlay_finite_on_valid():
    thermal, rgb, valid = _sample()
    g, _ = method_rgb_guided(thermal, rgb, valid, GuidedColorizationConfig())
    o, _ = method_rgb_overlay(thermal, rgb, valid, OverlayConfig())
    assert np.all(np.isfinite(g[valid]))
    assert np.all(np.isfinite(o[valid]))


def test_apply_all_methods_keys():
    thermal, rgb, valid = _sample()
    out = apply_all_methods(ColorizationInputs(thermal, rgb, valid))
    assert set(out) >= {"standard_colormap", "rgb_guided", "rgb_overlay", "thermal_kelvin_unchanged"}
    assert out["thermal_kelvin_unchanged"]
