"""Shared metric accumulation for multi-scene evaluation (Kelvin uses per-scene z-score stats)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from dataset_loader.metadata import ChannelNormStats
from training.metrics import compute_psnr, compute_ssim, denormalize_kelvin

METRIC_NAMES = (
    "mae_z",
    "rmse_z",
    "mae_kelvin",
    "rmse_kelvin",
    "psnr_mean_per_patch",
    "ssim_mean_per_patch",
)


@dataclass
class MethodSums:
    n_valid: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_abs_k: float = 0.0
    sum_sq_k: float = 0.0
    psnr: list[float] = field(default_factory=list)
    ssim: list[float] = field(default_factory=list)
    psnr_infinite: int = 0
    nonfinite_at_valid: int = 0
    forward_s: float = 0.0


@dataclass
class SceneSums:
    scene_id: str
    mean_k: float
    std_k: float
    methods: dict[str, MethodSums]
    patches_total: int = 0
    patches_scored: int = 0
    patches_empty_valid: int = 0
    valid_pixels: int = 0


def _hw(t: torch.Tensor) -> torch.Tensor:
    return t[0] if t.ndim == 3 else t


def accumulate_method_on_patch(
    m: MethodSums,
    pred: torch.Tensor,
    hr: torch.Tensor,
    valid: torch.Tensor,
    stats: ChannelNormStats,
) -> dict[str, float]:
    """Update ``m`` from one patch; returns per-patch metric dict for CSV rows."""
    pred, hr, valid = _hw(pred), _hw(hr), _hw(valid)
    p_v = pred[valid].double()
    hr_v = hr[valid].double()
    n = int(valid.sum())
    m.nonfinite_at_valid += int((~torch.isfinite(p_v)).sum())
    diff = p_v - hr_v
    hr_k = denormalize_kelvin(hr_v, stats)
    diff_k = denormalize_kelvin(p_v, stats) - hr_k
    m.n_valid += n
    m.sum_abs += float(diff.abs().sum())
    m.sum_sq += float((diff**2).sum())
    m.sum_abs_k += float(diff_k.abs().sum())
    m.sum_sq_k += float((diff_k**2).sum())
    pred4 = pred.unsqueeze(0).unsqueeze(0)
    hr4 = hr.unsqueeze(0).unsqueeze(0)
    valid4 = valid.unsqueeze(0).unsqueeze(0)
    psnr = compute_psnr(pred4, hr4, valid4)
    if math.isinf(psnr):
        m.psnr_infinite += 1
    m.psnr.append(psnr)
    ssim = compute_ssim(torch.nan_to_num(pred4), torch.nan_to_num(hr4), valid4)
    m.ssim.append(ssim)
    return {
        "mae_kelvin": float(diff_k.abs().mean()),
        "rmse_kelvin": float(diff_k.pow(2).mean().sqrt()),
        "mae_z": float(diff.abs().mean()),
        "psnr": psnr,
        "ssim": ssim,
    }


def finalize_method(m: MethodSums) -> dict[str, Any]:
    finite_psnr = [v for v in m.psnr if math.isfinite(v)]
    return {
        "mae_z": m.sum_abs / m.n_valid,
        "rmse_z": math.sqrt(m.sum_sq / m.n_valid),
        "mae_kelvin": m.sum_abs_k / m.n_valid,
        "rmse_kelvin": math.sqrt(m.sum_sq_k / m.n_valid),
        "psnr_mean_per_patch": float(np.mean(finite_psnr)) if finite_psnr else None,
        "ssim_mean_per_patch": float(np.mean(m.ssim)) if m.ssim else None,
        "psnr_infinite_patches": m.psnr_infinite,
        "nonfinite_predictions_at_valid_pixels": m.nonfinite_at_valid,
        "forward_s": m.forward_s,
    }


def finalize_scene(s: SceneSums, baseline_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scene_id": s.scene_id,
        "thermal_mean_k": s.mean_k,
        "thermal_std_k": s.std_k,
        "patches_total": s.patches_total,
        "patches_scored": s.patches_scored,
        "patches_excluded_zero_valid": s.patches_empty_valid,
        "valid_pixels": s.valid_pixels,
    }
    if s.valid_pixels == 0:
        out["metrics"] = None
        out["reason"] = "no valid HR pixels in the selected patches"
        return out
    out["metrics"] = {k: finalize_method(m) for k, m in s.methods.items()}
    base = out["metrics"].get(baseline_key)
    if base is not None:
        out["delta_vs_" + baseline_key] = {
            k: {
                name: (v[name] - base[name]) if v[name] is not None and base[name] is not None else None
                for name in METRIC_NAMES
            }
            for k, v in out["metrics"].items()
            if k != baseline_key
        }
    return out


def aggregate(
    scenes: list[dict[str, Any]],
    sums: list[SceneSums],
    method_keys: Sequence[str],
    baseline_key: str,
) -> dict[str, Any]:
    scored = [s for s in scenes if s["metrics"] is not None]
    agg: dict[str, Any] = {
        "scenes_with_metrics": len(scored),
        "scenes_without_metrics": len(scenes) - len(scored),
    }
    for key in method_keys:
        per_metric = {}
        for name in METRIC_NAMES:
            vals = [s["metrics"][key][name] for s in scored if s["metrics"][key][name] is not None]
            per_metric[name] = (
                {
                    "macro_mean_over_scenes": float(np.mean(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
                if vals
                else None
            )
        n = sum(s.methods[key].n_valid for s in sums)
        per_metric["pooled_mae_kelvin"] = sum(s.methods[key].sum_abs_k for s in sums) / n if n else None
        per_metric["pooled_rmse_kelvin"] = (
            math.sqrt(sum(s.methods[key].sum_sq_k for s in sums) / n) if n else None
        )
        if key != baseline_key:
            per_metric["scenes_better_than_baseline_rmse_kelvin"] = sum(
                s["metrics"][key]["rmse_kelvin"] < s["metrics"][baseline_key]["rmse_kelvin"] for s in scored
            )
        agg[key] = per_metric
    agg["note"] = (
        "Macro means weight every scene equally; pooled Kelvin metrics weight scenes by valid pixels. "
        "z-space metrics are not pooled across scenes because each scene has its own std."
    )
    return agg


def kelvin_mae_from_z(mae_z: float, std_k: float) -> float:
    """When |error| is constant in z, MAE in Kelvin equals MAE_z * std."""
    return mae_z * std_k
