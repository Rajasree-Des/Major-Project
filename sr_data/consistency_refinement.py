"""Post-SwinIR correction using the project's synthetic forward operator (not a radiative-transfer model).

The training degradation is: HR Kelvin -> masked Gaussian blur (sigma) -> area downsample -> noise -> LR.
This module enforces *noise-free* consistency between the SwinIR HR estimate and the observed LR under that
blur+downsample chain, by injecting a smooth (Gaussian-filtered) upsampled LR residual. SwinIR high-frequency
content is unchanged by the correction term by construction (correction lives in the same low-pass band).

Does not read reference HR pixels. Uses LR thermal, SwinIR output, LR validity mask, and scene normalization stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dataset_loader.metadata import ChannelNormStats
from sr_data.degradation import (
    downsample_kelvin_and_mask,
    masked_gaussian_blur_kelvin,
    normalize_kelvin_to_training_space,
)
from sr_data.metadata import DegradationConfig


@dataclass(frozen=True)
class ConsistencyRefinementConfig:
    """Validation-tuned strength; beta=0 returns the SwinIR input unchanged."""

    beta: float = 0.5
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    def resolved_degradation(self) -> DegradationConfig:
        return self.degradation

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")


def _repeat_upsample(lr: np.ndarray, scale: int) -> np.ndarray:
    """Block replication (adjoint of area downsample on a uniform grid)."""
    return np.repeat(np.repeat(lr, scale, axis=0), scale, axis=1)


def synthetic_forward_lr_kelvin(
    hr_kelvin: np.ndarray,
    hr_valid: np.ndarray,
    config: DegradationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free forward: blur then area downsample (matches ``degrade_thermal_patch`` without noise)."""
    blurred = masked_gaussian_blur_kelvin(hr_kelvin, hr_valid, config.blur_sigma_hr_pixels)
    lr_k, lr_valid = downsample_kelvin_and_mask(blurred, hr_valid, config.scale)
    return lr_k, lr_valid


def refine_hr_kelvin(
    swinir_k: np.ndarray,
    lr_k: np.ndarray,
    lr_valid: np.ndarray,
    config: ConsistencyRefinementConfig,
) -> np.ndarray:
    """Apply smooth LR consistency correction in Kelvin."""
    deg = config.resolved_degradation()
    if config.beta == 0.0:
        return swinir_k.astype(np.float64, copy=True)

    pred_valid = np.isfinite(swinir_k)
    hr_k = np.where(pred_valid, swinir_k, np.nan)
    forward_lr, _ = synthetic_forward_lr_kelvin(hr_k, pred_valid, deg)

    residual = np.full_like(forward_lr, np.nan, dtype=np.float64)
    use = lr_valid & np.isfinite(lr_k) & np.isfinite(forward_lr)
    residual[use] = lr_k[use] - forward_lr[use]

    h, w = swinir_k.shape
    up = _repeat_upsample(np.nan_to_num(residual, nan=0.0), deg.scale)[:h, :w]
    up_mask = _repeat_upsample(lr_valid.astype(np.float64), deg.scale)[:h, :w] >= 0.5
    smooth = masked_gaussian_blur_kelvin(
        np.where(up_mask, up, np.nan),
        up_mask,
        deg.blur_sigma_hr_pixels,
    )
    out = swinir_k.astype(np.float64, copy=True)
    apply = pred_valid & np.isfinite(smooth)
    out[apply] = swinir_k[apply] + config.beta * smooth[apply]
    return out


def refine_hr_normalized(
    swinir_z: np.ndarray,
    lr_z: np.ndarray,
    lr_valid: np.ndarray,
    stats: ChannelNormStats,
    config: ConsistencyRefinementConfig,
) -> np.ndarray:
    """Refine in Kelvin, return z-score normalized HR (same convention as preprocessing)."""
    swinir_k = swinir_z.astype(np.float64) * stats.std + stats.mean
    lr_k = lr_z.astype(np.float64) * stats.std + stats.mean
    refined_k = refine_hr_kelvin(swinir_k, lr_k, lr_valid, config)
    valid = np.isfinite(refined_k)
    return normalize_kelvin_to_training_space(refined_k, valid, stats)


def refine_torch_batch(
    lr: torch.Tensor,
    swinir: torch.Tensor,
    lr_valid_mask: torch.Tensor,
    stats: ChannelNormStats,
    config: ConsistencyRefinementConfig,
) -> torch.Tensor:
    """(B,1,H,W) tensors -> refined HR in z-space."""
    if lr.shape[0] != swinir.shape[0]:
        raise ValueError(f"batch size mismatch: {lr.shape[0]} vs {swinir.shape[0]}")
    outs = []
    for i in range(lr.shape[0]):
        z = refine_hr_normalized(
            swinir[i, 0].detach().cpu().numpy(),
            lr[i, 0].detach().cpu().numpy(),
            lr_valid_mask[i, 0].detach().cpu().numpy().astype(bool),
            stats,
            config,
        )
        outs.append(torch.from_numpy(z).unsqueeze(0))
    return torch.stack(outs, dim=0).to(device=swinir.device, dtype=swinir.dtype)


def lr_forward_consistency_rmse_kelvin(
    hr_z: np.ndarray,
    lr_z: np.ndarray,
    lr_valid: np.ndarray,
    stats: ChannelNormStats,
    config: DegradationConfig,
) -> float:
    """Diagnostic: RMSE in Kelvin between observed LR and noise-free forward(HR) on valid LR pixels."""
    hr_k = hr_z.astype(np.float64) * stats.std + stats.mean
    lr_k = lr_z.astype(np.float64) * stats.std + stats.mean
    pred_valid = np.isfinite(hr_k)
    forward_lr, _ = synthetic_forward_lr_kelvin(np.where(pred_valid, hr_k, np.nan), pred_valid, config)
    use = lr_valid & np.isfinite(lr_k) & np.isfinite(forward_lr)
    if not use.any():
        return float("nan")
    diff = forward_lr[use] - lr_k[use]
    return float(np.sqrt(np.mean(diff**2)))
