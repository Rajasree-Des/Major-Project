"""Synthetic thermal super-resolution degradation (Kelvin-space, mask-aware)."""

from __future__ import annotations

import hashlib
import math
import numpy as np

from dataset_loader.metadata import ChannelNormStats
from sr_data.metadata import DegradationConfig


def _patch_rng(patch_key: str, base_seed: int | None) -> np.random.Generator:
    if base_seed is None:
        return np.random.default_rng()
    digest = hashlib.sha256(f"{base_seed}:{patch_key}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32 - 1)
    return np.random.default_rng(seed)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve_axis(
    data: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(data, dtype=np.float64)
        for i in range(data.shape[0]):
            out[i, :] = np.sum(
                padded[i : i + len(kernel), :] * kernel[:, np.newaxis],
                axis=0,
            )
        return out
    padded = np.pad(data, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(data, dtype=np.float64)
    for j in range(data.shape[1]):
        out[:, j] = np.sum(
            padded[:, j : j + len(kernel)] * kernel[np.newaxis, :],
            axis=1,
        )
    return out


def masked_gaussian_blur_kelvin(
    kelvin: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Low-pass filter in Kelvin; invalid pixels do not contribute."""
    kelvin = kelvin.astype(np.float64, copy=False)
    values = np.where(valid_mask, kelvin, 0.0)
    weights = valid_mask.astype(np.float64)

    kernel = _gaussian_kernel_1d(sigma)
    num = _convolve_axis(_convolve_axis(values, kernel, axis=0), kernel, axis=1)
    den = _convolve_axis(_convolve_axis(weights, kernel, axis=0), kernel, axis=1)

    blurred = np.full(kelvin.shape, np.nan, dtype=np.float64)
    ok = den > 1e-8
    blurred[ok] = num[ok] / den[ok]
    return blurred


def downsample_kelvin_and_mask(
    kelvin: np.ndarray,
    valid_mask: np.ndarray,
    scale: int,
    *,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Area downsample Kelvin and propagate validity mask."""
    if scale == 1:
        return kelvin.copy(), valid_mask.copy()

    h, w = kelvin.shape
    h_trim = (h // scale) * scale
    w_trim = (w // scale) * scale
    kelvin = kelvin[:h_trim, :w_trim]
    valid_mask = valid_mask[:h_trim, :w_trim]

    h_lr, w_lr = h_trim // scale, w_trim // scale
    k_blocks = kelvin.reshape(h_lr, scale, w_lr, scale)
    m_blocks = valid_mask.reshape(h_lr, scale, w_lr, scale)

    weights = m_blocks.sum(axis=(1, 3)).astype(np.float64)
    weighted = np.where(m_blocks, k_blocks, 0.0).sum(axis=(1, 3))

    lr_k = np.full((h_lr, w_lr), np.nan, dtype=np.float64)
    positive = weights > 0
    lr_k[positive] = weighted[positive] / weights[positive]

    lr_valid = (weights / (scale * scale)) >= min_valid_fraction
    lr_k[~lr_valid] = np.nan
    return lr_k, lr_valid


def normalize_kelvin_to_training_space(
    kelvin: np.ndarray,
    valid_mask: np.ndarray,
    stats: ChannelNormStats,
) -> np.ndarray:
    """Map Kelvin back to scene z-score space (same as preprocessing export)."""
    out = np.full(kelvin.shape, np.nan, dtype=np.float32)
    valid = valid_mask & np.isfinite(kelvin)
    out[valid] = ((kelvin[valid] - stats.mean) / stats.std).astype(np.float32)
    return out


def degrade_thermal_patch(
    hr_normalized: np.ndarray,
    hr_valid_mask: np.ndarray,
    thermal_norm: ChannelNormStats,
    config: DegradationConfig,
    *,
    patch_key: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic LR from HR normalized thermal patch.

    Degradation sequence (Kelvin space):
        HR (norm) → Kelvin → masked Gaussian blur → area downsample → optional noise → LR (norm)

    HR normalized array is returned unchanged as the target.
    """
    if hr_normalized.ndim != 2:
        raise ValueError(f"HR thermal must be H×W, got {hr_normalized.shape}")

    hr_h, hr_w = hr_normalized.shape
    if hr_h % config.scale != 0 or hr_w % config.scale != 0:
        raise ValueError(
            f"HR shape {hr_normalized.shape} must be divisible by scale={config.scale}"
        )

    hr_norm = hr_normalized.astype(np.float32, copy=True)
    hr_mask = hr_valid_mask.astype(bool, copy=True)

    hr_k = hr_norm.astype(np.float64) * thermal_norm.std + thermal_norm.mean
    hr_k[~hr_mask] = np.nan

    blurred_k = masked_gaussian_blur_kelvin(hr_k, hr_mask, config.blur_sigma_hr_pixels)
    lr_k, lr_mask = downsample_kelvin_and_mask(blurred_k, hr_mask, config.scale)

    if config.noise_std_kelvin > 0:
        rng = _patch_rng(patch_key, config.base_seed)
        noise = rng.normal(0.0, config.noise_std_kelvin, size=lr_k.shape)
        apply = lr_mask & np.isfinite(lr_k)
        lr_k[apply] = lr_k[apply] + noise[apply]

    lr_norm = normalize_kelvin_to_training_space(lr_k, lr_mask, thermal_norm)

    return hr_norm, lr_norm.astype(np.float32), hr_mask, lr_mask
