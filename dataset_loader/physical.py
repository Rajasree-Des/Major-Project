"""Explicit utilities to recover physical units from normalized patches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from dataset_loader.metadata import ChannelNormStats, ProcessedSceneMetadata

if TYPE_CHECKING:
    import torch


def _is_torch_tensor(value: Any) -> bool:
    cls = type(value)
    return cls.__module__ == "torch" and cls.__name__ == "Tensor"


def denormalize_thermal_kelvin(
    normalized: np.ndarray | torch.Tensor,
    stats: ChannelNormStats,
) -> np.ndarray | torch.Tensor:
    """Inverse z-score: ``physical_k = normalized * std + mean``."""
    if _is_torch_tensor(normalized):
        return normalized * stats.std + stats.mean
    return normalized.astype(np.float64) * stats.std + stats.mean


def denormalize_rgb_reflectance(
    normalized: np.ndarray | torch.Tensor,
    stats: tuple[ChannelNormStats, ChannelNormStats, ChannelNormStats],
) -> np.ndarray | torch.Tensor:
    """Inverse z-score per RGB channel."""
    if _is_torch_tensor(normalized):
        out = normalized.clone()
        for c, ch in enumerate(stats):
            out[..., c, :, :] = out[..., c, :, :] * ch.std + ch.mean
        return out

    out = normalized.astype(np.float64, copy=True)
    for c, ch in enumerate(stats):
        out[..., c] = out[..., c] * ch.std + ch.mean
    return out


def denormalize_from_scene_metadata(
    thermal_norm: np.ndarray,
    rgb_norm: np.ndarray,
    metadata: ProcessedSceneMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover Kelvin thermal and reflectance RGB from normalized arrays."""
    thermal_k = denormalize_thermal_kelvin(thermal_norm, metadata.thermal_norm)
    rgb_refl = denormalize_rgb_reflectance(rgb_norm, metadata.rgb_norm)
    return np.asarray(thermal_k), np.asarray(rgb_refl)
