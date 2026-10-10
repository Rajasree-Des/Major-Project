"""Integral-image box filter (numpy only)."""

from __future__ import annotations

import numpy as np


def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter with square window radius ``r`` (edge-padded integral image)."""
    if radius < 1:
        return img.astype(np.float64, copy=True)
    r = radius
    pad = np.pad(img.astype(np.float64), ((r, r), (r, r)), mode="edge")
    s = pad.cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    a = s[2 * r : 2 * r + h, 2 * r : 2 * r + w]
    b = s[0:h, 2 * r : 2 * r + w]
    c = s[2 * r : 2 * r + h, 0:w]
    d = s[0:h, 0:w]
    return (a - b - c + d) / ((2 * r + 1) ** 2)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Single-channel guided filter (He et al.), deterministic."""
    I = guide.astype(np.float64)
    p = src.astype(np.float64)
    mean_I = box_filter(I, radius)
    mean_p = box_filter(p, radius)
    mean_Ip = box_filter(I * p, radius)
    mean_II = box_filter(I * I, radius)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    return mean_a * I + mean_b
