"""Training losses for thermal SR."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 loss over valid HR pixels only (mask True = valid)."""
    valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred[valid], target[valid])


def build_loss(name: str):
    if name.lower() in {"l1", "mae"}:
        return masked_l1_loss
    raise ValueError(f"Unsupported loss: {name}")
