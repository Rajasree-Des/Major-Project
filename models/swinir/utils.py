"""Small model utilities."""

from __future__ import annotations

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Number of parameters with ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
