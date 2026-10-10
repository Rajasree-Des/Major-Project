"""Minimal stand-ins for the three ``timm`` helpers used by the official SwinIR network.

Behaviour matches ``timm.layers`` (DropPath / to_2tuple / trunc_normal_).
"""

from __future__ import annotations

import collections.abc
from itertools import repeat

import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x * random_tensor / keep_prob


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
