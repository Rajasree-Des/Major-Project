"""Load pretrained SwinIR for inference (no weight changes)."""

from __future__ import annotations

from typing import Any

import torch

from pipeline.frozen_config import FrozenExperimentConfig


def load_swinir_model(config: FrozenExperimentConfig) -> tuple[torch.nn.Module, dict[str, Any]]:
    from evaluation.evaluate_multiscene import build_methods

    class Args:
        scale = config.degradation.scale
        variant = config.swinir_variant
        pretrained_checkpoint = config.pretrained_checkpoint

    methods, info = build_methods(Args())
    model = methods[1][2]
    model.eval()
    return model, info
