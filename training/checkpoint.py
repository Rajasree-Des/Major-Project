"""Checkpoint save/load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

REQUIRED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "train_loss",
    "val_loss",
    "model_config",
    "training_config",
    "seed",
    "pretrained",
)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Checkpoint payload missing keys: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model (strict) and optionally optimizer state; returns the full payload."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a training checkpoint (no 'model_state_dict')")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def save_training_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
