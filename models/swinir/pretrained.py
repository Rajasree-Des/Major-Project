"""Download and adapt official SwinIR checkpoints (RGB, DIV2K) to the 1-channel thermal model.

Every checkpoint tensor is accounted for explicitly; the final ``load_state_dict`` is strict.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from models.swinir.model import ThermalSwinIR

logger = logging.getLogger(__name__)

OFFICIAL_RELEASE_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"
OFFICIAL_CHECKPOINTS: dict[str, str] = {
    "lightweight_x2": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
}
VARIANT_CHOICES = tuple(OFFICIAL_CHECKPOINTS)
DEFAULT_PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained" / "swinir"

REGENERATED_BUFFER_SUFFIXES = ("attn_mask", "relative_position_index")


@dataclass
class AdaptationReport:
    """Per-key accounting of how a pretrained checkpoint was mapped onto the thermal model."""

    checkpoint_path: str
    source_url: str | None
    sha256: str
    state_key: str
    matched: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    adapted: list[tuple[str, tuple[int, ...], tuple[int, ...], str]] = field(default_factory=list)
    newly_initialized: list[tuple[str, tuple[int, ...], str]] = field(default_factory=list)
    unexpected: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    regenerated_buffers: list[str] = field(default_factory=list)
    matched_numel: int = 0
    adapted_numel: int = 0
    new_numel: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "source_url": self.source_url,
            "sha256": self.sha256,
            "state_key": self.state_key,
            "matched_tensors": len(self.matched),
            "matched_parameters": self.matched_numel,
            "adapted_tensors": [k for k, *_ in self.adapted],
            "adapted_parameters": self.adapted_numel,
            "newly_initialized_tensors": [k for k, *_ in self.newly_initialized],
            "newly_initialized_parameters": self.new_numel,
            "unexpected_tensors": [k for k, _ in self.unexpected],
            "regenerated_buffers": len(self.regenerated_buffers),
        }

    def log(self) -> None:
        logger.info("Pretrained checkpoint: %s (key '%s', sha256 %s...)", self.checkpoint_path, self.state_key, self.sha256[:12])
        logger.info("Pretrained parameters matched:  %d tensors / %d values (copied unchanged)", len(self.matched), self.matched_numel)
        logger.info("Adapted parameters:             %d tensors / %d values", len(self.adapted), self.adapted_numel)
        for key, src, dst, how in self.adapted:
            logger.info("    %s  %s -> %s  [%s]", key, src, dst, how)
        logger.info("Newly initialized parameters:   %d tensors / %d values", len(self.newly_initialized), self.new_numel)
        for key, shape, reason in self.newly_initialized:
            logger.info("    %s  %s  [%s]", key, shape, reason)
        logger.info("Unexpected checkpoint tensors:  %d (ignored)", len(self.unexpected))
        for key, shape in self.unexpected:
            logger.info("    %s  %s", key, shape)
        logger.info(
            "Regenerated buffers:            %d (attn_mask / relative_position_index rebuilt from config)",
            len(self.regenerated_buffers),
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def official_checkpoint_url(variant: str) -> str:
    if variant not in OFFICIAL_CHECKPOINTS:
        raise ValueError(f"No official checkpoint registered for variant '{variant}'")
    return f"{OFFICIAL_RELEASE_URL}/{OFFICIAL_CHECKPOINTS[variant]}"


def download_official_checkpoint(variant: str = "lightweight_x2", dest_dir: Path | None = None) -> Path:
    """Download the official SwinIR checkpoint for ``variant`` (skipped if already present)."""
    dest_dir = Path(dest_dir) if dest_dir is not None else DEFAULT_PRETRAINED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / OFFICIAL_CHECKPOINTS[variant]
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached official checkpoint: %s", dest)
        return dest
    url = official_checkpoint_url(variant)
    logger.info("Downloading official SwinIR checkpoint: %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        torch.hub.download_url_to_file(url, str(tmp), progress=True)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Download it manually and pass --pretrained-checkpoint <path>."
        ) from exc
    tmp.replace(dest)
    return dest


def _extract_state_dict(raw: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(raw, dict):
        for key in ("params_ema", "params", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key], key
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw, "<root>"
    raise ValueError("Unrecognized checkpoint format: expected a dict with 'params' or 'params_ema'")


def _adapt_tensor(key: str, src: torch.Tensor, dst: torch.Tensor, upscale: int) -> tuple[torch.Tensor, str] | None:
    """Map an RGB pretrained tensor onto a 1-channel target, or return None if not adaptable."""
    if key == "conv_first.weight" and src.ndim == 4 and src.shape[1] == 3 and dst.shape[1] == 1 and src.shape[0] == dst.shape[0]:
        return src.sum(dim=1, keepdim=True), "sum over RGB input channels (equivalent to grey replicated as RGB)"

    r2 = upscale**2
    if key in ("upsample.0.weight", "upsample.0.bias") and src.shape[0] == 3 * r2 and dst.shape[0] == r2:
        grouped = src.reshape(3, r2, *src.shape[1:])
        return grouped.mean(dim=0), "mean over RGB output groups (PixelShuffle channel order preserved)"

    if key == "conv_last.weight" and src.ndim == 4 and src.shape[0] == 3 and dst.shape[0] == 1:
        return src.mean(dim=0, keepdim=True), "mean over RGB output channels"
    if key == "conv_last.bias" and src.shape[0] == 3 and dst.shape[0] == 1:
        return src.mean(dim=0, keepdim=True), "mean over RGB output channels"
    return None


def load_pretrained_swinir(model: ThermalSwinIR, checkpoint_path: Path, source_url: str | None = None) -> AdaptationReport:
    """Load an official SwinIR checkpoint into ``model.net`` with explicit adaptation."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pretrained, state_key = _extract_state_dict(raw)
    target = model.net.state_dict()
    upscale = model.config.upscale

    report = AdaptationReport(
        checkpoint_path=str(checkpoint_path),
        source_url=source_url,
        sha256=sha256_file(checkpoint_path),
        state_key=state_key,
    )
    assembled: dict[str, torch.Tensor] = {}

    for key, dst in target.items():
        if key.endswith(REGENERATED_BUFFER_SUFFIXES):
            assembled[key] = dst
            report.regenerated_buffers.append(key)
            continue
        src = pretrained.get(key)
        if src is None:
            assembled[key] = dst
            report.newly_initialized.append((key, tuple(dst.shape), "absent from checkpoint"))
            report.new_numel += dst.numel()
            continue
        if src.shape == dst.shape:
            assembled[key] = src.to(dst.dtype)
            report.matched.append((key, tuple(dst.shape)))
            report.matched_numel += dst.numel()
            continue
        adapted = _adapt_tensor(key, src, dst, upscale)
        if adapted is not None and adapted[0].shape == dst.shape:
            assembled[key] = adapted[0].to(dst.dtype)
            report.adapted.append((key, tuple(src.shape), tuple(dst.shape), adapted[1]))
            report.adapted_numel += dst.numel()
        else:
            assembled[key] = dst
            report.newly_initialized.append(
                (key, tuple(dst.shape), f"shape mismatch with checkpoint {tuple(src.shape)}; no adaptation rule")
            )
            report.new_numel += dst.numel()

    for key, src in pretrained.items():
        if key not in target:
            report.unexpected.append((key, tuple(src.shape)))

    model.net.load_state_dict(assembled, strict=True)
    return report
