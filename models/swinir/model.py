"""Single-channel thermal wrapper around the official SwinIR network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from models.swinir.network_swinir import SwinIR


@dataclass
class ThermalSwinIRConfig:
    """Architecture of official SwinIR-S (lightweight, x2), with 1 thermal channel.

    ``img_size`` is the training patch size of the pretrained checkpoint; it only sets
    the precomputed attention-mask size. Other input sizes are handled at runtime.
    """

    upscale: int = 2
    in_channels: int = 1
    out_channels: int = 1
    img_size: int = 64
    window_size: int = 8
    img_range: float = 1.0
    depths: tuple[int, ...] = (6, 6, 6, 6)
    embed_dim: int = 60
    num_heads: tuple[int, ...] = (6, 6, 6, 6)
    mlp_ratio: float = 2.0
    upsampler: str = "pixelshuffledirect"
    resi_connection: str = "1conv"
    input_scale: float = 0.125
    variant: str = "lightweight_x2"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["depths"] = list(self.depths)
        d["num_heads"] = list(self.num_heads)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThermalSwinIRConfig":
        data = dict(data)
        data["depths"] = tuple(data.get("depths", cls.depths))
        data["num_heads"] = tuple(data.get("num_heads", cls.num_heads))
        return cls(**data)


VARIANTS: dict[str, ThermalSwinIRConfig] = {
    "lightweight_x2": ThermalSwinIRConfig(),
}


def config_for_variant(variant: str) -> ThermalSwinIRConfig:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown SwinIR variant '{variant}'. Available: {sorted(VARIANTS)}")
    return ThermalSwinIRConfig.from_dict(VARIANTS[variant].to_dict())


class ThermalSwinIR(nn.Module):
    """Official SwinIR operating on one normalized (z-score) thermal channel.

    The z-score input is multiplied by ``input_scale`` so its dynamic range resembles the
    mean-subtracted [0, 1] images the pretrained network was trained on; the output is
    divided by the same factor, so predictions stay in the dataset's normalized units.
    """

    def __init__(self, config: ThermalSwinIRConfig | None = None) -> None:
        super().__init__()
        self.config = config or ThermalSwinIRConfig()
        c = self.config
        if c.in_channels != c.out_channels:
            raise ValueError("Official SwinIR uses in_chans for both input and output; they must match.")
        if c.input_scale <= 0:
            raise ValueError("input_scale must be positive")
        self.net = SwinIR(
            img_size=c.img_size,
            patch_size=1,
            in_chans=c.in_channels,
            embed_dim=c.embed_dim,
            depths=list(c.depths),
            num_heads=list(c.num_heads),
            window_size=c.window_size,
            mlp_ratio=c.mlp_ratio,
            upscale=c.upscale,
            img_range=c.img_range,
            upsampler=c.upsampler,
            resi_connection=c.resi_connection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected (B,{self.config.in_channels},H,W), got {tuple(x.shape)}")
        s = self.config.input_scale
        return self.net(x * s) / s


def build_thermal_swinir(variant: str = "lightweight_x2", **overrides: Any) -> ThermalSwinIR:
    cfg = config_for_variant(variant)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown ThermalSwinIRConfig field '{key}'")
        setattr(cfg, key, value)
    return ThermalSwinIR(cfg)
