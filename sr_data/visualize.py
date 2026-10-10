"""Debug visualization for HR/LR thermal degradation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import DegradationConfig


def save_degradation_comparison(
    dataset: ThermalSRDataset,
    index: int,
    output_path: Path,
) -> None:
    """Save HR, LR, and nearest-upsampled LR for visual inspection."""
    sample = dataset[index]
    hr = sample["hr"].squeeze(0).numpy()
    lr = sample["lr"].squeeze(0).numpy()
    hr_hw = tuple(sample["hr"].shape[-2:])

    lr_up = (
        F.interpolate(
            sample["lr"].unsqueeze(0),
            size=hr_hw,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        [hr, lr, lr_up],
        ["HR (256, target)", "LR (128, input)", "LR upsampled (bilinear)"],
    ):
        finite = np.isfinite(data)
        display = np.where(finite, data, np.nan)
        im = ax.imshow(display, cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Patch {sample['patch_id']} — synthetic thermal SR degradation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
