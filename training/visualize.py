"""Validation figure export for thermal SR."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from training.metrics import bicubic_upsample


def save_validation_panel(
    lr: torch.Tensor,
    pred: torch.Tensor,
    hr: torch.Tensor,
    mask: torch.Tensor,
    output_path: Path,
    *,
    title: str = "Thermal SR validation",
) -> None:
    lr_np = lr.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()
    hr_np = hr.squeeze().detach().cpu().numpy()
    mask_np = mask.squeeze().detach().cpu().numpy().astype(bool)
    scale = hr.shape[-1] // lr.shape[-1]
    lr4 = lr.unsqueeze(0).unsqueeze(0) if lr.ndim == 2 else lr.unsqueeze(0)
    bic = bicubic_upsample(lr4, scale=scale).squeeze().detach().cpu().numpy()
    err = np.abs(pred_np - hr_np)
    err[~mask_np] = np.nan
    bic_err = np.abs(bic - hr_np)[mask_np]

    finite = np.concatenate([v[np.isfinite(v)] for v in (lr_np, bic, pred_np, hr_np) if np.isfinite(v).any()])
    vmin, vmax = np.percentile(finite, [2, 98]) if finite.size else (-1, 1)
    # Error range anchored to the bicubic error of the same sample, which is identical across epochs.
    bic_err = bic_err[np.isfinite(bic_err)]
    err_vmax = float(np.percentile(bic_err, 99)) if bic_err.size else 1.0

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for ax, (data, label) in zip(
        axes,
        [(lr_np, "LR"), (bic, "Bicubic"), (pred_np, "SwinIR"), (hr_np, "HR ref"), (err, "|SwinIR - HR|")],
    ):
        if label.startswith("|"):
            im = ax.imshow(data, cmap="magma", vmin=0.0, vmax=err_vmax)
        else:
            im = ax.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
