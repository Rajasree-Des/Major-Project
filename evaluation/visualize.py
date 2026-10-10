"""Multi-method comparison figures for thermal SR evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().detach().cpu().numpy().astype(np.float64)


def save_comparison_figure(
    lr: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    hr: torch.Tensor,
    mask: torch.Tensor,
    output_path: Path,
    *,
    title: str,
    units: str = "z-score",
    subtitle: str = "synthetic x2 SR vs 30 m reference (single-scene development evaluation)",
) -> None:
    """Top row: LR, each method, HR. Bottom row: valid mask and |method - HR| per method.

    All thermal panels share one colour range and all error panels share one error range.
    Invalid HR pixels are blank (NaN) in the error maps.
    """
    lr_np, hr_np = _np(lr), _np(hr)
    mask_np = _np(mask).astype(bool)
    pred_np = {name: _np(p) for name, p in predictions.items()}

    values = np.concatenate([v[np.isfinite(v)].ravel() for v in (lr_np, hr_np, *pred_np.values())])
    vmin, vmax = np.percentile(values, [2, 98]) if values.size else (-1.0, 1.0)

    errors = {}
    for name, p in pred_np.items():
        e = np.abs(p - hr_np)
        e[~mask_np] = np.nan
        errors[name] = e
    pooled = np.concatenate([e[np.isfinite(e)] for e in errors.values()])
    err_vmax = float(np.percentile(pooled, 99)) if pooled.size else 1.0

    n_cols = len(predictions) + 2
    fig, axes = plt.subplots(2, n_cols, figsize=(3.3 * n_cols, 6.6))
    top = [("LR (128x128)", lr_np), *[(n, p) for n, p in pred_np.items()], ("HR 30 m reference", hr_np)]
    for ax, (label, data) in zip(axes[0], top):
        im = ax.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    axes[1, 0].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"HR valid mask ({mask_np.mean() * 100:.1f}% valid)", fontsize=9)
    for ax, (name, e) in zip(axes[1, 1:], errors.items()):
        im = ax.imshow(e, cmap="magma", vmin=0.0, vmax=err_vmax)
        valid = e[np.isfinite(e)]
        ax.set_title(f"|{name} - HR|  MAE={valid.mean():.4f}" if valid.size else f"|{name} - HR|", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[1, -1].axis("off")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{title}\nvalues in {units}; {subtitle}", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
