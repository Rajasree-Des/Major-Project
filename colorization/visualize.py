"""Save colorized figures with fixed Kelvin legends."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def save_colormap_with_legend(
    rgba: np.ndarray,
    path: Path,
    *,
    vmin_k: float,
    vmax_k: float,
    cmap_name: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(rgba)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin_k, vmax=vmax_k), cmap=cmap_name)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_panel(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    *,
    vmin_k: float,
    vmax_k: float,
    cmap_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    mappable = None
    for ax, (title, rgba) in zip(axes, panels):
        im = ax.imshow(rgba)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        mappable = im
    if mappable is not None:
        sm = cm.ScalarMappable(norm=Normalize(vmin=vmin_k, vmax=vmax_k), cmap=cmap_name)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, label="Temperature (K)")
    fig.suptitle("Fixed Kelvin scale across methods (visualization only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
