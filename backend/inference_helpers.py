"""Single-patch SR comparisons and metrics (reuse existing training/eval helpers)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from evaluation.metric_core import MethodSums, accumulate_method_on_patch
from pipeline.frozen_config import FrozenExperimentConfig
from pipeline.patch_refs import resolve_patch
from sr_data.consistency_refinement import ConsistencyRefinementConfig, refine_hr_normalized
from sr_data.dataset import ThermalSRDataset
from training.metrics import bicubic_upsample
from training.train_swinir import prepare_lr


def compute_sr_comparison(
    config: FrozenExperimentConfig,
    model: torch.nn.Module,
    scene_id: str,
    patch_id: str,
) -> dict[str, Any]:
    """Bicubic, SwinIR, refined vs HR reference for one patch (live metrics, not frozen eval)."""
    pref = resolve_patch(config.processed_root, scene_id, patch_id, degradation=config.degradation)
    sr_ds = ThermalSRDataset(
        config.processed_root,
        degradation=config.degradation,
        scene_ids=[scene_id],
        patch_indices=[pref.sr_dataset_index],
    )
    sample = sr_ds[0]
    meta = sr_ds.base_dataset.get_scene_metadata(scene_id)
    lr_raw = sample["lr"].unsqueeze(0)
    lr = prepare_lr(lr_raw)
    hr = sample["hr"]
    valid = sample["hr_valid_mask"] & torch.isfinite(hr)

    with torch.no_grad():
        bic = bicubic_upsample(lr, config.degradation.scale)[0, 0]
        swin = model(lr)[0, 0]
    conf = ConsistencyRefinementConfig(beta=config.refinement_beta, degradation=config.degradation)
    refined_z = refine_hr_normalized(
        swin.cpu().numpy(),
        sample["lr"][0].numpy(),
        sample["lr_valid_mask"][0].numpy().astype(bool),
        meta.thermal_norm,
        conf,
    )
    refined = torch.from_numpy(refined_z)

    stats = meta.thermal_norm
    metrics = {}
    for name, pred in [("bicubic", bic), ("swinir_pretrained", swin), ("swinir_consistency_refined", refined)]:
        v = valid[0]
        if not v.any():
            metrics[name] = None
            continue
        m = MethodSums()
        metrics[name] = accumulate_method_on_patch(m, pred, hr[0], valid[0], stats)

    return {
        "scene_id": scene_id,
        "patch_id": pref.patch_key,
        "valid_pixels": int(valid.sum()),
        "metrics_vs_hr_reference": metrics,
        "metrics_note": (
            "Per-patch metrics vs 30 m HR reference for this inference run only; "
            "not part of locked multiscene evaluation."
        ),
    }
