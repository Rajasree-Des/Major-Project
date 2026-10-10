"""Run one patch: synthetic LR → SwinIR → refinement → Kelvin + colorization."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from colorization.audit_inputs import audit_scene_inputs
from colorization.pipeline import ColorizationInputs, apply_all_methods
from colorization.thermal_predict import predict_refined_thermal_kelvin
from colorization.visualize import save_colormap_with_legend, save_comparison_panel
from dataset_loader.physical import denormalize_thermal_kelvin
from evaluation.evaluate_multiscene import select_patch_indices
from pipeline.frozen_config import FrozenExperimentConfig, VALIDATION_SCENE_ID
from pipeline.patch_refs import PatchRef, normalize_patch_key, resolve_patch
from sr_data.dataset import ThermalSRDataset
from training.repro import set_seed


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path)


def run_integrated_patch(
    config: FrozenExperimentConfig,
    scene_id: str,
    patch_id: str,
    model: torch.nn.Module,
    output_dir: Path,
    *,
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end inference + visualization for one patch."""
    config.require_beta(config.refinement_beta)
    root = config.processed_root
    if not root.is_dir():
        raise FileNotFoundError(f"processed root missing: {root}")

    pref = resolve_patch(root, scene_id, patch_id, degradation=config.degradation)
    patch_out = output_dir / "patches" / pref.scene_id / f"patch_{pref.patch_key}"
    patch_out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pred = predict_refined_thermal_kelvin(
        root,
        scene_id,
        pref.thermal_dataset_index,
        model,
        degradation=config.degradation,
        beta=config.refinement_beta,
        seed=config.eval_seed,
    )
    if normalize_patch_key(pred["patch_id"]) != pref.patch_key:
        raise RuntimeError("patch_id mismatch between resolver and prediction")

    audit = audit_scene_inputs(pred["metadata"], pred["thermal_kelvin"], pred["rgb_reflectance"])
    kelvin = np.asarray(pred["thermal_kelvin"], dtype=np.float64)
    kelvin_path = patch_out / "thermal_kelvin_refined.npy"
    np.save(kelvin_path, kelvin)
    np.save(patch_out / "valid_mask.npy", pred["valid_mask"])

    inputs = ColorizationInputs(kelvin, pred["rgb_reflectance"], pred["valid_mask"])
    kelvin_before = inputs.thermal_kelvin.copy()
    colors = apply_all_methods(inputs, guided=config.guided, overlay=config.overlay)
    if not colors["thermal_kelvin_unchanged"]:
        raise RuntimeError("colorization mutated Kelvin prediction")
    if not np.allclose(kelvin_before, inputs.thermal_kelvin, equal_nan=True, rtol=0, atol=0):
        raise RuntimeError("inputs thermal_kelvin mutated after colorization")

    method_files = {}
    for method, label in [
        ("standard_colormap", "A_standard_colormap"),
        ("rgb_guided", "B_rgb_guided"),
        ("rgb_overlay", "C_rgb_overlay"),
    ]:
        _save_rgba(patch_out / f"{label}.png", colors[method]["rgba"])
        save_colormap_with_legend(
            colors[method]["rgba"],
            patch_out / f"{label}_legend.png",
            vmin_k=config.colormap.vmin_kelvin,
            vmax_k=config.colormap.vmax_kelvin,
            cmap_name=config.colormap.cmap_name,
            title=f"{label} patch {pref.patch_key}",
        )
        method_files[label] = str((patch_out / f"{label}.png").relative_to(output_dir))

    save_comparison_panel(
        patch_out / "comparison_side_by_side.png",
        [
            ("A colormap", colors["standard_colormap"]["rgba"]),
            ("B RGB-guided", colors["rgb_guided"]["rgba"]),
            ("C overlay", colors["rgb_overlay"]["rgba"]),
        ],
        vmin_k=config.colormap.vmin_kelvin,
        vmax_k=config.colormap.vmax_kelvin,
        cmap_name=config.colormap.cmap_name,
    )

    elapsed = time.perf_counter() - t0
    meta = pred["metadata"]
    return {
        "scene_id": scene_id,
        "patch_id": pref.patch_key,
        "thermal_dataset_index": pref.thermal_dataset_index,
        "sr_dataset_index": pref.sr_dataset_index,
        "valid_pixels": int(pred["valid_mask"].sum()),
        "thermal_kelvin_file": str(kelvin_path.relative_to(output_dir)),
        "valid_mask_file": str((patch_out / "valid_mask.npy").relative_to(output_dir)),
        "colorization_outputs": method_files,
        "audit": audit,
        "runtime_s": elapsed,
        "thermal_source": "swinir_pretrained + synthetic_forward_consistency_refinement",
        "thermal_not_hr_reference": True,
        "normalization": {
            "thermal": asdict(meta.thermal_norm),
            "rgb": [asdict(ch) for ch in meta.rgb_norm],
        },
        "model": model_info,
    }


def run_validation_sample(
    config: FrozenExperimentConfig,
    model: torch.nn.Module,
    output_dir: Path,
    *,
    num_patches: int,
    seed: int,
    model_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic validation-scene sample (non-overlapping patches)."""
    import random

    scene_id = VALIDATION_SCENE_ID
    sr = ThermalSRDataset(config.processed_root, degradation=config.degradation, scene_ids=[scene_id])
    indices = select_patch_indices(sr, "non_overlapping")
    rng = random.Random(seed)
    picked = sorted(rng.sample(indices, min(num_patches, len(indices))))
    results = []
    for sr_idx in picked:
        ref = sr.base_dataset.patch_refs[sr_idx]
        results.append(
            run_integrated_patch(
                config,
                scene_id,
                ref.patch_key,
                model,
                output_dir,
                model_info=model_info,
            )
        )
    return results


def verify_prediction_not_hr_reference(
    config: FrozenExperimentConfig,
    pref: PatchRef,
    kelvin: np.ndarray,
) -> dict[str, Any]:
    """Check refined Kelvin differs from HR reference Kelvin on valid pixels (sanity, not metric)."""
    sr = ThermalSRDataset(
        config.processed_root,
        degradation=config.degradation,
        scene_ids=[pref.scene_id],
        patch_indices=[pref.sr_dataset_index],
    )
    sample = sr[0]
    hr_z = sample["hr"][0].numpy()
    valid = sample["hr_valid_mask"][0].numpy().astype(bool) & np.isfinite(hr_z)
    meta = sr.base_dataset.get_scene_metadata(pref.scene_id)
    hr_k = denormalize_thermal_kelvin(hr_z, meta.thermal_norm)
    diff = np.abs(kelvin[valid] - hr_k[valid])
    return {
        "mean_abs_diff_kelvin_vs_hr_reference": float(np.mean(diff)),
        "max_abs_diff_kelvin_vs_hr_reference": float(np.max(diff)),
        "note": "HR used only for this audit check; pipeline colorization uses refined prediction.",
    }
