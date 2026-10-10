"""CPU validation: RGB-guided thermal colorization on Kerala patches (visualization only)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from colorization.audit_inputs import audit_scene_inputs
from colorization.pipeline import ColorizationInputs, apply_all_methods
from colorization.thermal_predict import predict_refined_thermal_kelvin
from colorization.types import (
    COLORIZATION_VERSION,
    ColormapConfig,
    GuidedColorizationConfig,
    OverlayConfig,
)
from colorization.visualize import save_colormap_with_legend, save_comparison_panel
from dataset_loader.thermal_dataset import ThermalSatelliteDataset
from evaluation.evaluate_multiscene import select_patch_indices
from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import DegradationConfig
from training.repro import set_seed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "dataset" / "processed_qa"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "colorization" / "guided_colorization_v1"
KERALA = "LC08_L2SP_144053_20250225_02_T1"
FIXED_KELVIN = ColormapConfig(vmin_kelvin=295.0, vmax_kelvin=320.0, cmap_name="inferno")
GUIDED = GuidedColorizationConfig(colormap=FIXED_KELVIN, radius=8, eps=0.02, edge_mix=0.35)
OVERLAY = OverlayConfig(colormap=FIXED_KELVIN, thermal_weight=0.55)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def pick_patch_indices(root: Path, scene_id: str, n: int, seed: int) -> list[int]:
    sr = ThermalSRDataset(root, degradation=DegradationConfig(base_seed=42), scene_ids=[scene_id])
    coords = select_patch_indices(sr, "non_overlapping")
    rng = random.Random(seed)
    return sorted(rng.sample(coords, min(n, len(coords))))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--num-patches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refinement-beta", type=float, default=1.0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        logger.error("Output %s exists and is not empty", out)
        return 1
    out.mkdir(parents=True)

    set_seed(args.seed)
    root = args.processed_root.resolve()
    degradation = DegradationConfig(scale=2, base_seed=args.seed)

    from evaluation.evaluate_physics_refinement import build_methods

    class A:
        scale = 2
        variant = "lightweight_x2"
        pretrained_checkpoint = "auto"

    methods, model_info = build_methods(A())
    model = methods[1][2]

    sr_ds = ThermalSRDataset(root, degradation=degradation, scene_ids=[KERALA])
    scene_ds = ThermalSatelliteDataset(root, scene_ids=[KERALA])
    key_to_scene_idx = {ref.patch_key: i for i, ref in enumerate(scene_ds.patch_refs)}

    picked_sr_idx = pick_patch_indices(root, KERALA, args.num_patches, args.seed)
    manifest_patches = []
    audit_scene = None
    t0 = time.perf_counter()

    for sr_idx in picked_sr_idx:
        ref = sr_ds.base_dataset.patch_refs[sr_idx]
        patch_id = ref.patch_key
        scene_idx = key_to_scene_idx[patch_id]
        pred = predict_refined_thermal_kelvin(
            root,
            KERALA,
            scene_idx,
            model,
            degradation=degradation,
            beta=args.refinement_beta,
            seed=args.seed,
        )
        audit = audit_scene_inputs(pred["metadata"], pred["thermal_kelvin"], pred["rgb_reflectance"])
        audit_scene = audit
        inputs = ColorizationInputs(
            thermal_kelvin=pred["thermal_kelvin"],
            rgb_reflectance=pred["rgb_reflectance"],
            valid_mask=pred["valid_mask"],
        )
        thermal_before = inputs.thermal_kelvin.copy()
        results = apply_all_methods(inputs, guided=GUIDED, overlay=OVERLAY)
        assert results["thermal_kelvin_unchanged"]
        assert np.array_equal(thermal_before, inputs.thermal_kelvin)

        patch_dir = out / "patches" / f"patch_{patch_id}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        for method, key in [
            ("standard_colormap", "A_standard_colormap"),
            ("rgb_guided", "B_rgb_guided"),
            ("rgb_overlay", "C_rgb_overlay"),
        ]:
            rgba = results[method]["rgba"]
            Image.fromarray(rgba).save(patch_dir / f"{key}.png")
            save_colormap_with_legend(
                rgba,
                patch_dir / f"{key}_legend.png",
                vmin_k=FIXED_KELVIN.vmin_kelvin,
                vmax_k=FIXED_KELVIN.vmax_kelvin,
                cmap_name=FIXED_KELVIN.cmap_name,
                title=f"{key} — patch {patch_id}",
            )

        save_comparison_panel(
            patch_dir / "comparison_side_by_side.png",
            [
                ("A colormap", results["standard_colormap"]["rgba"]),
                ("B RGB-guided", results["rgb_guided"]["rgba"]),
                ("C overlay", results["rgb_overlay"]["rgba"]),
            ],
            vmin_k=FIXED_KELVIN.vmin_kelvin,
            vmax_k=FIXED_KELVIN.vmax_kelvin,
            cmap_name=FIXED_KELVIN.cmap_name,
        )
        np.save(patch_dir / "thermal_kelvin.npy", pred["thermal_kelvin"])

        manifest_patches.append(
            {
                "patch_id": patch_id,
                "sr_dataset_index": sr_idx,
                "valid_pixels": int(pred["valid_mask"].sum()),
                "thermal_kelvin_file": str((patch_dir / "thermal_kelvin.npy").relative_to(out)),
                "outputs": {
                    "A_standard_colormap": f"patches/patch_{patch_id}/A_standard_colormap.png",
                    "B_rgb_guided": f"patches/patch_{patch_id}/B_rgb_guided.png",
                    "C_rgb_overlay": f"patches/patch_{patch_id}/C_rgb_overlay.png",
                },
            }
        )

    elapsed = time.perf_counter() - t0
    manifest = {
        "version": COLORIZATION_VERSION,
        "scene_id": KERALA,
        "git_commit": _git_head(),
        "command": "py -m scripts.run_guided_colorization_v1",
        "seed": args.seed,
        "num_patches": len(manifest_patches),
        "patches": manifest_patches,
        "thermal_source": "pretrained SwinIR-S + synthetic forward consistency refinement (beta=1.0)",
        "thermal_resolution_note": audit_scene["thermal_resolution_note"] if audit_scene else None,
        "alignment": audit_scene,
        "colormap": {**asdict(FIXED_KELVIN), "invalid_color": list(FIXED_KELVIN.invalid_color)},
        "guided_params": {k: v for k, v in asdict(GUIDED).items() if k != "colormap"},
        "overlay_params": {k: v for k, v in asdict(OVERLAY).items() if k != "colormap"},
        "degradation": degradation.to_dict(),
        "refinement_beta": args.refinement_beta,
        "model": model_info,
        "runtime_s": elapsed,
        "limitations": [
            "Visualization only; colorization does not change thermal Kelvin arrays used for SR.",
            "RGB (surface reflectance) and thermal radiance are different physical quantities.",
            "RGB edges do not necessarily coincide with thermal boundaries.",
            "RGB is 30 m reflectance on the same patch grid as the SR display; not 15 m thermal truth.",
            "Fixed Kelvin color limits (295–320 K) used for cross-patch comparability.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "RGB-GUIDED THERMAL COLORIZATION — validation prototype",
        "=" * 60,
        "Synthetic ×2 thermal SR display experiment; not real 15 m thermal accuracy.",
        "",
        f"Scene: {KERALA}",
        f"Patches: {len(manifest_patches)} (seed={args.seed}, non-overlapping SR indices)",
        f"Runtime: {elapsed:.1f} s CPU",
        "",
        "Alignment / inputs:",
        json.dumps(audit_scene, indent=2) if audit_scene else "",
        "",
        "Methods:",
        "  A — standard inferno colormap with fixed 295–320 K limits",
        "  B — RGB luminance guided filter on colormap RGB (conservative edge_mix=0.35)",
        "  C — overlay baseline (55% thermal colormap + 45% RGB context)",
        "",
        "Invalid pixels: transparent (alpha=0) with neutral RGB placeholder.",
        "",
        "Limitations:",
    ]
    report += [f"  - {x}" for x in manifest["limitations"]]
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote %s in %.1f s", out, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
