"""Integrated thermal SR + refinement + colorization (validation sample)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from pipeline.frozen_config import METHOD_SPEC, load_frozen_config
from pipeline.integrated import run_integrated_patch, run_validation_sample, verify_prediction_not_hr_reference
from pipeline.model_loader import load_swinir_model
from pipeline.patch_refs import resolve_patch
from training.repro import set_seed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "integrated_pipeline_v1"

AUDIT_NOTES = """
Phase 1 audit (code-backed):
- Thermal Kelvin for colorization comes from colorization.thermal_predict.predict_refined_thermal_kelvin:
  SwinIR(LR) -> refine_hr_normalized(swinir_z, lr, lr_mask, stats) -> denormalize; HR in ThermalSRDataset is
  used only inside degrade_thermal_patch to synthesize LR, not as the colorization input.
- RGB + valid_mask from ThermalSatelliteDataset (same patch_key, H×W validated in loader).
- Refinement beta loaded from outputs/evaluation/physics_refinement_v1/METHOD.txt (frozen 1.0).
- Degradation matches evaluation: DegradationConfig(scale=2, base_seed=42, blur 1.2, noise 0.05 K).
- Duplicate inference paths consolidated in pipeline.integrated (colorization script still usable separately).
- Masks: combined thermal∧RGB valid for colorization; LR mask used only in refinement step.
"""


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--scene-id", type=str, default=None, help="Single-patch mode")
    p.add_argument("--patch-id", type=str, default=None, help="Single-patch mode")
    p.add_argument("--num-patches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    out = args.output_dir.resolve()
    if out.exists():
        contents = [p for p in out.iterdir() if not p.name.startswith("_frozen")]
        if contents:
            logger.error("Refusing to overwrite non-empty %s", out)
            return 1
    else:
        out.mkdir(parents=True)

    config = load_frozen_config()
    set_seed(config.eval_seed)
    torch.set_grad_enabled(False)
    model, model_info = load_swinir_model(config)

    t0 = time.perf_counter()
    if args.scene_id and args.patch_id:
        patches = [
            run_integrated_patch(
                config, args.scene_id, args.patch_id, model, out, model_info=model_info
            )
        ]
    else:
        patches = run_validation_sample(
            config, model, out, num_patches=args.num_patches, seed=args.seed, model_info=model_info
        )

    hr_checks = []
    for entry in patches:
        pref = resolve_patch(
            config.processed_root,
            entry["scene_id"],
            entry["patch_id"],
            degradation=config.degradation,
        )
        import numpy as np

        kelvin = np.load(out / entry["thermal_kelvin_file"])
        hr_checks.append(
            {"patch_id": entry["patch_id"], **verify_prediction_not_hr_reference(config, pref, kelvin)}
        )

    manifest = {
        "pipeline_version": config.pipeline_version,
        "colorization_version": config.colorization_version,
        "git_commit": _git_head(),
        "command": "py -m scripts.run_integrated_pipeline_v1",
        "eval_seed": config.eval_seed,
        "refinement_beta": config.refinement_beta,
        "method_spec": str(METHOD_SPEC.resolve()),
        "degradation": config.degradation.to_dict(),
        "colormap": {**asdict(config.colormap), "invalid_color": list(config.colormap.invalid_color)},
        "guided_params": {k: v for k, v in asdict(config.guided).items() if k != "colormap"},
        "overlay_params": {k: v for k, v in asdict(config.overlay).items() if k != "colormap"},
        "model": model_info,
        "patches": patches,
        "hr_reference_separation_checks": hr_checks,
        "limitations": [
            "Synthetic ×2 SR vs 30 m reference; not real 15 m thermal accuracy.",
            "Consistency refinement is not a validated radiative-transfer model.",
            "RGB guidance is 30 m reflectance visualization only.",
            "Kelvin arrays are stored separately from all colorized PNGs.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "INTEGRATED PIPELINE v1 — validation report",
        "=" * 60,
        "Synthetic ×2 thermal SR + frozen consistency refinement + colorization.",
        "Not real 15 m thermal accuracy.",
        "",
        AUDIT_NOTES.strip(),
        "",
        f"Runtime total: {time.perf_counter() - t0:.1f} s",
        f"Git: {manifest['git_commit']}",
        f"Patches: {len(patches)}",
        f"Output: {out}",
        "",
        "HR separation checks (refined prediction ≠ HR reference Kelvin):",
        json.dumps(hr_checks, indent=2),
        "",
        "Limitations:",
    ]
    report += [f"  - {x}" for x in manifest["limitations"]]
    (out / "integration_report.txt").write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
