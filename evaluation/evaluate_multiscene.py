"""Per-scene CPU baseline evaluation on a scene-level split: bicubic x2 vs pretrained SwinIR-S x2.

Synthetic x2 reconstruction against the 30 m Landsat thermal reference; this is not evidence of
real 15 m thermal accuracy. Every method receives identical LR inputs and is scored on identical
valid HR pixels. Kelvin metrics use each patch's own scene normalization statistics. No training,
fine-tuning or weight updates happen here.

    py -m evaluation.evaluate_multiscene --roles validation test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loader.split import load_scene_split_manifest, list_processed_scene_ids, scene_ids_for_role
from evaluation.visualize import save_comparison_figure
from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import DegradationConfig
from evaluation.metric_core import (
    METRIC_NAMES,
    MethodSums,
    SceneSums,
    accumulate_method_on_patch,
    aggregate,
    finalize_scene,
)
from training.metrics import bicubic_upsample
from training.repro import set_seed
from training.train_swinir import collate_sr, prepare_lr

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "dataset" / "processed_qa"
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "splits" / "scene_split_qa_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "multiscene_qa_v1"
ROLE_DIRS = {"validation": "validation", "test": "test_locked"}
PROTECTED_OUTPUTS = (PROJECT_ROOT / "outputs" / "evaluation" / "sr_baseline", PROJECT_ROOT / "outputs" / "models")

EXPERIMENT_LABEL = ("Synthetic x2 reconstruction against the 30 m Landsat thermal reference; not evidence of "
                    "real 15 m thermal accuracy.")
TEST_LOCK_LABEL = "LOCKED - one-time baseline; do not use for model selection"
Method = tuple[str, str, Callable[[torch.Tensor], torch.Tensor]]


def select_patch_indices(dataset: ThermalSRDataset, selection: str) -> list[int]:
    """Indices into ``dataset``: all patches, or only the non-overlapping grid (row, col multiples of patch size)."""
    if selection == "all":
        return list(range(len(dataset)))
    if selection != "non_overlapping":
        raise ValueError(f"unknown patch selection {selection!r}")
    coords = {}
    for sid, meta in dataset.base_dataset.metadata_by_scene.items():
        size = meta.patch_size
        for c in meta.raw["patch_coordinates"]:
            coords[(sid, int(c["patch_id"]))] = (c["row"] % size == 0) and (c["col"] % size == 0)
    return [i for i, ref in enumerate(dataset.base_dataset.patch_refs) if coords[(ref.scene_id, ref.patch_id)]]


LrContextFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]], torch.Tensor]


def evaluate_role(
    root: Path,
    scene_ids: Sequence[str],
    methods: Sequence[Method],
    *,
    degradation: DegradationConfig,
    selection: str = "non_overlapping",
    batch_size: int = 4,
    max_patches_per_scene: int | None = None,
    log_every: int = 25,
    lr_context_method_keys: frozenset[str] = frozenset(),
    lr_context_fns: dict[str, LrContextFn] | None = None,
    patch_indices_override: list[int] | None = None,
) -> dict[str, Any]:
    """Score every method on the same patches of the given scenes; returns results plus per-patch rows."""
    full = ThermalSRDataset(root, degradation=degradation, scene_ids=list(scene_ids))
    found = set(full.base_dataset.metadata_by_scene)
    if found != set(scene_ids):
        raise ValueError(f"scenes requested {sorted(scene_ids)} but found {sorted(found)}")
    indices = list(patch_indices_override) if patch_indices_override is not None else select_patch_indices(full, selection)
    if max_patches_per_scene is not None:
        kept, per_scene = [], {}
        for i in indices:
            sid = full.base_dataset.patch_refs[i].scene_id
            if per_scene.get(sid, 0) < max_patches_per_scene:
                kept.append(i)
                per_scene[sid] = per_scene.get(sid, 0) + 1
        indices = kept
    ds = ThermalSRDataset(root, degradation=degradation, scene_ids=list(scene_ids), patch_indices=indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_sr)
    stats = {sid: m.thermal_norm for sid, m in ds.base_dataset.metadata_by_scene.items()}
    sums = {sid: SceneSums(sid, stats[sid].mean, stats[sid].std, {k: MethodSums() for k, _, _ in methods})
            for sid in scene_ids}
    expected_lr, expected_hr = ds.metadata.lr_patch_size, ds.metadata.hr_patch_size

    sanity = {"lr_shape_ok": True, "output_shape_ok": True, "masked_but_nonfinite_hr": 0,
              "identical_valid_pixels_all_methods": True}
    rows: list[dict[str, Any]] = []
    patch_keys: list[str] = []
    position = 0
    for b_idx, batch in enumerate(loader):
        lr_raw, hr, hr_mask = batch["lr"], batch["hr"], batch["hr_valid_mask"]
        sanity["lr_shape_ok"] &= tuple(lr_raw.shape[1:]) == (1, expected_lr, expected_lr)
        valid = hr_mask & torch.isfinite(hr)
        sanity["masked_but_nonfinite_hr"] += int((hr_mask & ~torch.isfinite(hr)).sum())
        lr = prepare_lr(lr_raw)
        preds = {}
        ctx_fns = lr_context_fns or {}
        for key, _, fn in methods:
            t0 = time.perf_counter()
            with torch.no_grad():
                if key in lr_context_method_keys:
                    preds[key] = ctx_fns[key](lr_raw, lr, batch["lr_valid_mask"], batch["scene_id"])
                else:
                    preds[key] = fn(lr)
            dt = time.perf_counter() - t0
            sanity["output_shape_ok"] &= tuple(preds[key].shape) == tuple(hr.shape)
            for sid in set(batch["scene_id"]):
                sums[sid].methods[key].forward_s += dt * batch["scene_id"].count(sid) / len(batch["scene_id"])

        for i in range(lr.shape[0]):
            sid, pid = batch["scene_id"][i], batch["patch_id"][i]
            s = sums[sid]
            v = valid[i]
            n = int(v.sum())
            s.patches_total += 1
            patch_keys.append(f"{sid}/{pid}")
            row: dict[str, Any] = {"position": position + i, "scene_id": sid, "patch_id": pid,
                                   "valid_pixels": n, "valid_fraction": n / v.numel()}
            if n == 0:
                s.patches_empty_valid += 1
                row["excluded"] = "zero_valid_pixels"
                rows.append(row)
                continue
            s.patches_scored += 1
            s.valid_pixels += n
            st = stats[sid]
            row["hr_std_kelvin"] = float(hr[i][v].double().std()) if n > 1 else 0.0
            counted = set()
            for key, _, _ in methods:
                p = preds[key][i]
                m = s.methods[key]
                before = m.n_valid
                patch_m = accumulate_method_on_patch(m, p, hr[i], v, st)
                counted.add(m.n_valid)
                if m.n_valid != before + n:
                    raise RuntimeError("valid pixel count drift between methods")
                row[f"{key}_mae_kelvin"] = patch_m["mae_kelvin"]
                row[f"{key}_rmse_kelvin"] = patch_m["rmse_kelvin"]
                row[f"{key}_mae_z"] = patch_m["mae_z"]
                row[f"{key}_psnr"] = patch_m["psnr"]
                row[f"{key}_ssim"] = patch_m["ssim"]
            sanity["identical_valid_pixels_all_methods"] &= len(counted) == 1
            rows.append(row)
        position += lr.shape[0]
        if log_every and b_idx % log_every == 0:
            logger.info("batch %d/%d  patches %d", b_idx + 1, len(loader), position)

    baseline_key = methods[0][0]
    scenes = [finalize_scene(sums[sid], baseline_key) for sid in scene_ids]
    sanity["all_predictions_finite_at_valid_pixels"] = all(
        m.nonfinite_at_valid == 0 for s in sums.values() for m in s.methods.values())
    sanity["passed"] = (sanity["lr_shape_ok"] and sanity["output_shape_ok"] and sanity["masked_but_nonfinite_hr"] == 0
                        and sanity["identical_valid_pixels_all_methods"]
                        and sanity["all_predictions_finite_at_valid_pixels"])
    return {
        "scene_ids": list(scene_ids),
        "patch_selection": selection,
        "max_patches_per_scene": max_patches_per_scene,
        "num_patches": position,
        "patch_key_sha256": hashlib.sha256("\n".join(patch_keys).encode()).hexdigest(),
        "per_scene": scenes,
        "aggregate": aggregate(scenes, [sums[s] for s in scene_ids], [k for k, _, _ in methods], baseline_key),
        "sanity_checks": sanity,
        "methods": {k: name for k, name, _ in methods},
        "baseline_method": baseline_key,
        "_rows": rows,
        "_dataset": ds,
    }


def build_methods(args: argparse.Namespace) -> tuple[list[Method], dict[str, Any]]:
    from evaluation.evaluate_sr import build_pretrained_model
    from models.swinir.utils import count_parameters

    model, info = build_pretrained_model(args.pretrained_checkpoint, args.variant)
    model.eval()
    methods: list[Method] = [
        ("bicubic", "Bicubic x2 (torch, align_corners=False)", lambda x: bicubic_upsample(x, args.scale)),
        ("swinir_pretrained", "SwinIR-S x2 pretrained, thermal-adapted, no fine-tuning", model),
    ]
    return methods, {"variant": args.variant, "config": model.config.to_dict(),
                     "parameters": count_parameters(model), "pretrained_checkpoint": info}


def render_report(r: dict[str, Any]) -> str:
    keys = list(r["methods"])
    lines = [f"MULTI-SCENE BASELINE EVALUATION - role: {r['role']}", "=" * 78, r["label"]]
    if r["role"] == "test":
        lines.append(TEST_LOCK_LABEL)
    lines += [f"split manifest: {r['split_manifest']}", f"scenes: {', '.join(r['scene_ids'])}",
              f"patch selection: {r['patch_selection']} ({r['num_patches']} patches)", "",
              "PER SCENE (Kelvin metrics use each scene's own mean/std)"]
    for s in r["per_scene"]:
        lines.append(f"{s['scene_id']}: patches scored {s['patches_scored']}/{s['patches_total']} "
                     f"(zero-valid excluded {s['patches_excluded_zero_valid']}), valid px {s['valid_pixels']:,}, "
                     f"thermal std {s['thermal_std_k']:.3f} K")
        if s["metrics"] is None:
            lines.append(f"  no metrics: {s['reason']}")
            continue
        for k in keys:
            m = s["metrics"][k]
            lines.append(f"  {k:18s} MAE {m['mae_kelvin']:.4f} K  RMSE {m['rmse_kelvin']:.4f} K  "
                         f"PSNR {m['psnr_mean_per_patch']:.3f} dB  SSIM {m['ssim_mean_per_patch']:.4f}  "
                         f"(z: MAE {m['mae_z']:.5f}, RMSE {m['rmse_z']:.5f})")
        for k, d in s.get(f"delta_vs_{r['baseline_method']}", {}).items():
            lines.append(f"  {k} - {r['baseline_method']}: MAE {d['mae_kelvin']:+.4f} K, RMSE {d['rmse_kelvin']:+.4f} K, "
                         f"PSNR {d['psnr_mean_per_patch']:+.3f} dB, SSIM {d['ssim_mean_per_patch']:+.4f}")
    a = r["aggregate"]
    lines += ["", "AGGREGATE (does not replace the per-scene table)"]
    for k in keys:
        m = a[k]
        extra = (f", scenes better than bicubic (RMSE K): {m['scenes_better_than_baseline_rmse_kelvin']}"
                 f"/{a['scenes_with_metrics']}" if "scenes_better_than_baseline_rmse_kelvin" in m else "")
        if m["rmse_kelvin"] is None:
            lines.append(f"  {k}: no scored scenes")
            continue
        lines.append(f"  {k:18s} macro RMSE {m['rmse_kelvin']['macro_mean_over_scenes']:.4f} K "
                     f"[{m['rmse_kelvin']['min']:.4f}-{m['rmse_kelvin']['max']:.4f}], pooled RMSE "
                     f"{m['pooled_rmse_kelvin']:.4f} K, pooled MAE {m['pooled_mae_kelvin']:.4f} K{extra}")
    lines += [f"  {a['note']}", "", f"sanity checks passed: {r['sanity_checks']['passed']}",
              f"training performed: {r['training_performed']}", "", "LIMITATIONS"]
    lines += [f"  - {x}" for x in r["limitations"]]
    return "\n".join(lines)


LIMITATIONS = [
    "Synthetic degradation (Gaussian blur sigma 1.2 HR px, area downsample, 0.05 K noise) against the 30 m "
    "reference; real 100 m TIRS acquisition physics and real 15 m truth are not represented.",
    "Few scenes: 1 validation and 2 test scenes; per-scene results are case studies, not population estimates.",
    "Degradation noise is keyed by seed and scene-local patch ID, so equal patch IDs in different scenes share a "
    "noise field (identical for every method).",
    "Per-scene z-score normalization: z-space metrics scale with 1/std and SSIM constants are fixed in z-units, "
    "so only Kelvin metrics and per-patch PSNR are comparable across scenes.",
    "Pretrained SwinIR weights are DIV2K RGB weights adapted to one channel; no thermal fine-tuning.",
    "Raster reading used the tifffile fallback; rasterio-vs-tifffile equivalence was tested only on the "
    "reference scene (see audit).",
]


def save_role_outputs(result: dict[str, Any], out_dir: Path, methods: Sequence[Method], figures_per_scene: int) -> None:
    rows, ds = result.pop("_rows"), result.pop("_dataset")
    out_dir.mkdir(parents=True, exist_ok=False)
    figures = []
    scored = [r for r in rows if "excluded" not in r]
    base, other = methods[0][0], methods[1][0] if len(methods) > 1 else methods[0][0]
    for sid in result["scene_ids"]:
        scene_rows = sorted((r for r in scored if r["scene_id"] == sid),
                            key=lambda r: r[f"{base}_rmse_kelvin"] - r[f"{other}_rmse_kelvin"])
        for rank, r in enumerate(scene_rows[len(scene_rows) // 2: len(scene_rows) // 2 + figures_per_scene]):
            sample = collate_sr([ds[r["position"]]])
            lr = prepare_lr(sample["lr"])
            with torch.no_grad():
                preds = {name.split(" (")[0].split(",")[0]: fn(lr) for _, name, fn in methods}
            path = out_dir / "figures" / f"{sid}_median_gain_{rank + 1}_patch_{r['patch_id']}.png"
            save_comparison_figure(sample["lr"][0], preds, sample["hr"][0],
                                   sample["hr_valid_mask"][0] & torch.isfinite(sample["hr"][0]), path,
                                   title=f"{sid} / patch {r['patch_id']} - median RMSE gain patch",
                                   subtitle=f"synthetic x2 SR vs 30 m reference ({result['role']} scene)")
            figures.append(str(path.relative_to(out_dir)))
    result["figures"] = figures
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "per_patch_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("position", "scene_id", "patch_id"), k))
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["scene_id", "method", "patches_scored", "patches_excluded_zero_valid", "valid_pixels",
                    "thermal_std_k", *METRIC_NAMES])
        for s in result["per_scene"]:
            for k, _, _ in methods:
                m = (s["metrics"] or {}).get(k) or {}
                w.writerow([s["scene_id"], k, s["patches_scored"], s["patches_excluded_zero_valid"],
                            s["valid_pixels"], s["thermal_std_k"], *[m.get(n) for n in METRIC_NAMES]])
    (out_dir / "report.txt").write_text(render_report(result), encoding="utf-8")


def write_readme(output_root: Path) -> None:
    parts = ["MULTI-SCENE QA_V1 BASELINE EVALUATION", "=" * 60, EXPERIMENT_LABEL, "",
             "split_manifest.json  frozen scene-level split (copy of dataset/splits/scene_split_qa_v1.json)",
             "audit/               read-only dataset audit", "validation/          baseline on validation scene(s)",
             f"test_locked/         {TEST_LOCK_LABEL}", ""]
    for role, sub in ROLE_DIRS.items():
        rp = output_root / sub / "report.txt"
        if rp.exists():
            parts += [f"--- {sub}/report.txt ---", rp.read_text(encoding="utf-8"), ""]
    (output_root / "README_results.txt").write_text("\n".join(parts), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--roles", nargs="+", choices=list(ROLE_DIRS), default=["validation"])
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--patch-selection", choices=("non_overlapping", "all"), default="non_overlapping")
    p.add_argument("--max-patches-per-scene", type=int, default=None, help="Debug only; recorded in results")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--variant", default="lightweight_x2")
    p.add_argument("--pretrained-checkpoint", default="auto")
    p.add_argument("--figures-per-scene", type=int, default=2)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    output_root = args.output_root.resolve()
    for protected in PROTECTED_OUTPUTS:
        if output_root == protected.resolve() or output_root.is_relative_to(protected.resolve()):
            logger.error("Refusing to write into protected output %s", protected)
            return 1
    for role in args.roles:
        d = output_root / ROLE_DIRS[role]
        if d.exists() and any(d.iterdir()):
            logger.error("Output %s already exists and is not empty; refusing to overwrite.", d)
            return 1

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    root = args.processed_root.resolve()
    split = load_scene_split_manifest(args.split_manifest, list_processed_scene_ids(root))
    degradation = DegradationConfig(scale=args.scale, base_seed=args.seed)
    methods, model_info = build_methods(args)
    logger.info(EXPERIMENT_LABEL)

    for role in args.roles:
        scene_ids = list(scene_ids_for_role(split, role))
        logger.info("Role %s: scenes %s", role, scene_ids)
        if role == "test":
            logger.warning(TEST_LOCK_LABEL)
        t0 = time.perf_counter()
        result = evaluate_role(root, scene_ids, methods, degradation=degradation, selection=args.patch_selection,
                               batch_size=args.batch_size, max_patches_per_scene=args.max_patches_per_scene,
                               log_every=args.log_every)
        result.update({
            "role": role,
            "label": EXPERIMENT_LABEL,
            "lock_label": TEST_LOCK_LABEL if role == "test" else None,
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
            "processed_root": str(root),
            "degradation": degradation.to_dict(),
            "seed": args.seed,
            "model": model_info,
            "metric_definitions": {
                "mae_z/rmse_z": "pooled over all valid HR pixels of the scene, in that scene's z-score units",
                "mae_kelvin/rmse_kelvin": "pred and HR de-normalized with the scene's own mean/std (z*std+mean), "
                                          "then pooled over valid pixels; exact inverse of the preprocessing z-score",
                "psnr_mean_per_patch": "training.metrics.compute_psnr per patch (data range = patch HR valid range; "
                                       "invariant to the linear normalization), averaged over scored patches",
                "ssim_mean_per_patch": "training.metrics.compute_ssim per patch (mask-weighted 11x11 Gaussian, "
                                       "constants in z-units), averaged over scored patches",
                "valid_pixels": "hr_valid_mask & finite(HR); identical for every method",
                "lr_input": "invalid LR pixels zero-filled (prepare_lr) for every method",
            },
            "limitations": LIMITATIONS,
            "training_performed": False,
            "timing": {"device": "cpu", "cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
                       "torch_version": torch.__version__, "role_total_s": time.perf_counter() - t0},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        save_role_outputs(result, output_root / ROLE_DIRS[role], methods, args.figures_per_scene)
        logger.info("Wrote %s (sanity passed: %s)", output_root / ROLE_DIRS[role], result["sanity_checks"]["passed"])
    write_readme(output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
