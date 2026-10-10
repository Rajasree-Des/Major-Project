"""Inference-only CPU evaluation: bicubic vs pretrained thermal-adapted SwinIR-S x2 (vs optional checkpoint).

All methods see the same LR inputs, HR references and validity masks. MAE/RMSE (normalized and Kelvin)
are pooled over every valid HR pixel of the split; PSNR/SSIM use the project metric implementation per
patch and are averaged over patches. No training or weight updates happen here.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loader.split import (
    DEVELOPMENT_SPLIT_LABEL,
    SINGLE_SCENE_SPLIT_WARNING,
    development_patch_split,
    list_processed_scene_ids,
)
from evaluation.visualize import save_comparison_figure
from models.swinir.model import ThermalSwinIR, ThermalSwinIRConfig, config_for_variant
from models.swinir.pretrained import download_official_checkpoint, load_pretrained_swinir, official_checkpoint_url
from models.swinir.utils import count_parameters
from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import SCIENTIFIC_LIMITATIONS, DegradationConfig
from training.checkpoint import load_checkpoint
from training.metrics import bicubic_upsample, compute_psnr, compute_ssim, denormalize_kelvin
from training.repro import set_seed
from training.train_swinir import collate_sr, prepare_lr

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "sr_baseline"
DEFAULT_FINETUNED = PROJECT_ROOT / "outputs" / "models" / "swinir" / "best.pt"

EVALUATION_LABEL = "SINGLE-SCENE DEVELOPMENT EVALUATION — NOT FINAL SCIENTIFIC EVALUATION"
EXPERIMENT_DESCRIPTION = (
    "Synthetic x2 super-resolution evaluation: 30 m Landsat thermal reference (256x256) -> synthetic "
    "degradation -> 128x128 LR -> reconstruction to 256x256, scored against the 30 m reference. "
    "The HR target is NOT true 15 m thermal ground truth, and results from one scene do not establish "
    "generalization."
)


@dataclass
class MethodStats:
    """Pooled valid-pixel error sums plus per-patch PSNR/SSIM for one method."""

    key: str
    name: str
    description: str
    n_valid: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_abs_k: float = 0.0
    sum_sq_k: float = 0.0
    psnr: list[float] = field(default_factory=list)
    ssim: list[float] = field(default_factory=list)
    forward_s: float = 0.0
    batches: int = 0
    patches: int = 0
    nonfinite_outputs: int = 0

    def finalize(self, hr_min: float, hr_max: float) -> dict[str, Any]:
        mse = self.sum_sq / self.n_valid
        data_range = hr_max - hr_min
        return {
            "name": self.name,
            "description": self.description,
            "valid_pixels": self.n_valid,
            "mae": self.sum_abs / self.n_valid,
            "rmse": math.sqrt(mse),
            "psnr_mean_per_patch": float(np.mean(self.psnr)),
            "psnr_pooled_global_range": 10.0 * math.log10(data_range**2 / mse),
            "ssim_mean_per_patch": float(np.mean(self.ssim)),
            "mae_kelvin": self.sum_abs_k / self.n_valid,
            "rmse_kelvin": math.sqrt(self.sum_sq_k / self.n_valid),
            "patches_scored": len(self.psnr),
            "nonfinite_output_values": self.nonfinite_outputs,
            "timing": {
                "total_forward_s": self.forward_s,
                "avg_s_per_batch": self.forward_s / max(self.batches, 1),
                "avg_s_per_patch": self.forward_s / max(self.patches, 1),
            },
        }


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")


def select_indices(split_name: str, num_patches: int, seed: int) -> tuple[int, ...]:
    """Same development patch split as training (identical function and seed)."""
    if split_name == "all":
        return tuple(range(num_patches))
    split = development_patch_split(num_patches, seed=seed)
    return {"train": split.train_indices, "val": split.validation_indices, "test": split.test_indices}[split_name]


def build_pretrained_model(spec: str, variant: str) -> tuple[ThermalSwinIR, dict[str, Any]]:
    model = ThermalSwinIR(config_for_variant(variant))
    if spec == "auto":
        path, url = download_official_checkpoint(variant), official_checkpoint_url(variant)
    else:
        path, url = Path(spec), None
    report = load_pretrained_swinir(model, path, source_url=url)
    report.log()
    return model.eval(), report.summary()


def build_finetuned_model(path: Path, reference: ThermalSwinIRConfig) -> tuple[ThermalSwinIR, dict[str, Any]] | None:
    if not path.is_file():
        logger.info("No fine-tuned checkpoint at %s; method C skipped.", path)
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ThermalSwinIRConfig.from_dict(payload["model_config"])
    if cfg.to_dict() != reference.to_dict():
        raise ValueError(f"{path} model_config differs from the pretrained SwinIR configuration")
    model = ThermalSwinIR(cfg)
    load_checkpoint(path, model)
    tc = payload.get("training_config", {})
    info = {
        "path": str(path),
        "epoch": payload.get("epoch"),
        "train_loss": payload.get("train_loss"),
        "dev_val_loss_at_save": payload.get("val_loss"),
        "batch_size": tc.get("batch_size"),
        "max_train_batches": tc.get("max_train_batches"),
        "freeze_epochs": tc.get("freeze_epochs"),
        "resumed_from": tc.get("resume"),
        "note": "PRELIMINARY sanity-check checkpoint from a few CPU optimizer steps; not a converged fine-tune.",
    }
    return model.eval(), info


def pick_representative(rows: list[dict[str, Any]], primary: str, k: int) -> list[tuple[str, dict[str, Any]]]:
    """Largest/median/smallest gain of ``primary`` over bicubic, most textured, least valid."""
    scored = [r for r in rows if r["valid_fraction"] > 0]
    for r in scored:
        r["_gain"] = r["bicubic_mae"] - r[f"{primary}_mae"]
    by_gain = sorted(scored, key=lambda r: r["_gain"])
    candidates = [
        ("largest_gain_vs_bicubic", by_gain[-1]),
        ("median_gain_vs_bicubic", by_gain[len(by_gain) // 2]),
        ("smallest_gain_vs_bicubic", by_gain[0]),
        ("most_textured_hr", max(scored, key=lambda r: r["hr_std"])),
    ]
    partial = [r for r in scored if r["valid_fraction"] < 1.0]
    if partial:
        candidates.append(("partially_invalid_mask", min(partial, key=lambda r: r["valid_fraction"])))
    else:
        candidates.append(("least_textured_hr", min(scored, key=lambda r: r["hr_std"])))
    for extra in by_gain[::-1]:
        candidates.append(("additional_high_gain", extra))

    picked, seen = [], set()
    for reason, row in candidates:
        if row["position"] not in seen:
            picked.append((reason, row))
            seen.add(row["position"])
        if len(picked) == k:
            break
    for r in scored:
        r.pop("_gain", None)
    return picked


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cpu")

    scene_ids = list_processed_scene_ids(args.processed_root)
    if len(scene_ids) <= 1:
        logger.warning("WARNING: %s", EVALUATION_LABEL)
        logger.warning(SINGLE_SCENE_SPLIT_WARNING)
    logger.warning(DEVELOPMENT_SPLIT_LABEL)
    logger.info(EXPERIMENT_DESCRIPTION)

    degradation = DegradationConfig(scale=args.scale, base_seed=args.seed)
    root = args.processed_root.resolve()
    full_ds = ThermalSRDataset(root, degradation=degradation)
    indices = select_indices(args.split, len(full_ds), args.seed)
    ds = ThermalSRDataset(root, degradation=degradation, patch_indices=indices)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_sr)
    stats_by_scene = {sid: meta.thermal_norm for sid, meta in ds.base_dataset.metadata_by_scene.items()}
    logger.info("Split '%s': %d patches from scene(s) %s", args.split, len(ds), sorted(stats_by_scene))

    pretrained, pretrained_info = build_pretrained_model(args.pretrained_checkpoint, args.variant)
    methods: list[tuple[MethodStats, Callable[[torch.Tensor], torch.Tensor]]] = [
        (MethodStats("bicubic", "Bicubic x2", "torch bicubic interpolation (align_corners=False)"),
         lambda x: bicubic_upsample(x, args.scale)),
        (MethodStats("swinir_pretrained", "SwinIR-S x2 pretrained (thermal-adapted)",
                     "Official DIV2K SwinIR-S x2 weights, RGB->1-channel adapted, no fine-tuning"),
         pretrained),
    ]
    finetuned_info = None
    if args.finetuned_checkpoint.lower() != "none":
        loaded = build_finetuned_model(Path(args.finetuned_checkpoint), pretrained.config)
        if loaded is not None:
            ft_model, finetuned_info = loaded
            methods.append((MethodStats("swinir_finetuned_preliminary", "SwinIR-S x2 short CPU fine-tune (PRELIMINARY)",
                                        "Existing sanity-check checkpoint; a few optimizer steps only"), ft_model))

    sanity = {"lr_shape_ok": True, "hr_shape_ok": True, "output_shape_ok": True, "masked_but_nonfinite_hr": 0,
              "lr_invalid_pixels_filled_with_zero": 0, "hr_invalid_pixels_excluded": 0, "patches_without_valid_pixels": 0}
    rows: list[dict[str, Any]] = []
    hr_min, hr_max = math.inf, -math.inf
    patch_order: list[str] = []
    data_s = 0.0
    position = 0
    expected_lr = ds.metadata.lr_patch_size
    expected_hr = ds.metadata.hr_patch_size

    t_data = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        data_s += time.perf_counter() - t_data
        lr_raw, hr, hr_mask = batch["lr"], batch["hr"], batch["hr_valid_mask"]
        b = lr_raw.shape[0]
        sanity["lr_shape_ok"] &= tuple(lr_raw.shape[1:]) == (1, expected_lr, expected_lr)
        sanity["hr_shape_ok"] &= tuple(hr.shape[1:]) == (1, expected_hr, expected_hr)
        valid = hr_mask & torch.isfinite(hr)
        sanity["masked_but_nonfinite_hr"] += int((hr_mask & ~torch.isfinite(hr)).sum())
        sanity["hr_invalid_pixels_excluded"] += int((~valid).sum())
        sanity["lr_invalid_pixels_filled_with_zero"] += int((~torch.isfinite(lr_raw)).sum())
        lr = prepare_lr(lr_raw).to(device)
        hr_valid_vals = hr[valid]
        if hr_valid_vals.numel():
            hr_min = min(hr_min, float(hr_valid_vals.min()))
            hr_max = max(hr_max, float(hr_valid_vals.max()))

        batch_rows = []
        for i in range(b):
            v = valid[i]
            batch_rows.append({
                "position": position + i,
                "dataset_index": indices[position + i],
                "patch_id": batch["patch_id"][i],
                "scene_id": batch["scene_id"][i],
                "valid_fraction": float(v.float().mean()),
                "hr_std": float(hr[i][v].std()) if int(v.sum()) > 1 else 0.0,
            })
            if not v.any():
                sanity["patches_without_valid_pixels"] += 1
        patch_order.extend(batch["patch_id"])

        for stats, fn in methods:
            t0 = time.perf_counter()
            pred = fn(lr)
            stats.forward_s += time.perf_counter() - t0
            stats.batches += 1
            stats.patches += b
            sanity["output_shape_ok"] &= tuple(pred.shape) == tuple(hr.shape)
            stats.nonfinite_outputs += int((~torch.isfinite(pred)).sum())

            for i in range(b):
                v = valid[i]
                row = batch_rows[i]
                n = int(v.sum())
                if n == 0:
                    for metric in ("mae", "rmse", "psnr", "ssim", "mae_kelvin"):
                        row[f"{stats.key}_{metric}"] = float("nan")
                    continue
                diff = (pred[i][v] - hr[i][v]).double()
                st = stats_by_scene[row["scene_id"]]
                diff_k = (denormalize_kelvin(pred[i][v].double(), st) - denormalize_kelvin(hr[i][v].double(), st))
                stats.n_valid += n
                stats.sum_abs += float(diff.abs().sum())
                stats.sum_sq += float((diff**2).sum())
                stats.sum_abs_k += float(diff_k.abs().sum())
                stats.sum_sq_k += float((diff_k**2).sum())
                psnr = compute_psnr(pred[i : i + 1], hr[i : i + 1], valid[i : i + 1])
                # Zero-filled copies: compute_ssim weights by the mask, but NaN * 0 would still be NaN.
                ssim = compute_ssim(torch.nan_to_num(pred[i : i + 1]), torch.nan_to_num(hr[i : i + 1]), valid[i : i + 1])
                stats.psnr.append(psnr)
                stats.ssim.append(ssim)
                row[f"{stats.key}_mae"] = float(diff.abs().mean())
                row[f"{stats.key}_rmse"] = float(diff.pow(2).mean().sqrt())
                row[f"{stats.key}_psnr"] = psnr
                row[f"{stats.key}_ssim"] = ssim
                row[f"{stats.key}_mae_kelvin"] = float(diff_k.abs().mean())

        rows.extend(batch_rows)
        position += b
        if batch_idx % args.log_every == 0:
            logger.info("batch %d/%d  patches %d", batch_idx + 1, len(loader), position)
        t_data = time.perf_counter()

    results_by_method = {s.key: s.finalize(hr_min, hr_max) for s, _ in methods}
    valid_counts = {k: r["valid_pixels"] for k, r in results_by_method.items()}
    metric_keys = ("mae", "rmse", "psnr_mean_per_patch", "psnr_pooled_global_range", "ssim_mean_per_patch",
                   "mae_kelvin", "rmse_kelvin")
    sanity.update({
        "identical_samples_all_methods": len(set(valid_counts.values())) == 1
                                          and all(s.patches == position for s, _ in methods),
        "patch_id_sha256": hashlib.sha256("\n".join(patch_order).encode()).hexdigest(),
        "valid_pixel_counts_per_method": valid_counts,
        "all_outputs_finite": all(s.nonfinite_outputs == 0 for s, _ in methods),
        "all_final_metrics_finite": all(math.isfinite(r[m]) for r in results_by_method.values() for m in metric_keys),
        "no_gradients_computed": not torch.is_grad_enabled(),
    })
    sanity["passed"] = all([sanity["lr_shape_ok"], sanity["hr_shape_ok"], sanity["output_shape_ok"],
                            sanity["masked_but_nonfinite_hr"] == 0, sanity["identical_samples_all_methods"],
                            sanity["all_outputs_finite"], sanity["all_final_metrics_finite"]])

    out = args.output_dir.resolve()
    fig_dir = out / "figures"
    figures = []
    if args.num_figures > 0 and rows:
        primary = "swinir_pretrained"
        for rank, (reason, row) in enumerate(pick_representative(rows, primary, args.num_figures), start=1):
            sample = collate_sr([ds[row["position"]]])
            lr = prepare_lr(sample["lr"])
            preds = {s.name.split(" (")[0] if s.key != "bicubic" else "Bicubic": fn(lr) for s, fn in methods}
            path = fig_dir / f"{rank:02d}_{reason}_patch_{row['patch_id']}.png"
            gain = row["bicubic_mae"] - row[f"{primary}_mae"]
            save_comparison_figure(
                sample["lr"][0], preds, sample["hr"][0], sample["hr_valid_mask"][0] & torch.isfinite(sample["hr"][0]),
                path, title=f"{row['scene_id']} / patch {row['patch_id']} — {reason.replace('_', ' ')} "
                            f"(MAE gain vs bicubic {gain:+.4f})",
            )
            figures.append({"file": str(path.relative_to(out)), "reason": reason, "patch_id": row["patch_id"],
                            "valid_fraction": row["valid_fraction"], "mae_gain_vs_bicubic": gain})

    total_s = time.perf_counter() - t_start
    stats0 = next(iter(stats_by_scene.values()))
    results = {
        "label": EVALUATION_LABEL,
        "experiment": EXPERIMENT_DESCRIPTION,
        "scientific_limitations": SCIENTIFIC_LIMITATIONS,
        "split_label": DEVELOPMENT_SPLIT_LABEL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene_ids": sorted(stats_by_scene),
        "processed_root": str(root),
        "split": args.split,
        "num_patches": position,
        "num_valid_pixels": valid_counts.get("bicubic", 0),
        "num_hr_pixels_total": position * expected_hr * expected_hr,
        "scale": args.scale,
        "lr_size": expected_lr,
        "hr_size": expected_hr,
        "seed": args.seed,
        "degradation": degradation.to_dict(),
        "thermal_normalization": {"mean_kelvin": stats0.mean, "std_kelvin": stats0.std, "method": stats0.method},
        "model": {
            "variant": args.variant,
            "config": pretrained.config.to_dict(),
            "parameters": count_parameters(pretrained),
            "pretrained_checkpoint": pretrained_info,
            "finetuned_checkpoint": finetuned_info,
        },
        "aggregation": {
            "mae_rmse": "pooled over all valid HR pixels of the split (not averaged per batch)",
            "kelvin": "normalized values de-normalized with scene thermal mean/std, then pooled like MAE/RMSE",
            "psnr_mean_per_patch": "project compute_psnr per patch (data range = patch HR valid range), averaged",
            "psnr_pooled_global_range": "10*log10(R^2/MSE) with pooled MSE and R = HR valid range over the split",
            "ssim_mean_per_patch": "project compute_ssim per patch (mask-weighted, 11x11 Gaussian), averaged",
            "invalid_pixels": "excluded via hr_valid_mask & finite(HR); LR invalid pixels zero-filled for both methods",
        },
        "metrics": results_by_method,
        "sanity_checks": sanity,
        "timing": {
            "device": "cpu",
            "cpu": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "total_evaluation_s": total_s,
            "data_loading_and_degradation_s": data_s,
        },
        "figures": figures,
        "training_performed": False,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "per_patch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "report.txt").write_text(render_report(results), encoding="utf-8")
    logger.info("Wrote %s, %s, %s and %d figures", out / "results.json", out / "report.txt",
                out / "per_patch_metrics.csv", len(figures))
    return results


def render_report(r: dict[str, Any]) -> str:
    lines = [
        "THERMAL SUPER-RESOLUTION BASELINE EVALUATION",
        "=" * 60,
        f"*** {r['label']} ***",
        "",
        r["experiment"],
        "",
        f"Timestamp (UTC):   {r['timestamp_utc']}",
        f"Scene(s):          {', '.join(r['scene_ids'])}",
        f"Split:             {r['split']}  ({r['split_label']})",
        f"Patches:           {r['num_patches']}  (LR {r['lr_size']}x{r['lr_size']} -> HR {r['hr_size']}x{r['hr_size']}, x{r['scale']})",
        f"Valid HR pixels:   {r['num_valid_pixels']:,} of {r['num_hr_pixels_total']:,}",
        f"Seed:              {r['seed']}",
        f"Degradation:       blur sigma {r['degradation']['blur_sigma_hr_pixels']} HR px, "
        f"{r['degradation']['downsample_method']} downsample, noise {r['degradation']['noise_std_kelvin']} K",
        f"Model:             official SwinIR-S x2 ({r['model']['parameters']:,} params), "
        f"checkpoint {Path(r['model']['pretrained_checkpoint']['checkpoint_path']).name}",
        "",
        "METRICS (MAE/RMSE pooled over valid pixels; PSNR/SSIM mean over patches)",
        "-" * 60,
        f"{'Method':46s} {'MAE':>8s} {'RMSE':>8s} {'PSNR':>7s} {'SSIM':>7s} {'MAE_K':>7s} {'RMSE_K':>7s}",
    ]
    for m in r["metrics"].values():
        lines.append(f"{m['name']:46s} {m['mae']:8.5f} {m['rmse']:8.5f} {m['psnr_mean_per_patch']:7.2f} "
                     f"{m['ssim_mean_per_patch']:7.4f} {m['mae_kelvin']:7.4f} {m['rmse_kelvin']:7.4f}")
    lines += ["", "Pooled PSNR (global data range):"]
    for m in r["metrics"].values():
        lines.append(f"  {m['name']:44s} {m['psnr_pooled_global_range']:.2f} dB")
    bic = r["metrics"]["bicubic"]
    lines += ["", "Relative to bicubic:"]
    for key, m in r["metrics"].items():
        if key == "bicubic":
            continue
        lines.append(f"  {m['name']:44s} MAE {100 * (m['mae'] - bic['mae']) / bic['mae']:+.1f}%  "
                     f"RMSE_K {m['rmse_kelvin'] - bic['rmse_kelvin']:+.4f} K  "
                     f"PSNR {m['psnr_mean_per_patch'] - bic['psnr_mean_per_patch']:+.2f} dB  "
                     f"SSIM {m['ssim_mean_per_patch'] - bic['ssim_mean_per_patch']:+.4f}")
    lines += ["", "INFERENCE TIME (CPU)", "-" * 60,
              f"Total evaluation wall-clock: {r['timing']['total_evaluation_s']:.1f} s "
              f"(data loading + degradation {r['timing']['data_loading_and_degradation_s']:.1f} s)"]
    for m in r["metrics"].values():
        t = m["timing"]
        lines.append(f"  {m['name']:44s} {t['total_forward_s']:8.1f} s total, {t['avg_s_per_batch']:.3f} s/batch, "
                     f"{t['avg_s_per_patch']:.3f} s/patch")
    s = r["sanity_checks"]
    lines += ["", "SANITY CHECKS", "-" * 60,
              f"Overall: {'PASS' if s['passed'] else 'FAIL'}",
              f"  LR/HR/output shapes correct:        {s['lr_shape_ok'] and s['hr_shape_ok'] and s['output_shape_ok']}",
              f"  All outputs finite:                 {s['all_outputs_finite']}",
              f"  All final metrics finite:           {s['all_final_metrics_finite']}",
              f"  Identical samples for all methods:  {s['identical_samples_all_methods']}",
              f"  Masked-valid but non-finite HR px:  {s['masked_but_nonfinite_hr']}",
              f"  Invalid HR pixels excluded:         {s['hr_invalid_pixels_excluded']:,}",
              f"  Invalid LR pixels zero-filled:      {s['lr_invalid_pixels_filled_with_zero']:,}",
              f"  Patches with no valid pixels:       {s['patches_without_valid_pixels']}"]
    if r["model"]["finetuned_checkpoint"]:
        ft = r["model"]["finetuned_checkpoint"]
        lines += ["", "PRELIMINARY CHECKPOINT", "-" * 60, ft["note"],
                  f"  {ft['path']} (epoch {ft['epoch']}, batch size {ft['batch_size']}, "
                  f"max_train_batches {ft['max_train_batches']}, freeze_epochs {ft['freeze_epochs']})"]
    lines += ["", "FIGURES", "-" * 60]
    lines += [f"  {f['file']}  ({f['reason']}, valid {100 * f['valid_fraction']:.1f}%)" for f in r["figures"]]
    lines += ["", "LIMITATIONS", "-" * 60, r["scientific_limitations"],
              "Single scene, random patch split within the scene: overlapping neighbouring patches mean the "
              "validation patches are not independent of the training patches.",
              "No training was performed in this evaluation.", ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU inference-only evaluation: bicubic vs thermal-adapted SwinIR-S x2.")
    p.add_argument("--processed-root", "--data-dir", dest="processed_root", type=Path,
                   default=Path("dataset/processed"), help="Processed scene directory or dataset/processed root")
    p.add_argument("--split", default="val", choices=("val", "test", "train", "all"),
                   help="Development patch split (same function and seed as training)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--variant", default="lightweight_x2")
    p.add_argument("--pretrained-checkpoint", default="auto")
    p.add_argument("--finetuned-checkpoint", default=str(DEFAULT_FINETUNED),
                   help="Existing short fine-tune checkpoint reported as PRELIMINARY; 'none' to skip")
    p.add_argument("--num-figures", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="Debug only: limit batches")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        results = evaluate(args)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 1
    print("\n" + render_report(results))
    return 0 if results["sanity_checks"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
