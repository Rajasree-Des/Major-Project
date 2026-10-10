"""Sanity check for synthetic thermal SR data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sr_data.dataset import ThermalSRDataset
from sr_data.degradation import degrade_thermal_patch
from sr_data.metadata import DegradationConfig, HR_REFERENCE_DISCLAIMER

logger = logging.getLogger(__name__)


def collate_sr(batch: list[dict]) -> dict:
    return {
        "lr": torch.stack([b["lr"] for b in batch]),
        "hr": torch.stack([b["hr"] for b in batch]),
        "lr_valid_mask": torch.stack([b["lr_valid_mask"] for b in batch]),
        "hr_valid_mask": torch.stack([b["hr_valid_mask"] for b in batch]),
        "patch_id": [b["patch_id"] for b in batch],
    }


def run_checks(
    processed_root: Path,
    *,
    config: DegradationConfig,
    num_samples: int = 32,
    batch_size: int = 8,
) -> dict:
    dataset = ThermalSRDataset(processed_root, degradation=config)
    scale = config.scale
    hr_size = dataset.metadata.hr_patch_size
    lr_size = dataset.metadata.lr_patch_size

    if lr_size * scale != hr_size:
        raise ValueError("LR/HR size inconsistent with scale")

    # Determinism on first sample
    base = dataset.base_dataset[0]
    scene_meta = dataset.base_dataset.get_scene_metadata(base["scene_id"])
    hr_np = base["thermal"].squeeze(0).numpy()
    hr_mask = base["valid_mask"].squeeze(0).numpy().astype(bool)
    patch_key = base["patch_id"]

    _, lr_a, _, _ = degrade_thermal_patch(
        hr_np, hr_mask, scene_meta.thermal_norm, config, patch_key=patch_key
    )
    _, lr_b, _, _ = degrade_thermal_patch(
        hr_np, hr_mask, scene_meta.thermal_norm, config, patch_key=patch_key
    )
    deterministic = bool(np.allclose(lr_a, lr_b, equal_nan=True))

    hr_finite_min = hr_finite_max = lr_finite_min = lr_finite_max = None
    hr_finite_mean = lr_finite_mean = None
    hr_valid_pct = lr_valid_pct = []

    tested = min(num_samples, len(dataset))
    for i in range(tested):
        item = dataset[i]
        assert item["hr"].dtype == torch.float32 and item["lr"].dtype == torch.float32
        hr = item["hr"].numpy()
        lr = item["lr"].numpy()
        hr_m = item["hr_valid_mask"].numpy().astype(bool)
        lr_m = item["lr_valid_mask"].numpy().astype(bool)

        assert hr.shape == (1, hr_size, hr_size)
        assert lr.shape == (1, lr_size, lr_size)
        assert item["lr_valid_mask"].shape == (1, lr_size, lr_size)
        assert item["hr_valid_mask"].shape == (1, hr_size, hr_size)
        assert hr.dtype == np.float32 and lr.dtype == np.float32

        hr_fin = hr[0][hr_m[0] & np.isfinite(hr[0])]
        lr_fin = lr[0][lr_m[0] & np.isfinite(lr[0])]
        if hr_fin.size:
            hr_finite_min = hr_fin.min() if hr_finite_min is None else min(hr_finite_min, hr_fin.min())
            hr_finite_max = hr_fin.max() if hr_finite_max is None else max(hr_finite_max, hr_fin.max())
            hr_finite_mean = hr_fin.mean() if hr_finite_mean is None else (hr_finite_mean + hr_fin.mean()) / 2
        if lr_fin.size:
            lr_finite_min = lr_fin.min() if lr_finite_min is None else min(lr_finite_min, lr_fin.min())
            lr_finite_max = lr_fin.max() if lr_finite_max is None else max(lr_finite_max, lr_fin.max())
            lr_finite_mean = lr_fin.mean() if lr_finite_mean is None else (lr_finite_mean + lr_fin.mean()) / 2

        hr_valid_pct.append(100.0 * hr_m.mean())
        lr_valid_pct.append(100.0 * lr_m.mean())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sr)
    batch = next(iter(loader))
    loader_ok = (
        batch["lr"].shape[1:] == (1, lr_size, lr_size)
        and batch["hr"].shape[1:] == (1, hr_size, hr_size)
        and batch["lr_valid_mask"].shape[1:] == (1, lr_size, lr_size)
        and batch["hr_valid_mask"].shape[1:] == (1, hr_size, hr_size)
    )

    passes = deterministic and loader_ok and tested > 0

    return {
        "dataset_length": len(dataset),
        "samples_tested": tested,
        "scale": scale,
        "hr_shape": (1, hr_size, hr_size),
        "lr_shape": (1, lr_size, lr_size),
        "dtype": "float32",
        "deterministic_with_seed": deterministic,
        "dataloader_ok": loader_ok,
        "mean_hr_valid_percent": float(np.mean(hr_valid_pct)),
        "mean_lr_valid_percent": float(np.mean(lr_valid_pct)),
        "hr_finite_min": None if hr_finite_min is None else float(hr_finite_min),
        "hr_finite_max": None if hr_finite_max is None else float(hr_finite_max),
        "hr_finite_mean_approx": None if hr_finite_mean is None else float(hr_finite_mean),
        "lr_finite_min": None if lr_finite_min is None else float(lr_finite_min),
        "lr_finite_max": None if lr_finite_max is None else float(lr_finite_max),
        "lr_finite_mean_approx": None if lr_finite_mean is None else float(lr_finite_mean),
        "degradation_config": config.to_dict(),
        "hr_reference_disclaimer": HR_REFERENCE_DISCLAIMER,
        "second_normalization_applied": False,
        "passes_validation": passes,
    }


def print_report(report: dict) -> None:
    print("\n=== Thermal SR Data Sanity Check ===")
    print(f"  Dataset length:           {report['dataset_length']}")
    print(f"  Samples tested:           {report['samples_tested']}")
    print(f"  Scale:                    {report['scale']}")
    print(f"  HR shape:                 {report['hr_shape']}")
    print(f"  LR shape:                 {report['lr_shape']}")
    print(f"  dtype:                    {report['dtype']}")
    print(f"  Deterministic (seed):     {report['deterministic_with_seed']}")
    print(f"  DataLoader batch OK:      {report['dataloader_ok']}")
    print(f"  Mean HR valid %:          {report['mean_hr_valid_percent']:.2f}")
    print(f"  Mean LR valid %:          {report['mean_lr_valid_percent']:.2f}")
    print(f"  HR finite min/max:        {report['hr_finite_min']} .. {report['hr_finite_max']}")
    print(f"  LR finite min/max:        {report['lr_finite_min']} .. {report['lr_finite_max']}")
    print(f"  Second normalization:     {report['second_normalization_applied']}")
    print(f"  Degradation config:       {json.dumps(report['degradation_config'])}")
    print(f"  Disclaimer:               {report['hr_reference_disclaimer']}")
    print(f"  PASSES validation:        {report['passes_validation']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check synthetic thermal SR pipeline.")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=1.2)
    parser.add_argument("--noise-std-kelvin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--visualize-out",
        type=Path,
        default=None,
        help="Optional PNG path for HR/LR comparison of sample 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)
    config = DegradationConfig(
        scale=args.scale,
        blur_sigma_hr_pixels=args.blur_sigma,
        noise_std_kelvin=args.noise_std_kelvin,
        base_seed=args.seed,
    )
    report = run_checks(
        args.processed_root.resolve(),
        config=config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    print_report(report)

    if args.visualize_out is not None:
        from sr_data.visualize import save_degradation_comparison

        ds = ThermalSRDataset(args.processed_root.resolve(), degradation=config)
        save_degradation_comparison(ds, 0, args.visualize_out.resolve())
        print(f"  Saved visualization:      {args.visualize_out}")

    return 0 if report["passes_validation"] else 1


if __name__ == "__main__":
    sys.exit(main())
