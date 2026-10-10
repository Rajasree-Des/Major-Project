"""Optional precomputation of synthetic SR patch pairs to disk."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from dataset_loader.thermal_dataset import ThermalSatelliteDataset
from sr_data.degradation import degrade_thermal_patch
from sr_data.metadata import DegradationConfig, SRDatasetMetadata

logger = logging.getLogger(__name__)


def precompute_sr_patches(
    processed_root: Path,
    output_dir: Path,
    config: DegradationConfig,
    *,
    max_patches: int | None = None,
) -> int:
    """Write lr/hr/mask arrays for reproducibility or debugging."""
    output_dir = output_dir.resolve()
    for sub in ("lr", "hr", "lr_mask", "hr_mask"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    base = ThermalSatelliteDataset(processed_root, return_mask=True)
    meta = next(iter(base.metadata_by_scene.values()))
    sr_meta = SRDatasetMetadata(
        degradation=config,
        hr_patch_size=meta.patch_size,
        lr_patch_size=meta.patch_size // config.scale,
    )

    count = 0
    for i in range(len(base)):
        if max_patches is not None and count >= max_patches:
            break
        sample = base[i]
        patch_key = sample["patch_id"]
        scene_id = sample["scene_id"]
        scene_meta = base.get_scene_metadata(scene_id)

        hr, lr, hr_mask, lr_mask = degrade_thermal_patch(
            sample["thermal"].squeeze(0).numpy(),
            sample["valid_mask"].squeeze(0).numpy().astype(bool),
            scene_meta.thermal_norm,
            config,
            patch_key=patch_key,
        )

        np.save(output_dir / "hr" / f"patch_{patch_key}.npy", hr)
        np.save(output_dir / "lr" / f"patch_{patch_key}.npy", lr)
        np.save(output_dir / "hr_mask" / f"patch_{patch_key}.npy", hr_mask.astype(np.uint8))
        np.save(output_dir / "lr_mask" / f"patch_{patch_key}.npy", lr_mask.astype(np.uint8))
        count += 1

    with (output_dir / "sr_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(sr_meta.to_dict(), handle, indent=2)

    logger.info("Precomputed %d SR patch pairs to %s", count, output_dir)
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute synthetic SR thermal patches.")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=1.2)
    parser.add_argument("--noise-std-kelvin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patches", type=int, default=None)
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
    precompute_sr_patches(
        args.processed_root.resolve(),
        args.output_dir,
        config,
        max_patches=args.max_patches,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
