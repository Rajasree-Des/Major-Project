"""CLI sanity check for processed thermal/RGB patch datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from dataset_loader.metadata import discover_scene_directories, load_scene_metadata
from dataset_loader.thermal_dataset import (
    DatasetStructureError,
    build_combined_valid_mask,
    index_scene_patches,
    parse_patch_filename,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def analyze_scene(scene_dir: Path) -> dict:
    """Collect summary statistics for one processed scene."""
    meta = load_scene_metadata(scene_dir)
    refs = index_scene_patches(scene_dir, meta)

    thermal_dir = scene_dir / "thermal"
    rgb_dir = scene_dir / "rgb"
    thermal_names = {parse_patch_filename(p.name) for p in thermal_dir.glob("patch_*.npy")}
    rgb_names = {parse_patch_filename(p.name) for p in rgb_dir.glob("patch_*.npy")}
    thermal_names.discard(None)
    rgb_names.discard(None)

    missing_rgb = len(thermal_names - rgb_names)
    missing_thermal = len(rgb_names - thermal_names)

    nan_patches = 0
    inf_patches = 0
    valid_fractions: list[float] = []
    t_min, t_max = float("inf"), float("-inf")
    r_min, r_max = float("inf"), float("-inf")

    for ref in refs:
        thermal = np.load(ref.thermal_path)
        rgb = np.load(ref.rgb_path)
        if np.isnan(thermal).any() or np.isnan(rgb).any():
            nan_patches += 1
        if np.isinf(thermal).any() or np.isinf(rgb).any():
            inf_patches += 1

        mask = build_combined_valid_mask(thermal, rgb)
        valid_fractions.append(float(np.count_nonzero(mask) / mask.size))

        t_fin = thermal[np.isfinite(thermal)]
        r_fin = rgb[np.isfinite(rgb)]
        if t_fin.size:
            t_min = min(t_min, float(np.min(t_fin)))
            t_max = max(t_max, float(np.max(t_fin)))
        if r_fin.size:
            r_min = min(r_min, float(np.min(r_fin)))
            r_max = max(r_max, float(np.max(r_fin)))

    mean_valid_pct = 100.0 * float(np.mean(valid_fractions)) if valid_fractions else 0.0

    metadata_ok = (
        meta.patches_generated == len(refs)
        and meta.data_already_normalized
        and meta.thermal_representation == "z_score_normalized_kelvin"
    )

    structurally_valid = (
        missing_rgb == 0
        and missing_thermal == 0
        and len(refs) > 0
        and inf_patches == 0
        and metadata_ok
    )

    return {
        "scene_id": meta.scene_id,
        "scene_dir": str(scene_dir),
        "thermal_patch_count": len(thermal_names),
        "rgb_patch_count": len(rgb_names),
        "matched_pairs": len(refs),
        "missing_rgb_for_thermal": missing_rgb,
        "missing_thermal_for_rgb": missing_thermal,
        "thermal_shape": (meta.patch_size, meta.patch_size),
        "rgb_shape": (meta.patch_size, meta.patch_size, 3),
        "dtype": "float32",
        "patches_with_nan": nan_patches,
        "patches_with_inf": inf_patches,
        "finite_thermal_min": None if t_min == float("inf") else t_min,
        "finite_thermal_max": None if t_max == float("-inf") else t_max,
        "finite_rgb_min": None if r_min == float("inf") else r_min,
        "finite_rgb_max": None if r_max == float("-inf") else r_max,
        "mean_valid_pixel_percent": mean_valid_pct,
        "metadata_patches_generated": meta.patches_generated,
        "metadata_patches_skipped": meta.patches_skipped,
        "normalization_method": meta.normalization_method,
        "metadata_consistent": metadata_ok,
        "passes_validation": structurally_valid,
    }


def print_report(reports: list[dict]) -> None:
    for rep in reports:
        print(f"\n=== Scene: {rep['scene_id']} ===")
        print(f"  Matched pairs:        {rep['matched_pairs']}")
        print(f"  Thermal patches:      {rep['thermal_patch_count']}")
        print(f"  RGB patches:          {rep['rgb_patch_count']}")
        print(f"  Missing RGB IDs:      {rep['missing_rgb_for_thermal']}")
        print(f"  Missing thermal IDs:  {rep['missing_thermal_for_rgb']}")
        print(f"  Thermal shape:        {rep['thermal_shape']}")
        print(f"  RGB shape:            {rep['rgb_shape']}")
        print(f"  dtype:                {rep['dtype']}")
        print(f"  Patches with NaN:     {rep['patches_with_nan']}")
        print(f"  Patches with Inf:     {rep['patches_with_inf']}")
        print(f"  Finite thermal range: {rep['finite_thermal_min']} .. {rep['finite_thermal_max']}")
        print(f"  Finite RGB range:     {rep['finite_rgb_min']} .. {rep['finite_rgb_max']}")
        print(f"  Mean valid pixels:    {rep['mean_valid_pixel_percent']:.2f}%")
        print(f"  Normalization:        {rep['normalization_method']} (stored in .npy; no re-normalize)")
        print(f"  Metadata consistent:  {rep['metadata_consistent']}")
        print(f"  PASSES validation:    {rep['passes_validation']}")


def run_sanity_check(data_dir: Path) -> int:
    scenes = discover_scene_directories(data_dir)
    if not scenes:
        logger.error("No processed scenes found at %s", data_dir)
        return 1

    reports = [analyze_scene(s) for s in scenes]
    print_report(reports)

    all_pass = all(r["passes_validation"] for r in reports)
    return 0 if all_pass else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check processed patch dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Processed scene dir or dataset/processed root.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return run_sanity_check(args.data_dir.resolve())
    except DatasetStructureError as exc:
        logger.error("Dataset structure error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
