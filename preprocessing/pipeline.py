"""End-to-end preprocessing pipeline CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from preprocessing.constants import (
    DEFAULT_MAX_INVALID_PERCENT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_STRIDE,
    REQUIRED_BANDS,
)
from preprocessing.export_numpy import export_patches, write_metadata
from preprocessing.masking import (
    build_per_band_valid_masks,
    combine_valid_mask,
    summarize_mask,
)
from preprocessing.normalize import normalize_scene_arrays
from preprocessing.patchify import extract_patches
from preprocessing.radiometric import radiometric_convert_scene
from preprocessing.paths import ensure_directory
from preprocessing.validate import SceneValidationError, band_path, read_band_array, validate_scene

logger = logging.getLogger(__name__)
console = Console()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _physical_stats_thermal(thermal_k: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    vals = thermal_k[valid_mask & np.isfinite(thermal_k)]
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def _physical_stats_rgb(rgb: np.ndarray, valid_mask: np.ndarray) -> dict[str, list[float]]:
    stats: dict[str, list[float]] = {"min": [], "max": [], "mean": [], "std": []}
    for ch in range(3):
        vals = rgb[..., ch][valid_mask & np.isfinite(rgb[..., ch])]
        stats["min"].append(float(np.min(vals)))
        stats["max"].append(float(np.max(vals)))
        stats["mean"].append(float(np.mean(vals)))
        stats["std"].append(float(np.std(vals)))
    return stats


def run_preprocessing_pipeline(
    scene_dir: Path,
    output_dir: Path,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    max_invalid_percent: float = DEFAULT_MAX_INVALID_PERCENT,
) -> Path:
    """Run validation through NumPy export for one scene."""
    validation = validate_scene(scene_dir)

    dn_bands: dict[str, np.ndarray] = {}
    for name in REQUIRED_BANDS:
        path = band_path(scene_dir, name)
        dn_bands[name], _ = read_band_array(path)

    per_band_valid = build_per_band_valid_masks(dn_bands, validation.nodata_values)
    valid_mask = combine_valid_mask(per_band_valid)
    mask_report = summarize_mask(valid_mask, per_band_valid)

    thermal_k, rgb_refl, scaling = radiometric_convert_scene(dn_bands, per_band_valid)

    thermal_phys_stats = _physical_stats_thermal(thermal_k, valid_mask)
    rgb_phys_stats = _physical_stats_rgb(rgb_refl, valid_mask)

    thermal_norm, rgb_norm, norm_params = normalize_scene_arrays(
        thermal_k, rgb_refl, valid_mask
    )

    extraction = extract_patches(
        thermal_norm,
        rgb_norm,
        valid_mask,
        patch_size=patch_size,
        stride=stride,
        max_invalid_percent=max_invalid_percent,
    )

    if not extraction.patches:
        raise RuntimeError(
            "No patches exported: all windows exceeded invalid-pixel threshold or were empty."
        )

    scene_output = ensure_directory(output_dir / validation.scene_id)
    export_patches(scene_output, extraction)

    patch_coords = [
        {
            "patch_id": p.patch_id,
            "row": p.row,
            "col": p.col,
            "invalid_fraction": p.invalid_fraction,
        }
        for p in extraction.patches
    ]

    write_metadata(
        scene_output / "metadata.json",
        validation=validation,
        radiometric_scaling=scaling,
        normalization_metadata=norm_params.to_metadata(),
        mask_summary={
            "total_pixels": mask_report.total_pixels,
            "valid_pixels": mask_report.valid_pixels,
            "invalid_pixels": mask_report.invalid_pixels,
            "valid_fraction": mask_report.valid_fraction,
            "per_band_invalid_counts": mask_report.per_band_invalid_counts,
        },
        extraction=extraction,
        thermal_physical_stats=thermal_phys_stats,
        rgb_physical_stats=rgb_phys_stats,
        patch_coords=patch_coords,
    )

    return scene_output


def print_summary(scene_output: Path, metadata: dict) -> None:
    meta_path = scene_output / "metadata.json"
    ps = metadata["patch_size"]
    table = Table(title="Preprocessing Summary", show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output directory", str(scene_output))
    table.add_row("Patches generated", str(metadata["patches_generated"]))
    table.add_row("Patches skipped", str(metadata["patches_skipped"]))
    table.add_row("Thermal patch", f"float32 ({ps}, {ps})")
    table.add_row("RGB patch", f"float32 ({ps}, {ps}, 3)")
    table.add_row("Metadata", str(meta_path))
    console.print(Panel(table, border_style="green"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Landsat C2 L2 scene bands into aligned NumPy patches.",
    )
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument("--scene-id", help="Scene folder under --raw-dir.")
    scene_group.add_argument("--scene-dir", type=Path, help="Path to raw scene folder.")

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Raw dataset root (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed output root (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--max-invalid-percent",
        type=float,
        default=DEFAULT_MAX_INVALID_PERCENT,
        help="Skip patches with more than this percent invalid pixels.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    if args.scene_dir is not None:
        scene_dir = Path(args.scene_dir).resolve()
    else:
        scene_dir = (Path(args.raw_dir) / args.scene_id).resolve()

    if args.patch_size <= 0 or args.stride <= 0:
        logger.error("patch-size and stride must be positive.")
        return 1
    if not 0.0 <= args.max_invalid_percent <= 100.0:
        logger.error("max-invalid-percent must be between 0 and 100.")
        return 1

    logger.info("Processing scene: %s", scene_dir)
    try:
        processed_root = ensure_directory(Path(args.output_dir).resolve())
        scene_output = run_preprocessing_pipeline(
            scene_dir,
            processed_root,
            patch_size=args.patch_size,
            stride=args.stride,
            max_invalid_percent=args.max_invalid_percent,
        )
    except SceneValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1

    meta = json.loads((scene_output / "metadata.json").read_text(encoding="utf-8"))
    print_summary(scene_output, meta)
    logger.info("Preprocessing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
