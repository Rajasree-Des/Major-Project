"""Export processed patches and metadata to disk."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.patchify import PatchExtractionResult, PatchRecord
from preprocessing.radiometric import RadiometricScaling
from preprocessing.validate import SceneValidationReport

logger = logging.getLogger(__name__)


class ExportQualityError(Exception):
    """Raised when exported patches fail sanity checks."""


def _patch_filename(patch_id: int) -> str:
    return f"patch_{patch_id:06d}.npy"


def verify_patch_record(record: PatchRecord, patch_size: int) -> None:
    """Sanity-check one patch pair before export."""
    if record.thermal.dtype != np.float32:
        raise ExportQualityError(f"Patch {record.patch_id}: thermal dtype {record.thermal.dtype}")
    if record.rgb.dtype != np.float32:
        raise ExportQualityError(f"Patch {record.patch_id}: rgb dtype {record.rgb.dtype}")
    if record.thermal.shape != (patch_size, patch_size):
        raise ExportQualityError(
            f"Patch {record.patch_id}: thermal shape {record.thermal.shape}"
        )
    if record.rgb.shape != (patch_size, patch_size, 3):
        raise ExportQualityError(f"Patch {record.patch_id}: rgb shape {record.rgb.shape}")

    thermal_finite = record.thermal[np.isfinite(record.thermal)]
    rgb_finite = record.rgb[np.isfinite(record.rgb)]
    if thermal_finite.size == 0 or rgb_finite.size == 0:
        raise ExportQualityError(f"Patch {record.patch_id}: no finite values")

    if np.isinf(thermal_finite).any() or np.isinf(rgb_finite).any():
        raise ExportQualityError(f"Patch {record.patch_id}: Inf in exported patch values")


def verify_export_result(
    extraction: PatchExtractionResult,
    thermal_dir: Path,
    rgb_dir: Path,
) -> None:
    """Verify patch counts, files, and correspondence on disk."""
    n = len(extraction.patches)
    thermal_files = sorted(thermal_dir.glob("patch_*.npy"))
    rgb_files = sorted(rgb_dir.glob("patch_*.npy"))
    if len(thermal_files) != n or len(rgb_files) != n:
        raise ExportQualityError(
            f"File count mismatch: patches={n}, thermal_files={len(thermal_files)}, "
            f"rgb_files={len(rgb_files)}"
        )
    for record in extraction.patches:
        verify_patch_record(record, extraction.patch_size)
        t_path = thermal_dir / _patch_filename(record.patch_id)
        r_path = rgb_dir / _patch_filename(record.patch_id)
        if not t_path.is_file() or not r_path.is_file():
            raise ExportQualityError(f"Missing exported files for patch {record.patch_id}")


def export_patches(
    scene_output_dir: Path,
    extraction: PatchExtractionResult,
) -> tuple[Path, Path]:
    """Write thermal and RGB patches to ``thermal/`` and ``rgb/`` subdirectories."""
    thermal_dir = scene_output_dir / "thermal"
    rgb_dir = scene_output_dir / "rgb"
    thermal_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    for record in extraction.patches:
        verify_patch_record(record, extraction.patch_size)
        np.save(thermal_dir / _patch_filename(record.patch_id), record.thermal)
        np.save(rgb_dir / _patch_filename(record.patch_id), record.rgb)

    verify_export_result(extraction, thermal_dir, rgb_dir)
    logger.info("Exported %d patch pairs to %s", len(extraction.patches), scene_output_dir)
    return thermal_dir, rgb_dir


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write_metadata(
    path: Path,
    *,
    validation: SceneValidationReport,
    radiometric_scaling: dict[str, RadiometricScaling],
    normalization_metadata: dict[str, Any],
    mask_summary: dict[str, Any],
    extraction: PatchExtractionResult,
    thermal_physical_stats: dict[str, float],
    rgb_physical_stats: dict[str, list[float]],
    patch_coords: list[dict[str, int | float]],
) -> None:
    """Write ``metadata.json`` for a processed scene."""
    ref = validation.bands["ST_B10"]
    metadata = {
        "scene_id": validation.scene_id,
        "source_directory": str(validation.scene_dir),
        "source_bands": list(validation.bands.keys()),
        "crs": ref.crs,
        "original_dimensions": {"width": ref.width, "height": ref.height},
        "original_resolution_m": {
            "x": ref.resolution_x,
            "y": ref.resolution_y,
        },
        "bounds": {
            "left": ref.bounds[0],
            "bottom": ref.bounds[1],
            "right": ref.bounds[2],
            "top": ref.bounds[3],
        },
        "nodata_values": validation.nodata_values,
        "radiometric_scaling": {
            k: _json_safe(v) for k, v in radiometric_scaling.items()
        },
        "normalization": normalization_metadata,
        "mask_summary": mask_summary,
        "patch_size": extraction.patch_size,
        "stride": extraction.stride,
        "max_invalid_percent": extraction.max_invalid_percent,
        "patches_generated": len(extraction.patches),
        "patches_skipped": extraction.skipped_count,
        "patch_coordinates": patch_coords,
        "valid_pixel_statistics_physical": {
            "thermal_kelvin": thermal_physical_stats,
            "rgb_reflectance": rgb_physical_stats,
        },
        "thermal_data_representation": {
            "exported": "z_score_normalized_kelvin",
            "physical_units_before_normalization": "kelvin",
        },
        "rgb_data_representation": {
            "exported": "z_score_normalized_reflectance",
            "physical_units_before_normalization": "unitless_surface_reflectance",
        },
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Wrote metadata: %s", path)
