"""QA_PIXEL-aware preprocessing (versioned ``qa_v1``), written to a separate root.

Same validation, radiometric conversion, normalization, patch extraction and
export as :mod:`preprocessing.pipeline`; the only difference is that the valid
mask additionally excludes QA_PIXEL fill, dilated cloud, cirrus, cloud, cloud
shadow and snow/ice pixels. The original ``dataset/processed`` root is never
written to, and existing scene outputs are never overwritten.

    py -m preprocessing.pipeline_qa --scene-ids LC09_L2SP_143050_20240327_02_T1 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from preprocessing.constants import (
    DEFAULT_MAX_INVALID_PERCENT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_STRIDE,
    PROJECT_ROOT,
    REQUIRED_BANDS,
    THERMAL_BAND,
)
from preprocessing.export_numpy import export_patches, write_metadata
from preprocessing.masking import build_per_band_valid_masks, combine_valid_mask, summarize_mask
from preprocessing.normalize import normalize_scene_arrays
from preprocessing.patchify import extract_patches
from preprocessing.pipeline import _physical_stats_rgb, _physical_stats_thermal
from preprocessing.qa_mask import (
    EXCLUDED_FLAGS,
    QA_RULES_VERSION,
    check_alignment,
    decode_flags,
    exclusion_codes,
    find_qa_path,
    qa_exclusion_mask,
    read_qa,
    rules_metadata,
    summarize_flags,
    verify_bitfields_against_stac,
)
from preprocessing.radiometric import radiometric_convert_scene
from preprocessing.raster_io import BACKEND as RASTER_BACKEND
from preprocessing.validate import band_path, read_band_array, validate_scene

logger = logging.getLogger(__name__)

DEFAULT_QA_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed_qa"
DEFAULT_SUPPLEMENT_DIR = PROJECT_ROOT / "dataset" / "raw_supplement"
MANIFEST_NAME = "processing_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_accepted_windows(valid_mask: np.ndarray, patch_size: int, stride: int,
                           max_invalid_percent: float) -> tuple[int, int]:
    """Accepted/rejected window counts under the extract_patches threshold rule (no arrays built)."""
    height, width = valid_mask.shape
    integral = np.pad(valid_mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    accepted = rejected = 0
    limit = max_invalid_percent / 100.0
    area = patch_size * patch_size
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            r1, c1 = row + patch_size, col + patch_size
            valid = integral[r1, c1] - integral[row, c1] - integral[r1, col] + integral[row, col]
            if valid == 0 or 1.0 - valid / area > limit:
                rejected += 1
            else:
                accepted += 1
    return accepted, rejected


def stac_bitfield_check(scene_id: str) -> dict[str, Any]:
    """Verify bit positions against the STAC item; never fatal if the network is unavailable."""
    try:
        from scripts.download_dataset import build_stac_client, fetch_scenes_by_id

        item = fetch_scenes_by_id(build_stac_client(), [scene_id])[0]
        bitfields = item.assets["qa_pixel"].extra_fields["classification:bitfields"]
    except Exception as exc:  # network or catalog problem
        logger.warning("Could not fetch STAC bitfields for %s: %s", scene_id, exc)
        return {"verified": False, "reason": f"STAC lookup failed: {exc}"}
    return verify_bitfields_against_stac(bitfields)


def run_qa_pipeline(
    scene_dir: Path,
    output_root: Path,
    *,
    supplement_dirs: Sequence[Path] = (DEFAULT_SUPPLEMENT_DIR,),
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    max_invalid_percent: float = DEFAULT_MAX_INVALID_PERCENT,
    verify_stac: bool = True,
) -> dict[str, Any]:
    """Process one scene with QA masking; return the per-scene summary."""
    validation = validate_scene(scene_dir)
    scene_id = validation.scene_id
    scene_output = output_root / scene_id
    if scene_output.exists() and any(scene_output.iterdir()):
        raise FileExistsError(f"Output already exists, refusing to overwrite: {scene_output}")

    qa_path = find_qa_path(scene_dir, supplement_dirs)
    qa, qa_info = read_qa(qa_path)
    alignment = check_alignment(qa_info, validation.bands[THERMAL_BAND])
    for name in REQUIRED_BANDS:
        check_alignment(qa_info, validation.bands[name])
    alignment["also_identical_to"] = list(REQUIRED_BANDS)
    if qa.shape != (validation.bands[THERMAL_BAND].height, validation.bands[THERMAL_BAND].width):
        raise ValueError(f"QA array shape {qa.shape} differs from thermal grid")

    dn_bands = {name: read_band_array(band_path(scene_dir, name))[0] for name in REQUIRED_BANDS}
    per_band_valid = build_per_band_valid_masks(dn_bands, validation.nodata_values)
    base_valid = combine_valid_mask(per_band_valid)

    flags = decode_flags(qa)
    valid_mask = base_valid & ~qa_exclusion_mask(flags)
    flag_summary = summarize_flags(base_valid, flags)
    codes = exclusion_codes(base_valid, flags)
    del qa

    before_accepted, before_rejected = count_accepted_windows(
        base_valid, patch_size, stride, max_invalid_percent
    )

    thermal_k, rgb_refl, scaling = radiometric_convert_scene(dn_bands, per_band_valid)
    del dn_bands
    thermal_phys = _physical_stats_thermal(thermal_k, valid_mask)
    rgb_phys = _physical_stats_rgb(rgb_refl, valid_mask)
    thermal_norm, rgb_norm, norm_params = normalize_scene_arrays(thermal_k, rgb_refl, valid_mask)
    del thermal_k, rgb_refl

    extraction = extract_patches(
        thermal_norm, rgb_norm, valid_mask,
        patch_size=patch_size, stride=stride, max_invalid_percent=max_invalid_percent,
    )
    if not extraction.patches:
        raise RuntimeError("No patches exported after QA masking.")

    scene_output.mkdir(parents=True, exist_ok=True)
    export_patches(scene_output, extraction)
    np.savez_compressed(scene_output / "exclusion_codes.npz", codes=codes)

    mask_report = summarize_mask(valid_mask, per_band_valid)
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
            "definition": "valid = finite and not NoData in every required band AND no excluded "
                          "QA_PIXEL flag set",
        },
        extraction=extraction,
        thermal_physical_stats=thermal_phys,
        rgb_physical_stats=rgb_phys,
        patch_coords=[
            {"patch_id": p.patch_id, "row": p.row, "col": p.col, "invalid_fraction": p.invalid_fraction}
            for p in extraction.patches
        ],
    )

    meta_path = scene_output / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_files = {
        name: {"path": str(band_path(scene_dir, name)), "size": band_path(scene_dir, name).stat().st_size,
               "sha256": sha256(band_path(scene_dir, name))}
        for name in REQUIRED_BANDS
    }
    source_files["QA_PIXEL"] = {"path": str(qa_path), "size": qa_path.stat().st_size, "sha256": sha256(qa_path)}
    meta.update({
        "processing_version": QA_RULES_VERSION,
        "raster_reader_backend": RASTER_BACKEND,
        "source_files": source_files,
        "qa_masking": {
            "rules": rules_metadata(),
            "stac_bitfield_check": stac_bitfield_check(scene_id) if verify_stac else {"verified": False,
                                                                                      "reason": "skipped"},
            "qa_file": {k: v for k, v in qa_info.items() if k != "path"} | {"path": str(qa_path)},
            "alignment": alignment,
            "pixel_counts": flag_summary,
            "exclusion_codes_file": "exclusion_codes.npz",
            "exclusion_codes_bits": {"0": "base_invalid (NoData or non-finite in any required band)"}
            | {str(i + 1): f.name for i, f in enumerate(EXCLUDED_FLAGS)},
        },
        "patch_counts": {
            "before_qa": {"accepted": before_accepted, "rejected": before_rejected},
            "after_qa": {"accepted": len(extraction.patches), "rejected": extraction.skipped_count},
            "rule": f"window rejected if invalid fraction > {max_invalid_percent}% (same as original pipeline)",
        },
        "patch_id_note": "patch IDs are sequential within this scene and this processing version; "
                         "they are not comparable with IDs in dataset/processed",
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "scene_id": scene_id,
        "output": str(scene_output),
        "valid_fraction_before_qa": flag_summary["valid_fraction_before_qa"],
        "valid_fraction_after_qa": flag_summary["valid_fraction_after_qa"],
        "patches_after_qa": len(extraction.patches),
        "patches_rejected_after_qa": extraction.skipped_count,
        "patches_before_qa": before_accepted,
        "thermal_norm": meta["normalization"]["thermal"],
    }


def update_manifest(output_root: Path, summary: dict[str, Any], settings: dict[str, Any]) -> None:
    path = output_root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "processing_version": QA_RULES_VERSION,
        "qa_rules": rules_metadata(),
        "settings": settings,
        "normalization": "per-scene z-score over that scene's valid pixels; statistics are never pooled",
        "hr_reference_note": "thermal patches are the 30 m Landsat reference, not sub-30 m ground truth",
        "scenes": {},
    }
    if manifest.get("settings") != settings:
        raise ValueError(f"Settings {settings} differ from manifest {manifest.get('settings')}")
    summary = dict(summary, processed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    manifest["scenes"][summary["scene_id"]] = summary
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QA_PIXEL-aware preprocessing into a separate root.")
    p.add_argument("--scene-ids", nargs="+", required=True)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--qa-supplement-dir", type=Path, action="append", default=None,
                   help=f"Extra root holding <scene_id>/QA_PIXEL.tif (default: {DEFAULT_SUPPLEMENT_DIR}).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_QA_OUTPUT_DIR)
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--max-invalid-percent", type=float, default=DEFAULT_MAX_INVALID_PERCENT)
    p.add_argument("--skip-existing", action="store_true", help="Skip scenes already processed.")
    p.add_argument("--no-stac-check", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    output_root = args.output_dir.resolve()
    if output_root == DEFAULT_PROCESSED_DIR.resolve() or output_root.is_relative_to(DEFAULT_PROCESSED_DIR.resolve()):
        logger.error("Refusing to write into the original processed root %s", DEFAULT_PROCESSED_DIR)
        return 1
    output_root.mkdir(parents=True, exist_ok=True)
    supplements = args.qa_supplement_dir or [DEFAULT_SUPPLEMENT_DIR]
    settings = {"patch_size": args.patch_size, "stride": args.stride,
                "max_invalid_percent": args.max_invalid_percent}

    failures = []
    for scene_id in args.scene_ids:
        if args.skip_existing and (output_root / scene_id / "metadata.json").exists():
            logger.info("Skipping existing %s", scene_id)
            continue
        logger.info("QA processing %s", scene_id)
        try:
            summary = run_qa_pipeline(
                (args.raw_dir / scene_id).resolve(), output_root,
                supplement_dirs=supplements, patch_size=args.patch_size, stride=args.stride,
                max_invalid_percent=args.max_invalid_percent, verify_stac=not args.no_stac_check,
            )
            update_manifest(output_root, summary, settings)
            logger.info("Done %s: valid %.2f%% -> %.2f%%, patches %d -> %d", scene_id,
                        100 * summary["valid_fraction_before_qa"], 100 * summary["valid_fraction_after_qa"],
                        summary["patches_before_qa"], summary["patches_after_qa"])
        except Exception as exc:
            logger.error("FAILED %s: %s", scene_id, exc)
            failures.append(scene_id)
    if failures:
        logger.error("Failed scenes: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
