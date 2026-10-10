"""Check that the active raster backend reproduces an existing processed scene exactly.

Re-runs the original preprocessing steps (validation, masks, radiometric
conversion, normalization, patch extraction) in memory from the raw bands, then
compares every patch array and key metadata with ``dataset/processed/<scene>``.
Nothing is written except the JSON report.

    py -m scripts.verify_reader_equivalence --scene-id LC09_L2SP_143050_20240327_02_T1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from preprocessing.constants import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR, REQUIRED_BANDS
from preprocessing.masking import build_per_band_valid_masks, combine_valid_mask
from preprocessing.normalize import normalize_scene_arrays
from preprocessing.patchify import extract_patches
from preprocessing.radiometric import radiometric_convert_scene
from preprocessing.raster_io import BACKEND
from preprocessing.validate import band_path, read_band_array, validate_scene


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene-id", required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/dataset_expansion/reader_equivalence.json"))
    args = p.parse_args(argv)

    processed = DEFAULT_PROCESSED_DIR / args.scene_id
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    validation = validate_scene(DEFAULT_RAW_DIR / args.scene_id)
    ref = validation.bands["ST_B10"]

    dn = {n: read_band_array(band_path(validation.scene_dir, n))[0] for n in REQUIRED_BANDS}
    per_band = build_per_band_valid_masks(dn, validation.nodata_values)
    valid = combine_valid_mask(per_band)
    thermal_k, rgb, _ = radiometric_convert_scene(dn, per_band)
    t_norm, rgb_norm, params = normalize_scene_arrays(thermal_k, rgb, valid)
    extraction = extract_patches(t_norm, rgb_norm, valid, patch_size=meta["patch_size"],
                                 stride=meta["stride"], max_invalid_percent=meta["max_invalid_percent"])

    checks = {
        "crs": ref.crs == meta["crs"],
        "dimensions": {"width": ref.width, "height": ref.height} == meta["original_dimensions"],
        "bounds": list(ref.bounds) == [meta["bounds"][k] for k in ("left", "bottom", "right", "top")],
        "nodata": validation.nodata_values == meta["nodata_values"],
        "valid_pixels": int(np.count_nonzero(valid)) == meta["mask_summary"]["valid_pixels"],
        "normalization": params.to_metadata()["thermal"] == meta["normalization"]["thermal"]
                         and params.to_metadata()["rgb"] == meta["normalization"]["rgb"],
        "patch_counts": (len(extraction.patches), extraction.skipped_count)
                        == (meta["patches_generated"], meta["patches_skipped"]),
        "patch_coordinates": [(r.patch_id, r.row, r.col) for r in extraction.patches]
                             == [(c["patch_id"], c["row"], c["col"]) for c in meta["patch_coordinates"]],
    }

    mismatched = []
    for rec in extraction.patches:
        name = f"patch_{rec.patch_id:06d}.npy"
        t_old = np.load(processed / "thermal" / name)
        r_old = np.load(processed / "rgb" / name)
        if t_old.tobytes() != rec.thermal.tobytes() or r_old.tobytes() != rec.rgb.tobytes():
            mismatched.append(rec.patch_id)
    checks["patch_arrays_byte_identical"] = not mismatched

    report = {
        "scene_id": args.scene_id,
        "backend": BACKEND,
        "patches_compared": len(extraction.patches),
        "mismatched_patch_ids": mismatched[:50],
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
