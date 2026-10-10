"""Read-only audit of the qa_v1 processed dataset and its raw inputs.

Checks, per scene: manifest vs metadata vs files, patch-ID uniqueness, thermal
normalization recomputed independently from raw ST_B10, patch values vs raw
Kelvin, QA_PIXEL value histogram and flag counts vs STAC cloud cover, reader
cross-checks (tifffile vs STAC proj:* grid, tifffile vs Pillow pixel decode,
STAC raster:bands scale/offset vs pipeline constants), patch masks vs exclusion
codes, and whether the development patch split would mix scenes.

    py -m scripts.audit_qa_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset_loader.split import development_patch_split
from dataset_loader.thermal_dataset import ThermalSatelliteDataset
from preprocessing.qa_mask import EXCLUDED_FLAGS, find_qa_path
from preprocessing.radiometric import (
    SR_REFLECTANCE_ADD,
    SR_REFLECTANCE_MULT,
    ST_TEMPERATURE_ADD_K,
    ST_TEMPERATURE_MULT,
)
from preprocessing.raster_io import BACKEND, raster_info, read_band1

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None

BANDS = ("ST_B10", "SR_B2", "SR_B3", "SR_B4", "QA_PIXEL")
STAC_KEYS = {"ST_B10": "lwir11", "SR_B2": "blue", "SR_B3": "green", "SR_B4": "red", "QA_PIXEL": "qa_pixel"}
SPOT_PATCHES = 20
NOISE_STD_K = 0.05


def fetch_stac(scene_ids: list[str]) -> dict[str, Any]:
    try:
        from scripts.download_dataset import build_stac_client, fetch_scenes_by_id

        return {it.id: it for it in fetch_scenes_by_id(build_stac_client(), scene_ids)}
    except Exception as exc:
        logger.warning("STAC unavailable: %s", exc)
        return {}


def stac_grid_check(item, info) -> dict[str, Any]:
    p = item.properties
    a, b, c, d, e, f = p["proj:transform"][:6]
    stac_gdal = (c, a, b, f, d, e)
    rows, cols = p["proj:shape"]
    epsg = p.get("proj:epsg") or int(str(p.get("proj:code", "EPSG:0")).split(":")[1])
    result = {
        "stac_transform_gdal_order": list(stac_gdal),
        "reader_transform": list(info.transform),
        "transform_equal": tuple(info.transform) == stac_gdal,
        "shape_equal": (info.height, info.width) == (rows, cols),
        "epsg_equal": info.crs == f"EPSG:{epsg}",
    }
    result["passed"] = result["transform_equal"] and result["shape_equal"] and result["epsg_equal"]
    return result


def stac_scaling_check(item) -> dict[str, Any]:
    out = {}
    expected = {"lwir11": (ST_TEMPERATURE_MULT, ST_TEMPERATURE_ADD_K)}
    expected.update({k: (SR_REFLECTANCE_MULT, SR_REFLECTANCE_ADD) for k in ("blue", "green", "red")})
    for key, (mult, add) in expected.items():
        rb = item.assets[key].extra_fields.get("raster:bands", [{}])[0]
        out[key] = {"stac_scale": rb.get("scale"), "stac_offset": rb.get("offset"),
                    "pipeline_mult": mult, "pipeline_add": add,
                    "equal": rb.get("scale") is not None and abs(rb["scale"] - mult) < 1e-12
                             and abs(rb["offset"] - add) < 1e-9}
    out["passed"] = all(v["equal"] for v in out.values() if isinstance(v, dict))
    return out


def audit_scene(scene_dir: Path, manifest_entry: dict[str, Any] | None, item) -> dict[str, Any]:
    meta = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    sid = meta["scene_id"]
    raw_dir = Path(meta["source_directory"])
    rec: dict[str, Any] = {"scene_id": sid, "platform": item.properties.get("platform") if item else None}

    # Manifest / metadata / files / patch IDs
    t_files = sorted(p.name for p in (scene_dir / "thermal").glob("patch_*.npy"))
    r_files = sorted(p.name for p in (scene_dir / "rgb").glob("patch_*.npy"))
    ids = [c["patch_id"] for c in meta["patch_coordinates"]]
    rec["structure"] = {
        "patches_generated": meta["patches_generated"],
        "manifest_patches": manifest_entry.get("patches_after_qa") if manifest_entry else None,
        "thermal_files": len(t_files),
        "rgb_files": len(r_files),
        "thermal_rgb_names_match": t_files == r_files,
        "patch_ids_unique": len(ids) == len(set(ids)),
        "coordinates_unique": len({(c["row"], c["col"]) for c in meta["patch_coordinates"]}) == len(ids),
    }
    s = rec["structure"]
    s["passed"] = (s["patches_generated"] == s["manifest_patches"] == s["thermal_files"] == s["rgb_files"]
                   and s["thermal_rgb_names_match"] and s["patch_ids_unique"] and s["coordinates_unique"])

    codes = np.load(scene_dir / "exclusion_codes.npz")["codes"]
    valid = codes == 0

    # Independent thermal normalization recompute and patch spot-check
    st_path = raw_dir / "ST_B10.tif"
    dn = read_band1(st_path)
    kelvin = dn.astype(np.float64) * ST_TEMPERATURE_MULT + ST_TEMPERATURE_ADD_K
    vals = kelvin[valid]
    mean, std = float(vals.mean()), float(vals.std())
    stored = meta["normalization"]["thermal"]
    rec["thermal_normalization"] = {
        "stored_mean": stored["mean"], "stored_std": stored["std"],
        "recomputed_mean_float64": mean, "recomputed_std_float64": std,
        "abs_diff_mean_k": abs(mean - stored["mean"]), "abs_diff_std_k": abs(std - stored["std"]),
        "valid_pixels_used": int(valid.sum()),
        "passed": abs(mean - stored["mean"]) < 1e-3 and abs(std - stored["std"]) < 1e-3,
    }
    rng = np.random.default_rng(0)
    coords = meta["patch_coordinates"]
    pick = rng.choice(len(coords), size=min(SPOT_PATCHES, len(coords)), replace=False)
    ps = meta["patch_size"]
    max_err = 0.0
    nan_mismatch = 0
    for i in pick:
        c = coords[int(i)]
        patch = np.load(scene_dir / "thermal" / f"patch_{c['patch_id']:06d}.npy").astype(np.float64)
        ref_k = kelvin[c["row"]:c["row"] + ps, c["col"]:c["col"] + ps]
        ref_valid = valid[c["row"]:c["row"] + ps, c["col"]:c["col"] + ps]
        fin = np.isfinite(patch)
        nan_mismatch += int((fin != ref_valid).sum())
        if fin.any():
            max_err = max(max_err, float(np.abs(patch[fin] * stored["std"] + stored["mean"] - ref_k[fin]).max()))
    rec["patch_value_spot_check"] = {
        "patches_checked": int(len(pick)),
        "max_abs_kelvin_error_vs_raw": max_err,
        "finite_vs_valid_mismatches": nan_mismatch,
        "passed": max_err < 1e-3 and nan_mismatch == 0,
        "note": "patch*std+mean compared with raw DN*0.00341802+149 at the same pixels (float32 rounding expected)",
    }
    base_invalid = int((codes & 1).sum())
    del dn, kelvin, vals

    # QA histogram and flag consistency
    qa_path = find_qa_path(raw_dir, [Path("dataset/raw_supplement")])
    qa = read_band1(qa_path)
    uniq, counts = np.unique(qa, return_counts=True)
    order = np.argsort(-counts)
    flag_bits_all = {f.name: int(((qa >> f.bit) & 1).sum()) for f in EXCLUDED_FLAGS}
    fill = ((qa & 1) == 1)
    nodata_any = (codes & 1) == 1
    rec["qa"] = {
        "distinct_values": int(len(uniq)),
        "top_values": [{"value": int(uniq[j]), "count": int(counts[j]),
                        "set_bits": [b for b in range(16) if (int(uniq[j]) >> b) & 1]} for j in order[:10]],
        "flag_counts_all_pixels": flag_bits_all,
        "fill_pixels": int(fill.sum()),
        "band_nodata_or_nonfinite_pixels": base_invalid,
        "fill_equals_band_nodata": bool(np.array_equal(fill, nodata_any)),
        "fill_and_not_nodata": int((fill & ~nodata_any).sum()),
        "nodata_and_not_fill": int((nodata_any & ~fill).sum()),
        "stac_eo_cloud_cover": item.properties.get("eo:cloud_cover") if item else None,
        "stac_cloud_cover_land": item.properties.get("landsat:cloud_cover_land") if item else None,
        "metadata_matches_recount": {
            name: meta["qa_masking"]["pixel_counts"]["per_flag"][name]["set_all_pixels"] == n
            for name, n in flag_bits_all.items()
        },
    }
    nonfill = ~fill
    cloudy_bits = sum(flag_bits_all[n] for n in ("dilated_cloud", "cirrus", "cloud", "cloud_shadow"))
    rec["qa"]["cloud_related_fraction_of_image_pixels"] = cloudy_bits / max(int(nonfill.sum()), 1)
    del qa, fill, nodata_any, nonfill

    # Reader cross-checks
    cross: dict[str, Any] = {"reader_backend": BACKEND}
    if item is not None:
        cross["stac_proj_grid"] = {b: stac_grid_check(item, raster_info(raw_dir / f"{b}.tif"))
                                   for b in BANDS[:4]}
        cross["stac_proj_grid"]["QA_PIXEL"] = stac_grid_check(item, raster_info(qa_path))
        cross["stac_proj_grid_performed"] = True
        cross["stac_proj_grid_passed"] = all(v["passed"] for k, v in cross["stac_proj_grid"].items()
                                              if isinstance(v, dict))
        cross["stac_scaling"] = stac_scaling_check(item)
    else:
        cross["stac_proj_grid_performed"] = False
    pillow = {}
    for b in BANDS:
        path = qa_path if b == "QA_PIXEL" else raw_dir / f"{b}.tif"
        try:
            with Image.open(path) as im:
                pil = np.array(im)
            tf = read_band1(path)
            pillow[b] = {"pillow_mode_dtype": str(pil.dtype), "shape_equal": pil.shape == tf.shape,
                         "pixels_identical": bool(pil.shape == tf.shape and np.array_equal(pil.astype(tf.dtype), tf))}
        except Exception as exc:
            pillow[b] = {"error": str(exc), "pixels_identical": False}
    cross["pillow_vs_tifffile_pixels"] = pillow
    cross["pillow_decode_performed"] = all("error" not in v for v in pillow.values())
    cross["pillow_decode_passed"] = all(v["pixels_identical"] for v in pillow.values())
    cross["rasterio_vs_tifffile_performed"] = False
    rec["reader_cross_checks"] = cross

    rec["normalization_effects"] = {
        "thermal_std_k": stored["std"],
        "synthetic_noise_0.05K_in_z_units": NOISE_STD_K / stored["std"],
        "z_error_to_kelvin_factor": stored["std"],
    }
    return rec


def dev_split_mixing(root: Path) -> dict[str, Any]:
    ds = ThermalSatelliteDataset(root, return_mask=False)
    scene_of = [ref.scene_id for ref in ds.patch_refs]
    split = development_patch_split(len(scene_of), seed=42)
    parts = {"train": split.train_indices, "validation": split.validation_indices, "test": split.test_indices}
    per_part = {k: dict(Counter(scene_of[i] for i in v)) for k, v in parts.items()}
    scenes_in_multiple = sorted(s for s in set(scene_of) if sum(s in per_part[k] for k in per_part) > 1)
    return {
        "total_patches": len(scene_of),
        "scenes_per_partition": {k: len(v) for k, v in per_part.items()},
        "patches_per_scene_per_partition": per_part,
        "scenes_present_in_more_than_one_partition": scenes_in_multiple,
        "mixes_scenes": bool(scenes_in_multiple),
        "conclusion": "development_patch_split is NOT scene-safe on a multi-scene root; use the scene split manifest",
    }


def render(report: dict[str, Any]) -> str:
    L = ["QA_V1 DATASET AUDIT", "=" * 78, f"root: {report['root']} | scenes: {report['scene_count']} | "
         f"patches: {report['total_patches']} | reader: {report['reader_backend']}", ""]
    for r in report["scenes"]:
        q, n, x = r["qa"], r["thermal_normalization"], r["reader_cross_checks"]
        L += [
            f"{r['scene_id']} ({r['platform']})",
            f"  structure ok: {r['structure']['passed']} (patches {r['structure']['patches_generated']})",
            f"  thermal mean/std stored {n['stored_mean']:.4f}/{n['stored_std']:.4f} K, recomputed float64 "
            f"{n['recomputed_mean_float64']:.4f}/{n['recomputed_std_float64']:.4f} K -> ok {n['passed']}",
            f"  patch values vs raw Kelvin: max err {r['patch_value_spot_check']['max_abs_kelvin_error_vs_raw']:.2e} K, "
            f"mask mismatches {r['patch_value_spot_check']['finite_vs_valid_mismatches']} -> ok "
            f"{r['patch_value_spot_check']['passed']}",
            f"  QA: {q['distinct_values']} distinct values; flags(all px) {q['flag_counts_all_pixels']}; "
            f"fill==band NoData: {q['fill_equals_band_nodata']}; STAC cloud {q['stac_eo_cloud_cover']}% "
            f"(land {q['stac_cloud_cover_land']}%)",
            f"  QA top values: " + ", ".join(f"{v['value']}x{v['count']}" for v in q["top_values"][:5]),
            f"  readers: STAC proj grid performed {x.get('stac_proj_grid_performed')} passed "
            f"{x.get('stac_proj_grid_passed')}; STAC scale/offset passed {x.get('stac_scaling', {}).get('passed')}; "
            f"Pillow pixels performed {x['pillow_decode_performed']} passed {x['pillow_decode_passed']}; "
            f"rasterio vs tifffile performed {x['rasterio_vs_tifffile_performed']}",
            f"  normalization: std {r['normalization_effects']['thermal_std_k']:.3f} K, 0.05 K noise = "
            f"{r['normalization_effects']['synthetic_noise_0.05K_in_z_units']:.4f} z",
            "",
        ]
    m = report["dev_patch_split"]
    L += ["DEVELOPMENT PATCH SPLIT ON THIS ROOT", f"  scenes per partition: {m['scenes_per_partition']}; "
          f"mixes scenes: {m['mixes_scenes']}", "", "OTHER FINDINGS"]
    L += [f"  - {f}" for f in report["findings"]]
    L += ["", f"mask check (scripts.verify_qa_patches): {report['mask_check_passed']}",
          f"rasterio-vs-tifffile evidence: {report['rasterio_equivalence_evidence']}"]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("dataset/processed_qa"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/multiscene_qa_v1/audit"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    manifest = json.loads((args.root / "processing_manifest.json").read_text(encoding="utf-8"))
    scene_dirs = sorted(d for d in args.root.iterdir() if (d / "metadata.json").is_file())
    items = fetch_stac([d.name for d in scene_dirs])
    scenes = []
    for d in scene_dirs:
        logger.info("Auditing %s", d.name)
        scenes.append(audit_scene(d, manifest["scenes"].get(d.name), items.get(d.name)))

    from scripts.verify_qa_patches import main as verify_masks

    mask_ok = verify_masks(["--root", str(args.root)]) == 0
    eq_path = Path("outputs/dataset_expansion/reader_equivalence.json")
    eq = json.loads(eq_path.read_text(encoding="utf-8")) if eq_path.exists() else None

    report = {
        "root": str(args.root),
        "scene_count": len(scenes),
        "manifest_scene_count": len(manifest["scenes"]),
        "total_patches": sum(s["structure"]["patches_generated"] for s in scenes),
        "reader_backend": BACKEND,
        "scenes": scenes,
        "dev_patch_split": dev_split_mixing(args.root),
        "mask_check_passed": mask_ok,
        "rasterio_equivalence_evidence": (
            f"only {eq['scene_id']}: original pipeline re-run with tifffile reproduced dataset/processed "
            f"(produced earlier with rasterio) byte-identically, passed={eq['passed']}" if eq else "none"),
        "findings": [
            "training/train_swinir.py build_datasets uses development_patch_split on the whole root (mixes "
            "scenes) and its Kelvin metrics use the first scene's normalization stats; guarded in this task.",
            "Degradation noise RNG is keyed by seed and scene-local patch_id only, so equal patch IDs in "
            "different scenes receive identical noise fields (not changed; same inputs for every method).",
            "Normalized-space MAE/RMSE equal Kelvin error divided by each scene's thermal std, so they are not "
            "comparable across scenes; Kelvin metrics are. Per-patch PSNR (patch-range) is invariant to the "
            "linear normalization; SSIM constants are fixed in z-units, so SSIM depends on scene std.",
            "Kelvin reconstruction value_K = z*std+mean is exact (up to float32 rounding) only with the "
            "patch's own scene statistics; never with pooled or another scene's statistics.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    text = render(report)
    (args.output_dir / "audit_report.txt").write_text(text, encoding="utf-8")
    print(text)
    ok = (mask_ok and all(s["structure"]["passed"] and s["thermal_normalization"]["passed"]
                          and s["patch_value_spot_check"]["passed"] for s in scenes))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
