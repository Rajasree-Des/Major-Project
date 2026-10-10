"""Summarise a QA-processed dataset root into JSON and text reports (read-only).

    py -m scripts.report_qa_dataset --root dataset/processed_qa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from preprocessing.qa_mask import EXCLUDED_FLAGS


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def scene_record(scene_dir: Path) -> dict[str, Any]:
    meta = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    qa = meta["qa_masking"]
    counts = qa["pixel_counts"]
    raw_dir = Path(meta["source_directory"])
    return {
        "scene_id": meta["scene_id"],
        "processing_version": meta["processing_version"],
        "raster_reader_backend": meta.get("raster_reader_backend"),
        "source_files": meta["source_files"],
        "raw_size_bytes": sum(f["size"] for f in meta["source_files"].values()),
        "grid": {
            "crs": meta["crs"],
            "resolution_m": meta["original_resolution_m"],
            "dimensions": meta["original_dimensions"],
            "bounds": meta["bounds"],
            "qa_alignment": qa["alignment"],
            "qa_dtype": qa["qa_file"]["dtype"],
            "qa_nodata": qa["qa_file"]["nodata"],
        },
        "stac_bitfield_check": qa["stac_bitfield_check"].get("verified"),
        "valid_percent_before_qa": round(100 * counts["valid_fraction_before_qa"], 3),
        "valid_percent_after_qa": round(100 * counts["valid_fraction_after_qa"], 3),
        "pixels_excluded_by_qa": counts["excluded_by_qa_among_base_valid"],
        "per_flag_among_base_valid": {k: v["set_among_base_valid"] for k, v in counts["per_flag"].items()},
        "per_flag_all_pixels": {k: v["set_all_pixels"] for k, v in counts["per_flag"].items()},
        "patches": meta["patch_counts"],
        "normalization": meta["normalization"],
        "thermal_kelvin_valid_stats": meta["valid_pixel_statistics_physical"]["thermal_kelvin"],
        "processed_size_bytes": dir_size(scene_dir),
        "raw_dir": str(raw_dir),
    }


def render(records: list[dict[str, Any]], root: Path) -> str:
    flags = [f.name for f in EXCLUDED_FLAGS]
    lines = [f"QA-PROCESSED DATASET REPORT ({root})", "=" * 78,
             "HR target is the 30 m Landsat thermal reference, not sub-30 m ground truth.",
             "Per-scene z-score statistics; never pooled. Splits must be scene-level.", ""]
    for r in records:
        g = r["grid"]
        p = r["patches"]
        t = r["normalization"]["thermal"]
        lines += [
            r["scene_id"],
            f"  grid: {g['crs']} {g['resolution_m']['x']} m, {g['dimensions']['width']}x{g['dimensions']['height']}, "
            f"QA aligned: {g['qa_alignment']['result']} ({', '.join(g['qa_alignment']['also_identical_to'])})",
            f"  STAC bit definitions verified: {r['stac_bitfield_check']} | reader: {r['raster_reader_backend']}",
            f"  valid pixels: {r['valid_percent_before_qa']:.2f}% -> {r['valid_percent_after_qa']:.2f}% "
            f"({r['pixels_excluded_by_qa']:,} excluded by QA)",
            "  QA flag set among otherwise-valid pixels: "
            + ", ".join(f"{k}={r['per_flag_among_base_valid'][k]:,}" for k in flags),
            f"  patches: before QA {p['before_qa']['accepted']} accepted / {p['before_qa']['rejected']} rejected; "
            f"after QA {p['after_qa']['accepted']} / {p['after_qa']['rejected']}",
            f"  thermal z-score: mean {t['mean']:.3f} K, std {t['std']:.3f} K | "
            f"valid LST range {r['thermal_kelvin_valid_stats']['min']:.1f}-{r['thermal_kelvin_valid_stats']['max']:.1f} K",
            f"  storage: raw {r['raw_size_bytes'] / 2**20:,.0f} MiB, processed {r['processed_size_bytes'] / 2**30:.2f} GiB",
            "",
        ]
    total_raw = sum(r["raw_size_bytes"] for r in records)
    total_proc = sum(r["processed_size_bytes"] for r in records)
    total_patches = sum(r["patches"]["after_qa"]["accepted"] for r in records)
    lines += [f"TOTAL: {len(records)} scenes, {total_patches} patches, raw {total_raw / 2**30:.2f} GiB, "
              f"processed {total_proc / 2**30:.2f} GiB"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("dataset/processed_qa"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_expansion"))
    args = p.parse_args(argv)

    scenes = sorted(d for d in args.root.iterdir() if (d / "metadata.json").is_file())
    records = [scene_record(d) for d in scenes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qa_dataset_report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    text = render(records, args.root)
    (args.output_dir / "qa_dataset_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
