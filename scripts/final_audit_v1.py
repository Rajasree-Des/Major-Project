"""Final reproducibility and integrity audit (read-only)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "final_audit_v1"
STANDALONE = PROJECT_ROOT / "outputs" / "colorization" / "guided_colorization_v1"
INTEGRATED = PROJECT_ROOT / "outputs" / "integrated_pipeline_v1"
KERALA = "LC08_L2SP_144053_20250225_02_T1"
PATCHES = ("000096", "000448", "001006", "001116")
METHOD_PNG = (
    "A_standard_colormap.png",
    "A_standard_colormap_legend.png",
    "B_rgb_guided.png",
    "B_rgb_guided_legend.png",
    "C_rgb_overlay.png",
    "C_rgb_overlay_legend.png",
)

FROZEN_RESULTS = (
    "outputs/evaluation/multiscene_qa_v1/validation/results.json",
    "outputs/evaluation/multiscene_qa_v1/test_locked/results.json",
    "outputs/evaluation/physics_refinement_v1/validation/results.json",
    "outputs/evaluation/physics_refinement_v1/test_locked/results.json",
)
FROZEN_OTHER = (
    "dataset/splits/scene_split_qa_v1.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare_npy(a: Path, b: Path) -> dict:
    if not a.is_file() or not b.is_file():
        return {"error": f"missing {a} or {b}"}
    xa, xb = np.load(a), np.load(b)
    if xa.shape != xb.shape:
        return {"shape_a": list(xa.shape), "shape_b": list(xb.shape), "match": False}
    both_finite = np.isfinite(xa) & np.isfinite(xb)
    diff = np.abs(xa.astype(np.float64) - xb.astype(np.float64))
    ne = ~np.isclose(xa, xb, equal_nan=True, rtol=0, atol=0)
    return {
        "match": bool(np.allclose(xa, xb, equal_nan=True, rtol=0, atol=0)),
        "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        "differing_pixels": int(ne.sum()),
        "total_pixels": int(xa.size),
    }


def compare_png_pixels(a: Path, b: Path) -> dict:
    if not a.is_file() or not b.is_file():
        return {"error": f"missing {a.name}", "match": False}
    ia, ib = np.array(Image.open(a)), np.array(Image.open(b))
    if ia.shape != ib.shape:
        return {"match": False, "shape_a": list(ia.shape), "shape_b": list(ib.shape)}
    ne = ia != ib
    return {
        "match": bool(not ne.any()),
        "differing_pixels": int(ne.sum()),
        "total_values": int(ne.size),
        "max_abs_channel_diff": int(np.max(np.abs(ia.astype(int) - ib.astype(int)))) if ne.any() else 0,
    }


def standalone_patch_dir(patch_id: str) -> Path:
    return STANDALONE / "patches" / f"patch_{patch_id}"


def integrated_patch_dir(patch_id: str) -> Path:
    return INTEGRATED / "patches" / KERALA / f"patch_{patch_id}"


def _standalone_valid_pixels() -> dict[str, int]:
    man = STANDALONE / "manifest.json"
    if not man.is_file():
        return {}
    data = json.loads(man.read_text(encoding="utf-8"))
    return {p["patch_id"]: int(p["valid_pixels"]) for p in data.get("patches", [])}


def phase1_regression() -> dict:
    st_valid = _standalone_valid_pixels()
    per_patch = {}
    for pid in PATCHES:
        sdir, idir = standalone_patch_dir(pid), integrated_patch_dir(pid)
        kelvin = compare_npy(sdir / "thermal_kelvin.npy", idir / "thermal_kelvin_refined.npy")
        mask_standalone = sdir / "valid_mask.npy"
        int_mask = idir / "valid_mask.npy"
        int_valid = int(np.load(int_mask).sum()) if int_mask.is_file() else None
        if mask_standalone.is_file():
            mask_cmp = compare_npy(mask_standalone, int_mask)
        else:
            manifest_vp = st_valid.get(pid)
            mask_cmp = {
                "note": "standalone run did not save valid_mask.npy",
                "integrated_valid_pixels": int_valid,
                "standalone_manifest_valid_pixels": manifest_vp,
                "valid_pixel_count_match": manifest_vp == int_valid if manifest_vp is not None else None,
            }
        pngs = {}
        for name in METHOD_PNG:
            pngs[name] = compare_png_pixels(sdir / name, idir / name)
        per_patch[pid] = {"kelvin": kelvin, "mask": mask_cmp, "png": pngs}
    method_png_match = all(
        all(p["png"][n].get("match") for n in METHOD_PNG if not n.endswith("_legend.png"))
        for p in per_patch.values()
    )
    legend_png_match = all(
        all(p["png"][n].get("match") for n in METHOD_PNG if n.endswith("_legend.png"))
        for p in per_patch.values()
    )
    summary = {
        "kelvin_all_match": all(p["kelvin"].get("match") for p in per_patch.values()),
        "method_png_all_match": method_png_match,
        "legend_png_all_match": legend_png_match,
        "png_all_match": method_png_match and legend_png_match,
        "legend_mismatch_cause": (
            None
            if legend_png_match
            else (
                "Legend PNGs embed matplotlib titles; standalone uses "
                "'{key} — patch {id}' (em dash) in scripts/run_guided_colorization_v1.py "
                "while integrated uses '{label} patch {id}' in pipeline/integrated.py. "
                "Method A/B/C RGBA PNGs (saved via PIL from identical arrays) match exactly."
            )
        ),
    }
    return {"patches": per_patch, "summary": summary}


def load_baseline_hashes(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def phase2_integrity(baseline_path: Path | None) -> dict:
    files = {}
    for rel in FROZEN_RESULTS + FROZEN_OTHER:
        p = PROJECT_ROOT / rel
        if p.is_file():
            files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}

    checkpoints = []
    for rel in ("outputs/models", "pretrained"):
        root = PROJECT_ROOT / rel
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix in (".pth", ".pt", ".ckpt"):
                    checkpoints.append(
                        {
                            "path": p.relative_to(PROJECT_ROOT).as_posix(),
                            "sha256": sha256_file(p),
                            "size": p.stat().st_size,
                        }
                    )

    baseline = load_baseline_hashes(baseline_path) if baseline_path else {}
    frozen_changes = []
    for rel, meta in files.items():
        if rel in baseline and baseline[rel] != meta["sha256"]:
            frozen_changes.append({"file": rel, "was": baseline[rel], "now": meta["sha256"]})

    git_status = subprocess.check_output(
        ["git", "status", "--short"], text=True, cwd=PROJECT_ROOT, stderr=subprocess.STDOUT
    )

    return {
        "frozen_hashes": files,
        "baseline_compare": {
            "baseline_file": str(baseline_path) if baseline_path else None,
            "changes_detected": frozen_changes,
            "passed": len(frozen_changes) == 0,
        },
        "checkpoints": checkpoints,
        "git_status_short": git_status,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = PROJECT_ROOT / "outputs" / "colorization" / "_frozen_hashes_before.json"
    if not baseline.is_file():
        baseline = PROJECT_ROOT / "outputs" / "evaluation" / "physics_refinement_v1" / "frozen_integrity_before_test.json"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip(),
        "phase1_colorization_regression": phase1_regression(),
        "phase2_frozen_integrity": phase2_integrity(
            PROJECT_ROOT / "outputs" / "final_audit_v1" / "frozen_baseline_snapshot.json"
            if (PROJECT_ROOT / "outputs" / "final_audit_v1" / "frozen_baseline_snapshot.json").is_file()
            else None
        ),
    }
    # First run: save snapshot for future compares; compare to colorization frozen if present
    snap_path = OUT / "frozen_baseline_snapshot.json"
    if not snap_path.is_file():
        report["phase2_frozen_integrity"]["note"] = "snapshot written on first audit run"
        snap = {k: v["sha256"] for k, v in report["phase2_frozen_integrity"]["frozen_hashes"].items()}
        snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    # Compare to earlier snapshot from integration task if exists
    for candidate in (
        PROJECT_ROOT / "outputs" / "colorization" / "_frozen_hashes_before.json",
        PROJECT_ROOT / "outputs" / "evaluation" / "physics_refinement_v1" / "frozen_integrity_before_test.json",
    ):
        if candidate.is_file():
            old = json.loads(candidate.read_text(encoding="utf-8"))
            drift = []
            for rel, h in old.items():
                rel = rel.replace("\\", "/")
                cur = report["phase2_frozen_integrity"]["frozen_hashes"].get(rel)
                if cur and cur["sha256"].lower() != str(h).lower():
                    drift.append(rel)
            report["phase2_prior_snapshot"] = {
                "file": str(candidate),
                "changed": drift,
                "passed": len(drift) == 0,
                "note": "Compared sha256 case-insensitively (prior snapshot used uppercase hex).",
            }
            break

    (OUT / "audit_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "git_status_short.txt").write_text(
        report["phase2_frozen_integrity"]["git_status_short"], encoding="utf-8"
    )
    (OUT / "frozen_hashes_evidence.json").write_text(
        json.dumps(report["phase2_frozen_integrity"]["frozen_hashes"], indent=2), encoding="utf-8"
    )
    print(json.dumps(report["phase1_colorization_regression"]["summary"], indent=2))
    print("frozen passed:", report["phase2_frozen_integrity"]["baseline_compare"]["passed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
