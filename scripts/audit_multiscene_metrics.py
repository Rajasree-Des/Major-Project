"""Read-only audit: recompute pooled metrics from saved per-patch CSV and verify Kelvin weighting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "outputs" / "evaluation" / "multiscene_qa_v1"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "evaluation" / "metrics_audit_v1"


def pooled_from_csv(rows: list[dict], method: str) -> dict[str, float]:
    vp = sum(int(r["valid_pixels"]) for r in rows)
    mae_k = sum(int(r["valid_pixels"]) * float(r[f"{method}_mae_kelvin"]) for r in rows) / vp
    rmse_k = math.sqrt(
        sum(int(r["valid_pixels"]) * float(r[f"{method}_rmse_kelvin"]) ** 2 for r in rows) / vp
    )
    mae_z = sum(int(r["valid_pixels"]) * float(r[f"{method}_mae_z"]) for r in rows) / vp
    psnr = sum(float(r[f"{method}_psnr"]) for r in rows if math.isfinite(float(r[f"{method}_psnr"]))) / len(rows)
    ssim = sum(float(r[f"{method}_ssim"]) for r in rows) / len(rows)
    return {"pooled_mae_kelvin": mae_k, "pooled_rmse_kelvin": rmse_k, "pooled_mae_z": mae_z,
            "mean_psnr_per_patch": psnr, "mean_ssim_per_patch": ssim, "valid_pixels": vp}


def audit_role(role_dir: Path) -> dict:
    results = json.loads((role_dir / "results.json").read_text(encoding="utf-8"))
    with (role_dir / "per_patch_metrics.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("excluded")]
    methods = list(results["methods"])
    checks = []
    for scene in results["per_scene"]:
        sid = scene["scene_id"]
        scene_rows = [r for r in rows if r["scene_id"] == sid]
        std_k = scene["thermal_std_k"]
        for method in methods:
            pooled = pooled_from_csv(scene_rows, method)
            m = scene["metrics"][method]
            tol = 1e-5
            checks.append({
                "scene_id": sid,
                "method": method,
                "mae_kelvin_match": abs(pooled["pooled_mae_kelvin"] - m["mae_kelvin"]) < tol,
                "rmse_kelvin_match": abs(pooled["pooled_rmse_kelvin"] - m["rmse_kelvin"]) < tol,
                "mae_z_match": abs(pooled["pooled_mae_z"] - m["mae_z"]) < tol,
                "kelvin_linear_check": abs(pooled["pooled_mae_kelvin"] - pooled["pooled_mae_z"] * std_k) < 0.02,
                "reported_mae_kelvin": m["mae_kelvin"],
                "csv_pooled_mae_kelvin": pooled["pooled_mae_kelvin"],
                "thermal_std_k": std_k,
            })
    agg = results["aggregate"]
    for method in methods:
        pooled_csv = pooled_from_csv(rows, method)
        checks.append({
            "method": method,
            "aggregate_pooled_rmse_matches_csv": abs(
                agg[method]["pooled_rmse_kelvin"] - pooled_csv["pooled_rmse_kelvin"]
            ) < 1e-4,
            "aggregate_pooled_mae_matches_csv": abs(
                agg[method]["pooled_mae_kelvin"] - pooled_csv["pooled_mae_kelvin"]
            ) < 1e-4,
        })
        scene_rmses = [s["metrics"][method]["rmse_kelvin"] for s in results["per_scene"] if s["metrics"]]
        if scene_rmses:
            checks.append({
                "method": method,
                "macro_rmse_mean_over_scenes": abs(
                    agg[method]["rmse_kelvin"]["macro_mean_over_scenes"] - float(sum(scene_rmses) / len(scene_rmses))
                ) < 1e-9,
            })
    def _ok(c: dict) -> bool:
        if "scene_id" in c:
            return c["mae_kelvin_match"] and c["rmse_kelvin_match"] and c["mae_z_match"]
        if "macro_rmse_mean_over_scenes" in c:
            return bool(c["macro_rmse_mean_over_scenes"])
        return bool(c.get("aggregate_pooled_rmse_matches_csv", True)) and bool(
            c.get("aggregate_pooled_mae_matches_csv", True)
        )

    passed = all(_ok(c) for c in checks)
    return {"role_dir": str(role_dir), "checks": checks, "passed": passed, "methods": methods}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--roles", nargs="+", default=["validation", "test_locked"])
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for role in args.roles:
        role_dir = args.baseline_root / role
        if not role_dir.exists():
            print(f"skip missing {role_dir}")
            continue
        reports.append(audit_role(role_dir))
    summary = {"baseline_root": str(args.baseline_root), "roles": reports,
               "passed": all(r["passed"] for r in reports)}
    (out / "audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["MULTISCENE METRIC AUDIT", "=" * 60,
             "Kelvin: per-pixel (pred_z*std+mean - hr_z*std+mean), pooled by valid pixel count.",
             "Scene mae_kelvin/rmse_kelvin must match valid-pixel-weighted means from per_patch_metrics.csv.",
             f"Overall passed: {summary['passed']}", ""]
    for rep in reports:
        lines.append(f"--- {rep['role_dir']} passed={rep['passed']} ---")
        for c in rep["checks"]:
            if "scene_id" in c:
                lines.append(
                    f"  {c['scene_id']}/{c['method']}: mae_k={c['mae_kelvin_match']} rmse_k={c['rmse_kelvin_match']} "
                    f"z={c['mae_z_match']} linear~={c.get('kelvin_linear_check')}"
                )
    (out / "audit_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} passed={summary['passed']}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
