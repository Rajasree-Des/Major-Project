"""Background pipeline jobs (one at a time, unique output dirs)."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.config import UI_RUNS_ROOT
from pipeline.frozen_config import load_frozen_config

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "completed", "failed"]

_lock = threading.Lock()
_active: dict[str, dict[str, Any]] = {}
_model_cache: tuple[Any, dict] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_model():
    global _model_cache
    if _model_cache is None:
        import torch

        from pipeline.model_loader import load_swinir_model
        from training.repro import set_seed

        cfg = load_frozen_config()
        set_seed(cfg.eval_seed)
        torch.set_grad_enabled(False)
        _model_cache = load_swinir_model(cfg)
    return _model_cache


def _legacy_runs() -> list[dict[str, Any]]:
    from backend.config import ARTIFACT_ROOTS

    legacy: list[dict[str, Any]] = []
    for run_id, key in [
        ("integrated_pipeline_v1", "integrated_pipeline_v1"),
        ("guided_colorization_v1", "guided_colorization_v1"),
    ]:
        root = ARTIFACT_ROOTS.get(key)
        if not root or not (root / "manifest.json").is_file():
            continue
        data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        data["run_id"] = run_id
        data["status"] = "completed"
        data["output_root_key"] = key
        data["kind"] = "historical_batch_run"
        legacy.append(data)
    return legacy


def _merge_active_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with _lock:
        for job_id, job in _active.items():
            if job.get("persisted"):
                continue
            runs.insert(
                0,
                {
                    "run_id": job_id,
                    "status": job["status"],
                    "created_at": job.get("created_at"),
                    "scene_id": job.get("scene_id"),
                    "patch_ids": job.get("patch_ids"),
                    "mode": job.get("mode"),
                    "runtime_s": job.get("runtime_s"),
                    "error": job.get("error"),
                    "in_memory_only": True,
                },
            )
    return runs


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = _legacy_runs()
    if UI_RUNS_ROOT.is_dir():
        for run_dir in sorted(UI_RUNS_ROOT.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    data["run_id"] = run_dir.name
                    data["output_root_key"] = "ui_runs"
                    data["output_root_relative"] = f"ui_runs/{run_dir.name}"
                    runs.append(data)
                except json.JSONDecodeError:
                    runs.append({"run_id": run_dir.name, "status": "unknown", "error": "invalid manifest"})
    return _merge_active_runs(runs)


def _load_legacy_manifest(run_id: str) -> dict[str, Any] | None:
    from backend.config import ARTIFACT_ROOTS

    key_map = {
        "integrated_pipeline_v1": "integrated_pipeline_v1",
        "guided_colorization_v1": "guided_colorization_v1",
    }
    key = key_map.get(run_id)
    if not key:
        return None
    root = ARTIFACT_ROOTS.get(key)
    if not root:
        return None
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["run_id"] = run_id
    data["output_root_key"] = key
    data["status"] = data.get("status", "completed")
    return data


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        if job_id in _active:
            return dict(_active[job_id])
    legacy = _load_legacy_manifest(job_id)
    if legacy:
        return legacy
    manifest = UI_RUNS_ROOT / job_id / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["run_id"] = job_id
        data.setdefault("output_root_key", "ui_runs")
        log_file = UI_RUNS_ROOT / job_id / "job.log"
        if log_file.is_file() and not data.get("logs"):
            data["logs"] = log_file.read_text(encoding="utf-8").splitlines()
        return data
    return None


def _append_log(job: dict[str, Any], line: str) -> None:
    job["logs"].append(f"[{_utc_now()}] {line}")
    job["progress_message"] = line


def _set_stage(job: dict[str, Any], stage: str, message: str | None = None) -> None:
    job["stage"] = stage
    if message:
        _append_log(job, message)
    else:
        job["progress_message"] = stage.replace("_", " ").title()


def is_inference_busy() -> bool:
    with _lock:
        return any(j["status"] in ("queued", "running") for j in _active.values())


def _run_job_body(job_id: str, job: dict[str, Any]) -> None:
    from backend.inference_helpers import compute_sr_comparison
    from backend.stage_outputs import render_stage_pngs
    from pipeline.integrated import run_integrated_patch, run_validation_sample

    cfg = load_frozen_config()
    out = UI_RUNS_ROOT / job_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stage_artifacts: dict[str, str] = {}
    try:
        job["status"] = "running"
        _set_stage(job, "validating", "Validating scene and patch against QA manifest...")
        _set_stage(job, "loading_model", "Loading SwinIR checkpoint (CPU)...")
        model, model_info = _get_model()
        patches_out: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []

        if job["mode"] == "single":
            _set_stage(job, "preparing_input", "Preparing synthetic LR thermal input...")
            _set_stage(job, "swinir_inference", "Running SwinIR ×2 inference...")
            _set_stage(job, "consistency_refinement", "Applying consistency refinement (β=1.0)...")
            _set_stage(job, "colorization", "Generating colorization outputs...")
            entry = run_integrated_patch(
                cfg, job["scene_id"], job["patch_id"], model, out, model_info=model_info
            )
            patches_out.append(entry)
            _set_stage(job, "computing_metrics", "Rendering stage previews and computing metrics...")
            stage_artifacts = render_stage_pngs(
                cfg, model, job["scene_id"], job["patch_id"], out
            )
            comparisons.append(compute_sr_comparison(cfg, model, job["scene_id"], job["patch_id"]))
        else:
            _set_stage(
                job,
                "preparing_input",
                f"Validation sample: {job['num_patches']} patches, seed {job['seed']}...",
            )
            _set_stage(job, "swinir_inference", "Running integrated pipeline on sample patches...")
            patches_out = run_validation_sample(
                cfg, model, out, num_patches=job["num_patches"], seed=job["seed"], model_info=model_info
            )
            _set_stage(job, "computing_metrics", "Computing per-patch metrics vs HR reference...")
            for entry in patches_out:
                comparisons.append(
                    compute_sr_comparison(cfg, model, entry["scene_id"], entry["patch_id"])
                )

        _set_stage(job, "saving_artifacts", "Writing manifest and logs...")
        elapsed = time.perf_counter() - t0
        manifest = {
            "run_id": job_id,
            "status": "completed",
            "stage": "completed",
            "created_at": job["created_at"],
            "completed_at": _utc_now(),
            "pipeline_version": cfg.pipeline_version,
            "mode": job["mode"],
            "scene_id": job.get("scene_id"),
            "patch_id": job.get("patch_id"),
            "patch_ids": [p["patch_id"] for p in patches_out],
            "num_patches": len(patches_out),
            "seed": job.get("seed"),
            "refinement_beta": cfg.refinement_beta,
            "degradation": cfg.degradation.to_dict(),
            "model": model_info,
            "runtime_s": elapsed,
            "patches": patches_out,
            "sr_comparisons": comparisons,
            "stage_artifacts": stage_artifacts,
            "output_root_key": "ui_runs",
            "limitations": [
                "Synthetic ×2 SR vs 30 m reference; not real 15 m thermal accuracy.",
                "Per-run metrics are not locked multiscene evaluation results.",
            ],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out / "job.log").write_text("\n".join(job["logs"]), encoding="utf-8")
        job["status"] = "completed"
        job["stage"] = "completed"
        job["runtime_s"] = elapsed
        job["manifest"] = manifest
        job["stage_artifacts"] = stage_artifacts
        job["persisted"] = True
        _append_log(job, f"Completed in {elapsed:.1f}s")
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job["status"] = "failed"
        job["stage"] = "failed"
        job["error"] = str(exc)
        _append_log(job, f"FAILED: {exc}")
        (out / "job.log").write_text("\n".join(job["logs"]), encoding="utf-8")
        fail_manifest = {
            "run_id": job_id,
            "status": "failed",
            "error": str(exc),
            "created_at": job["created_at"],
            "logs": job["logs"],
        }
        (out / "manifest.json").write_text(json.dumps(fail_manifest, indent=2), encoding="utf-8")
        job["persisted"] = True


def submit_job(
    *,
    mode: Literal["single", "validation_sample"],
    scene_id: str | None = None,
    patch_id: str | None = None,
    num_patches: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    with _lock:
        for j in _active.values():
            if j["status"] in ("queued", "running"):
                raise RuntimeError("Another job is already running. Wait for it to finish.")

    cfg = load_frozen_config()
    if mode == "single":
        if not scene_id or not patch_id:
            raise ValueError("scene_id and patch_id required for single mode")
        from pipeline.patch_refs import resolve_patch

        resolve_patch(cfg.processed_root, scene_id, patch_id, degradation=cfg.degradation)
    else:
        scene_id = scene_id or "LC08_L2SP_144053_20250225_02_T1"

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job: dict[str, Any] = {
        "run_id": job_id,
        "status": "queued",
        "created_at": _utc_now(),
        "mode": mode,
        "scene_id": scene_id,
        "patch_id": patch_id,
        "patch_ids": [patch_id] if patch_id else None,
        "num_patches": num_patches if mode == "validation_sample" else 1,
        "seed": seed,
        "stage": "queued",
        "progress_message": "Queued",
        "logs": [],
        "error": None,
        "runtime_s": None,
        "persisted": False,
    }
    with _lock:
        _active[job_id] = job

    thread = threading.Thread(target=_run_job_body, args=(job_id, job), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}
