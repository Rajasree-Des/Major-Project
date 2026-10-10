"""ThermaScope FastAPI backend — local CPU inference and artifact serving."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.config import ARTIFACT_ROOTS, PROJECT_ROOT, PROCESSED_ROOT
from backend.artifacts import list_job_artifacts, read_job_logs
from backend.frozen_eval import frozen_evaluation_summaries
from backend.frozen_results import get_frozen_results
from backend.jobs import get_job, is_inference_busy, list_runs, submit_job
from backend.paths import PathSecurityError, resolve_artifact
from backend.previews import ensure_preview
from backend.scenes import (
    DEMO_PATCH_ID,
    DEMO_SCENE_ID,
    demo_patch_available,
    list_patch_ids,
    list_scenes,
    patch_metadata,
)
from pipeline.frozen_config import load_frozen_config

app = FastAPI(title="ThermaScope API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobSubmitBody(BaseModel):
    mode: Literal["single", "validation_sample"] = "single"
    scene_id: str | None = None
    patch_id: str | None = None
    num_patches: int = Field(default=4, ge=1, le=8)
    seed: int = 42


@app.get("/api/health")
def health() -> dict[str, Any]:
    cfg = load_frozen_config()
    pretrained = PROJECT_ROOT / "pretrained" / "swinir" / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    missing: list[str] = []
    if not PROCESSED_ROOT.is_dir():
        missing.append("dataset/processed_qa")
    if not pretrained.is_file():
        missing.append("pretrained/swinir checkpoint")
    return {
        "status": "ok" if not missing else "degraded",
        "project_root": str(PROJECT_ROOT),
        "processed_qa_available": PROCESSED_ROOT.is_dir(),
        "scene_count": len(list_scenes()) if PROCESSED_ROOT.is_dir() else 0,
        "pretrained_checkpoint_available": pretrained.is_file(),
        "frozen_refinement_beta": cfg.refinement_beta,
        "device": "cpu",
        "inference_busy": is_inference_busy(),
        "missing_requirements": missing,
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    scenes = list_scenes()
    runs = list_runs()
    legacy: list[dict[str, Any]] = []
    for name, key in [
        ("integrated_pipeline_v1", "integrated_pipeline_v1"),
        ("guided_colorization_v1", "guided_colorization_v1"),
    ]:
        manifest = ARTIFACT_ROOTS[key] / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            legacy.append(
                {
                    "label": name,
                    "output_root_key": key,
                    "created": data.get("runtime_s"),
                    "patches": data.get("patches") or data.get("num_patches"),
                    "kind": "historical_batch_run",
                }
            )
    latest = runs[0] if runs else (legacy[0] if legacy else None)
    total_patches = sum(int(s.get("patches_after_qa") or 0) for s in scenes)
    cfg = load_frozen_config()
    return {
        "project_name": "ThermaScope",
        "tagline": "Thermal Satellite Intelligence & Super-Resolution Studio",
        "subtitle": "Physics-aware deep learning for Landsat thermal SR and guided colorization",
        "scenes_prepared": len(scenes),
        "total_patches": total_patches,
        "demo_scene_id": DEMO_SCENE_ID,
        "demo_patch_id": DEMO_PATCH_ID,
        "demo_patch_available": demo_patch_available(),
        "scenes": scenes,
        "checkpoint": {
            "variant": cfg.swinir_variant,
            "pretrained": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
            "available": (PROJECT_ROOT / "pretrained" / "swinir" / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth").is_file(),
        },
        "frozen_config_summary": {
            "refinement_beta": cfg.refinement_beta,
            "degradation": cfg.degradation.to_dict(),
            "colormap_kelvin": [cfg.colormap.vmin_kelvin, cfg.colormap.vmax_kelvin],
            "eval_seed": cfg.eval_seed,
            "locked_note": "Scientific parameters are frozen; UI does not expose tuning controls.",
        },
        "latest_run": latest,
        "historical_runs": legacy,
        "frozen_evaluations_available": len(frozen_evaluation_summaries()) > 0,
    }


@app.get("/api/scenes")
def api_list_scenes() -> list[dict[str, Any]]:
    return list_scenes()


@app.get("/api/scenes/{scene_id}/patches")
def api_list_patches(
    scene_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    try:
        return list_patch_ids(scene_id, limit=limit, offset=offset)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/scenes/{scene_id}/patches/{patch_id}")
def api_patch_meta(scene_id: str, patch_id: str) -> dict[str, Any]:
    try:
        return patch_metadata(scene_id, patch_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/scenes/{scene_id}/patches/{patch_id}/preview/{kind}")
def api_patch_preview(scene_id: str, patch_id: str, kind: str):
    if kind not in ("thermal_colormap", "rgb"):
        raise HTTPException(400, "kind must be thermal_colormap or rgb")
    try:
        path = ensure_preview(scene_id, patch_id, kind)
    except Exception as e:
        raise HTTPException(404, str(e)) from e
    return FileResponse(path, media_type="image/png")


@app.get("/api/frozen-evaluations")
def api_frozen_eval() -> list[dict]:
    return frozen_evaluation_summaries()


@app.get("/api/results/frozen")
def api_frozen_results(
    experiment: Literal["multiscene", "physics"] = Query(default="multiscene"),
    role: Literal["validation", "test_locked"] = Query(default="test_locked"),
) -> dict[str, Any]:
    try:
        return get_frozen_results(experiment, role)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/history")
def api_history() -> list[dict[str, Any]]:
    return list_runs()


@app.get("/api/runs")
def api_runs() -> list[dict[str, Any]]:
    return list_runs()


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict[str, Any]:
    job = get_job(run_id)
    if not job:
        raise HTTPException(404, "Run not found")
    return job


@app.post("/api/jobs")
def api_submit_job(body: JobSubmitBody) -> dict[str, Any]:
    try:
        return submit_job(
            mode=body.mode,
            scene_id=body.scene_id,
            patch_id=body.patch_id,
            num_patches=body.num_patches,
            seed=body.seed,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/logs")
def api_job_logs(job_id: str) -> dict[str, Any]:
    if not get_job(job_id):
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "lines": read_job_logs(job_id)}


@app.get("/api/jobs/{job_id}/artifacts")
def api_job_artifacts(job_id: str) -> dict[str, Any]:
    if not get_job(job_id):
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "artifacts": list_job_artifacts(job_id)}


@app.get("/api/artifacts/{root_key}/{path:path}")
def serve_artifact(root_key: str, path: str):
    try:
        full = resolve_artifact(root_key, path)
    except PathSecurityError as e:
        raise HTTPException(403, str(e)) from e
    if not full.is_file():
        raise HTTPException(404, "File not found")
    media, _ = mimetypes.guess_type(str(full))
    return FileResponse(full, media_type=media or "application/octet-stream")


def run() -> None:
    import uvicorn

    from backend.config import API_HOST, API_PORT

    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    run()
