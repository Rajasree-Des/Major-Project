"""List job artifacts from manifest (no absolute paths)."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from backend.config import UI_RUNS_ROOT
from backend.jobs import get_job


def _guess_media(path: str) -> str:
    media, _ = mimetypes.guess_type(path)
    return media or "application/octet-stream"


def list_job_artifacts(job_id: str) -> list[dict[str, Any]]:
    job = get_job(job_id)
    if not job:
        return []

    root_key = job.get("output_root_key") or "ui_runs"
    items: list[dict[str, Any]] = []

    if root_key == "ui_runs":
        run_dir = UI_RUNS_ROOT / job_id
        if not run_dir.is_dir():
            return items
        prefix = f"{job_id}/"

        stage = job.get("stage_artifacts") or {}
        for name, rel in stage.items():
            items.append(
                {
                    "name": f"stage_{name}",
                    "relative_path": prefix + rel.replace("\\", "/"),
                    "root_key": root_key,
                    "media_type": "image/png",
                }
            )

        for patch in job.get("patches") or []:
            pid = patch.get("patch_id", "")
            for label, rel in (patch.get("colorization_outputs") or {}).items():
                r = str(rel).replace("\\", "/")
                items.append(
                    {
                        "name": f"{pid}_{label}",
                        "relative_path": prefix + r if not r.startswith(job_id) else r,
                        "root_key": root_key,
                        "media_type": "image/png",
                    }
                )
            tk = patch.get("thermal_kelvin_file")
            if tk:
                r = str(tk).replace("\\", "/")
                items.append(
                    {
                        "name": f"{pid}_thermal_kelvin",
                        "relative_path": prefix + r,
                        "root_key": root_key,
                        "media_type": "application/octet-stream",
                    }
                )
            vm = patch.get("valid_mask_file")
            if vm:
                r = str(vm).replace("\\", "/")
                items.append(
                    {
                        "name": f"{pid}_valid_mask",
                        "relative_path": prefix + r,
                        "root_key": root_key,
                        "media_type": "application/octet-stream",
                    }
                )

        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            items.append(
                {
                    "name": "manifest",
                    "relative_path": f"{job_id}/manifest.json",
                    "root_key": root_key,
                    "media_type": "application/json",
                }
            )
        log_path = run_dir / "job.log"
        if log_path.is_file():
            items.append(
                {
                    "name": "job_log",
                    "relative_path": f"{job_id}/job.log",
                    "root_key": root_key,
                    "media_type": "text/plain",
                }
            )
    else:
        manifest_path = Path(job.get("source_manifest") or "")
        for patch in job.get("patches") or []:
            for label, rel in (patch.get("colorization_outputs") or {}).items():
                r = str(rel).replace("\\", "/")
                items.append(
                    {
                        "name": f"{patch.get('patch_id')}_{label}",
                        "relative_path": r,
                        "root_key": root_key,
                        "media_type": _guess_media(r),
                    }
                )

    return items


def read_job_logs(job_id: str) -> list[str]:
    job = get_job(job_id)
    if job and job.get("logs"):
        return list(job["logs"])
    log_file = UI_RUNS_ROOT / job_id / "job.log"
    if log_file.is_file():
        return log_file.read_text(encoding="utf-8").splitlines()
    return []
