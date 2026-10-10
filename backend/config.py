"""Backend configuration and safe path roots."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed_qa"
SPLIT_PATH = PROJECT_ROOT / "dataset" / "splits" / "scene_split_qa_v1.json"
PROCESSING_MANIFEST = PROCESSED_ROOT / "processing_manifest.json"

UI_RUNS_ROOT = PROJECT_ROOT / "outputs" / "ui_runs"
UI_CACHE_ROOT = PROJECT_ROOT / "outputs" / "ui_cache"

# Read-only artifact roots (relative keys for API)
ARTIFACT_ROOTS: dict[str, Path] = {
    "ui_runs": UI_RUNS_ROOT,
    "ui_cache": UI_CACHE_ROOT,
    "integrated_pipeline_v1": PROJECT_ROOT / "outputs" / "integrated_pipeline_v1",
    "guided_colorization_v1": PROJECT_ROOT / "outputs" / "colorization" / "guided_colorization_v1",
    "previews": PROJECT_ROOT / "outputs" / "previews",
}

API_HOST = "127.0.0.1"
API_PORT = 8000
