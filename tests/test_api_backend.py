"""FastAPI backend smoke tests (no full inference by default)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "scene_count" in data
    assert "inference_busy" in data


def test_scenes_match_manifest(client: TestClient):
    r = client.get("/api/scenes")
    assert r.status_code == 200
    scenes = r.json()
    if scenes:
        assert len(scenes) == 7
        assert all("scene_id" in s for s in scenes)


def test_patch_list_validation_scene(client: TestClient):
    scene = "LC08_L2SP_144053_20250225_02_T1"
    r = client.get(f"/api/scenes/{scene}/patches", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    assert len(body["patch_ids"]) == 5


def test_artifact_path_traversal_blocked(client: TestClient):
    r = client.get("/api/artifacts/ui_runs/../../dataset/processed_qa/processing_manifest.json")
    assert r.status_code in (403, 404)


def test_frozen_eval_excerpts(client: TestClient):
    r = client.get("/api/frozen-evaluations")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_frozen_results_multiscene_locked(client: TestClient):
    r = client.get("/api/results/frozen", params={"experiment": "multiscene", "role": "test_locked"})
    assert r.status_code == 200
    data = r.json()
    assert data["locked"] is True
    assert "LC08_L2SP_144048_20240427_02_T1" in data["scene_ids"]
    assert "LC09_L2SP_137042_20230110_02_T1" in data["scene_ids"]
    assert "bicubic" in data["methods"]


def test_history_endpoint(client: TestClient):
    r = client.get("/api/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.slow
def test_single_patch_job_cpu(client: TestClient):
    """Full CPU inference (~15–60 s); run with pytest -m slow."""
    r = client.post(
        "/api/jobs",
        json={
            "mode": "single",
            "scene_id": "LC08_L2SP_144053_20250225_02_T1",
            "patch_id": "000096",
        },
    )
    if r.status_code == 409:
        pytest.skip("Another job running")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    import time

    for _ in range(120):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(2)
    assert st["status"] == "completed", st.get("error")
    assert st.get("patches") or st.get("manifest")
    assert st.get("stage_artifacts")
    arts = client.get(f"/api/jobs/{job_id}/artifacts").json()["artifacts"]
    assert any("stage_bicubic" in a["name"] for a in arts)
    from backend.config import UI_RUNS_ROOT

    assert (UI_RUNS_ROOT / job_id / "stages" / "refined.png").is_file()
