"""Read-only structured access to frozen evaluation results.json files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from backend.config import PROJECT_ROOT

DISCLAIMER = (
    "Synthetic ×2 reconstruction against the 30 m Landsat thermal reference; "
    "not evidence of real 15 m thermal accuracy."
)

FROZEN_PATHS: dict[tuple[str, str], Path] = {
    ("multiscene", "validation"): PROJECT_ROOT
    / "outputs/evaluation/multiscene_qa_v1/validation/results.json",
    ("multiscene", "test_locked"): PROJECT_ROOT
    / "outputs/evaluation/multiscene_qa_v1/test_locked/results.json",
    ("physics", "validation"): PROJECT_ROOT
    / "outputs/evaluation/physics_refinement_v1/validation/results.json",
    ("physics", "test_locked"): PROJECT_ROOT
    / "outputs/evaluation/physics_refinement_v1/test_locked/results.json",
}

Experiment = Literal["multiscene", "physics"]
Role = Literal["validation", "test_locked"]


@lru_cache(maxsize=8)
def _load_raw(experiment: str, role: str) -> dict[str, Any] | None:
    path = FROZEN_PATHS.get((experiment, role))
    if not path or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _methods_from_per_scene(per_scene: list[dict[str, Any]]) -> list[str]:
    if not per_scene:
        return []
    m = per_scene[0].get("metrics") or {}
    return sorted(m.keys())


def get_frozen_results(experiment: Experiment, role: Role) -> dict[str, Any]:
    raw = _load_raw(experiment, role)
    if raw is None:
        raise FileNotFoundError(f"Frozen results not found for {experiment}/{role}")

    per_scene = raw.get("per_scene") or []
    payload: dict[str, Any] = {
        "experiment": experiment,
        "role": role,
        "locked": role == "test_locked",
        "disclaimer": DISCLAIMER,
        "source_path": str(FROZEN_PATHS[(experiment, role)].relative_to(PROJECT_ROOT)),
        "scene_ids": raw.get("scene_ids") or [s.get("scene_id") for s in per_scene],
        "num_patches": raw.get("num_patches"),
        "patch_selection": raw.get("patch_selection"),
        "per_scene": per_scene,
        "aggregate": raw.get("aggregate"),
        "methods": _methods_from_per_scene(per_scene),
    }
    if "patch_distribution" in raw:
        payload["patch_distribution"] = raw["patch_distribution"]
    if "runtime" in raw:
        payload["runtime"] = raw["runtime"]
    return payload
