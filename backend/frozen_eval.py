"""Read-only summaries from frozen evaluation reports (no recomputation)."""

from __future__ import annotations

from pathlib import Path

from backend.config import PROJECT_ROOT

REPORTS = {
    "multiscene_validation": PROJECT_ROOT / "outputs/evaluation/multiscene_qa_v1/validation/report.txt",
    "multiscene_test_locked": PROJECT_ROOT / "outputs/evaluation/multiscene_qa_v1/test_locked/report.txt",
    "physics_validation": PROJECT_ROOT / "outputs/evaluation/physics_refinement_v1/validation/report.txt",
    "physics_test_locked": PROJECT_ROOT / "outputs/evaluation/physics_refinement_v1/test_locked/report.txt",
}


def frozen_evaluation_summaries() -> list[dict]:
    items = []
    for key, path in REPORTS.items():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        items.append(
            {
                "id": key,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "label": key.replace("_", " "),
                "excerpt_lines": text.splitlines()[:28],
                "is_locked_test": "test_locked" in key,
                "disclaimer": "Synthetic ×2 reconstruction against 30 m reference; not real 15 m thermal accuracy.",
            }
        )
    return items
