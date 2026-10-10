"""CLI guards for physics refinement evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.evaluate_physics_refinement import load_frozen_beta


def test_load_frozen_beta_from_method_txt(tmp_path: Path):
    root = tmp_path / "exp"
    root.mkdir()
    (root / "METHOD.txt").write_text(json.dumps({"beta": 1.0}), encoding="utf-8")
    assert load_frozen_beta(root, None) == 1.0
    assert load_frozen_beta(root, 0.5) == 0.5
