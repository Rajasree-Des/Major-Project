"""Validate and resolve paths under approved artifact roots."""

from __future__ import annotations

from pathlib import Path

from backend.config import ARTIFACT_ROOTS, PROJECT_ROOT


class PathSecurityError(ValueError):
    pass


def resolve_artifact(root_key: str, relative: str) -> Path:
    if root_key not in ARTIFACT_ROOTS:
        raise PathSecurityError(f"Unknown artifact root: {root_key}")
    root = ARTIFACT_ROOTS[root_key].resolve()
    rel = relative.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise PathSecurityError("Path traversal not allowed")
    full = (root / rel).resolve()
    if not str(full).startswith(str(root)):
        raise PathSecurityError("Resolved path outside approved root")
    return full


def assert_under_project(path: Path) -> Path:
    resolved = path.resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
        raise PathSecurityError("Path outside project")
    return resolved
