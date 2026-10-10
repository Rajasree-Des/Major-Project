"""Filesystem helpers for preprocessing outputs."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Create ``path`` as a directory, removing blocking placeholder files."""
    path = path.resolve()
    for ancestor in [path, *path.parents]:
        if ancestor == ancestor.parent:
            break
        if ancestor.exists() and ancestor.is_file():
            logger.warning("Removing file blocking directory: %s", ancestor)
            ancestor.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path
