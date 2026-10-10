"""Train/validation/test splitting utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dataset_loader.metadata import discover_scene_directories

DEVELOPMENT_SPLIT_LABEL = (
    "DEVELOPMENT SPLIT — NOT FINAL SCIENTIFIC EVALUATION "
    "(random patch split within scene; spatial leakage likely)"
)

SINGLE_SCENE_SPLIT_WARNING = (
    "Only one processed scene is available. A true scene-independent "
    "train/validation/test split is NOT yet possible. Use scene-level "
    "splitting once multiple scenes exist under dataset/processed/."
)


@dataclass(frozen=True)
class SceneLevelSplit:
    """Scene IDs assigned to each split."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    label: str = "scene-level split"


@dataclass(frozen=True)
class DevelopmentPatchSplit:
    """Patch indices for temporary single-scene experimentation."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    label: str = DEVELOPMENT_SPLIT_LABEL
    warning: str = SINGLE_SCENE_SPLIT_WARNING


def list_processed_scene_ids(processed_root) -> list[str]:
    """Return scene directory names under the processed root."""
    from pathlib import Path

    root = Path(processed_root).resolve()
    return [p.name for p in discover_scene_directories(root)]


def split_scenes_deterministic(
    scene_ids: Sequence[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SceneLevelSplit:
    """Deterministic scene-level split for multiple scenes."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    unique = sorted(set(scene_ids))
    if len(unique) < 2:
        raise ValueError(SINGLE_SCENE_SPLIT_WARNING)

    ordered = sorted(unique, key=lambda s: _hash_seed(f"{seed}:{s}"))
    n = len(ordered)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n - n_val - n_test)

    train = tuple(ordered[:n_train])
    val = tuple(ordered[n_train : n_train + n_val])
    test = tuple(ordered[n_train + n_val :])
    return SceneLevelSplit(train=train, validation=val, test=test)


def development_patch_split(
    num_patches: int,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DevelopmentPatchSplit:
    """Temporary patch-index split for development only (single-scene leakage risk)."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    if num_patches < 3:
        raise ValueError("Need at least 3 patches for a development split.")

    indices = list(range(num_patches))
    indices.sort(key=lambda i: _hash_seed(f"{seed}:patch:{i}"))

    n_train = int(num_patches * train_ratio)
    n_val = int(num_patches * val_ratio)
    n_test = num_patches - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        n_train = max(1, num_patches - 2)
        n_val = 1
        n_test = num_patches - n_train - n_val

    train_idx = tuple(indices[:n_train])
    val_idx = tuple(indices[n_train : n_train + n_val])
    test_idx = tuple(indices[n_train + n_val :])
    return DevelopmentPatchSplit(
        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx,
    )


SPLIT_ROLES = ("train", "validation", "test")


def require_single_scene_for_patch_split(scene_ids: Sequence[str]) -> None:
    """Refuse a random patch split on a multi-scene root, where it would mix scenes across partitions."""
    unique = sorted(set(scene_ids))
    if len(unique) > 1:
        raise ValueError(
            f"development_patch_split would mix {len(unique)} scenes across train/validation/test. "
            "Use a scene-level split manifest (dataset/splits/*.json) for multi-scene data."
        )


class SceneSplitError(ValueError):
    """Raised when a scene split is not a valid partition of the available scenes."""


def validate_scene_split(split: SceneLevelSplit, available_scene_ids: Sequence[str] | None = None) -> None:
    """Require disjoint, duplicate-free roles that together cover exactly the available scenes.

    Raises:
        SceneSplitError: On any violation.
    """
    roles = {"train": split.train, "validation": split.validation, "test": split.test}
    problems: list[str] = []
    for role, ids in roles.items():
        dupes = sorted({s for s in ids if list(ids).count(s) > 1})
        if dupes:
            problems.append(f"duplicate IDs in {role}: {dupes}")
    for i, a in enumerate(SPLIT_ROLES):
        for b in SPLIT_ROLES[i + 1:]:
            shared = sorted(set(roles[a]) & set(roles[b]))
            if shared:
                problems.append(f"scenes in both {a} and {b}: {shared}")
    if not split.test:
        problems.append("test role is empty")
    if available_scene_ids is not None:
        assigned = set().union(*roles.values())
        available = set(available_scene_ids)
        if assigned - available:
            problems.append(f"unknown scene IDs: {sorted(assigned - available)}")
        if available - assigned:
            problems.append(f"unassigned scenes: {sorted(available - assigned)}")
    if problems:
        raise SceneSplitError("; ".join(problems))


def save_scene_split_manifest(
    path: Path,
    split: SceneLevelSplit,
    *,
    seed: int | None,
    available_scene_ids: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and write a scene split manifest; returns the written dict."""
    validate_scene_split(split, available_scene_ids)
    manifest = {
        "split_type": "scene_level",
        "label": split.label,
        "seed": seed,
        "roles": {"train": list(split.train), "validation": list(split.validation), "test": list(split.test)},
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_scene_split_manifest(path: Path, available_scene_ids: Sequence[str] | None = None) -> SceneLevelSplit:
    """Read and validate a scene split manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("split_type") != "scene_level":
        raise SceneSplitError(f"{path} is not a scene-level split manifest")
    roles = data["roles"]
    split = SceneLevelSplit(
        train=tuple(roles["train"]),
        validation=tuple(roles["validation"]),
        test=tuple(roles["test"]),
        label=data.get("label", "scene-level split"),
    )
    validate_scene_split(split, available_scene_ids)
    return split


def scene_ids_for_role(split: SceneLevelSplit, role: str) -> tuple[str, ...]:
    if role not in SPLIT_ROLES:
        raise SceneSplitError(f"unknown role {role!r}; expected one of {SPLIT_ROLES}")
    return {"train": split.train, "validation": split.validation, "test": split.test}[role]


def _hash_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
