"""Record or compare a fingerprint of protected data, checkpoints and evaluation outputs.

Read-only: hashes raw scene bands and checkpoints, and summarises processed patch
directories by file count, total size and newest modification time.

    py -m scripts.integrity_snapshot --save outputs/dataset_expansion/integrity_before.json
    py -m scripts.integrity_snapshot --compare outputs/dataset_expansion/integrity_before.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASHED_DIRS = ("dataset/raw", "outputs/models", "outputs/evaluation", "pretrained")
SUMMARISED_DIRS = ("dataset/processed",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot() -> dict:
    files: dict[str, dict] = {}
    for rel in HASHED_DIRS:
        root = PROJECT_ROOT / rel
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            stat = path.stat()
            files[path.relative_to(PROJECT_ROOT).as_posix()] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }

    summaries: dict[str, dict] = {}
    for rel in SUMMARISED_DIRS:
        root = PROJECT_ROOT / rel
        if not root.exists():
            continue
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            entries = [p for p in scene_dir.rglob("*") if p.is_file()]
            summaries[scene_dir.relative_to(PROJECT_ROOT).as_posix()] = {
                "file_count": len(entries),
                "total_size": sum(p.stat().st_size for p in entries),
                "newest_mtime_ns": max((p.stat().st_mtime_ns for p in entries), default=0),
            }
    return {"hashed_files": files, "directory_summaries": summaries}


def compare(before: dict, after: dict) -> list[str]:
    problems: list[str] = []
    for section in ("hashed_files", "directory_summaries"):
        old, new = before[section], after[section]
        for key in sorted(old):
            if key not in new:
                problems.append(f"missing: {key}")
            elif old[key] != new[key]:
                problems.append(f"changed: {key}")
        problems.extend(f"added: {key}" for key in sorted(set(new) - set(old)))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", type=Path)
    group.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)

    current = snapshot()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Saved {len(current['hashed_files'])} hashed files and "
              f"{len(current['directory_summaries'])} directory summaries to {args.save}")
        return 0

    before = json.loads(args.compare.read_text(encoding="utf-8"))
    problems = compare(before, current)
    if problems:
        print("INTEGRITY CHANGES DETECTED:")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"UNCHANGED: {len(current['hashed_files'])} hashed files, "
          f"{len(current['directory_summaries'])} processed scene directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
