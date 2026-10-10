"""Check that every exported qa_v1 patch is finite exactly where the scene exclusion codes are 0.

    py -m scripts.verify_qa_patches --root dataset/processed_qa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("dataset/processed_qa"))
    args = p.parse_args(argv)

    all_ok = True
    for scene in sorted(d for d in args.root.iterdir() if (d / "metadata.json").is_file()):
        meta = json.loads((scene / "metadata.json").read_text(encoding="utf-8"))
        size = meta["patch_size"]
        codes = np.load(scene / "exclusion_codes.npz")["codes"]
        mismatched = []
        for c in meta["patch_coordinates"]:
            name = f"patch_{c['patch_id']:06d}.npy"
            thermal = np.load(scene / "thermal" / name)
            rgb = np.load(scene / "rgb" / name)
            expected = codes[c["row"]:c["row"] + size, c["col"]:c["col"] + size] == 0
            finite = np.isfinite(thermal) & np.all(np.isfinite(rgb), axis=-1)
            if not np.array_equal(finite, expected):
                mismatched.append(c["patch_id"])
        all_ok &= not mismatched
        print(f"{scene.name}: {len(meta['patch_coordinates'])} patches, mask mismatches {len(mismatched)}")
    print("ALL PATCH MASKS MATCH QA EXCLUSION CODES" if all_ok else "MISMATCH FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
