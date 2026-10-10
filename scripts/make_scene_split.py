"""Write the fixed, diversity-stratified scene-level split for dataset/processed_qa (qa_v1).

    py -m scripts.make_scene_split
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dataset_loader.split import SceneLevelSplit, list_processed_scene_ids, save_scene_split_manifest

SCENES = {
    "LC09_L2SP_143050_20240327_02_T1": ("train", "Southern Andhra Pradesh (original reference scene)",
                                        "dry tropical plains/coast, March"),
    "LC09_L2SP_150041_20250416_02_T1": ("train", "Thar Desert, Rajasthan", "hot arid desert, April"),
    "LC09_L2SP_144041_20231213_02_T1": ("train", "Indo-Gangetic Plain, Uttar Pradesh",
                                        "winter irrigated cropland and cities, December"),
    "LC08_L2SP_146039_20241119_02_T1": ("train", "Himalayan foothills, Uttarakhand",
                                        "mountain terrain with snow/ice, November"),
    "LC08_L2SP_144053_20250225_02_T1": ("validation", "Kerala coast / Western Ghats",
                                        "humid tropical coast and forest, February"),
    "LC08_L2SP_144048_20240427_02_T1": ("test", "Deccan Plateau, Telangana", "semi-arid plateau and city, April"),
    "LC09_L2SP_137042_20230110_02_T1": ("test", "Brahmaputra valley, Assam",
                                        "humid winter floodplain, January (UTM 46N)"),
}

RATIONALE = (
    "Manual diversity-stratified split (no random seed). Train spans the thermal extremes (hot desert, "
    "snowy mountains), a winter plain and the original reference scene, which a preliminary checkpoint has "
    "already seen and therefore must not be in validation or test. Validation is one humid coastal scene. "
    "Test holds one climate close to training (semi-arid Deccan) and one farther away (humid Assam floodplain, "
    "different UTM zone), and mixes Landsat 8 and 9. No scene footprints overlap."
)
REJECTED = {
    "hard_ood": "train=Reference,Thar,IGP,Assam; val=Deccan; test=Kerala,Himalaya. Harsher test, but training "
                "would contain no mountain terrain or snow.",
    "seeded_hash_seed42": "dataset_loader.split.split_scenes_deterministic(seed=42): train=Thar,IGP,Reference,"
                          "Assam; val=Himalaya; test=Kerala,Deccan. Train is all Landsat 9 and val/test all "
                          "Landsat 8, confounding sensor with partition.",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("dataset/processed_qa"))
    p.add_argument("--output", type=Path, default=Path("dataset/splits/scene_split_qa_v1.json"))
    p.add_argument("--copy-to", type=Path, default=Path("outputs/evaluation/multiscene_qa_v1/split_manifest.json"))
    args = p.parse_args(argv)

    if args.output.exists():
        print(f"{args.output} already exists; refusing to overwrite a frozen split.")
        return 1
    available = list_processed_scene_ids(args.root)
    roles = {r: tuple(s for s, v in SCENES.items() if v[0] == r) for r in ("train", "validation", "test")}
    split = SceneLevelSplit(train=roles["train"], validation=roles["validation"], test=roles["test"],
                            label="qa_v1 diversity-stratified scene-level split")
    scenes = {}
    for sid, (role, region, climate) in SCENES.items():
        meta = json.loads((args.root / sid / "metadata.json").read_text(encoding="utf-8"))
        scenes[sid] = {
            "role": role, "region": region, "climate_season": climate,
            "sensor": {"LC08": "Landsat 8", "LC09": "Landsat 9"}[sid[:4]],
            "wrs_path_row": f"{sid[10:13]}/{sid[13:16]}", "acquisition_date": sid[17:25],
            "patches": meta["patches_generated"], "crs": meta["crs"],
        }
    save_scene_split_manifest(
        args.output, split, seed=None, available_scene_ids=available,
        extra={
            "dataset_root": str(args.root), "processing_version": "qa_v1",
            "rationale": RATIONALE, "rejected_alternatives": REJECTED, "scenes": scenes,
            "rules": [
                "All patches of a scene belong to exactly one role.",
                "Test scenes are locked: never used for model selection, early stopping or tuning.",
                "Normalization statistics are per scene and never pooled across roles.",
                "Random patch splits must not be used for generalization claims.",
            ],
        },
    )
    args.copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.output, args.copy_to)
    print(json.dumps({r: list(v) for r, v in roles.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
