"""Load one DataLoader batch for smoke testing (no training)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_loader.thermal_dataset import ThermalSatelliteDataset

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def collate_samples(batch: list[dict]) -> dict:
    """Stack tensor fields into a batch."""
    out: dict = {
        "thermal": torch.stack([b["thermal"] for b in batch], dim=0),
        "rgb": torch.stack([b["rgb"] for b in batch], dim=0),
        "patch_id": [b["patch_id"] for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
    }
    if "valid_mask" in batch[0]:
        out["valid_mask"] = torch.stack([b["valid_mask"] for b in batch], dim=0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test PyTorch DataLoader on processed patches.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("dataset/processed"),
        help="Processed root or single scene directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    dataset = ThermalSatelliteDataset(args.processed_root.resolve(), return_mask=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )

    batch = next(iter(loader))
    print(f"Dataset length: {len(dataset)} scenes metadata: {list(dataset.metadata_by_scene.keys())}")
    print(f"thermal shape: {tuple(batch['thermal'].shape)} dtype: {batch['thermal'].dtype}")
    print(f"rgb shape:     {tuple(batch['rgb'].shape)} dtype: {batch['rgb'].dtype}")
    if "valid_mask" in batch:
        mask = batch["valid_mask"]
        valid_pct = mask.float().mean().item() * 100.0
        print(f"valid_mask shape: {tuple(mask.shape)} dtype: {mask.dtype}")
        print(f"batch mean valid pixel %: {valid_pct:.2f}%")

    meta = next(iter(dataset.metadata_by_scene.values()))
    print(
        "normalization (for denorm only, not applied in Dataset): "
        f"thermal mean={meta.thermal_norm.mean:.4f} std={meta.thermal_norm.std:.4f}"
    )
    logger.info("DataLoader batch smoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
