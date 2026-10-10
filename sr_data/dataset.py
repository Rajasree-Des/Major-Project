"""PyTorch Dataset for synthetic thermal super-resolution pairs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loader.thermal_dataset import ThermalSatelliteDataset
from sr_data.degradation import degrade_thermal_patch
from sr_data.metadata import DegradationConfig, SRDatasetMetadata

logger = logging.getLogger(__name__)


class ThermalSRDataset(Dataset):
    """LR/HR thermal pairs via on-the-fly or precomputed degradation."""

    def __init__(
        self,
        processed_root: Path,
        *,
        degradation: DegradationConfig | None = None,
        scene_ids: Sequence[str] | None = None,
        patch_indices: Sequence[int] | None = None,
        precomputed_dir: Path | None = None,
    ) -> None:
        """
        Args:
            processed_root: Processed scene directory or ``dataset/processed`` root.
            degradation: Synthetic degradation config (on-the-fly mode).
            precomputed_dir: If set, load ``lr/`` and ``hr/`` NumPy arrays from disk.
        """
        self.processed_root = processed_root.resolve()
        self.degradation = degradation or DegradationConfig()
        self.precomputed_dir = precomputed_dir.resolve() if precomputed_dir else None

        self.base_dataset = ThermalSatelliteDataset(
            self.processed_root,
            scene_ids=scene_ids,
            patch_indices=patch_indices,
            return_mask=True,
        )

        sample_meta = next(iter(self.base_dataset.metadata_by_scene.values()))
        hr_size = sample_meta.patch_size
        if hr_size % self.degradation.scale != 0:
            raise ValueError(
                f"HR patch size {hr_size} not divisible by scale {self.degradation.scale}"
            )
        lr_size = hr_size // self.degradation.scale

        self.sr_metadata = SRDatasetMetadata(
            degradation=self.degradation,
            hr_patch_size=hr_size,
            lr_patch_size=lr_size,
        )

        if self.precomputed_dir is not None:
            logger.info("ThermalSRDataset using precomputed arrays from %s", self.precomputed_dir)

    @property
    def metadata(self) -> SRDatasetMetadata:
        return self.sr_metadata

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _load_precomputed(
        self,
        patch_key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lr_path = self.precomputed_dir / "lr" / f"patch_{patch_key}.npy"
        hr_path = self.precomputed_dir / "hr" / f"patch_{patch_key}.npy"
        lr_mask_path = self.precomputed_dir / "lr_mask" / f"patch_{patch_key}.npy"
        hr_mask_path = self.precomputed_dir / "hr_mask" / f"patch_{patch_key}.npy"
        for path in (lr_path, hr_path, lr_mask_path, hr_mask_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing precomputed file: {path}")
        return (
            np.load(lr_path),
            np.load(hr_path),
            np.load(lr_mask_path).astype(bool),
            np.load(hr_mask_path).astype(bool),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        base = self.base_dataset[index]
        patch_key = str(base["patch_id"])
        scene_id = str(base["scene_id"])

        hr_t = base["thermal"].squeeze(0).numpy()
        hr_mask = base["valid_mask"].squeeze(0).numpy().astype(bool)

        if self.precomputed_dir is not None:
            lr, hr, lr_mask, hr_mask_out = self._load_precomputed(patch_key)
        else:
            scene_meta = self.base_dataset.get_scene_metadata(scene_id)
            hr, lr, hr_mask_out, lr_mask = degrade_thermal_patch(
                hr_t,
                hr_mask,
                scene_meta.thermal_norm,
                self.degradation,
                patch_key=patch_key,
            )

        return {
            "lr": torch.from_numpy(lr).unsqueeze(0).to(dtype=torch.float32),
            "hr": torch.from_numpy(hr).unsqueeze(0).to(dtype=torch.float32),
            "lr_valid_mask": torch.from_numpy(lr_mask.astype(np.bool_)).unsqueeze(0),
            "hr_valid_mask": torch.from_numpy(hr_mask_out.astype(np.bool_)).unsqueeze(0),
            "patch_id": patch_key,
            "scene_id": scene_id,
        }
