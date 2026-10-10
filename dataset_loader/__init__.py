"""PyTorch dataset layer for preprocessed Landsat thermal/RGB patches.

Import ``ThermalSatelliteDataset`` from ``dataset_loader.thermal_dataset`` directly
when needed so lightweight API routes avoid loading PyTorch at startup.
"""

from dataset_loader.metadata import ProcessedSceneMetadata, load_scene_metadata
from dataset_loader.physical import denormalize_from_scene_metadata, denormalize_thermal_kelvin
from dataset_loader.split import (
    DEVELOPMENT_SPLIT_LABEL,
    SINGLE_SCENE_SPLIT_WARNING,
    development_patch_split,
    split_scenes_deterministic,
)

__all__ = [
    "ThermalSatelliteDataset",
    "ProcessedSceneMetadata",
    "load_scene_metadata",
    "denormalize_thermal_kelvin",
    "denormalize_from_scene_metadata",
    "split_scenes_deterministic",
    "development_patch_split",
    "DEVELOPMENT_SPLIT_LABEL",
    "SINGLE_SCENE_SPLIT_WARNING",
]


def __getattr__(name: str):
    if name == "ThermalSatelliteDataset":
        from dataset_loader.thermal_dataset import ThermalSatelliteDataset

        return ThermalSatelliteDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
