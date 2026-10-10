"""Metadata and scientific documentation for synthetic thermal SR data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

HR_REFERENCE_DISCLAIMER = (
    "HR is the 30 m Landsat thermal reference used for synthetic SR training; "
    "it is not sub-30-m ground truth."
)

SCIENTIFIC_LIMITATIONS = (
    "The 30 m thermal patch is treated as the HR reference for a synthetic SR experiment. "
    "This does not provide true higher-resolution thermal ground truth. "
    "The degradation operator defines the synthetic LR problem. "
    "Performance on synthetic degradation does not automatically prove real-world sub-30-m "
    "thermal reconstruction accuracy. "
    "Later evaluation should include synthetic benchmark performance and physical/thermal "
    "consistency checks."
)

DATA_LEAKAGE_NOTE = (
    "With a single processed scene, only development patch splits are available "
    "(DEVELOPMENT SPLIT — NOT FINAL SCIENTIFIC EVALUATION). "
    "Scientific train/validation/test separation requires multiple distinct scenes."
)


@dataclass(frozen=True)
class DegradationConfig:
    """Configurable synthetic thermal degradation parameters."""

    scale: int = 2
    blur_sigma_hr_pixels: float = 1.2
    noise_std_kelvin: float = 0.05
    downsample_method: str = "area"
    interpolation_note: str = "Masked area averaging after Gaussian low-pass in Kelvin"
    base_seed: int | None = 42

    def __post_init__(self) -> None:
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if self.blur_sigma_hr_pixels < 0:
            raise ValueError("blur_sigma_hr_pixels must be non-negative")
        if self.noise_std_kelvin < 0:
            raise ValueError("noise_std_kelvin must be non-negative")

    @property
    def degradation_type(self) -> str:
        return "thermal_gaussian_blur_area_downsample"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SRDatasetMetadata:
    """Description of an SR dataset instance."""

    degradation: DegradationConfig
    hr_patch_size: int
    lr_patch_size: int
    hr_reference_disclaimer: str = HR_REFERENCE_DISCLAIMER
    scientific_limitations: str = SCIENTIFIC_LIMITATIONS
    data_leakage_note: str = DATA_LEAKAGE_NOTE
    normalization_representation: str = "z_score_normalized_kelvin (scene-level, from preprocessing)"
    physical_interpretation: str = (
        "Degradation is applied in Kelvin using scene normalization metadata, "
        "then LR is mapped back to the same z-score space as HR. No second normalization pass."
    )
    rgb_usage: str = "RGB is not used as SR target or input in this module."

    def to_dict(self) -> dict[str, Any]:
        return {
            "degradation": self.degradation.to_dict(),
            "hr_patch_size": self.hr_patch_size,
            "lr_patch_size": self.lr_patch_size,
            "hr_reference_disclaimer": self.hr_reference_disclaimer,
            "scientific_limitations": self.scientific_limitations,
            "data_leakage_note": self.data_leakage_note,
            "normalization_representation": self.normalization_representation,
            "physical_interpretation": self.physical_interpretation,
            "rgb_usage": self.rgb_usage,
        }
