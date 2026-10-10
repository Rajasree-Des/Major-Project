"""Synthetic thermal super-resolution training data generation.

Dataset and degradation helpers live in submodules; import them directly
(e.g. ``from sr_data.dataset import ThermalSRDataset``) to avoid loading
PyTorch when only metadata is needed.
"""

from sr_data.metadata import (
    DATA_LEAKAGE_NOTE,
    HR_REFERENCE_DISCLAIMER,
    SCIENTIFIC_LIMITATIONS,
    DegradationConfig,
    SRDatasetMetadata,
)

__all__ = [
    "DegradationConfig",
    "SRDatasetMetadata",
    "HR_REFERENCE_DISCLAIMER",
    "SCIENTIFIC_LIMITATIONS",
    "DATA_LEAKAGE_NOTE",
]
