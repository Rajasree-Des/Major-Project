"""End-to-end thermal SR + refinement + colorization orchestration.

Heavy modules (integrated, model loading) are imported from their submodules
so the API can start without loading PyTorch.
"""

from pipeline.frozen_config import FrozenExperimentConfig, load_frozen_config

__all__ = [
    "FrozenExperimentConfig",
    "load_frozen_config",
]
