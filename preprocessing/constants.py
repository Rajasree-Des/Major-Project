"""Shared band names and paths for preprocessing."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

REQUIRED_BANDS = ("ST_B10", "SR_B2", "SR_B3", "SR_B4")
THERMAL_BAND = "ST_B10"
RGB_BANDS = ("SR_B2", "SR_B3", "SR_B4")

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 128
DEFAULT_MAX_INVALID_PERCENT = 10.0

BOUNDS_ABS_TOLERANCE_M = 0.01
RESOLUTION_REL_TOLERANCE = 1e-3
