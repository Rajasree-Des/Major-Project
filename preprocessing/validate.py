"""Structural validation for downloaded Landsat scene bands."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preprocessing.raster_io import raster_info, read_band1
from preprocessing.constants import (
    BOUNDS_ABS_TOLERANCE_M,
    REQUIRED_BANDS,
    RESOLUTION_REL_TOLERANCE,
    THERMAL_BAND,
)

logger = logging.getLogger(__name__)


class SceneValidationError(Exception):
    """Raised when a scene fails structural validation."""


@dataclass
class BandInfo:
    """Geospatial and raster metadata for one band file."""

    name: str
    path: Path
    width: int
    height: int
    crs: str
    transform: tuple[float, ...]
    bounds: tuple[float, float, float, float]
    resolution_x: float
    resolution_y: float
    nodata: float | None
    dtype: str


@dataclass
class SceneValidationReport:
    """Result of validating all required bands in a scene directory."""

    scene_id: str
    scene_dir: Path
    bands: dict[str, BandInfo]
    nodata_values: dict[str, float | None]
    valid: bool
    messages: list[str] = field(default_factory=list)


def band_path(scene_dir: Path, name: str) -> Path:
    """Return expected GeoTIFF path for a logical band name."""
    return scene_dir / f"{name}.tif"


def _approx_equal(a: float, b: float, *, rel: float = RESOLUTION_REL_TOLERANCE, abs_tol: float = 1e-6) -> bool:
    if math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol):
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= rel


def _bounds_match(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return all(_approx_equal(x, y, abs_tol=BOUNDS_ABS_TOLERANCE_M) for x, y in zip(a, b))


def _load_band_info(path: Path, name: str) -> BandInfo:
    if not path.is_file() or path.stat().st_size == 0:
        raise SceneValidationError(f"Missing or empty band file: {path}")

    info = raster_info(path)
    if info.count < 1:
        raise SceneValidationError(f"Band file has no raster bands: {path}")
    return BandInfo(
        name=name,
        path=path,
        width=info.width,
        height=info.height,
        crs=info.crs,
        transform=info.transform,
        bounds=info.bounds,
        resolution_x=abs(info.transform[1]),
        resolution_y=abs(info.transform[5]),
        nodata=info.nodata,
        dtype=info.dtype,
    )


def validate_scene(scene_dir: Path) -> SceneValidationReport:
    """Validate required bands and geospatial consistency."""
    scene_dir = scene_dir.resolve()
    if not scene_dir.is_dir():
        raise SceneValidationError(f"Scene directory not found: {scene_dir}")

    scene_id = scene_dir.name
    messages: list[str] = []
    bands: dict[str, BandInfo] = {}

    for name in REQUIRED_BANDS:
        path = band_path(scene_dir, name)
        bands[name] = _load_band_info(path, name)

    messages.append("All required band files present.")

    reference = bands[THERMAL_BAND]
    if reference.crs == "unknown":
        raise SceneValidationError("Thermal band CRS is unknown.")

    for name, info in bands.items():
        if info.crs != reference.crs:
            raise SceneValidationError(
                f"CRS mismatch: {THERMAL_BAND}={reference.crs}, {name}={info.crs}"
            )
    messages.append(f"CRS consistent ({reference.crs}).")

    for name, info in bands.items():
        if info.width != reference.width or info.height != reference.height:
            raise SceneValidationError(
                f"Dimension mismatch: {THERMAL_BAND}={reference.width}×{reference.height}, "
                f"{name}={info.width}×{info.height}"
            )
    messages.append(f"Dimensions consistent ({reference.width}×{reference.height}).")

    ref_res = (reference.resolution_x + reference.resolution_y) / 2.0
    for name, info in bands.items():
        res = (info.resolution_x + info.resolution_y) / 2.0
        if not _approx_equal(res, ref_res):
            raise SceneValidationError(
                f"Resolution mismatch: {THERMAL_BAND}≈{ref_res:.4f} m, {name}≈{res:.4f} m"
            )
    messages.append(f"Resolution consistent (~{ref_res:.2f} m).")

    for name, info in bands.items():
        if not _bounds_match(reference.bounds, info.bounds):
            raise SceneValidationError(f"Bounds mismatch between {THERMAL_BAND} and {name}.")
    messages.append("Bounds aligned across bands.")

    for name, info in bands.items():
        if info.transform != reference.transform:
            raise SceneValidationError(
                f"Affine transform mismatch between {THERMAL_BAND} and {name}."
            )
    messages.append("Geotransforms identical across bands.")

    nodata_values = {name: info.nodata for name, info in bands.items()}
    messages.append(f"NoData values: {nodata_values}.")

    report = SceneValidationReport(
        scene_id=scene_id,
        scene_dir=scene_dir,
        bands=bands,
        nodata_values=nodata_values,
        valid=True,
        messages=messages,
    )
    logger.info("Scene validation passed for %s", scene_id)
    return report


def read_band_array(path: Path) -> tuple[np.ndarray, float | None]:
    """Read band 1 as float64 (read-only)."""
    nodata = raster_info(path).nodata
    data = np.array(read_band1(path), dtype=np.float64, copy=True)
    return data, nodata
