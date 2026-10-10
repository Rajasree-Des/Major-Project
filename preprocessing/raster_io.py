"""Single-band GeoTIFF access with a rasterio backend and a pure-Python tifffile fallback.

rasterio is used whenever it imports. If it cannot be loaded (for example when
Windows Smart App Control blocks its bundled DLLs), ``tifffile`` is used. The
fallback reproduces GDAL's georeferencing for north-up GeoTIFFs with an EPSG code,
including the half-pixel origin shift for ``RasterPixelIsPoint``, and refuses
anything it cannot interpret exactly.

Set ``RASTER_BACKEND=tifffile`` or ``RASTER_BACKEND=rasterio`` to force a backend.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PIXEL_IS_POINT = 2


@dataclass(frozen=True)
class RasterInfo:
    width: int
    height: int
    count: int
    crs: str
    transform: tuple[float, float, float, float, float, float]  # GDAL order
    bounds: tuple[float, float, float, float]  # left, bottom, right, top
    nodata: float | None
    dtype: str


def _load_rasterio():
    try:
        import rasterio
    except (ImportError, OSError) as exc:
        return None, exc
    return rasterio, None


_FORCED = os.environ.get("RASTER_BACKEND", "").strip().lower()
if _FORCED not in ("", "rasterio", "tifffile"):
    raise ValueError(f"RASTER_BACKEND must be 'rasterio' or 'tifffile', got {_FORCED!r}")
_rasterio, _rasterio_error = (None, "forced tifffile") if _FORCED == "tifffile" else _load_rasterio()
if _rasterio is None and _FORCED == "rasterio":
    raise ImportError(f"RASTER_BACKEND=rasterio but rasterio failed to load: {_rasterio_error}")
BACKEND = "rasterio" if _rasterio is not None else "tifffile"
if BACKEND == "tifffile" and _FORCED != "tifffile":
    logger.warning("rasterio unavailable (%s); using tifffile fallback.", _rasterio_error)


def _info_rasterio(path: Path) -> RasterInfo:
    with _rasterio.open(path, "r") as ds:
        b = ds.bounds
        return RasterInfo(
            width=ds.width, height=ds.height, count=ds.count,
            crs=str(ds.crs) if ds.crs else "unknown",
            transform=tuple(ds.transform.to_gdal()),
            bounds=(b.left, b.bottom, b.right, b.top),
            nodata=float(ds.nodata) if ds.nodata is not None else None,
            dtype=str(ds.dtypes[0]),
        )


def _info_tifffile(path: Path) -> RasterInfo:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = page.tags
        if "ModelTransformationTag" in tags:
            raise ValueError(f"{path}: ModelTransformationTag not supported by the fallback reader")
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None or len(tie.value) != 6:
            raise ValueError(f"{path}: needs exactly one ModelTiepoint and a ModelPixelScale")
        sx, sy = float(scale.value[0]), float(scale.value[1])
        i, j, _, x, y, _ = (float(v) for v in tie.value)
        geo = tf.geotiff_metadata or {}
        epsg = geo.get("ProjectedCSTypeGeoKey") or geo.get("GeographicTypeGeoKey")
        if epsg is None or int(epsg) in (0, 32767):
            raise ValueError(f"{path}: no EPSG code in GeoKeys")
        origin_x = x - i * sx
        origin_y = y + j * sy
        if int(geo.get("GTRasterTypeGeoKey", 1)) == PIXEL_IS_POINT:
            origin_x -= sx / 2.0
            origin_y += sy / 2.0
        nodata_tag = tags.get("GDAL_NODATA")
        nodata = float(str(nodata_tag.value).strip("\x00 ")) if nodata_tag is not None else None
        height, width = page.imagelength, page.imagewidth
        transform = (origin_x, sx, 0.0, origin_y, 0.0, -sy)
        return RasterInfo(
            width=width, height=height, count=page.samplesperpixel,
            crs=f"EPSG:{int(epsg)}",
            transform=transform,
            bounds=(origin_x, origin_y - height * sy, origin_x + width * sx, origin_y),
            nodata=nodata,
            dtype=str(page.dtype),
        )


def raster_info(path: Path) -> RasterInfo:
    return _info_rasterio(path) if BACKEND == "rasterio" else _info_tifffile(path)


def read_band1(path: Path) -> np.ndarray:
    """Read band 1 with its native dtype."""
    if BACKEND == "rasterio":
        with _rasterio.open(path, "r") as ds:
            return ds.read(1)
    import tifffile

    with tifffile.TiffFile(path) as tf:
        data = tf.pages[0].asarray()
    return data if data.ndim == 2 else data[..., 0]
