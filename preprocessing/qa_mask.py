"""Landsat Collection 2 Level-2 QA_PIXEL decoding and exclusion masks.

Bit positions follow the USGS Landsat 8-9 Collection 2 Level-2 Science Product
Guide (LSDS-1619), QA_PIXEL table, and match the ``classification:bitfields``
published for the ``qa_pixel`` asset in the Planetary Computer ``landsat-c2-l2``
STAC collection:

    bit 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow/ice,
    6 clear, 7 water, 8-9 cloud confidence, 10-11 cloud shadow confidence,
    12-13 snow/ice confidence, 14-15 cirrus confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from preprocessing.raster_io import BACKEND, raster_info, read_band1

QA_BAND = "QA_PIXEL"
QA_RULES_VERSION = "qa_v1"


@dataclass(frozen=True)
class QaFlag:
    name: str
    bit: int
    rationale: str


EXCLUDED_FLAGS: tuple[QaFlag, ...] = (
    QaFlag("fill", 0, "No image data; not a surface observation."),
    QaFlag("dilated_cloud", 1,
           "CFMask cloud buffer; cloud edges and thin cloud bias LST cold and contaminate "
           "neighbouring pixels."),
    QaFlag("cirrus", 2,
           "High-confidence cirrus (OLI band 9); thin ice cloud depresses brightness temperature "
           "while optical bands may look clear."),
    QaFlag("cloud", 3, "Cloud top temperature, not land surface temperature."),
    QaFlag("cloud_shadow", 4,
           "Shadowed surfaces are cooled and darkened, so thermal and RGB do not represent normal "
           "illumination; also flags cloud-adjacent pixels."),
    QaFlag("snow", 5,
           "Snow/ice is excluded from the initial dataset: extreme cold and very high reflectance "
           "would dominate per-scene z-score statistics, and CFMask often confuses snow and cloud."),
)

IGNORED_BITS_NOTE = (
    "Bits 6 (clear) and 7 (water) are not used as exclusion rules: water is a valid surface "
    "for thermal SR. Confidence bit pairs 8-15 are not used; the binary CFMask decision bits "
    "1-5 already encode the thresholded result, and medium/low confidence levels would remove "
    "many clear pixels."
)

EXPECTED_STAC_BITFIELDS: dict[str, tuple[int, int]] = {
    "fill": (0, 1),
    "dilated_cloud": (1, 1),
    "cirrus": (2, 1),
    "cloud": (3, 1),
    "cloud_shadow": (4, 1),
    "snow": (5, 1),
}


def verify_bitfields_against_stac(bitfields: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare our bit positions with a STAC ``classification:bitfields`` list.

    Raises:
        ValueError: If any excluded flag is missing or has a different position.
    """
    published = {f["name"]: (int(f["offset"]), int(f["length"])) for f in bitfields}
    mismatches = {
        name: {"expected": exp, "stac": published.get(name)}
        for name, exp in EXPECTED_STAC_BITFIELDS.items()
        if published.get(name) != exp
    }
    if mismatches:
        raise ValueError(f"QA_PIXEL bit definitions differ from STAC: {mismatches}")
    return {"verified": True, "stac_bitfields": {k: list(v) for k, v in published.items()}}


def rules_metadata() -> dict[str, Any]:
    return {
        "version": QA_RULES_VERSION,
        "source": "USGS LSDS-1619 Landsat 8-9 C2 L2 Science Product Guide, QA_PIXEL bit table; "
                  "cross-checked with STAC classification:bitfields (landsat-c2-l2, qa_pixel)",
        "rule": "pixel excluded if any listed bit is set; combined (AND) with per-band NoData "
                "and finite-value masks of ST_B10, SR_B2, SR_B3, SR_B4",
        "excluded_flags": [{"name": f.name, "bit": f.bit, "rationale": f.rationale}
                           for f in EXCLUDED_FLAGS],
        "not_used": IGNORED_BITS_NOTE,
    }


def find_qa_path(scene_dir: Path, supplement_dirs: Iterable[Path] = ()) -> Path:
    """Locate ``QA_PIXEL.tif`` in the scene folder or a supplement root.

    Raises:
        FileNotFoundError: If no non-empty file is found.
    """
    candidates = [scene_dir / f"{QA_BAND}.tif"]
    candidates += [Path(d) / scene_dir.name / f"{QA_BAND}.tif" for d in supplement_dirs]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"{QA_BAND}.tif not found for {scene_dir.name}; looked in {candidates}")


def read_qa(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read QA_PIXEL as uint16 together with its grid description."""
    meta = raster_info(path)
    if meta.count < 1:
        raise ValueError(f"{path} has no raster bands")
    info = {
        "path": str(path),
        "crs": meta.crs,
        "width": meta.width,
        "height": meta.height,
        "transform": meta.transform,
        "bounds": meta.bounds,
        "dtype": meta.dtype,
        "nodata": meta.nodata,
        "reader_backend": BACKEND,
    }
    data = read_band1(path)
    if data.dtype != np.uint16:
        raise ValueError(f"{path} dtype is {data.dtype}, expected uint16")
    return data, info


def check_alignment(qa_info: Mapping[str, Any], ref_band: Any) -> dict[str, Any]:
    """Require identical CRS, dimensions and geotransform between QA and the thermal band.

    ``ref_band`` is a :class:`preprocessing.validate.BandInfo`.

    Raises:
        ValueError: On any mismatch.
    """
    checks = {
        "crs": (qa_info["crs"], ref_band.crs),
        "width": (qa_info["width"], ref_band.width),
        "height": (qa_info["height"], ref_band.height),
        "transform": (tuple(qa_info["transform"]), tuple(ref_band.transform)),
        "bounds": (tuple(qa_info["bounds"]), tuple(ref_band.bounds)),
    }
    failed = {k: {"qa": v[0], "reference": v[1]} for k, v in checks.items() if v[0] != v[1]}
    if failed:
        raise ValueError(f"QA_PIXEL is not aligned with {ref_band.name}: {failed}")
    return {"aligned_with": ref_band.name, "checked": list(checks), "result": "identical"}


def decode_flags(qa: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean array per excluded flag (True where the bit is set)."""
    return {f.name: (qa >> f.bit) & 1 == 1 for f in EXCLUDED_FLAGS}


def qa_exclusion_mask(flags: Mapping[str, np.ndarray]) -> np.ndarray:
    """True where any excluded flag is set."""
    out = np.zeros(next(iter(flags.values())).shape, dtype=bool)
    for arr in flags.values():
        out |= arr
    return out


def exclusion_codes(base_valid: np.ndarray, flags: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-pixel uint8 code: bit 0 = base (NoData/non-finite) invalid, bit k+1 = flag k set."""
    codes = (~base_valid).astype(np.uint8)
    for i, f in enumerate(EXCLUDED_FLAGS):
        codes |= flags[f.name].astype(np.uint8) << (i + 1)
    return codes


def summarize_flags(base_valid: np.ndarray, flags: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Pixel counts per flag: over the whole grid, and among pixels that were otherwise valid."""
    total = int(base_valid.size)
    qa_any = qa_exclusion_mask(flags)
    per_flag = {}
    for f in EXCLUDED_FLAGS:
        arr = flags[f.name]
        others = np.zeros_like(qa_any)
        for g in EXCLUDED_FLAGS:
            if g.name != f.name:
                others |= flags[g.name]
        per_flag[f.name] = {
            "bit": f.bit,
            "set_all_pixels": int(np.count_nonzero(arr)),
            "set_among_base_valid": int(np.count_nonzero(arr & base_valid)),
            "only_this_flag_among_base_valid": int(np.count_nonzero(arr & base_valid & ~others)),
        }
    base_n = int(np.count_nonzero(base_valid))
    final_n = int(np.count_nonzero(base_valid & ~qa_any))
    return {
        "total_pixels": total,
        "valid_before_qa": base_n,
        "valid_after_qa": final_n,
        "valid_fraction_before_qa": base_n / total if total else 0.0,
        "valid_fraction_after_qa": final_n / total if total else 0.0,
        "excluded_by_qa_among_base_valid": base_n - final_n,
        "per_flag": per_flag,
        "note": "flags overlap (e.g. cloud usually also sets dilated_cloud), so per-flag counts "
                "do not sum to the total excluded",
    }
