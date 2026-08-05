"""Inspect downloaded Landsat 8/9 scenes before preprocessing.

Read-only validation tool for scenes under ``dataset/raw/<scene_id>/`` containing
ST_B10, SR_B2, SR_B3, and SR_B4 GeoTIFF bands. Produces metadata summaries,
alignment checks, preview PNGs, and a JSON report under ``outputs/previews/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "previews"

REQUIRED_BANDS = ("ST_B10", "SR_B2", "SR_B3", "SR_B4")
RGB_COMPOSITE_BANDS = ("SR_B4", "SR_B3", "SR_B2")
THERMAL_BAND = "ST_B10"
RGB_BANDS = ("SR_B2", "SR_B3", "SR_B4")

EXPECTED_THERMAL_RES_M = 100.0
EXPECTED_RGB_RES_M = 30.0
EXPECTED_RES_RATIO = EXPECTED_THERMAL_RES_M / EXPECTED_RGB_RES_M
RATIO_TOLERANCE = 0.15
OVERLAP_FAIL_THRESHOLD = 0.80
MAX_PREVIEW_DIMENSION = 2048

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BandMetadata:
    """Raster metadata extracted from a single GeoTIFF band."""

    name: str
    path: str
    width: int
    height: int
    crs: str
    resolution_x: float
    resolution_y: float
    bounds: tuple[float, float, float, float]
    dtype: str
    nodata: float | None
    count: int


@dataclass
class BandStatistics:
    """Descriptive statistics for valid (non-nodata) pixels."""

    name: str
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    valid_pixel_count: int
    nodata_pixel_count: int


@dataclass
class AlignmentReport:
    """Thermal vs RGB geospatial alignment assessment."""

    crs_match: bool
    bounds_overlap_ratio: float | None
    thermal_resolution_m: float | None
    rgb_resolution_m: float | None
    resolution_ratio: float | None
    shape_ratio_height: float | None
    shape_ratio_width: float | None
    status: Literal["pass", "warn", "fail"]
    messages: list[str] = field(default_factory=list)


@dataclass
class SceneReport:
    """Complete inspection report for one scene."""

    scene_id: str
    scene_dir: str
    bands_present: list[str]
    bands_missing: list[str]
    metadata: dict[str, BandMetadata]
    statistics: dict[str, BandStatistics]
    alignment: AlignmentReport | None
    outputs: dict[str, str]
    generated_at: str


def setup_logging(level: str) -> None:
    """Configure root logging with a consistent timestamp format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directory(path: Path) -> Path:
    """Ensure ``path`` exists as a directory.

    Removes placeholder *files* that block ``mkdir`` on Windows (e.g. an empty
    ``outputs`` file at the project root).

    Args:
        path: Directory path to create.

    Returns:
        Resolved directory path.
    """
    path = path.resolve()
    for ancestor in [path, *path.parents]:
        if ancestor == ancestor.parent:
            break
        if ancestor.exists() and ancestor.is_file():
            logger.warning("Removing file blocking directory: %s", ancestor)
            ancestor.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def band_path(scene_dir: Path, name: str) -> Path:
    """Return the expected GeoTIFF path for a logical band name."""
    return scene_dir / f"{name}.tif"


def resolve_scene_dir(args: argparse.Namespace) -> Path:
    """Resolve the scene directory from CLI arguments."""
    if args.scene_dir is not None:
        scene_dir = Path(args.scene_dir).resolve()
    else:
        scene_dir = (Path(args.raw_dir) / args.scene_id).resolve()

    if not scene_dir.is_dir():
        raise ValueError(f"Scene directory not found: {scene_dir}")
    return scene_dir


def verify_required_bands(scene_dir: Path) -> tuple[list[str], list[str]]:
    """Check which required band files exist on disk."""
    present: list[str] = []
    missing: list[str] = []
    for name in REQUIRED_BANDS:
        path = band_path(scene_dir, name)
        if path.is_file() and path.stat().st_size > 0:
            present.append(name)
        else:
            missing.append(name)
    return present, missing


def extract_band_metadata(path: Path, name: str) -> BandMetadata:
    """Extract metadata from a GeoTIFF without modifying it."""
    with rasterio.open(path, "r") as dataset:
        bounds = dataset.bounds
        return BandMetadata(
            name=name,
            path=str(path),
            width=dataset.width,
            height=dataset.height,
            crs=str(dataset.crs) if dataset.crs else "unknown",
            resolution_x=abs(dataset.transform.a),
            resolution_y=abs(dataset.transform.e),
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            dtype=str(dataset.dtypes[0]),
            nodata=float(dataset.nodata) if dataset.nodata is not None else None,
            count=dataset.count,
        )


def _decimated_out_shape(width: int, height: int) -> tuple[int, int]:
    """Return ``(out_height, out_width)`` capped at MAX_PREVIEW_DIMENSION."""
    scale = max(width, height) / MAX_PREVIEW_DIMENSION
    if scale <= 1.0:
        return height, width
    return max(1, int(height / scale)), max(1, int(width / scale))


def _read_band_2d(path: Path, *, decimated: bool = False) -> tuple[np.ndarray, float | None, int, int]:
    """Read band 1 as a float64 array (read-only).

    Args:
        path: GeoTIFF path.
        decimated: If True, downsample for preview rendering.

    Returns:
        Tuple of ``(data_2d, nodata, width, height)`` using source dimensions.
    """
    with rasterio.open(path, "r") as dataset:
        nodata = float(dataset.nodata) if dataset.nodata is not None else None
        width, height = dataset.width, dataset.height
        dtype = dataset.dtypes[0]

        # Rasterio + NumPy 2.5 triggers a deprecation when reshaping read buffers.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting the shape on a NumPy array has been deprecated",
                category=DeprecationWarning,
            )
            if decimated:
                out_h, out_w = _decimated_out_shape(width, height)
                raw = dataset.read(
                    1,
                    window=Window(0, 0, width, height),
                    out_shape=(out_h, out_w),
                    resampling=Resampling.average,
                )
            else:
                buffer = np.empty((height, width), dtype=dtype)
                raw = dataset.read(1, out=buffer)

        data = np.asarray(raw, dtype=np.float64)
        return data, nodata, width, height


def _valid_pixel_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Build a boolean mask of finite, non-nodata pixels."""
    mask = np.isfinite(data)
    if nodata is not None:
        mask &= data != nodata
    return mask


def _read_valid_pixels(path: Path) -> tuple[np.ndarray, int, int]:
    """Read band 1 and return valid pixels plus pixel counts.

    Returns:
        Tuple of ``(valid_1d, valid_count, nodata_count)``.
    """
    data, nodata, width, height = _read_band_2d(path, decimated=False)
    total_pixels = width * height
    valid_mask = _valid_pixel_mask(data, nodata)
    valid = data[valid_mask]
    valid_count = int(valid.size)
    return valid, valid_count, total_pixels - valid_count


def compute_band_statistics(path: Path, name: str) -> BandStatistics:
    """Compute descriptive statistics on raw valid pixels."""
    valid, valid_count, nodata_count = _read_valid_pixels(path)

    if valid_count == 0:
        return BandStatistics(
            name=name,
            min=None,
            max=None,
            mean=None,
            median=None,
            std=None,
            valid_pixel_count=0,
            nodata_pixel_count=nodata_count,
        )

    return BandStatistics(
        name=name,
        min=float(np.min(valid)),
        max=float(np.max(valid)),
        mean=float(np.mean(valid)),
        median=float(np.median(valid)),
        std=float(np.std(valid)),
        valid_pixel_count=valid_count,
        nodata_pixel_count=nodata_count,
    )


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    """Compute axis-aligned bounding box area."""
    left, bottom, right, top = bounds
    return max(right - left, 0.0) * max(top - bottom, 0.0)


def _bounds_intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute intersection area of two axis-aligned bounds."""
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def _within_tolerance(value: float, expected: float, tolerance: float) -> bool:
    """Return True if value is within relative tolerance of expected."""
    if expected == 0:
        return value == 0
    return abs(value - expected) / abs(expected) <= tolerance


def check_thermal_rgb_alignment(
    thermal: BandMetadata,
    rgb_metas: dict[str, BandMetadata],
) -> AlignmentReport:
    """Assess geospatial alignment between thermal and RGB bands."""
    messages: list[str] = []
    rgb_list = [rgb_metas[b] for b in RGB_BANDS if b in rgb_metas]
    if not rgb_list:
        return AlignmentReport(
            crs_match=False,
            bounds_overlap_ratio=None,
            thermal_resolution_m=thermal.resolution_x,
            rgb_resolution_m=None,
            resolution_ratio=None,
            shape_ratio_height=None,
            shape_ratio_width=None,
            status="fail",
            messages=["No RGB band metadata available for alignment check."],
        )

    reference_rgb = rgb_list[0]
    crs_match = all(meta.crs == thermal.crs for meta in rgb_list)
    messages.append(
        f"CRS consistent across all bands ({thermal.crs})."
        if crs_match
        else f"CRS mismatch detected: "
        f"{ {m.name: m.crs for m in [thermal, *rgb_list]} }"
    )

    rgb_area = _bounds_area(reference_rgb.bounds)
    overlap_ratio = (
        _bounds_intersection_area(thermal.bounds, reference_rgb.bounds) / rgb_area
        if rgb_area > 0
        else 0.0
    )
    messages.append(f"Thermal/RGB bounds overlap ratio: {overlap_ratio:.3f}")

    thermal_res = (thermal.resolution_x + thermal.resolution_y) / 2.0
    rgb_res = (reference_rgb.resolution_x + reference_rgb.resolution_y) / 2.0
    resolution_ratio = thermal_res / rgb_res if rgb_res > 0 else None
    shape_ratio_height = (
        thermal.height / reference_rgb.height if reference_rgb.height else None
    )
    shape_ratio_width = (
        thermal.width / reference_rgb.width if reference_rgb.width else None
    )
    expected_shape_ratio = 1.0 / EXPECTED_RES_RATIO

    if resolution_ratio is not None:
        messages.append(
            f"Resolution ratio thermal/RGB: {resolution_ratio:.3f} "
            f"(expected ~{EXPECTED_RES_RATIO:.2f})"
        )
    if shape_ratio_height is not None and shape_ratio_width is not None:
        messages.append(
            f"Shape ratio H×W thermal/RGB: {shape_ratio_height:.3f} × "
            f"{shape_ratio_width:.3f} (expected ~{expected_shape_ratio:.3f})"
        )

    if not crs_match or overlap_ratio < OVERLAP_FAIL_THRESHOLD:
        status: Literal["pass", "warn", "fail"] = "fail"
        if not crs_match:
            messages.append("FAIL: CRS mismatch.")
        if overlap_ratio < OVERLAP_FAIL_THRESHOLD:
            messages.append(f"FAIL: overlap ratio below {OVERLAP_FAIL_THRESHOLD:.2f}.")
    else:
        ratio_ok = resolution_ratio is not None and _within_tolerance(
            resolution_ratio, EXPECTED_RES_RATIO, RATIO_TOLERANCE
        )
        shape_ok = (
            shape_ratio_height is not None
            and shape_ratio_width is not None
            and _within_tolerance(shape_ratio_height, expected_shape_ratio, RATIO_TOLERANCE)
            and _within_tolerance(shape_ratio_width, expected_shape_ratio, RATIO_TOLERANCE)
        )
        if ratio_ok and shape_ok:
            status = "pass"
            messages.append("PASS: thermal and RGB bands appear aligned.")
        else:
            status = "warn"
            messages.append("WARN: resolution or shape ratio outside ±15% tolerance.")

    return AlignmentReport(
        crs_match=crs_match,
        bounds_overlap_ratio=overlap_ratio,
        thermal_resolution_m=thermal_res,
        rgb_resolution_m=rgb_res,
        resolution_ratio=resolution_ratio,
        shape_ratio_height=shape_ratio_height,
        shape_ratio_width=shape_ratio_width,
        status=status,
        messages=messages,
    )


def _apply_nodata_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return a copy with nodata values replaced by NaN for display."""
    display = data.copy()
    if nodata is not None:
        display[display == nodata] = np.nan
    return display


def _percentile_stretch(channel: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    """Apply percentile stretch for display PNG only."""
    valid = channel[np.isfinite(channel)]
    if valid.size == 0:
        return np.zeros_like(channel, dtype=np.float64)

    p_low, p_high = np.percentile(valid, [lower, upper])
    if p_high <= p_low:
        out = np.zeros_like(channel, dtype=np.float64)
        out[np.isfinite(channel)] = 0.5
        return out

    return np.clip((channel - p_low) / (p_high - p_low), 0.0, 1.0)


def _save_figure(fig: plt.Figure, out_png: Path) -> None:
    """Save a matplotlib figure and close it."""
    ensure_directory(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_png)


def render_thermal_preview(path: Path, out_png: Path) -> None:
    """Render ST_B10 thermal preview with inferno colormap (display only)."""
    data, nodata, _, _ = _read_band_2d(path, decimated=True)
    display = _apply_nodata_mask(data, nodata)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(display, cmap="inferno")
    ax.set_title("ST_B10 Thermal Preview")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Raw pixel value")
    fig.patch.set_facecolor("black")
    _save_figure(fig, out_png)


def render_rgb_composite(band_paths: dict[str, Path], out_png: Path) -> None:
    """Build RGB composite from SR_B4, SR_B3, SR_B2 for display PNG only."""
    channels = [
        _apply_nodata_mask(*_read_band_2d(band_paths[name], decimated=True)[:2])
        for name in RGB_COMPOSITE_BANDS
    ]
    rgb = np.stack([_percentile_stretch(ch) for ch in channels], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.imshow(rgb)
    ax.set_title("RGB Composite (SR_B4, SR_B3, SR_B2)")
    ax.axis("off")
    _save_figure(fig, out_png)


def render_histograms(
    thermal_path: Path,
    rgb_paths: dict[str, Path],
    out_png: Path,
) -> None:
    """Render histograms for thermal and RGB bands (valid pixels only)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    plot_specs = [
        (THERMAL_BAND, thermal_path, axes[0, 0]),
        ("SR_B4 (Red)", rgb_paths["SR_B4"], axes[0, 1]),
        ("SR_B3 (Green)", rgb_paths["SR_B3"], axes[1, 0]),
        ("SR_B2 (Blue)", rgb_paths["SR_B2"], axes[1, 1]),
    ]

    for title, path, ax in plot_specs:
        valid, _, _ = _read_valid_pixels(path)
        if valid.size == 0:
            ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(valid, bins=100, color="#3B82F6", alpha=0.85, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Raw pixel value")
        ax.set_ylabel("Frequency")

    fig.suptitle("Band Histograms (valid pixels, raw values)")
    fig.tight_layout()
    _save_figure(fig, out_png)


def _format_file_size(num_bytes: int) -> str:
    """Format byte count for display."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size = float(num_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} PB"


def _format_stat(value: float | None, precision: int = 4) -> str:
    """Format a statistic value for table display."""
    return f"{value:.{precision}f}" if value is not None else "—"


def _to_json_safe(value: Any) -> Any:
    """Recursively convert report objects to JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "__dataclass_fields__"):
        return _to_json_safe(asdict(value))
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    return value


def export_json_report(report: SceneReport, path: Path) -> None:
    """Write the scene report as pretty-printed JSON."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(report), handle, indent=2)
    logger.info("Saved JSON report: %s", path)


def print_scene_summary(report: SceneReport) -> None:
    """Print a rich terminal summary of the scene inspection report."""
    console.print(
        Panel(
            f"[bold]Scene ID:[/bold] {report.scene_id}\n"
            f"[bold]Path:[/bold] {report.scene_dir}\n"
            f"[bold]Generated:[/bold] {report.generated_at}",
            title="Landsat Scene Inspection",
            border_style="blue",
        )
    )

    verify_table = Table(title="Band Verification", show_header=True, header_style="bold")
    verify_table.add_column("Band")
    verify_table.add_column("Status")
    verify_table.add_column("File Size")
    for name in REQUIRED_BANDS:
        path = Path(report.scene_dir) / f"{name}.tif"
        if name in report.bands_present:
            verify_table.add_row(name, "[green]Present[/green]", _format_file_size(path.stat().st_size))
        else:
            verify_table.add_row(name, "[red]Missing[/red]", "—")
    console.print(verify_table)

    if report.metadata:
        meta_table = Table(title="Band Metadata", show_header=True, header_style="bold")
        for col in ("Band", "Size", "CRS", "Resolution (m)", "Dtype", "NoData"):
            meta_table.add_column(col)
        for name, meta in report.metadata.items():
            meta_table.add_row(
                name,
                f"{meta.width}×{meta.height}",
                meta.crs,
                f"{meta.resolution_x:.2f} × {meta.resolution_y:.2f}",
                meta.dtype,
                str(meta.nodata),
            )
        console.print(meta_table)

    if report.statistics:
        stats_table = Table(title="Band Statistics (raw valid pixels)", show_header=True, header_style="bold")
        for col in ("Band", "Min", "Max", "Mean", "Median", "Std", "Valid px"):
            stats_table.add_column(col)
        for name, stats in report.statistics.items():
            stats_table.add_row(
                name,
                _format_stat(stats.min),
                _format_stat(stats.max),
                _format_stat(stats.mean),
                _format_stat(stats.median),
                _format_stat(stats.std),
                str(stats.valid_pixel_count),
            )
        console.print(stats_table)

    if report.alignment is not None:
        align = report.alignment
        status_style = {"pass": "green", "warn": "yellow", "fail": "red"}.get(align.status, "white")
        body = "\n".join(f"• {msg}" for msg in align.messages)
        console.print(
            Panel(
                body,
                title=f"Thermal/RGB Alignment [{align.status.upper()}]",
                border_style=status_style,
            )
        )

    if report.outputs:
        lines = "\n".join(f"[bold]{key}:[/bold] {path}" for key, path in report.outputs.items())
        console.print(Panel(lines, title="Generated Outputs", border_style="cyan"))


def inspect_scene(scene_dir: Path, output_root: Path) -> SceneReport:
    """Run full read-only inspection pipeline for one scene."""
    scene_id = scene_dir.name
    present, missing = verify_required_bands(scene_dir)

    metadata: dict[str, BandMetadata] = {}
    statistics: dict[str, BandStatistics] = {}
    for name in present:
        path = band_path(scene_dir, name)
        metadata[name] = extract_band_metadata(path, name)
        statistics[name] = compute_band_statistics(path, name)

    alignment: AlignmentReport | None = None
    if THERMAL_BAND in metadata and all(b in metadata for b in RGB_BANDS):
        alignment = check_thermal_rgb_alignment(
            metadata[THERMAL_BAND],
            {name: metadata[name] for name in RGB_BANDS},
        )

    scene_output_dir = ensure_directory(output_root / scene_id)
    outputs: dict[str, str] = {}

    if THERMAL_BAND in present:
        thermal_png = scene_output_dir / "thermal_preview.png"
        render_thermal_preview(band_path(scene_dir, THERMAL_BAND), thermal_png)
        outputs["thermal_preview"] = str(thermal_png)

    if all(b in present for b in RGB_COMPOSITE_BANDS):
        rgb_paths = {name: band_path(scene_dir, name) for name in RGB_COMPOSITE_BANDS}
        rgb_png = scene_output_dir / "rgb_composite.png"
        render_rgb_composite(rgb_paths, rgb_png)
        outputs["rgb_composite"] = str(rgb_png)

    if THERMAL_BAND in present and all(b in present for b in RGB_BANDS):
        rgb_paths = {name: band_path(scene_dir, name) for name in RGB_BANDS}
        hist_png = scene_output_dir / "histograms.png"
        render_histograms(band_path(scene_dir, THERMAL_BAND), rgb_paths, hist_png)
        outputs["histograms"] = str(hist_png)

    report = SceneReport(
        scene_id=scene_id,
        scene_dir=str(scene_dir),
        bands_present=present,
        bands_missing=missing,
        metadata=metadata,
        statistics=statistics,
        alignment=alignment,
        outputs=outputs,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    json_path = scene_output_dir / "report.json"
    export_json_report(report, json_path)
    outputs["report_json"] = str(json_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect downloaded Landsat 8/9 scenes (ST_B10, SR_B2, SR_B3, SR_B4) "
            "before preprocessing. Read-only — does not modify GeoTIFFs."
        ),
    )

    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--scene-id",
        help="Scene folder name under --raw-dir (e.g. LC09_L2SP_..._T1).",
    )
    scene_group.add_argument(
        "--scene-dir",
        type=Path,
        help="Direct path to scene folder containing band GeoTIFFs.",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Root raw dataset directory (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Preview and report output root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scene preview inspection CLI."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        scene_dir = resolve_scene_dir(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Inspecting scene: %s", scene_dir)
    report = inspect_scene(scene_dir, Path(args.output_dir).resolve())
    print_scene_summary(report)

    if report.bands_missing:
        logger.error("Missing required bands: %s", ", ".join(report.bands_missing))
        return 1

    if report.alignment is not None and report.alignment.status == "fail":
        logger.error("Thermal/RGB alignment check failed.")
        return 1

    if report.alignment is not None and report.alignment.status == "warn":
        logger.warning("Thermal/RGB alignment check produced warnings.")

    logger.info("Scene inspection completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
