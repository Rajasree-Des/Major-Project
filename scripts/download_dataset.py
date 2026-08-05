"""Download Landsat Collection 2 Level-2 scenes from Microsoft Planetary Computer.

Queries the Planetary Computer STAC API for Landsat 8/9 C2 L2 imagery and
downloads surface temperature (ST_B10) and surface reflectance bands
(SR_B2, SR_B3, SR_B4) into ``dataset/raw/<scene_id>/``.

Optional: set the ``PC_SDK_SUBSCRIPTION_KEY`` environment variable for higher
API rate limits (read automatically by the planetary-computer SDK).
"""

from __future__ import annotations

import argparse
import http.client
import logging
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import planetary_computer
import pystac_client
import requests
import urllib3.exceptions
from pystac import Item
from tqdm import tqdm

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "landsat-c2-l2"
DEFAULT_PLATFORMS: tuple[str, ...] = ("landsat-8", "landsat-9")
DEFAULT_ASSETS: tuple[str, ...] = ("ST_B10", "SR_B2", "SR_B3", "SR_B4")
ASSET_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "ST_B10": ("lwir11", "ST_B10"),
    "SR_B2": ("blue", "SR_B2"),
    "SR_B3": ("green", "SR_B3"),
    "SR_B4": ("red", "SR_B4"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "raw"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Download tuning (adjust here; CLI is unchanged).
DEFAULT_CHUNK_SIZE = 128 * 1024  # 128 KB
CONNECT_TIMEOUT = 30  # seconds
READ_TIMEOUT = 300  # seconds
BACKOFF_BASE_SECONDS = 2
BACKOFF_MAX_SECONDS = 60
BACKOFF_JITTER_MAX = 1.0
LOG_PROGRESS_INTERVAL = 5.0  # seconds between DEBUG speed/ETA logs

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """Configure root logging with a consistent timestamp format.

    Args:
        level: Logging level name (e.g. ``"INFO"`` or ``"DEBUG"``).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_bbox(args: argparse.Namespace) -> list[float]:
    """Build a WGS84 bounding box from CLI arguments.

    Args:
        args: Parsed CLI namespace with either ``bbox`` or ``lon``/``lat``/``buffer``.

    Returns:
        Bounding box as ``[minx, miny, maxx, maxy]``.

    Raises:
        ValueError: If required AOI arguments are missing or incomplete.
    """
    if args.bbox is not None:
        return list(args.bbox)

    if args.lon is None or args.lat is None:
        raise ValueError("Provide either --bbox or both --lon and --lat.")

    if args.buffer is None:
        raise ValueError("--buffer is required when using --lon and --lat.")

    minx = args.lon - args.buffer
    maxx = args.lon + args.buffer
    miny = args.lat - args.buffer
    maxy = args.lat + args.buffer
    return [minx, miny, maxx, maxy]


def validate_bbox(bbox: Sequence[float]) -> None:
    """Validate WGS84 bounding box values.

    Args:
        bbox: Bounding box as ``[minx, miny, maxx, maxy]``.

    Raises:
        ValueError: If bounds are invalid.
    """
    minx, miny, maxx, maxy = bbox

    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0):
        raise ValueError("Longitude values must be within [-180, 180].")

    if not (-90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        raise ValueError("Latitude values must be within [-90, 90].")

    if minx >= maxx:
        raise ValueError("Invalid bbox: min longitude must be less than max longitude.")

    if miny >= maxy:
        raise ValueError("Invalid bbox: min latitude must be less than max latitude.")


def parse_datetime_range(start_date: str, end_date: str) -> str:
    """Convert date strings into a STAC datetime interval.

    Args:
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.

    Returns:
        STAC datetime interval string, e.g. ``"2020-06-01/2020-08-31"``.

    Raises:
        ValueError: If dates are invalid or out of order.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            "Dates must use YYYY-MM-DD format, e.g. 2023-01-01."
        ) from exc

    if start > end:
        raise ValueError("start-date must be on or before end-date.")

    return f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"


def build_stac_client() -> pystac_client.Client:
    """Open a signed Planetary Computer STAC client.

    Returns:
        Configured STAC client with in-place SAS signing enabled.
    """
    return pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_landsat_scenes(
    client: pystac_client.Client,
    bbox: Sequence[float],
    datetime_range: str,
    max_cloud_cover: float,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    max_items: int | None = None,
) -> list[Item]:
    """Search Landsat C2 L2 scenes matching spatial, temporal, and cloud filters.

    Args:
        client: Signed Planetary Computer STAC client.
        bbox: Area of interest as ``[minx, miny, maxx, maxy]``.
        datetime_range: STAC datetime interval string.
        max_cloud_cover: Maximum allowed ``eo:cloud_cover`` percentage.
        platforms: Allowed STAC ``platform`` values (default: Landsat 8 and 9).
        max_items: Optional cap on the number of returned scenes.

    Returns:
        List of matching STAC items.

    Raises:
        RuntimeError: If the STAC search fails.
    """
    allowed_platforms = set(platforms)
    search_kwargs: dict[str, object] = {
        "collections": [COLLECTION_ID],
        "bbox": list(bbox),
        "datetime": datetime_range,
        "query": {
            "eo:cloud_cover": {"lte": max_cloud_cover},
            "platform": {"in": list(platforms)},
        },
    }
    if max_items is not None:
        search_kwargs["max_items"] = max_items

    try:
        search = client.search(**search_kwargs)
        items = list(search.items())
    except Exception as exc:
        raise RuntimeError(f"STAC search failed: {exc}") from exc

    items = [
        item
        for item in items
        if item.properties.get("platform") in allowed_platforms
    ]

    logger.info(
        "Found %d scene(s) matching search criteria (platforms: %s).",
        len(items),
        ", ".join(platforms),
    )
    return items


def is_transient_download_error(exc: BaseException) -> bool:
    """Return True if a download error is likely temporary and worth retrying.

    Args:
        exc: Exception raised during HTTP request or streaming.

    Returns:
        True for transient network/HTTP errors; False otherwise.
    """
    if isinstance(
        exc,
        (
            ConnectionResetError,
            ConnectionError,
            TimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.SSLError,
            urllib3.exceptions.ProtocolError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ),
    ):
        return True

    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None:
            return exc.response.status_code in RETRYABLE_STATUS_CODES
        return True

    return False


def _backoff_seconds(attempt: int) -> float:
    """Compute exponential backoff with jitter for a retry attempt.

    Args:
        attempt: One-based attempt number that just failed.

    Returns:
        Seconds to sleep before the next attempt.
    """
    base = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS ** attempt)
    return base + random.uniform(0, BACKOFF_JITTER_MAX)


def _part_path(dest: Path) -> Path:
    """Return the partial-download path for a destination file.

    Args:
        dest: Final asset path, e.g. ``ST_B10.tif``.

    Returns:
        Companion ``.part`` path, e.g. ``ST_B10.tif.part``.
    """
    return dest.with_suffix(dest.suffix + ".part")


def _existing_resume_offset(part_path: Path) -> int:
    """Return the number of bytes already stored in a partial file.

    Args:
        part_path: Path to the ``.part`` file.

    Returns:
        Existing file size in bytes, or ``0`` if the file does not exist.
    """
    if part_path.exists():
        return part_path.stat().st_size
    return 0


def _parse_total_size(
    headers: Mapping[str, str],
    resume_offset: int,
    status_code: int,
) -> int | None:
    """Derive the expected total object size from HTTP response headers.

    Args:
        headers: Response headers from the download request.
        resume_offset: Byte offset used for a resumed request.
        status_code: HTTP status code returned by the server.

    Returns:
        Expected total size in bytes, or ``None`` if unknown.
    """
    content_range = headers.get("Content-Range")
    if content_range:
        match = re.search(r"/(\d+)\s*$", content_range)
        if match:
            return int(match.group(1))

    content_length = headers.get("Content-Length")
    if content_length:
        length = int(content_length)
        if status_code == 206 and resume_offset > 0:
            return resume_offset + length
        return length

    return None


def _verify_file_size(path: Path, expected: int) -> bool:
    """Verify that a downloaded file matches the expected total size.

    Args:
        path: Path to the downloaded ``.part`` file.
        expected: Expected total size in bytes.

    Returns:
        True if sizes match.
    """
    return path.stat().st_size == expected


def _finalize_download(part_path: Path, dest: Path) -> None:
    """Atomically promote a completed partial file to the final destination.

    Args:
        part_path: Completed ``.part`` file.
        dest: Final destination path.
    """
    part_path.replace(dest)


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count for human-readable logging.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Short string such as ``"43.2 MB"``.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size = float(num_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} PB"


def _format_eta(seconds: float) -> str:
    """Format remaining seconds as ``MM:SS`` for logs.

    Args:
        seconds: Estimated seconds remaining.

    Returns:
        Formatted ETA string.
    """
    if seconds < 0 or not math.isfinite(seconds):
        return "--:--"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _log_download_complete(
    label: str,
    context: str,
    dest: Path,
    started_at: float,
) -> None:
    """Log a successful download summary.

    Args:
        label: Asset label for logs.
        context: Optional scene context prefix.
        dest: Final downloaded file path.
        started_at: ``time.monotonic()`` timestamp when download began.
    """
    elapsed = time.monotonic() - started_at
    size = dest.stat().st_size
    avg_speed = size / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Completed %s | %ssize=%s | elapsed=%s | avg_speed=%s/s",
        label,
        context,
        _format_bytes(size),
        _format_eta(elapsed),
        _format_bytes(int(avg_speed)),
    )


def _stream_response_to_part(
    response: requests.Response,
    part_path: Path,
    *,
    file_mode: str,
    resume_offset: int,
    expected_total: int | None,
    chunk_size: int,
    label: str,
    context: str,
) -> None:
    """Stream HTTP response bytes into a partial file with progress reporting.

    Args:
        response: Open streaming HTTP response.
        part_path: Partial file destination.
        file_mode: File open mode, ``"wb"`` or ``"ab"``.
        resume_offset: Bytes already on disk before this stream.
        expected_total: Expected final object size, if known.
        chunk_size: Read chunk size in bytes.
        label: Asset label for logs and progress bar.
        context: Optional scene context prefix.

    Raises:
        Exception: Propagates network/stream errors to the caller for retry.
    """
    progress = tqdm(
        total=expected_total if expected_total else None,
        initial=resume_offset if expected_total else 0,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=label,
        leave=False,
    )

    session_start = time.monotonic()
    last_log_time = session_start
    session_bytes = 0

    try:
        with part_path.open(file_mode) as file_handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_handle.write(chunk)
                file_handle.flush()
                chunk_len = len(chunk)
                session_bytes += chunk_len
                progress.update(chunk_len)

                now = time.monotonic()
                if now - last_log_time >= LOG_PROGRESS_INTERVAL:
                    current_size = part_path.stat().st_size
                    elapsed = now - session_start
                    speed = session_bytes / elapsed if elapsed > 0 else 0.0
                    if expected_total and speed > 0:
                        remaining = max(expected_total - current_size, 0)
                        eta = remaining / speed
                    else:
                        eta = float("inf")
                    logger.debug(
                        "Streaming %s | %sspeed=%s/s | eta=%s | written=%s",
                        label,
                        context,
                        _format_bytes(int(speed)),
                        _format_eta(eta),
                        _format_bytes(current_size),
                    )
                    last_log_time = now
    finally:
        progress.close()


def download_with_retry(
    url: str,
    dest: Path,
    session: requests.Session,
    max_retries: int = 3,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    scene_id: str | None = None,
    asset_name: str | None = None,
) -> bool:
    """Download a remote asset with resume, retry, and integrity verification.

    Downloads stream into ``<dest>.part`` and atomically renames on success.
    Partial files are preserved across transient failures to support HTTP Range
    resume.

    Args:
        url: Signed asset URL.
        dest: Local destination path.
        session: Shared requests session.
        max_retries: Maximum number of download attempts.
        chunk_size: Stream chunk size in bytes.
        scene_id: Optional scene identifier for structured logs.
        asset_name: Optional asset label for structured logs.

    Returns:
        True if the file was downloaded or already exists; False on failure.
    """
    label = asset_name or dest.name
    context = f"scene={scene_id} | " if scene_id else ""

    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("Skipping existing file: %s", dest)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = _part_path(dest)
    download_start = time.monotonic()

    for attempt in range(1, max_retries + 1):
        resume_offset = _existing_resume_offset(part_path)
        headers: dict[str, str] = {}
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"

        logger.info(
            "Downloading %s | %sattempt=%d/%d | resuming=%s",
            label,
            context,
            attempt,
            max_retries,
            _format_bytes(resume_offset),
        )

        response: requests.Response | None = None
        try:
            response = session.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                headers=headers,
            )

            if response.status_code == 416:
                if (
                    resume_offset > 0
                    and part_path.exists()
                    and part_path.stat().st_size == resume_offset
                ):
                    logger.info(
                        "Download already complete (HTTP 416); finalizing %s",
                        label,
                    )
                    _finalize_download(part_path, dest)
                    _log_download_complete(label, context, dest, download_start)
                    return True
                logger.warning(
                    "Range not satisfiable for %s; deleting partial and retrying",
                    label,
                )
                part_path.unlink(missing_ok=True)
                response.close()
                response = None
                if attempt == max_retries:
                    logger.error(
                        "Failed to download %s after %d attempt(s): HTTP 416 Range Not Satisfiable",
                        label,
                        attempt,
                    )
                    return False
                sleep_seconds = _backoff_seconds(attempt)
                logger.warning(
                    "Retrying %s | %sreason=HTTP416 | waiting=%.1fs | resume=0 B",
                    label,
                    context,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue

            if response.status_code == 200 and resume_offset > 0:
                logger.warning(
                    "Server ignored Range header for %s; restarting from byte 0",
                    label,
                )
                response.close()
                part_path.unlink(missing_ok=True)
                resume_offset = 0
                response = session.get(
                    url,
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )

            response.raise_for_status()

            current_resume = _existing_resume_offset(part_path)
            if response.status_code == 206:
                file_mode = "ab" if current_resume > 0 else "wb"
            else:
                file_mode = "wb"
                current_resume = 0

            expected_total = _parse_total_size(
                response.headers,
                current_resume,
                response.status_code,
            )

            _stream_response_to_part(
                response,
                part_path,
                file_mode=file_mode,
                resume_offset=current_resume,
                expected_total=expected_total,
                chunk_size=chunk_size,
                label=label,
                context=context,
            )
            response.close()
            response = None

            if expected_total is not None and not _verify_file_size(
                part_path, expected_total
            ):
                actual_size = part_path.stat().st_size
                logger.error(
                    "Size mismatch for %s | got=%s expected=%s | deleting .part",
                    label,
                    _format_bytes(actual_size),
                    _format_bytes(expected_total),
                )
                part_path.unlink(missing_ok=True)
                if attempt == max_retries:
                    logger.error(
                        "Failed to download %s after %d attempt(s): integrity check failed",
                        label,
                        attempt,
                    )
                    return False
                sleep_seconds = _backoff_seconds(attempt)
                logger.warning(
                    "Retrying %s | %sreason=integrity_mismatch | waiting=%.1fs | resume=0 B",
                    label,
                    context,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue

            _finalize_download(part_path, dest)
            _log_download_complete(label, context, dest, download_start)
            return True

        except OSError as exc:
            if response is not None:
                response.close()

            if is_transient_download_error(exc):
                if attempt == max_retries:
                    resume_after = _existing_resume_offset(part_path)
                    logger.error(
                        "Failed to download %s after %d attempt(s): %s | partial kept at %s (%s)",
                        label,
                        attempt,
                        exc,
                        part_path,
                        _format_bytes(resume_after),
                    )
                    return False

                resume_after = _existing_resume_offset(part_path)
                sleep_seconds = _backoff_seconds(attempt)
                logger.warning(
                    "Retrying %s | %sreason=%s | waiting=%.1fs | resume=%s",
                    label,
                    context,
                    type(exc).__name__,
                    sleep_seconds,
                    _format_bytes(resume_after),
                )
                time.sleep(sleep_seconds)
                continue

            logger.error(
                "File system error while downloading %s | %s%s",
                label,
                context,
                exc,
            )
            return False

        except Exception as exc:
            if response is not None:
                response.close()

            if not is_transient_download_error(exc):
                logger.error(
                    "Non-retryable error downloading %s | %s%s",
                    label,
                    context,
                    exc,
                )
                return False

            if attempt == max_retries:
                resume_after = _existing_resume_offset(part_path)
                logger.error(
                    "Failed to download %s after %d attempt(s): %s | partial kept at %s (%s)",
                    label,
                    attempt,
                    exc,
                    part_path,
                    _format_bytes(resume_after),
                )
                return False

            resume_after = _existing_resume_offset(part_path)
            sleep_seconds = _backoff_seconds(attempt)
            logger.warning(
                "Retrying %s | %sreason=%s | waiting=%.1fs | resume=%s",
                label,
                context,
                type(exc).__name__,
                sleep_seconds,
                _format_bytes(resume_after),
            )
            time.sleep(sleep_seconds)

    return False


def resolve_asset_key(item: Item, logical_name: str) -> str | None:
    """Resolve a logical band name to a STAC asset key on an item.

    Planetary Computer exposes Landsat 8/9 bands under common names (e.g.
    ``blue``, ``lwir11``) rather than USGS product names (e.g. ``SR_B2``,
    ``ST_B10``). Landsat 7 thermal (``lwir``) is intentionally excluded.

    Args:
        item: STAC item containing asset metadata.
        logical_name: Target band name used for output files.

    Returns:
        Matching STAC asset key, or ``None`` if no alias is present.
    """
    candidates = ASSET_KEY_ALIASES.get(logical_name, (logical_name,))
    for candidate in candidates:
        if candidate in item.assets:
            return candidate
    return None


def download_scene_assets(
    item: Item,
    output_dir: Path,
    session: requests.Session,
    assets: Iterable[str] = DEFAULT_ASSETS,
    max_retries: int = 3,
) -> tuple[int, int]:
    """Download selected assets for a single Landsat scene.

    Args:
        item: Signed STAC item (must already be signed in-place).
        output_dir: Directory where scene assets will be stored.
        session: Shared requests session.
        assets: Asset keys to download.
        max_retries: Maximum retry attempts per asset.

    Returns:
        Tuple of ``(downloaded_count, skipped_count)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for logical_name in assets:
        stac_key = resolve_asset_key(item, logical_name)
        if stac_key is None:
            logger.warning(
                "Scene %s is missing asset %s; skipping.",
                item.id,
                logical_name,
            )
            skipped += 1
            continue

        asset = item.assets[stac_key]
        dest_path = output_dir / f"{logical_name}.tif"
        success = download_with_retry(
            url=asset.href,
            dest=dest_path,
            session=session,
            max_retries=max_retries,
            scene_id=item.id,
            asset_name=logical_name,
        )
        if success:
            downloaded += 1
        else:
            skipped += 1

    return downloaded, skipped


def download_dataset(
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    max_items: int | None = None,
    max_retries: int = 3,
    assets: Sequence[str] = DEFAULT_ASSETS,
) -> int:
    """Search and download Landsat C2 L2 scenes for the given parameters.

    Args:
        bbox: Area of interest as ``[minx, miny, maxx, maxy]``.
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.
        max_cloud_cover: Maximum allowed cloud cover percentage.
        output_dir: Root directory for downloaded scenes.
        platforms: Allowed STAC ``platform`` values (default: Landsat 8 and 9).
        max_items: Optional cap on number of scenes to download.
        max_retries: Maximum retry attempts per asset download.
        assets: Asset keys to download for each scene.

    Returns:
        Exit code: ``0`` on success, ``1`` if any asset failed to download.
    """
    validate_bbox(bbox)
    datetime_range = parse_datetime_range(start_date, end_date)

    logger.info(
        "Searching collection=%s platforms=%s bbox=%s datetime=%s max_cloud_cover<=%.1f",
        COLLECTION_ID,
        list(platforms),
        list(bbox),
        datetime_range,
        max_cloud_cover,
    )

    client = build_stac_client()
    items = search_landsat_scenes(
        client=client,
        bbox=bbox,
        datetime_range=datetime_range,
        max_cloud_cover=max_cloud_cover,
        platforms=platforms,
        max_items=max_items,
    )

    if not items:
        logger.warning("No scenes found for the given search parameters.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    failures = 0

    for item in tqdm(items, desc="Scenes", unit="scene"):
        cloud_cover = item.properties.get("eo:cloud_cover", "unknown")
        scene_dir = output_dir / item.id
        logger.info(
            "Processing scene %s (cloud cover: %s%%)",
            item.id,
            cloud_cover,
        )

        downloaded, skipped = download_scene_assets(
            item=item,
            output_dir=scene_dir,
            session=session,
            assets=assets,
            max_retries=max_retries,
        )
        logger.info(
            "Scene %s complete: %d asset(s) ready, %d skipped/failed.",
            item.id,
            downloaded,
            skipped,
        )
        failures += skipped

    if failures:
        logger.error("Finished with %d asset download failure(s).", failures)
        return 1

    logger.info("All requested assets downloaded successfully.")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list override for testing.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download Landsat Collection 2 Level-2 thermal and RGB bands "
            "from Microsoft Planetary Computer."
        ),
    )

    aoi_group = parser.add_mutually_exclusive_group(required=True)
    aoi_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Bounding box in WGS84: min longitude, min latitude, max longitude, max latitude.",
    )
    aoi_group.add_argument(
        "--lon",
        type=float,
        help="Center longitude for point-based AOI (requires --lat and --buffer).",
    )

    parser.add_argument(
        "--lat",
        type=float,
        help="Center latitude for point-based AOI (requires --lon and --buffer).",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        help="Buffer in degrees around --lon/--lat to form a bounding box.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=20.0,
        help="Maximum cloud cover percentage (default: 20).",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=list(DEFAULT_PLATFORMS),
        choices=["landsat-8", "landsat-9"],
        help="Allowed Landsat platforms (default: landsat-8 landsat-9).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for raw scenes (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional maximum number of scenes to download.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum retry attempts per asset download (default: 3).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args(argv)

    if args.lon is not None and args.lat is None:
        parser.error("--lat is required when --lon is provided.")
    if args.lat is not None and args.lon is None:
        parser.error("--lon is required when --lat is provided.")
    if (args.lon is not None or args.lat is not None) and args.buffer is None:
        parser.error("--buffer is required when using --lon and --lat.")
    if args.bbox is not None and (args.lon is not None or args.lat is not None):
        parser.error("Use either --bbox or --lon/--lat, not both.")

    if not 0.0 <= args.max_cloud_cover <= 100.0:
        parser.error("--max-cloud-cover must be between 0 and 100.")

    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be a positive integer.")

    if args.retries <= 0:
        parser.error("--retries must be a positive integer.")

    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset downloader CLI.

    Args:
        argv: Optional argument list override for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        bbox = parse_bbox(args)
        validate_bbox(bbox)
    except ValueError as exc:
        logger.error("Invalid area of interest: %s", exc)
        return 1

    try:
        return download_dataset(
            bbox=bbox,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud_cover=args.max_cloud_cover,
            output_dir=args.output_dir,
            platforms=args.platforms,
            max_items=args.max_items,
            max_retries=args.retries,
        )
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
