"""Download Landsat Collection 2 Level-2 scenes from Microsoft Planetary Computer.

Queries the Planetary Computer STAC API for Landsat 8/9 C2 L2 imagery and
downloads surface temperature (ST_B10) and surface reflectance bands
(SR_B2, SR_B3, SR_B4) into ``dataset/raw/<scene_id>/``.

Two modes:

* Search mode (``--bbox`` or ``--lon/--lat/--buffer`` with dates) downloads every
  matching scene, as before.
* Scene-ID mode (``--scene-ids``) fetches only the exact scenes requested, plus
  ``QA_PIXEL``. Before writing anything it checks that every scene exists in STAC,
  that every required asset is present, gets server-reported sizes via HTTP HEAD,
  checks local files, and checks free disk space. Use ``--dry-run`` to stop after
  these checks.

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
import shutil
import sys
import time
from dataclasses import dataclass, field
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
    "QA_PIXEL": ("qa_pixel", "QA_PIXEL"),
}
SCENE_ID_ASSETS: tuple[str, ...] = DEFAULT_ASSETS + ("QA_PIXEL",)
SCENE_ID_PATTERN = re.compile(r"^L[CO]0[89]_L2SP_\d{6}_\d{8}_\d{2}_(T1|T2|RT)$")
DEFAULT_MIN_FREE_GB = 2.0
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
    expected_size: int | None = None,
    cleanup_partial_on_failure: bool = False,
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
        expected_size: Optional known object size in bytes. When given, an
            existing ``dest`` counts as complete only if it has exactly this size
            (a mismatching file is reported and never overwritten), and a new
            download must end at exactly this size.
        cleanup_partial_on_failure: Delete the ``.part`` file if every attempt
            fails, instead of keeping it for a later resume.

    Returns:
        True if the file was downloaded or already exists; False on failure.
    """
    label = asset_name or dest.name
    context = f"scene={scene_id} | " if scene_id else ""
    part_path = _part_path(dest)

    if expected_size is not None and dest.exists():
        actual = dest.stat().st_size
        if actual == expected_size:
            logger.info("Skipping complete file %s | %ssize=%s", label, context, _format_bytes(actual))
            return True
        logger.error(
            "Existing file %s has size %d but server reports %d; not overwriting. "
            "Inspect or move it manually.",
            dest,
            actual,
            expected_size,
        )
        return False

    if expected_size is not None and part_path.exists() and part_path.stat().st_size > expected_size:
        logger.warning("Partial file %s is larger than the expected size; deleting it.", part_path)
        part_path.unlink()

    existed_before = dest.exists()
    success = _download_with_retry_attempts(
        url, dest, session, max_retries, chunk_size, label, context, expected_size
    )

    if success and expected_size is not None and not existed_before:
        actual = dest.stat().st_size
        if actual != expected_size:
            logger.error(
                "Downloaded %s has size %d, expected %d; deleting it.", dest, actual, expected_size
            )
            dest.unlink()
            success = False

    if not success and cleanup_partial_on_failure and part_path.exists():
        logger.warning("Deleting incomplete partial file %s", part_path)
        part_path.unlink()

    return success


def _download_with_retry_attempts(
    url: str,
    dest: Path,
    session: requests.Session,
    max_retries: int,
    chunk_size: int,
    label: str,
    context: str,
    expected_size: int | None,
) -> bool:
    """Retry loop behind :func:`download_with_retry`."""
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
            if expected_size is not None:
                if expected_total is not None and expected_total != expected_size:
                    logger.error(
                        "Server size for %s changed: GET reports %d, HEAD reported %d",
                        label,
                        expected_total,
                        expected_size,
                    )
                    response.close()
                    return False
                expected_total = expected_size

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
    expected_sizes: Mapping[str, int] | None = None,
    cleanup_partial_on_failure: bool = False,
) -> tuple[int, int]:
    """Download selected assets for a single Landsat scene.

    Args:
        item: Signed STAC item (must already be signed in-place).
        output_dir: Directory where scene assets will be stored.
        session: Shared requests session.
        assets: Asset keys to download.
        max_retries: Maximum retry attempts per asset.
        expected_sizes: Optional server-reported size per logical asset name.
        cleanup_partial_on_failure: Delete ``.part`` files of failed assets.

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
            expected_size=(expected_sizes or {}).get(logical_name),
            cleanup_partial_on_failure=cleanup_partial_on_failure,
        )
        if success:
            downloaded += 1
        else:
            skipped += 1

    return downloaded, skipped


def validate_scene_ids(scene_ids: Sequence[str]) -> list[str]:
    """Normalise and validate exact Landsat C2 L2 scene IDs.

    Args:
        scene_ids: Requested IDs, e.g. ``LC09_L2SP_150041_20250416_02_T1``.

    Returns:
        Stripped IDs with duplicates removed, in request order.

    Raises:
        ValueError: If the list is empty or any ID is malformed.
    """
    cleaned = list(dict.fromkeys(s.strip() for s in scene_ids if s.strip()))
    if not cleaned:
        raise ValueError("--scene-ids needs at least one scene ID.")
    bad = [s for s in cleaned if not SCENE_ID_PATTERN.match(s)]
    if bad:
        raise ValueError(
            f"Malformed Landsat 8/9 C2 L2 scene ID(s): {bad}. "
            "Expected e.g. LC09_L2SP_150041_20250416_02_T1."
        )
    return cleaned


def fetch_scenes_by_id(
    client: pystac_client.Client,
    scene_ids: Sequence[str],
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
) -> list[Item]:
    """Fetch exactly the requested scenes from the STAC catalog.

    Any returned item whose ID was not requested is discarded, so this can never
    select an unrelated scene.

    Args:
        client: Signed Planetary Computer STAC client.
        scene_ids: Validated scene IDs.
        platforms: Allowed STAC ``platform`` values.

    Returns:
        One item per requested ID, in request order.

    Raises:
        RuntimeError: If the search fails, any ID is not in the catalog, or a
            scene is from a disallowed platform.
    """
    requested = list(scene_ids)
    try:
        search = client.search(collections=[COLLECTION_ID], ids=requested, max_items=len(requested) * 2)
        returned = list(search.items())
    except Exception as exc:
        raise RuntimeError(f"STAC lookup by scene ID failed: {exc}") from exc

    by_id: dict[str, Item] = {}
    for item in returned:
        if item.id not in requested:
            logger.warning("Ignoring unrequested STAC item %s", item.id)
            continue
        if item.collection_id not in (None, COLLECTION_ID):
            logger.warning("Ignoring %s from collection %s", item.id, item.collection_id)
            continue
        by_id.setdefault(item.id, item)

    missing = [s for s in requested if s not in by_id]
    if missing:
        raise RuntimeError(f"Scene(s) not found in STAC collection {COLLECTION_ID}: {missing}")

    wrong_platform = [
        f"{s} ({by_id[s].properties.get('platform')})"
        for s in requested
        if by_id[s].properties.get("platform") not in set(platforms)
    ]
    if wrong_platform:
        raise RuntimeError(f"Scene(s) not on allowed platforms {list(platforms)}: {wrong_platform}")

    return [by_id[s] for s in requested]


def head_asset_size(session: requests.Session, url: str, max_retries: int = 3) -> int:
    """Return the server-reported size of an asset via HTTP HEAD.

    Raises:
        RuntimeError: If the asset is unreachable or reports no size.
    """
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.head(url, allow_redirects=True, timeout=(CONNECT_TIMEOUT, 60))
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length is None or int(length) <= 0:
                raise RuntimeError("no Content-Length in HEAD response")
            return int(length)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries and (
                is_transient_download_error(exc) or isinstance(exc, requests.RequestException)
            ):
                time.sleep(_backoff_seconds(attempt))
                continue
            break
    raise RuntimeError(f"HEAD request failed: {last_error}")


@dataclass
class AssetPlan:
    """Planned download of one asset of one scene."""

    logical_name: str
    stac_key: str
    href: str
    dest: Path
    expected_size: int
    local_status: str  # "complete", "partial", "missing" or "size_mismatch"
    local_bytes: int = 0

    @property
    def bytes_to_download(self) -> int:
        if self.local_status == "complete":
            return 0
        if self.local_status == "partial":
            return self.expected_size - self.local_bytes
        return self.expected_size


@dataclass
class ScenePlan:
    """Planned download of one scene."""

    item: Item
    scene_dir: Path
    assets: list[AssetPlan] = field(default_factory=list)


def _local_status(dest: Path, expected_size: int) -> tuple[str, int]:
    if dest.exists():
        size = dest.stat().st_size
        return ("complete" if size == expected_size else "size_mismatch"), size
    part = _part_path(dest)
    if part.exists() and 0 < part.stat().st_size <= expected_size:
        return "partial", part.stat().st_size
    return "missing", 0


def plan_scene_downloads(
    items: Sequence[Item],
    assets: Sequence[str],
    output_dir: Path,
    session: requests.Session,
    max_retries: int = 3,
) -> list[ScenePlan]:
    """Verify assets and measure sizes for each scene without downloading.

    Raises:
        RuntimeError: If any scene lacks a required asset or a size cannot be
            obtained. Nothing is downloaded in that case.
    """
    missing: list[str] = []
    for item in items:
        for name in assets:
            if resolve_asset_key(item, name) is None:
                missing.append(f"{item.id}:{name}")
    if missing:
        raise RuntimeError(f"Required asset(s) missing from STAC items: {missing}")

    plans: list[ScenePlan] = []
    for item in items:
        plan = ScenePlan(item=item, scene_dir=output_dir / item.id)
        for name in assets:
            key = resolve_asset_key(item, name)
            href = item.assets[key].href
            try:
                size = head_asset_size(session, href, max_retries)
            except RuntimeError as exc:
                raise RuntimeError(f"{item.id}:{name}: {exc}") from exc
            dest = plan.scene_dir / f"{name}.tif"
            status, local_bytes = _local_status(dest, size)
            plan.assets.append(AssetPlan(name, key, href, dest, size, status, local_bytes))
        plans.append(plan)
    return plans


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the volume holding ``path`` (or its nearest existing parent)."""
    probe = path.resolve()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def log_download_plan(plans: Sequence[ScenePlan], free_bytes: int, min_free_bytes: int) -> int:
    """Print the per-asset plan and disk summary; return bytes still to download."""
    total_expected = 0
    to_download = 0
    for plan in plans:
        props = plan.item.properties
        logger.info(
            "Scene %s | %s | %s | eo:cloud_cover=%s%% | dest=%s",
            plan.item.id,
            props.get("platform"),
            str(props.get("datetime", ""))[:10],
            props.get("eo:cloud_cover"),
            plan.scene_dir,
        )
        for asset in plan.assets:
            total_expected += asset.expected_size
            to_download += asset.bytes_to_download
            logger.info(
                "    %-8s (stac key %-8s) expected=%s local=%s (%s) to_download=%s",
                asset.logical_name,
                asset.stac_key,
                _format_bytes(asset.expected_size),
                asset.local_status,
                _format_bytes(asset.local_bytes),
                _format_bytes(asset.bytes_to_download),
            )
    logger.info(
        "Total server-reported size: %s | still to download: %s | free disk: %s | "
        "required free after download: %s",
        _format_bytes(total_expected),
        _format_bytes(to_download),
        _format_bytes(free_bytes),
        _format_bytes(min_free_bytes),
    )
    return to_download


def download_scenes_by_id(
    scene_ids: Sequence[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    assets: Sequence[str] = SCENE_ID_ASSETS,
    max_retries: int = 3,
    dry_run: bool = False,
    min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * 1024**3),
    keep_partial: bool = False,
    client: pystac_client.Client | None = None,
    session: requests.Session | None = None,
) -> int:
    """Verify and download exactly the requested scenes.

    Every check (catalog lookup, required assets, HEAD sizes, local files, disk
    space) completes for all scenes before any byte is written.

    Returns:
        ``0`` if every asset is complete (or, with ``dry_run``, every check passed);
        ``1`` otherwise.
    """
    ids = validate_scene_ids(scene_ids)
    logger.info("Scene-ID mode: %d scene(s) requested, assets=%s", len(ids), list(assets))

    client = client or build_stac_client()
    items = fetch_scenes_by_id(client, ids, platforms)
    logger.info("All %d requested scene(s) found in STAC.", len(items))

    session = session or requests.Session()
    plans = plan_scene_downloads(items, assets, output_dir, session, max_retries)

    mismatched = [
        str(asset.dest) for plan in plans for asset in plan.assets if asset.local_status == "size_mismatch"
    ]
    free_bytes = free_disk_bytes(output_dir)
    to_download = log_download_plan(plans, free_bytes, min_free_bytes)

    if mismatched:
        logger.error(
            "Existing file(s) do not match server-reported sizes and will not be overwritten: %s",
            mismatched,
        )
        return 1
    if free_bytes - to_download < min_free_bytes:
        logger.error(
            "Insufficient disk space: need %s plus %s reserve, only %s free. Nothing downloaded.",
            _format_bytes(to_download),
            _format_bytes(min_free_bytes),
            _format_bytes(free_bytes),
        )
        return 1
    if dry_run:
        logger.info("Dry run: all checks passed; nothing downloaded.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete: list[str] = []
    for plan in tqdm(plans, desc="Scenes", unit="scene"):
        fresh_item = fetch_scenes_by_id(client, [plan.item.id], platforms)[0]
        download_scene_assets(
            item=fresh_item,
            output_dir=plan.scene_dir,
            session=session,
            assets=[a.logical_name for a in plan.assets],
            max_retries=max_retries,
            expected_sizes={a.logical_name: a.expected_size for a in plan.assets},
            cleanup_partial_on_failure=not keep_partial,
        )
        for asset in plan.assets:
            status, _ = _local_status(asset.dest, asset.expected_size)
            if status != "complete":
                incomplete.append(f"{plan.item.id}:{asset.logical_name} ({status})")
        logger.info("Scene %s verified: %s", plan.item.id,
                    "complete" if not any(i.startswith(plan.item.id) for i in incomplete) else "INCOMPLETE")

    if incomplete:
        logger.error("Download INCOMPLETE for %d asset(s): %s", len(incomplete), incomplete)
        return 1
    logger.info("All %d scene(s) complete and size-verified.", len(plans))
    return 0


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
    aoi_group.add_argument(
        "--scene-ids",
        nargs="+",
        metavar="SCENE_ID",
        help=(
            "Download only these exact Landsat 8/9 C2 L2 scene IDs (no search). "
            "Cannot be combined with AOI, date, cloud or --max-items options."
        ),
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
        help="Start date in YYYY-MM-DD format (required in search mode).",
    )
    parser.add_argument(
        "--end-date",
        help="End date in YYYY-MM-DD format (required in search mode).",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=None,
        help="Maximum cloud cover percentage (search mode only; default: 20).",
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
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=list(ASSET_KEY_ALIASES),
        default=None,
        help=(
            "Assets to download (scene-ID mode only; default: "
            f"{' '.join(SCENE_ID_ASSETS)})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scene-ID mode: run every check and print the plan, but download nothing.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            "Scene-ID mode: free space that must remain after downloading "
            f"(default: {DEFAULT_MIN_FREE_GB})."
        ),
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Scene-ID mode: keep .part files of failed assets for a later resume.",
    )

    args = parser.parse_args(argv)

    if args.scene_ids is not None:
        conflicting = [
            flag
            for flag, value in (
                ("--lat", args.lat),
                ("--buffer", args.buffer),
                ("--start-date", args.start_date),
                ("--end-date", args.end_date),
                ("--max-cloud-cover", args.max_cloud_cover),
                ("--max-items", args.max_items),
            )
            if value is not None
        ]
        if conflicting:
            parser.error(f"--scene-ids cannot be combined with {', '.join(conflicting)}.")
        try:
            args.scene_ids = validate_scene_ids(args.scene_ids)
        except ValueError as exc:
            parser.error(str(exc))
        if args.min_free_gb < 0:
            parser.error("--min-free-gb must be non-negative.")
        if args.retries <= 0:
            parser.error("--retries must be a positive integer.")
        return args

    scene_only = [
        flag
        for flag, value in (
            ("--assets", args.assets),
            ("--dry-run", args.dry_run or None),
            ("--keep-partial", args.keep_partial or None),
        )
        if value is not None
    ]
    if scene_only:
        parser.error(f"{', '.join(scene_only)} require --scene-ids.")
    missing_dates = [f for f, v in (("--start-date", args.start_date), ("--end-date", args.end_date)) if v is None]
    if missing_dates:
        parser.error(f"the following arguments are required: {', '.join(missing_dates)}")
    if args.max_cloud_cover is None:
        args.max_cloud_cover = 20.0

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

    if args.scene_ids is not None:
        try:
            return download_scenes_by_id(
                scene_ids=args.scene_ids,
                output_dir=args.output_dir,
                platforms=args.platforms,
                assets=args.assets or SCENE_ID_ASSETS,
                max_retries=args.retries,
                dry_run=args.dry_run,
                min_free_bytes=int(args.min_free_gb * 1024**3),
                keep_partial=args.keep_partial,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error("%s", exc)
            return 1

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
