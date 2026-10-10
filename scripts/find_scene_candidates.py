"""Find diverse Landsat 8/9 C2 L2 candidate scenes across Indian climate zones (read-only).

Queries the Planetary Computer STAC API with the helpers in ``scripts.download_dataset`` and
writes a candidate report. Nothing is downloaded: asset sizes come from HTTP HEAD requests
(server-reported Content-Length) or are explicitly labelled estimates.

Usage (from the project root):
    py -m scripts.find_scene_candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests
from pystac import Item

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_dataset import (  # noqa: E402
    COLLECTION_ID,
    DEFAULT_ASSETS,
    STAC_API_URL,
    build_stac_client,
    resolve_asset_key,
    search_landsat_scenes,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_expansion"
REFERENCE_SCENE_ID = "LC09_L2SP_143050_20240327_02_T1"
MONSOON_MONTHS = {6, 7, 8, 9}
EDGE_MARGIN_WARN_DEG = 0.3
AOI_BUFFER_DEG = 0.05

Polygon = list[tuple[float, float]]


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    lon: float
    lat: float
    months: tuple[int, ...]
    season: str
    character: str
    snow_possible: bool = False


REGIONS: tuple[Region, ...] = (
    Region("thar", "Thar Desert, Rajasthan (Jaisalmer)", 70.91, 26.92, (4, 5, 6), "pre-monsoon (Apr-Jun)",
           "hot arid desert, sand dunes, sparse vegetation"),
    Region("igp", "Indo-Gangetic Plain, Uttar Pradesh (Lucknow)", 80.95, 26.85, (11, 12, 1, 2), "winter (Nov-Feb)",
           "irrigated cropland, dense urban area, rivers"),
    Region("deccan", "Deccan Plateau, Telangana (Hyderabad)", 78.47, 17.39, (3, 4, 5), "summer (Mar-May)",
           "semi-arid plateau, large city, exposed rock, reservoirs"),
    Region("kerala", "Kerala coast / Western Ghats (Kochi)", 76.27, 9.98, (1, 2, 3), "dry season (Jan-Mar)",
           "humid tropical coast, backwaters, forested ghats, sea surface"),
    Region("himalaya", "Himalayan foothills, Uttarakhand (Dehradun)", 78.03, 30.32, (10, 11), "post-monsoon (Oct-Nov)",
           "strong elevation gradient, forest, valley city, possible high-altitude snow", snow_possible=True),
    Region("assam", "Brahmaputra valley, Assam (Guwahati)", 91.74, 26.14, (11, 12, 1, 2), "winter (Nov-Feb)",
           "humid floodplain, wide braided river, tea/forest, hills"),
    Region("kutch", "Rann of Kutch, Gujarat", 70.20, 23.85, (12, 1, 2), "winter (Dec-Feb)",
           "salt flats and marsh with strong thermal contrast, coast"),
)


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")


def item_polygon(item: Item) -> Polygon:
    """Outer ring of the item footprint as (lon, lat) vertices (no closing duplicate)."""
    geom = item.geometry or {}
    if geom.get("type") == "Polygon":
        ring = geom["coordinates"][0]
    elif geom.get("type") == "MultiPolygon":
        ring = max((p[0] for p in geom["coordinates"]), key=lambda r: abs(polygon_area(r)))
    else:
        minx, miny, maxx, maxy = item.bbox
        ring = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    pts = [(float(x), float(y)) for x, y in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    """Signed shoelace area in degrees^2."""
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i][:2]
        x2, y2 = poly[(i + 1) % len(poly)][:2]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _ccw(poly: Polygon) -> Polygon:
    return poly if polygon_area(poly) > 0 else poly[::-1]


def clip_convex(subject: Polygon, clip: Polygon) -> Polygon:
    """Sutherland-Hodgman intersection; Landsat footprints are near-convex quadrilaterals."""
    subject, clip = _ccw(subject), _ccw(clip)

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        ex, ey = b[0] - a[0], b[1] - a[1]
        denom = dx * ey - dy * ex
        if denom == 0:
            return p2
        t = ((a[0] - p1[0]) * ey - (a[1] - p1[1]) * ex) / denom
        return (p1[0] + t * dx, p1[1] + t * dy)

    output = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, output = output, []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            if inside(cur, a, b):
                if not inside(prev, a, b):
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif inside(prev, a, b):
                output.append(intersect(prev, cur, a, b))
            prev = cur
    return output


def overlap_fraction(a: Polygon, b: Polygon) -> float:
    """Intersection area as a fraction of the smaller footprint (planar lon/lat approximation)."""
    inter = clip_convex(a, b)
    if len(inter) < 3:
        return 0.0
    return abs(polygon_area(inter)) / min(abs(polygon_area(a)), abs(polygon_area(b)))


def point_in_polygon(lon: float, lat: float, poly: Polygon) -> bool:
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def edge_margin_deg(lon: float, lat: float, poly: Polygon) -> float:
    """Distance (degrees) from the AOI point to the nearest footprint edge; negative if outside."""
    best = math.inf
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / (dx * dx + dy * dy or 1.0)))
        best = min(best, math.hypot(lon - (x1 + t * dx), lat - (y1 + t * dy)))
    return best if point_in_polygon(lon, lat, poly) else -best


def head_size(url: str, session: requests.Session) -> int | None:
    try:
        resp = session.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except Exception as exc:
        logger.warning("HEAD failed (%s); size will be estimated", type(exc).__name__)
        return None


def reference_band_sizes(raw_dir: Path) -> dict[str, int]:
    return {b: (raw_dir / f"{b}.tif").stat().st_size for b in DEFAULT_ASSETS if (raw_dir / f"{b}.tif").is_file()}


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def describe_item(item: Item) -> dict[str, Any]:
    p = item.properties
    poly = item_polygon(item)
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    return {
        "scene_id": item.id,
        "source": "stac",
        "platform": p.get("platform"),
        "acquisition_datetime_utc": p.get("datetime"),
        "wrs_path": p.get("landsat:wrs_path"),
        "wrs_row": p.get("landsat:wrs_row"),
        "collection_category": p.get("landsat:collection_category"),
        "eo_cloud_cover_pct": p.get("eo:cloud_cover"),
        "cloud_cover_land_pct": p.get("landsat:cloud_cover_land"),
        "sun_elevation_deg": p.get("view:sun_elevation"),
        "footprint_bbox": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
        "footprint_centroid": [round(sum(xs) / len(xs), 4), round(sum(ys) / len(ys), 4)],
    }


def required_assets(item: Item) -> dict[str, str | None]:
    return {name: resolve_asset_key(item, name) for name in DEFAULT_ASSETS}


def rank_key(item: Item, region: Region) -> tuple:
    p = item.properties
    land = p.get("landsat:cloud_cover_land")
    cloud = p.get("eo:cloud_cover", 100.0)
    margin = edge_margin_deg(region.lon, region.lat, item_polygon(item))
    return (
        p.get("landsat:collection_category") != "T1",
        margin < EDGE_MARGIN_WARN_DEG,
        land if land is not None and land >= 0 else cloud,
        cloud,
        -margin,
    )


def search_region(client, region: Region, start: str, end: str, max_cloud: float) -> list[Item]:
    bbox = [region.lon - AOI_BUFFER_DEG, region.lat - AOI_BUFFER_DEG,
            region.lon + AOI_BUFFER_DEG, region.lat + AOI_BUFFER_DEG]
    return search_landsat_scenes(client, bbox, f"{start}/{end}", max_cloud)


def candidate_record(item: Item, region: Region, *, off_season: bool, selected_polys: dict[str, Polygon],
                     ref_poly: Polygon, ref_item: Item, session: requests.Session | None,
                     ref_sizes: dict[str, int], max_cloud: float) -> dict[str, Any]:
    rec = describe_item(item)
    poly = item_polygon(item)
    month = datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00")).month
    margin = edge_margin_deg(region.lon, region.lat, poly)

    assets = {}
    for name, key in required_assets(item).items():
        entry: dict[str, Any] = {"stac_asset_key": key, "present": key is not None}
        if key is not None:
            size = head_size(item.assets[key].href, session) if session is not None else None
            if size is not None:
                entry.update(size_bytes=size, size_source="http_head (server-reported Content-Length, not downloaded)")
            elif name in ref_sizes:
                entry.update(size_bytes=ref_sizes[name],
                             size_source=f"estimate (same band of {ref_item.id})")
        assets[name] = entry
    rec["required_assets"] = assets
    rec["optional_qa_pixel_available"] = "qa_pixel" in item.assets
    rec["raw_size_bytes"] = sum(a.get("size_bytes", 0) for a in assets.values())
    rec["raw_size_source"] = ("http_head" if all("http_head" in a.get("size_source", "") for a in assets.values())
                              else "partly or fully estimated")

    overlaps = {"reference": round(overlap_fraction(poly, ref_poly), 4)}
    overlaps.update({k: round(overlap_fraction(poly, p), 4) for k, p in selected_polys.items()})
    rec["overlap_fraction_of_smaller_footprint"] = overlaps
    rec["aoi_edge_margin_deg"] = round(margin, 3)

    reason = (f"{region.character}; {region.season} acquisition; lowest land cloud cover among "
              f"{'all' if off_season else 'in-season'} T1 scenes with all required bands and the AOI "
              f"{margin:.2f} deg inside the footprint edge")
    rec["region"] = region.name
    rec["reason_for_selection"] = reason

    concerns = []
    cloud = item.properties.get("eo:cloud_cover")
    if cloud is not None and cloud > 5:
        concerns.append(f"eo:cloud_cover {cloud}% (> 5%)")
    if item.properties.get("landsat:collection_category") != "T1":
        concerns.append(f"collection category {item.properties.get('landsat:collection_category')} (not T1)")
    if not rec["optional_qa_pixel_available"]:
        concerns.append("no qa_pixel asset")
    if 0 <= margin < EDGE_MARGIN_WARN_DEG:
        concerns.append(f"AOI point only {margin:.2f} deg from footprint edge")
    if month in MONSOON_MONTHS:
        concerns.append("monsoon-season acquisition (higher cloud/haze risk)")
    if off_season:
        concerns.append(f"no scene <= {max_cloud}% cloud in target season; outside target months")
    if region.snow_possible:
        concerns.append("snow/ice possible at high elevations (thermal extremes)")
    if any(v > 0 for v in overlaps.values()):
        concerns.append("footprint overlaps another selected/reference scene")
    concerns.append("cloud cover is USGS STAC metadata only; scene is not verified cloud-free")
    rec["quality_concerns"] = concerns
    return rec


def find_candidates(args: argparse.Namespace) -> dict[str, Any]:
    client = build_stac_client()
    session = None if args.no_size_check else requests.Session()

    ref_item = next(client.search(collections=[COLLECTION_ID], ids=[REFERENCE_SCENE_ID]).items(), None)
    if ref_item is None:
        raise RuntimeError(f"Reference scene {REFERENCE_SCENE_ID} not found in STAC")
    ref_poly = item_polygon(ref_item)
    ref_pathrow = (ref_item.properties.get("landsat:wrs_path"), ref_item.properties.get("landsat:wrs_row"))
    ref_sizes = reference_band_sizes(PROJECT_ROOT / "dataset" / "raw" / REFERENCE_SCENE_ID)
    ref_processed = dir_size(PROJECT_ROOT / "dataset" / "processed" / REFERENCE_SCENE_ID)

    selected_polys: dict[str, Polygon] = {}
    regions_out: list[dict[str, Any]] = []
    for region in REGIONS:
        logger.info("Searching %s ...", region.name)
        items = search_region(client, region, args.start_date, args.end_date, args.max_cloud_cover)
        eligible = []
        rejected = {"missing_required_asset": 0, "aoi_outside_footprint": 0, "overlaps_reference": 0,
                    "overlaps_selected": 0}
        for it in items:
            if any(k is None for k in required_assets(it).values()):
                rejected["missing_required_asset"] += 1
                continue
            poly = item_polygon(it)
            if edge_margin_deg(region.lon, region.lat, poly) < 0:
                rejected["aoi_outside_footprint"] += 1
                continue
            pr = (it.properties.get("landsat:wrs_path"), it.properties.get("landsat:wrs_row"))
            if pr == ref_pathrow or overlap_fraction(poly, ref_poly) > 0:
                rejected["overlaps_reference"] += 1
                continue
            if any(overlap_fraction(poly, p) > 0 for p in selected_polys.values()):
                rejected["overlaps_selected"] += 1
                continue
            eligible.append(it)

        def month_of(it: Item) -> int:
            return datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")).month

        in_season = sorted((it for it in eligible if month_of(it) in region.months), key=lambda i: rank_key(i, region))
        off = sorted((it for it in eligible if month_of(it) not in region.months), key=lambda i: rank_key(i, region))
        ranked = [(it, False) for it in in_season] + [(it, True) for it in off]

        entry: dict[str, Any] = {
            "region": region.name, "aoi_point_lon_lat": [region.lon, region.lat], "target_season": region.season,
            "character": region.character, "stac_items_returned": len(items), "eligible_after_filters": len(eligible),
            "eligible_in_target_season": len(in_season), "rejected_counts": rejected,
            "selected": None, "alternates": [],
        }
        if ranked:
            top, off_flag = ranked[0]
            entry["selected"] = candidate_record(top, region, off_season=off_flag, selected_polys=selected_polys,
                                                 ref_poly=ref_poly, ref_item=ref_item, session=session,
                                                 ref_sizes=ref_sizes, max_cloud=args.max_cloud_cover)
            alt_seen = {top.id}
            for it, off_flag in ranked[1:]:
                if len(entry["alternates"]) >= args.alternates:
                    break
                pr = (it.properties.get("landsat:wrs_path"), it.properties.get("landsat:wrs_row"))
                if it.id in alt_seen:
                    continue
                alt_seen.add(it.id)
                rec = candidate_record(it, region, off_season=off_flag, selected_polys=selected_polys,
                                       ref_poly=ref_poly, ref_item=ref_item, session=session,
                                       ref_sizes=ref_sizes, max_cloud=args.max_cloud_cover)
                rec["same_path_row_as_selected"] = pr == (top.properties.get("landsat:wrs_path"),
                                                         top.properties.get("landsat:wrs_row"))
                entry["alternates"].append(rec)
            selected_polys[top.id] = item_polygon(top)
            logger.info("  selected %s (cloud %s%%, land %s%%)", top.id, top.properties.get("eo:cloud_cover"),
                        top.properties.get("landsat:cloud_cover_land"))
        else:
            logger.warning("  no eligible scene for %s", region.name)
        regions_out.append(entry)

    selected = [r["selected"] for r in regions_out if r["selected"]]
    raw_total = sum(s["raw_size_bytes"] for s in selected)
    ref_raw = sum(ref_sizes.values())
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "CANDIDATE REPORT ONLY - nothing downloaded; awaiting approval",
        "query": {
            "stac_api": STAC_API_URL, "collection": COLLECTION_ID, "platforms": ["landsat-8", "landsat-9"],
            "date_range": f"{args.start_date}/{args.end_date}", "max_eo_cloud_cover_pct": args.max_cloud_cover,
            "aoi_buffer_deg": AOI_BUFFER_DEG, "required_assets": list(DEFAULT_ASSETS),
            "selection_rules": [
                "all four required assets resolvable (ST_B10/lwir11, SR_B2/blue, SR_B3/green, SR_B4/red)",
                "AOI point inside footprint",
                "different WRS path/row from the reference and zero footprint overlap with the reference and "
                "previously selected scenes (scene-level independence)",
                f"prefer target-season months, then T1, then AOI >= {EDGE_MARGIN_WARN_DEG} deg inside the footprint "
                "edge, then lowest landsat:cloud_cover_land, then eo:cloud_cover, then largest AOI edge margin",
            ],
        },
        "reference_scene": {**describe_item(ref_item), "role": "existing scene (unchanged)",
                            "raw_size_bytes_on_disk": ref_raw, "processed_size_bytes_on_disk": ref_processed},
        "regions": regions_out,
        "summary": {
            "regions_searched": len(REGIONS),
            "candidates_selected": len(selected),
            "total_scenes_including_reference": len(selected) + 1,
            "proposed_scene_ids": [s["scene_id"] for s in selected],
            "raw_download_bytes": raw_total,
            "raw_download_source": ("http_head" if all(s["raw_size_source"] == "http_head" for s in selected)
                                    else "partly estimated"),
            "processed_bytes_estimate": ref_processed * len(selected),
            "processed_estimate_basis": (f"estimate: {len(selected)} x measured processed size of {REFERENCE_SCENE_ID} "
                                         "(256 px patches, stride 128, float32 thermal+RGB). Actual size scales with "
                                         "each scene's valid-pixel coverage, unknown before download."),
        },
        "scientific_notes": [
            "Future SR experiments remain synthetic x2 reconstruction against the 30 m Landsat thermal reference, "
            "not 15 m thermal ground truth.",
            "Train/validation/test splits must be made at scene level; overlapping patches from one scene must "
            "never be treated as geographically independent.",
            "Each scene keeps its own metadata, normalization statistics, validity masks and patch IDs.",
        ],
    }


def _mb(n: int) -> str:
    return f"{n / 2**20:,.0f} MB" if n < 2**30 else f"{n / 2**30:,.2f} GB"


def render_text(r: dict[str, Any]) -> str:
    ref = r["reference_scene"]
    s = r["summary"]
    lines = [
        "LANDSAT SCENE CANDIDATES FOR DATASET EXPANSION",
        "=" * 72,
        r["status"],
        f"Generated (UTC): {r['generated_utc']}",
        f"Query: {r['query']['collection']} | {', '.join(r['query']['platforms'])} | {r['query']['date_range']} | "
        f"eo:cloud_cover <= {r['query']['max_eo_cloud_cover_pct']}%",
        "",
        "Provenance: [STAC] = Planetary Computer STAC metadata; [HEAD] = server-reported file size via HTTP HEAD "
        "(not downloaded); [EST] = estimate.",
        "",
        "REFERENCE (existing, unchanged)",
        f"  {ref['scene_id']}  {ref['platform']}  {ref['acquisition_datetime_utc'][:10]}  path/row "
        f"{ref['wrs_path']}/{ref['wrs_row']}  cloud {ref['eo_cloud_cover_pct']}% (land {ref['cloud_cover_land_pct']}%) "
        f"[STAC]",
        f"  footprint bbox {ref['footprint_bbox']} [STAC]; raw {_mb(ref['raw_size_bytes_on_disk'])}, "
        f"processed {_mb(ref['processed_size_bytes_on_disk'])} (measured on disk)",
        "",
        "SELECTED CANDIDATES",
        "-" * 72,
    ]
    for i, reg in enumerate(r["regions"], start=1):
        c = reg["selected"]
        lines.append(f"{i}. {reg['region']}  (target {reg['target_season']})")
        lines.append(f"   STAC items {reg['stac_items_returned']}, eligible {reg['eligible_after_filters']} "
                     f"({reg['eligible_in_target_season']} in season); rejected {reg['rejected_counts']}")
        if c is None:
            lines += ["   NO SUITABLE SCENE FOUND", ""]
            continue
        sizes = ", ".join(f"{k} {_mb(v.get('size_bytes', 0))}" for k, v in c["required_assets"].items())
        lines += [
            f"   Scene:      {c['scene_id']}",
            f"   [STAC]      {c['platform']}, {c['acquisition_datetime_utc'][:10]}, path/row {c['wrs_path']}/{c['wrs_row']}, "
            f"category {c['collection_category']}, sun elev {c['sun_elevation_deg']:.1f} deg",
            f"   [STAC]      cloud {c['eo_cloud_cover_pct']}% (land {c['cloud_cover_land_pct']}%), "
            f"footprint bbox {c['footprint_bbox']}",
            f"   [STAC]      required assets present: {all(v['present'] for v in c['required_assets'].values())}; "
            f"qa_pixel: {c['optional_qa_pixel_available']}",
            f"   [{'HEAD' if c['raw_size_source'] == 'http_head' else 'EST'}]      raw size {_mb(c['raw_size_bytes'])} ({sizes})",
            f"   Overlap:    {c['overlap_fraction_of_smaller_footprint']}",
            f"   Reason:     {c['reason_for_selection']}",
            "   Concerns:   " + "; ".join(c["quality_concerns"]),
        ]
        if reg["alternates"]:
            alts = ", ".join(f"{a['scene_id']} ({a['acquisition_datetime_utc'][:10]}, {a['eo_cloud_cover_pct']}%)"
                             for a in reg["alternates"])
            lines.append(f"   Alternates: {alts}")
        lines.append("")
    lines += [
        "SUMMARY",
        "-" * 72,
        f"Candidates selected: {s['candidates_selected']} (total with reference: {s['total_scenes_including_reference']})",
        f"Proposed scene IDs:  {', '.join(s['proposed_scene_ids'])}",
        f"Raw download [{'HEAD' if s['raw_download_source'] == 'http_head' else 'partly EST'}]: "
        f"{_mb(s['raw_download_bytes'])}",
        f"Processed patches [EST]: ~{_mb(s['processed_bytes_estimate'])} ({s['processed_estimate_basis']})",
        "",
        "SCIENTIFIC NOTES",
        "-" * 72,
        *[f"- {n}" for n in r["scientific_notes"]],
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find diverse Landsat C2 L2 candidate scenes (no downloads).")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2025-06-30")
    p.add_argument("--max-cloud-cover", type=float, default=10.0)
    p.add_argument("--alternates", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-size-check", action="store_true", help="Skip HTTP HEAD size checks")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        report = find_candidates(args)
    except Exception as exc:
        logger.exception("Candidate search failed: %s", exc)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scene_candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = render_text(report)
    (args.output_dir / "scene_candidates.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
