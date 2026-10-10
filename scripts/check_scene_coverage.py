"""Check how well a Landsat scene footprint covers a target location (read-only STAC queries).

Reports, from the STAC footprint geometry: whether the target point is inside,
its distance to the footprint edge, and what fraction of an approximate target
extent the footprint covers. Optionally lists other low-cloud scenes in a season
window ranked by that coverage. Nothing is downloaded.

    py -m scripts.check_scene_coverage --scene-id LC08_L2SP_150044_20230214_02_T1 --region kutch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.download_dataset import build_stac_client, fetch_scenes_by_id, search_landsat_scenes
    from scripts.find_scene_candidates import (
        EDGE_MARGIN_WARN_DEG,
        clip_convex,
        edge_margin_deg,
        item_polygon,
        point_in_polygon,
        polygon_area,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_dataset import build_stac_client, fetch_scenes_by_id, search_landsat_scenes
    from find_scene_candidates import (
        EDGE_MARGIN_WARN_DEG,
        clip_convex,
        edge_margin_deg,
        item_polygon,
        point_in_polygon,
        polygon_area,
    )

# Hand-drawn approximate extents (lon, lat), not authoritative boundaries.
TARGETS: dict[str, dict[str, Any]] = {
    "kutch": {
        "name": "Rann of Kutch, Gujarat",
        "point": (70.20, 23.85),
        "season_months": (12, 1, 2),
        "extents": {
            "great_rann_approx": [(68.9, 23.6), (71.2, 23.6), (71.2, 24.6), (68.9, 24.6)],
            "little_rann_approx": [(70.8, 23.0), (71.8, 23.0), (71.8, 23.6), (70.8, 23.6)],
        },
    },
}


def coverage(item, target: dict[str, Any]) -> dict[str, Any]:
    poly = item_polygon(item)
    lon, lat = target["point"]
    covered = {}
    for name, extent in target["extents"].items():
        inter = clip_convex(extent, poly)
        covered[name] = round(abs(polygon_area(inter)) / abs(polygon_area(extent)), 3) if inter else 0.0
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return {
        "scene_id": item.id,
        "date": str(item.properties.get("datetime", ""))[:10],
        "path_row": f"{item.properties.get('landsat:wrs_path')}/{item.properties.get('landsat:wrs_row')}",
        "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
        "cloud_cover_land": item.properties.get("landsat:cloud_cover_land"),
        "footprint_bbox": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
        "point_inside_footprint": point_in_polygon(lon, lat, poly),
        "point_edge_margin_deg": round(edge_margin_deg(lon, lat, poly), 3),
        "extent_fraction_covered_estimate": covered,
    }


def alternatives(client, target: dict[str, Any], start: str, end: str, max_cloud: float) -> list[dict]:
    lon, lat = target["point"]
    xs = [p[0] for e in target["extents"].values() for p in e]
    ys = [p[1] for e in target["extents"].values() for p in e]
    bbox = [min(xs + [lon]), min(ys + [lat]), max(xs + [lon]), max(ys + [lat])]
    items = search_landsat_scenes(client, bbox, f"{start}/{end}", max_cloud)
    rows = []
    for item in items:
        if int(str(item.properties.get("datetime", ""))[5:7]) not in target["season_months"]:
            continue
        if item.properties.get("landsat:collection_category") != "T1":
            continue
        rec = coverage(item, target)
        if rec["point_inside_footprint"]:
            rows.append(rec)
    rows.sort(key=lambda r: (-r["point_edge_margin_deg"], r["cloud_cover_land"] or 0))
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check scene footprint coverage of a target (no downloads).")
    p.add_argument("--scene-id", required=True)
    p.add_argument("--region", choices=list(TARGETS), required=True)
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2025-06-30")
    p.add_argument("--max-cloud-cover", type=float, default=10.0)
    p.add_argument("--top", type=int, default=8, help="Alternatives to list.")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target = TARGETS[args.region]
    client = build_stac_client()
    scene = coverage(fetch_scenes_by_id(client, [args.scene_id])[0], target)
    alts = alternatives(client, target, args.start_date, args.end_date, args.max_cloud_cover)[: args.top]

    report = {
        "status": "COVERAGE CHECK ONLY - nothing downloaded",
        "target": {"name": target["name"], "point_lon_lat": target["point"],
                   "extents_note": "approximate hand-drawn extents; coverage fractions are estimates "
                                   "from STAC footprint geometry (planar lon/lat)",
                   "edge_margin_threshold_deg": EDGE_MARGIN_WARN_DEG},
        "scene": scene,
        "marginal": (not scene["point_inside_footprint"])
                    or scene["point_edge_margin_deg"] < EDGE_MARGIN_WARN_DEG,
        "in_season_alternatives_by_edge_margin": alts,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
