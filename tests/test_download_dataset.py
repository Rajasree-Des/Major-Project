"""Offline tests for scripts/download_dataset.py (no network access)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pystac import Asset, Item

from scripts import download_dataset as dd

SCENE_A = "LC09_L2SP_150041_20250416_02_T1"
SCENE_B = "LC08_L2SP_144048_20240427_02_T1"
UNRELATED = "LC08_L2SP_144048_20240411_02_T1"
ALL_KEYS = {"ST_B10": "lwir11", "SR_B2": "blue", "SR_B3": "green", "SR_B4": "red", "QA_PIXEL": "qa_pixel"}


def make_item(scene_id: str, platform: str = "landsat-9", skip: tuple[str, ...] = ()) -> Item:
    item = Item(
        id=scene_id,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        bbox=[0, 0, 1, 1],
        datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        properties={"platform": platform, "eo:cloud_cover": 0.1},
        collection=dd.COLLECTION_ID,
    )
    for logical, key in ALL_KEYS.items():
        if logical not in skip:
            item.add_asset(key, Asset(href=f"https://example.test/{scene_id}/{key}.tif"))
    return item


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class FakeClient:
    """Returns its catalog for any search, ignoring the requested IDs, to prove filtering."""

    def __init__(self, items):
        self.catalog = items
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return FakeSearch(self.catalog)


class FakeResponse:
    def __init__(self, status_code=200, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


class FakeSession:
    """Serves deterministic bytes per URL; ``truncate`` URLs send fewer bytes than advertised."""

    def __init__(self, size=1000, truncate=()):
        self.size = size
        self.truncate = set(truncate)
        self.head_calls: list[str] = []
        self.get_calls: list[str] = []

    def head(self, url, **kwargs):
        self.head_calls.append(url)
        return FakeResponse(headers={"Content-Length": str(self.size)})

    def get(self, url, stream=True, timeout=None, headers=None):
        self.get_calls.append(url)
        body = bytes(self.size)
        if url in self.truncate:
            body = body[: self.size // 2]
        return FakeResponse(headers={"Content-Length": str(self.size)}, body=body)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)


# ---------------------------------------------------------------- argument parsing

def test_search_mode_arguments_unchanged():
    args = dd.parse_args(["--bbox", "78", "13", "80", "15", "--start-date", "2024-01-01",
                          "--end-date", "2024-12-31"])
    assert args.scene_ids is None
    assert args.max_cloud_cover == 20.0
    assert args.bbox == [78.0, 13.0, 80.0, 15.0]


def test_search_mode_still_requires_dates():
    with pytest.raises(SystemExit):
        dd.parse_args(["--bbox", "78", "13", "80", "15"])


def test_an_aoi_or_scene_ids_is_required():
    with pytest.raises(SystemExit):
        dd.parse_args(["--start-date", "2024-01-01", "--end-date", "2024-12-31"])


def test_scene_ids_parsed_and_deduplicated():
    args = dd.parse_args(["--scene-ids", SCENE_A, SCENE_B, SCENE_A, "--dry-run"])
    assert args.scene_ids == [SCENE_A, SCENE_B]
    assert args.dry_run


@pytest.mark.parametrize("extra", [
    ["--bbox", "78", "13", "80", "15"],
    ["--lon", "78"],
    ["--start-date", "2024-01-01"],
    ["--end-date", "2024-12-31"],
    ["--max-cloud-cover", "5"],
    ["--max-items", "1"],
    ["--buffer", "0.1"],
])
def test_scene_ids_reject_search_options(extra):
    with pytest.raises(SystemExit):
        dd.parse_args(["--scene-ids", SCENE_A, *extra])


@pytest.mark.parametrize("bad", ["LC09_L2SP_150041", "LE07_L2SP_150041_20250416_02_T1",
                                 "lc09_l2sp_150041_20250416_02_t1", "LC09_L2SP_150041_20250416_02_T1x"])
def test_malformed_scene_ids_rejected(bad):
    with pytest.raises(SystemExit):
        dd.parse_args(["--scene-ids", bad])


@pytest.mark.parametrize("flag", ["--dry-run", "--keep-partial"])
def test_scene_only_flags_require_scene_ids(flag):
    with pytest.raises(SystemExit):
        dd.parse_args(["--bbox", "78", "13", "80", "15", "--start-date", "2024-01-01",
                       "--end-date", "2024-12-31", flag])


def test_main_search_mode_routes_to_original_downloader(monkeypatch):
    captured = {}
    monkeypatch.setattr(dd, "download_dataset", lambda **kw: captured.update(kw) or 0)
    monkeypatch.setattr(dd, "download_scenes_by_id", lambda **kw: pytest.fail("scene-ID path used"))
    assert dd.main(["--bbox", "78", "13", "80", "15", "--start-date", "2024-01-01",
                    "--end-date", "2024-12-31"]) == 0
    assert captured["max_cloud_cover"] == 20.0
    assert "assets" not in captured


# ---------------------------------------------------------------- exact scene selection

def test_fetch_returns_only_requested_scenes_in_order():
    client = FakeClient([make_item(UNRELATED), make_item(SCENE_B), make_item(SCENE_A)])
    items = dd.fetch_scenes_by_id(client, [SCENE_A, SCENE_B])
    assert [i.id for i in items] == [SCENE_A, SCENE_B]
    call = client.calls[0]
    assert call["ids"] == [SCENE_A, SCENE_B]
    assert "bbox" not in call and "datetime" not in call


def test_fetch_fails_if_any_scene_missing():
    client = FakeClient([make_item(SCENE_A), make_item(UNRELATED)])
    with pytest.raises(RuntimeError, match=SCENE_B):
        dd.fetch_scenes_by_id(client, [SCENE_A, SCENE_B])


def test_fetch_rejects_disallowed_platform():
    client = FakeClient([make_item(SCENE_A, platform="landsat-7")])
    with pytest.raises(RuntimeError, match="platform"):
        dd.fetch_scenes_by_id(client, [SCENE_A])


def test_missing_required_asset_aborts_before_any_request(tmp_path):
    session = FakeSession()
    with pytest.raises(RuntimeError, match="QA_PIXEL"):
        dd.plan_scene_downloads([make_item(SCENE_A, skip=("QA_PIXEL",))], dd.SCENE_ID_ASSETS,
                                tmp_path, session)
    assert session.head_calls == [] and session.get_calls == []
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------- download behaviour

def run(tmp_path, items, session, ids, **kw):
    return dd.download_scenes_by_id(ids, output_dir=tmp_path, client=FakeClient(items),
                                    session=session, min_free_bytes=0, **kw)


def test_dry_run_downloads_nothing(tmp_path):
    session = FakeSession()
    assert run(tmp_path, [make_item(SCENE_A), make_item(UNRELATED)], session, [SCENE_A], dry_run=True) == 0
    assert len(session.head_calls) == 5
    assert session.get_calls == []
    assert not any(tmp_path.iterdir())


def test_downloads_exact_scenes_and_skips_complete_assets(tmp_path):
    items = [make_item(SCENE_A), make_item(UNRELATED)]
    session = FakeSession()
    assert run(tmp_path, items, session, [SCENE_A]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [SCENE_A]
    files = sorted(p.name for p in (tmp_path / SCENE_A).iterdir())
    assert files == sorted(f"{n}.tif" for n in dd.SCENE_ID_ASSETS)
    assert all(UNRELATED not in url for url in session.get_calls)

    again = FakeSession()
    assert run(tmp_path, items, again, [SCENE_A]) == 0
    assert again.get_calls == []


def test_insufficient_disk_space_downloads_nothing(tmp_path):
    session = FakeSession()
    rc = dd.download_scenes_by_id([SCENE_A], output_dir=tmp_path, client=FakeClient([make_item(SCENE_A)]),
                                  session=session, min_free_bytes=10**18)
    assert rc == 1
    assert session.get_calls == []
    assert not any(tmp_path.iterdir())


def test_existing_mismatched_file_is_not_overwritten(tmp_path):
    scene_dir = tmp_path / SCENE_A
    scene_dir.mkdir()
    existing = scene_dir / "ST_B10.tif"
    existing.write_bytes(b"x" * 10)
    session = FakeSession()
    assert run(tmp_path, [make_item(SCENE_A)], session, [SCENE_A]) == 1
    assert existing.read_bytes() == b"x" * 10
    assert session.get_calls == []


def test_truncated_download_reported_as_failure_and_cleaned(tmp_path):
    item = make_item(SCENE_A)
    bad_url = item.assets["lwir11"].href
    session = FakeSession(truncate=[bad_url])
    assert run(tmp_path, [item], session, [SCENE_A]) == 1
    assert not (tmp_path / SCENE_A / "ST_B10.tif").exists()
    assert not (tmp_path / SCENE_A / "ST_B10.tif.part").exists()
    assert (tmp_path / SCENE_A / "SR_B2.tif").stat().st_size == session.size


def test_download_with_retry_default_behaviour_unchanged(tmp_path):
    dest = tmp_path / "ST_B10.tif"
    dest.write_bytes(b"abc")
    session = FakeSession()
    assert dd.download_with_retry("https://example.test/x", dest, session) is True
    assert session.get_calls == []
    assert dest.read_bytes() == b"abc"


def test_qa_pixel_alias_resolves():
    assert dd.resolve_asset_key(make_item(SCENE_A), "QA_PIXEL") == "qa_pixel"
    assert dd.DEFAULT_ASSETS == ("ST_B10", "SR_B2", "SR_B3", "SR_B4")
