"""Tests for QA_PIXEL decoding and the qa_v1 window counter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from preprocessing.patchify import extract_patches
from preprocessing.pipeline_qa import count_accepted_windows
from preprocessing.qa_mask import (
    EXCLUDED_FLAGS,
    check_alignment,
    decode_flags,
    exclusion_codes,
    qa_exclusion_mask,
    summarize_flags,
    verify_bitfields_against_stac,
)

# Common Landsat 8/9 C2 QA_PIXEL values listed in the USGS product guide.
CLEAR_LAND = 21824     # clear, low cloud confidence
CLEAR_WATER = 21952    # clear + water
FILL = 1
HIGH_CLOUD = 22280     # cloud, high cloud confidence (0b0101011100001000)
DILATED_CLOUD = 21826  # clear-land value with the dilated-cloud bit set
CLOUD_SHADOW = 23888   # cloud shadow
SNOW = 30048           # snow/ice
CIRRUS = 54596         # high-confidence cirrus


def flags_of(value: int) -> set[str]:
    decoded = decode_flags(np.array([[value]], dtype=np.uint16))
    return {name for name, arr in decoded.items() if arr[0, 0]}


@pytest.mark.parametrize("value, expected", [
    (CLEAR_LAND, set()),
    (CLEAR_WATER, set()),
    (FILL, {"fill"}),
    (HIGH_CLOUD, {"cloud"}),
    (DILATED_CLOUD, {"dilated_cloud"}),
    (HIGH_CLOUD | 0b10, {"cloud", "dilated_cloud"}),
    (CLOUD_SHADOW, {"cloud_shadow"}),
    (SNOW, {"snow"}),
    (CIRRUS, {"cirrus"}),
])
def test_documented_values_decode(value, expected):
    assert flags_of(value) == expected


def test_each_bit_maps_to_one_flag():
    for flag in EXCLUDED_FLAGS:
        assert flags_of(1 << flag.bit) == {flag.name}
    for bit in range(6, 16):
        assert flags_of(1 << bit) == set()


def test_exclusion_and_counts():
    qa = np.array([[CLEAR_LAND, HIGH_CLOUD], [SNOW, FILL]], dtype=np.uint16)
    base = np.array([[True, True], [True, False]])
    flags = decode_flags(qa)
    assert qa_exclusion_mask(flags).tolist() == [[False, True], [True, True]]
    s = summarize_flags(base, flags)
    assert s["valid_before_qa"] == 3 and s["valid_after_qa"] == 1
    assert s["per_flag"]["fill"]["set_all_pixels"] == 1
    assert s["per_flag"]["fill"]["set_among_base_valid"] == 0
    assert s["per_flag"]["cloud"]["only_this_flag_among_base_valid"] == 1
    codes = exclusion_codes(base, flags)
    assert codes[1, 1] & 1 and codes[1, 1] & 0b10
    assert codes[0, 0] == 0


def ref_band(**over):
    base = dict(name="ST_B10", crs="EPSG:32644", width=10, height=8,
                transform=(500000.0, 30.0, 0.0, 1600000.0, 0.0, -30.0),
                bounds=(500000.0, 1599760.0, 500300.0, 1600000.0))
    base.update(over)
    return SimpleNamespace(**base)


def qa_info(**over):
    b = ref_band()
    info = dict(crs=b.crs, width=b.width, height=b.height, transform=b.transform, bounds=b.bounds)
    info.update(over)
    return info


def test_alignment_accepts_identical_grid():
    assert check_alignment(qa_info(), ref_band())["result"] == "identical"


@pytest.mark.parametrize("over", [
    {"crs": "EPSG:32643"},
    {"width": 11},
    {"transform": (500015.0, 30.0, 0.0, 1600000.0, 0.0, -30.0)},
    {"bounds": (500000.0, 1599760.0, 500330.0, 1600000.0)},
])
def test_alignment_rejects_mismatch(over):
    with pytest.raises(ValueError, match="not aligned"):
        check_alignment(qa_info(**over), ref_band())


def test_stac_bitfield_verification():
    good = [{"name": f.name, "offset": f.bit, "length": 1} for f in EXCLUDED_FLAGS]
    assert verify_bitfields_against_stac(good)["verified"]
    bad = [dict(g, offset=g["offset"] + 1) if g["name"] == "cloud" else g for g in good]
    with pytest.raises(ValueError):
        verify_bitfields_against_stac(bad)


def test_window_counter_matches_extract_patches():
    rng = np.random.default_rng(0)
    mask = rng.random((96, 80)) > 0.06
    mask[:40, :40] = False
    arr = np.where(mask, 1.0, np.nan).astype(np.float32)
    rgb = np.repeat(arr[..., None], 3, axis=-1)
    for limit in (0.0, 5.0, 10.0, 50.0):
        res = extract_patches(arr, rgb, mask, patch_size=16, stride=8, max_invalid_percent=limit)
        assert count_accepted_windows(mask, 16, 8, limit) == (len(res.patches), res.skipped_count)
