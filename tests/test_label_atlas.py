"""Turning Muse's answers about map-rip tiles into atlas places, without a model."""

import importlib.util
import sys
from itertools import pairwise
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "label_atlas", Path(__file__).parents[1] / "scripts" / "label_atlas.py"
)
L = importlib.util.module_from_spec(spec)
sys.modules["label_atlas"] = L  # dataclasses look their module up
spec.loader.exec_module(L)


def test_tiles_cover_the_rip_with_overlap():
    boxes = L.tiles(1536, 1024, 320, 112)
    assert boxes[0] == (0, 0, 320, 320)
    assert max(b[2] for b in boxes) == 1536 and max(b[3] for b in boxes) == 1024
    xs = sorted({b[0] for b in boxes})
    assert all(b - a <= 320 - 112 for a, b in pairwise(xs))
    assert L.tiles(200, 100, 320, 112) == [(0, 0, 200, 100)]  # smaller than a tile


def test_answers_map_back_to_rip_pixels_and_cut_labels_are_dropped():
    ans = {
        "places": [
            {"name": "Truce Inn", "label": [300, 30, 420, 60], "icon": [330, 90, 390, 150]},
            {"name": "Leene", "label": [0, 30, 60, 60], "icon": [10, 90, 50, 120]},  # cut off
            {"name": "", "icon": [1, 2, 3, 4]},
            {"name": "No icon"},
        ]
    }
    got = L.to_rip(ans, (100, 200, 420, 520), scale=3)
    assert [s.name for s in got] == ["Truce Inn"]
    assert (got[0].x, got[0].y) == (100 + 120, 200 + 40)  # icon centre / 3 + tile corner


def seen(name, x, y, lx=None, ly=None):
    lx, ly = (x, y - 12) if lx is None else (lx, ly)
    return L.Seen(name, x, y, (lx - 20, ly - 4, lx + 20, ly + 4), (0, 0))


def test_merge_joins_tiles_that_saw_one_label_and_keeps_far_namesakes_apart():
    cands = L.merge(
        [
            seen("Residence", 100, 100),
            seen("Residence", 104, 98),  # one label, two tiles
            seen("Residence", 400, 300),  # another residence
            seen("Porre Inn", 200, 200),
            seen("Inn", 206, 201, 210, 188),  # a cut-off read
        ]
    )
    by = sorted((c.name, round(c.x), round(c.y), c.votes) for c in cands)
    assert by == [("Porre Inn", 203, 200, 2), ("Residence", 102, 99, 2), ("Residence", 400, 300, 1)]
    assert L.unique_names(cands) == {
        "Residence (1)": [102, 99],
        "Porre Inn": [203, 200],
        "Residence (2)": [400, 300],
    }


def test_median_of_two_is_their_middle():
    assert L.median([10, 30]) == 20 and L.median([5, 1, 9]) == 5


@pytest.mark.parametrize(
    ("xy", "result"), [((105, 100), "right"), ((130, 100), "off"), ((60, 180), "right")]
)
def test_score_counts_icon_centre_or_way_in(xy, result):
    truth = [["Cave", "Some Cave", [100, 100], [60, 176]]]
    s = L.score([L.Candidate("cave", *xy, 1)], truth)
    assert s["rows"][0]["result"] == result


def test_score_reports_missed_extra_and_zones():
    truth = [
        ["Lab 16", "Lab 16 (west)", [100, 100], None, "zone"],
        ["Market", "Truce Market", [300, 300]],
    ]
    s = L.score([L.Candidate("Lab 16", 120, 110, 1), L.Candidate("Ghost", 5, 5, 1)], truth)
    assert (s["read"], s["missed"], s["right"], s["total"]) == (1, 1, 0, 1)
    assert s["extra"] == ["Ghost @5,5"]
