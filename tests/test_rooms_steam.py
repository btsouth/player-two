"""Rooms on the Steam port of Chrono Trigger: real Steam screens found on the places the
SNES survey pictured, and screens of places not surveyed (battles, menus, the Mountain
of Woe, Mayor's Manor) found nowhere.

The data isn't in the repo: the survey in ~/.config/player-two/atlas/chrono-trigger/rooms,
Steam screenshots in ~/.cache/player-two/live/steam-2026-10-31/trips and
steam-2026-11-01/frames, recordings (rec_fair.npz, rec_down.npz) in
~/.cache/player-two/recordings (or P2_RECORDINGS). Each test is skipped without its data.
"""

import functools
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from playertwo.rooms import STEAM_FOLLOW, STEAM_TAP, Here, Rooms, lead, steam_range, steam_view

ROOMS = Path.home() / ".config/player-two/atlas/chrono-trigger/rooms"
LIVE = Path.home() / ".cache/player-two/live"
TRIPS = LIVE / "steam-2026-10-31/trips"
FRAMES = LIVE / "steam-2026-11-01/frames"
REC = Path(os.environ.get("P2_RECORDINGS") or Path.home() / ".cache/player-two/recordings")
# The places surveyed when these screens were taken: places surveyed since (the Mountain
# of Woe) are left out, so the screens of them still count as places not surveyed.
# The folder grows while the game is played: the screens up to this one (13:46 to 16:56).
LAST_FRAME = "165609-00.jpg"
FIRST_FRAME = "134600"  # (the next night's, after midnight, sort before it)
SURVEYED = {0x1, 0x2, 0x3, 0x4, 0x5, 0xC, 0x11, 0x13, 0x14, 0x1B2, 0x1B7}

# Steam screens from trips (2560x1440): the place and the view's top left on its picture.
TRIP = {"crono-s-house": (0x1, (-24, 31)), "guardia-forest": (0x13, (152, 575)),
        "lucca-s-house": (0x4, (-24, 143)), "truce-inn": (0xC, (744, 303)),
        "truce-market": (0x11, (232, 575))}

needs_rooms = pytest.mark.skipif(not all((ROOMS / f"loc-{loc:03x}.png").exists()
                                         for loc in SURVEYED), reason=f"no survey in {ROOMS}")


@functools.cache
def rooms() -> Rooms:
    r = Rooms(ROOMS)
    r.rooms = {k: v for k, v in r.rooms.items() if k in SURVEYED}
    return r


@functools.cache
def rec(name):
    return np.load(REC / f"{name}.npz")["frames"]


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


@needs_rooms
@pytest.mark.skipif(not TRIPS.is_dir(), reason=f"no trip screens in {TRIPS}")
def test_steam_trip_screens():
    r = rooms()
    for name, (loc, cam) in TRIP.items():
        for i in (0, 1):
            frame = rgb(TRIPS / f"{name}-{i}.png")
            assert r.is_steam(frame)
            h = r.locate(Image.fromarray(frame))
            assert h is not None, name
            assert (h.loc, h.cam, h.steam) == (loc, cam, True), name
            assert h.score > 0.9, name
            # The character itself is found, standing on ground the survey walked.
            assert h.sure, name
            assert h.tile in r.rooms[loc].walk, (name, h.tile)


@needs_rooms
@pytest.mark.skipif(not (REC / "rec_fair.npz").exists(), reason=f"no recordings in {REC}")
def test_steam_fair_track():
    """Leene Square, 410 screens at 480x270 while walking about: found in every one, the
    view moving smoothly as the picture scrolls, the character on walkable ground."""
    import cv2

    r = rooms()
    frames = rec("rec_fair")
    track, near = [], None
    for f in frames:
        h = r.locate(f, near)
        assert h is not None and h.loc == 0x5
        track.append(h)
        near = h
    cams = np.array([h.cam for h in track])
    assert np.abs(np.diff(cams, axis=0)).max() <= 12  # no jumps
    assert tuple(cams[0]) == (-1, 573) and tuple(cams[90]) == (-1, 362)
    assert tuple(cams[360]) == (125, 573)
    lo_x, lo_y, hi_x, hi_y = steam_range(r.rooms[0x5])
    assert lo_x - 2 <= cams[:, 0].min() and cams[:, 0].max() <= hi_x + 2
    assert lo_y - 2 <= cams[:, 1].min() and cams[:, 1].max() <= hi_y + 2
    # The view's moves match how far the screen scrolled.
    errs = []
    prev = None
    for f, h in zip(frames, track, strict=True):
        g = cv2.cvtColor(steam_view(f), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev[0], g)
            if resp > 0.3:
                errs.append(max(abs(h.cam[0] - prev[1][0] + dx), abs(h.cam[1] - prev[1][1] + dy)))
        prev = (g, h.cam)
    assert len(errs) > 300 and np.percentile(errs, 95) < 1.0
    sure = [h for h in track if h.sure]
    assert len(sure) > 300
    assert all(h.tile in r.rooms[0x5].walk for h in sure)
    # Where it's found while the view follows it is where the view keeps it.
    for h in sure:
        if lo_x + 2 < h.cam[0] < hi_x - 2:
            assert abs(h.x - h.cam[0] - STEAM_FOLLOW[0]) <= 6
        if lo_y + 2 < h.cam[1] < hi_y - 2:
            assert abs(h.y - h.cam[1] - STEAM_FOLLOW[1]) <= 6


@needs_rooms
@pytest.mark.skipif(not FRAMES.is_dir() or not (REC / "rec_down.npz").exists()
                    or not TRIPS.is_dir(), reason="no screens of places not surveyed")
def test_steam_not_surveyed():
    """383 Steam screens of places not surveyed (the Mountain of Woe, battles, menus, a
    cliff path, Mayor's Manor): none is taken for a surveyed place."""
    r = rooms()
    shots = [rgb(p) for p in sorted(FRAMES.glob("*.jpg")) if FIRST_FRAME <= p.name <= LAST_FRAME]
    shots += list(rec("rec_down"))
    shots += [rgb(TRIPS / f"truce-mayor-s-manor-{i}.png") for i in (0, 1)]
    assert len(shots) == 383
    hits = [i for i, s in enumerate(shots) if r.locate(s) is not None]
    assert hits == []


@needs_rooms
def test_steam_or_emulator():
    """The emulator's 4:3 picture (native, or boxed in a wide window) takes the SNES
    path; a wide window filled with the place, or Steam's fill round it, takes Steam's."""
    r = Rooms(ROOMS)
    snes = np.zeros((224, 256, 3), np.uint8)
    assert not r.is_steam(snes)
    boxed = np.zeros((1080, 1920, 3), np.uint8)
    boxed[:, 240:1680] = 120
    assert not r.is_steam(boxed)
    filled = np.full((1080, 1920, 3), (24, 8, 8), np.uint8)
    filled[:, 600:1320] = 120
    assert r.is_steam(filled)
    dark = np.zeros((1080, 1920, 3), np.uint8)  # a fade: as the last clear screen said
    assert r.is_steam(dark)
    assert not r.is_steam(np.zeros((900, 1600, 3), np.uint8))


def test_steam_lead():
    """On Steam a turn starts early by how far it walks in the lag; not on the emulator."""
    import time

    now = time.monotonic()
    trail = [(now - 0.4, 5, 100.0, 200.0), (now - 0.2, 5, 112.0, 200.0)]
    h = Here(5, (0, 0), 124.0, 200.0, 0.9, True, True)
    x, y = lead(h, trail)
    assert 130 < x <= 124 + 16 and y == 200.0
    assert lead(Here(5, (0, 0), 124.0, 200.0, 0.9), trail) == (124.0, 200.0)
    assert lead(h, []) == (124.0, 200.0)
    assert STEAM_TAP >= 0.1
    assert math.isclose(lead(h, [(now - 0.4, 6, 0.0, 0.0)])[0], 124.0)


@pytest.mark.skipif(not (ROOMS / "loc-189.png").exists() or not (LIVE / "steam-2026-11-01/w0.png").exists(),
                    reason="no Mountain of Woe survey or screens")
def test_steam_mountain_of_woe():
    """Steam tints the Mountain of Woe and draws its mist its own way: found by the edges.
    w0 is the owner's save on the ledge by the chain ladder, where the same save on the
    SNES stands at (440, 128); the rest crossing the chain bridge on the same ledges."""
    r = Rooms(ROOMS)
    h = r.locate(rgb(LIVE / "steam-2026-11-01/w0.png"))
    assert h is not None and h.loc == 0x189
    assert abs(h.cam[0] - 208) <= 3 and abs(h.cam[1] + 1) <= 3
    assert h.sure and math.dist((h.x, h.y), (440, 128)) <= 8
    crossing = [p for p in sorted(FRAMES.glob("*.jpg")) if "135812" <= p.name[:6] <= "135938"]
    assert len(crossing) == 15
    for p in crossing:
        h = r.locate(rgb(p))
        assert h is not None and h.loc == 0x189, p.name
        assert h.tile in r.rooms[0x189].walk, p.name


@pytest.mark.skipif(not (FRAMES / "forest-041307-04.jpg").exists(), reason="no forest frames")
def test_a_fight_on_the_field_is_told_from_the_field():
    """The walker stopped 13 times "stuck" in a Guardia Forest fight, then said blocked."""
    from playertwo.rooms import over_place

    assert over_place(Image.open(FRAMES / "forest-041307-04.jpg")) == "fight"
    assert over_place(Image.open(FRAMES / "forest-field-north.png")) == ""
    # A text box is the game's window and font, not the survey's picture looking unlike
    # Steam's (the fair's balustrade read as a box four times in a row).
    assert over_place(Image.open(FRAMES / "box-inn-lodging.png")) == "box"
