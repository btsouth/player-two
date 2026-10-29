"""Walking and mapping on real Chrono Trigger frames, recorded from the game at about
26 frames a second while holding the d-pad (the sticks array says which way, -1 for
none). The recordings aren't in the repo: keep rec_fair.npz (the Millennial Fair) and
rec_down.npz (a cliff path) in ~/.cache/player-two/recordings, or point P2_RECORDINGS
at them. Skipped without them.

The synthetic worlds in the other tests move the character a fixed amount a frame; these
check the thresholds against a real game's timing, sprites and crowds.
"""

import asyncio
import functools
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from playertwo.areamap import AreaMap
from playertwo.explore import keep_walking
from playertwo.walk import Look, Transition, phase_shift, small

DIR = Path(os.environ.get("P2_RECORDINGS") or Path.home() / ".cache/player-two/recordings")
pytestmark = pytest.mark.skipif(not (DIR / "rec_fair.npz").exists()
                                or not (DIR / "rec_down.npz").exists(),
                                reason=f"no recordings in {DIR}")

CRONO = (189.0, 157.0)  # where Crono stands in rec_fair frames 229 and 316, tracking size


@functools.cache
def rec(name):
    d = np.load(DIR / f"{name}.npz")
    return d["frames"], d["ts"]


def replay(name, start):
    """A grab that plays the recording from frame `start` in real time."""
    frames, ts = rec(name)
    t0 = []

    async def grab():
        if not t0:
            t0.append(time.monotonic())
        i = int(np.searchsorted(ts, ts[start] + time.monotonic() - t0[0], side="right") - 1)
        return Image.fromarray(frames[min(i, len(frames) - 1)])
    return grab


def crono(i):
    rgb = small(Image.fromarray(rec("rec_fair")[0][i]))[0]
    x, y = CRONO
    look = Look(rgb, (x - 8, y - 12.5, x + 8, y + 12.5))
    assert look.usable
    return look, np.array(CRONO), (16, 25)


def faded(steps):
    """Walking right through the fair, a fade out over `steps` frames, black, and a
    fade in to the cliff path, as through a door."""
    fair, down = rec("rec_fair")[0], rec("rec_down")[0]
    walk = [fair[i] for i in range(325, 345)]
    out = [(walk[-1] * (1 - k / steps)).astype(np.uint8) for k in range(1, steps + 1)]
    inn = [(down[10] * (k / steps)).astype(np.uint8) for k in range(1, steps + 1)]
    return walk, [*out, *[np.zeros_like(walk[0])] * 8, *inn], down[10]


def test_walking_about_the_fair_and_the_cliff_is_never_a_new_place():
    for name in ("rec_fair", "rec_down"):
        frames, ts = rec(name)
        scene, prev = Transition(), None
        for f, t in zip(frames, ts, strict=True):
            g = small(Image.fromarray(f))[1]
            scroll = (0.0, 0.0) if prev is None else phase_shift(prev, g)[:2]
            assert scene.see(g, scroll, float(t)) == "steady", name
            prev = g


@pytest.mark.parametrize(("direction", "start", "seconds", "outcome", "steps"), [
    ("left", 229, 3.0, "didnt_move", 0),    # facing into the trees
    ("left", 229, 0.7, "didnt_move", 0),    # the same, too short for the usual check
    ("right", 316, 3.5, "blocked", 15),     # along the path to a fence
])
async def test_keep_walking_on_real_frames(direction, start, seconds, outcome, steps):
    went = await keep_walking(replay("rec_fair", start), lambda _a: None, direction,
                              seconds, me=crono(start))
    assert went.outcome == outcome, went
    assert abs(abs(went.steps[0]) - steps) <= 3 and abs(went.steps[1]) < 2, went


async def test_a_guard_walking_by_isnt_taken_for_crono():
    # Crono can't go left here; a guard nearby walks about. Not knowing which one is
    # Crono, it mustn't say he went anywhere.
    went = await keep_walking(replay("rec_fair", 229), lambda _a: None, "left", 3.0)
    assert went.outcome == "unsure" and not went.steps.any(), went


@pytest.mark.parametrize("steps", [3, 7, 10])
async def test_a_fade_between_real_places_is_one_new_place(steps):
    walk, fade, there = faded(steps)
    seq = iter([*walk, *fade, *[there] * 200])

    async def grab():
        return Image.fromarray(next(seq))

    went = await keep_walking(grab, lambda _a: None, "right", 8.0)
    assert went.outcome == "new_area", went
    assert went.steps[1] == 0, went  # no steps made up from noise on the dark frames

    area, fresh = AreaMap(), []
    for i, f in enumerate([*walk, *fade, *[there] * 10]):
        fresh.append(area.track(f, now=i / 26))
    assert fresh.count(True) == 1 and fresh.index(True) >= len(walk) + len(fade) - 1


def test_a_text_box_over_the_fair_keeps_its_map():
    frames = rec("rec_fair")[0]
    area = AreaMap()
    for i, f in enumerate(frames[316:396]):
        area.track(f, now=i / 26)
    seen = area.bounds()
    boxed = frames[395].copy()
    boxed[180:262, 8:472] = (16, 24, 120)  # the blue text box, over the bottom third
    boxed[180:184, 8:472] = (230, 230, 230)
    assert not area.resume(boxed)
    assert area.bounds() == seen
    # Somewhere else entirely starts over.
    assert area.resume(rec("rec_down")[0][10])


@pytest.mark.parametrize("step", [4, 5, 6])
def test_a_slow_capture_of_walking_is_never_a_new_place(step):
    # Every 4th to 6th frame: about 4 to 7 a second, the view moving 25 to 45 px a frame.
    frames, ts = rec("rec_fair")
    scene, prev = Transition(), None
    for i in range(0, len(frames), step):
        g = small(Image.fromarray(frames[i]))[1]
        scroll = (0.0, 0.0) if prev is None else phase_shift(prev, g)[:2]
        assert scene.see(g, scroll, float(ts[i])) == "steady", i
        prev = g


async def test_keep_walking_counts_steps_on_a_slow_capture():
    real = replay("rec_fair", 316)

    async def grab():
        await asyncio.sleep(0.1)  # about 6 frames a second
        return await real()

    went = await keep_walking(grab, lambda _a: None, "right", 3.5, me=crono(316))
    assert went.outcome == "blocked" and went.steps[0] >= 10, went


def bands(f, frac):
    f = f.copy()
    k = int(f.shape[0] * frac / 2)
    f[:k] = 0
    f[f.shape[0] - k:] = 0
    return f


@pytest.mark.parametrize("frac", [0.0, 0.4, 0.5])
def test_a_cut_between_rooms_with_dark_margins_is_a_new_place(frac):
    fair, down = rec("rec_fair")[0], rec("rec_down")[0]
    a, b = bands(fair[300], frac), bands(down[10], frac)
    scene, t, states = Transition(), 0.0, []
    for f in [a] * 15 + [b] * 15:
        states.append(scene.see(small(Image.fromarray(f))[1], (0.0, 0.0), t))
        t += 0.04
    assert [s for s in states if s not in ("steady", "changing")] == ["new"], states


def test_a_cut_to_a_dark_place_starts_a_new_map():
    fair, down = rec("rec_fair")[0], rec("rec_down")[0]
    dim = (down[10] * 0.08).astype(np.uint8)  # a cave, nearly black
    area, fresh = AreaMap(), []
    for i, f in enumerate([*fair[316:346], *[dim] * 60]):
        fresh.append(area.track(f, now=i / 26))
    assert fresh.count(True) == 1
    assert area.frames > 20  # and the dark place is mapped, not skipped as a fade


def test_only_the_character_is_followed_not_the_ground():
    rgb = small(Image.fromarray(rec("rec_fair")[0][316]))[0]
    x, y = CRONO
    assert Look(rgb, (x - 8, y - 12.5, x + 8, y + 12.5)).usable
    for gx, gy in ((260, 200), (300, 150), (230, 120)):  # cobblestones
        assert not Look(rgb, (gx - 8, gy - 12.5, gx + 8, gy + 12.5)).usable, (gx, gy)


@pytest.mark.parametrize("lag", [0.0, 0.03, 0.06])
async def test_a_walk_on_a_slow_capture_never_arrives_short(lag):
    # Toward a stall past the fence Crono stops at. At 16 frames a second or fewer it
    # used to follow the cobblestones with the scroll and say it arrived 150 px short.
    from playertwo.walk import Located, Walker

    real = replay("rec_fair", 316)

    async def grab():
        await asyncio.sleep(lag)
        return await real()

    w = Walker(None, grab, lambda _a: None)
    w._start()
    img = await grab()
    t0 = time.monotonic()
    x, y = CRONO
    loc = Located((x - 8, y - 12.5, x + 8, y + 12.5), (592, 144.5, 608, 169.5), [], "")
    result = await w._walk("the stall", img, loc, t0, 80 / 26, w._done(t0))
    assert result.outcome != "arrived", result.text()
    # Where it last really saw Crono: at the fence, or (on a starved machine, frames too
    # far apart to pick him out) still where he started; never out on the ground.
    at = w.memory[1]
    assert min(np.hypot(*(at - (243, 161))), np.hypot(*(at - CRONO))) < 20, at
