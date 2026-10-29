"""Walking on where nothing is in sight yet, and remembering where walks went.

Holding the d-pad blind tells the model nothing until it looks again, and a screen that
scrolled a few tiles looks much like the last one, so it walks in circles. keep_walking
holds a direction and watches every frame instead: how far the scenery scrolled and the
character moved (so how far it went over the ground), when it stopped (a wall), and when
the picture changed to a new place (an exit, a door).

Trail keeps a rough map of one area from those distances: where each walk ended, in
steps from where the area was entered, so the model can tell what it has tried.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .walk import Look, Transition, _box_means, changed, motion, phase_shift, settle, small, spot

ANGLES = {"up": 0, "up_right": 45, "right": 90, "down_right": 135, "down": 180,
          "down_left": 225, "left": 270, "up_left": 315}


def words(v: np.ndarray) -> str:
    """A move in steps as words: "3 steps up and 2 right"."""
    x, y = round(float(v[0])), round(float(v[1]))
    parts = []
    if y:
        parts.append(f"{abs(y)} {'up' if y < 0 else 'down'}")
    if x:
        parts.append(f"{abs(x)} {'left' if x < 0 else 'right'}")
    if not parts:
        return "hardly anywhere"
    n, way = parts[0].split(" ")
    first = f"{n} {'step' if n == '1' else 'steps'} {way}"
    return " and ".join([first, *parts[1:]])


@dataclass
class Went:
    # "walked" (the time ran out, still going), "blocked", "new_area", "interrupted",
    # "stopped", "didnt_move", "unsure" (the character unknown and the view still)
    outcome: str
    steps: np.ndarray  # over the ground, in character sizes (x right, y down)
    screens: float     # how far the view scrolled, in screen widths
    seconds: float
    at: np.ndarray | None = None  # where the character ended up on screen, if followed

    def text(self, direction: str) -> str:
        went = f"went {words(self.steps)}"
        scrolled = (f"; the view moved on about {self.screens:.1f} screens"
                    if self.screens >= 0.3 else "")
        return {
            "walked": f"walked: {went} going {direction}, still moving when the time was "
                      f"up{scrolled}",
            "blocked": f"blocked: something stops you going {direction}, after you {went}"
                       f"{scrolled}",
            "new_area": f"new_area: the screen changed to a new place after you {went}",
            "interrupted": f"interrupted: something came up on screen after you {went}; "
                           "look before going on",
            "stopped": f"stopped by the player after you {went}",
            "didnt_move": f"didnt_move: holding {direction} didn't move your character; a "
                          "text box or menu may be up, or it's a wall",
            "unsure": "unsure: I couldn't pick out your character, and the view "
                      + (f"stopped scrolling after you {went}" if self.screens >= 0.05
                         else "didn't scroll")
                      + "; it may be at a wall, walking across a still screen, or a text "
                        "box or menu is up: look",
        }[self.outcome] + f" ({self.seconds:.1f} s)"


async def keep_walking(grab: Callable[[], Awaitable[Image.Image]],
                       steer: Callable[[float | None], None], direction: str,
                       seconds: float = 3.0,
                       keep_going: Callable[[], Awaitable[bool]] | None = None,
                       me: tuple[Look, np.ndarray, tuple[int, int]] | None = None,
                       log: Callable[[str], None] | None = None) -> Went:
    """Hold `direction` for up to `seconds`, stopping early at a wall, a new place or a
    popup. `me` is the character as last seen by a walk (its colours, where, its size),
    to follow it. Without it only the scroll says how far it went: on real screens
    someone else walking about is easily taken for it (see find_me to know which one
    it is), so then a still view is "unsure"."""
    angle = ANGLES[direction]
    unit = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
    t0 = time.monotonic()
    rgb, prev = small(await grab())
    scene = Transition()
    scene.see(prev)
    h, w = prev.shape
    if me is not None:
        look, at, (bw, bh) = me
        at = np.array(at, float)
    else:
        # About the size of a 2D game's sprite (Crono is a thirtieth of the screen wide),
        # where most 2D games keep the character, so it is left out of the scroll.
        look, at, (bw, bh) = None, np.array([w / 2, h / 2]), (round(w / 30), round(h / 11))
    known = me is not None
    size = float(max(6, min(bw, bh)))
    ground = np.zeros(2)   # how far the character went over the ground, in pixels
    view = np.zeros(2)     # how far the scenery scrolled
    history: list[tuple[float, np.ndarray]] = []
    popup: list[float] = []
    paused = 0.0
    outcome = "walked"
    ever = False
    steer(angle)
    try:
        while time.monotonic() - t0 - paused < seconds:
            if keep_going is not None:
                t = time.monotonic()
                go = await keep_going()
                paused += time.monotonic() - t
                if not go:
                    outcome = "stopped"
                    break
                steer(angle)
            await asyncio.sleep(0.04)
            rgb, g = small(await grab())
            mask = (int(at[0] - bw), int(at[1] - bh), int(at[0] + bw), int(at[1] + bh))
            sx, sy, _ = phase_shift(prev, g, mask)
            if scene.see(g, (sx, sy)) != "steady":
                # A fade, a door: the scroll and the steps mean nothing until it
                # settles, and holding on might walk back out.
                steer(None)
                t = time.monotonic()
                state, rgb, g = await settle(grab, scene, keep_going)
                if state == "same":
                    # A flash, or a false alarm: walk on from here.
                    paused += time.monotonic() - t
                    history, popup, prev = [], [], g
                    steer(angle)
                    continue
                outcome = "new_area" if state == "new" else "stopped"
                break
            scroll = np.array([sx, sy])
            # While the screen scrolls, the camera is following the character: that's how
            # far it went (followers and passers-by would only add noise). On a still
            # screen (a small room, the edge of the map) the character crosses it.
            follow = bool(np.hypot(sx, sy) > 1)
            # Where it is, by its colours, favouring what moves differently from the
            # scenery.
            if look is not None:
                x, y, score = look.find(rgb, (at[0], at[1]), int(2 * max(bw, bh)),
                                        motion(prev, g, (sx, sy)))
                if score > 0.3:
                    now = np.array([x, y])
                    if not follow:
                        ground += now - at
                    at = now
            if follow:
                ground -= scroll
            view += scroll
            # A text box, a menu or a battle stops the character, so the view stops too;
            # while it scrolls, layers moving at their own speed (clouds, a far-off
            # background) change much of the picture and mean nothing.
            popup = [*popup[-2:], 0.0 if follow else changed(prev, g, (sx, sy))]
            prev = g
            t = time.monotonic() - paused
            history.append((t, ground.copy()))
            if len(popup) == 3 and sum(popup) > 0.15:
                outcome = "interrupted"
                break
            # Progress the way it's held, over the last second: a wall stops it.
            old = [g0 for t1, g0 in history if t - t1 >= 0.8]
            if old:
                gone = float(np.dot(ground - old[-1], unit))
                if gone > 0.5 * size:
                    ever = True
                elif gone < 0.3 * size:
                    # A step or so and then a wall still counts as having moved.
                    went = float(np.dot(ground, unit)) > 0.5 * size
                    outcome = "blocked" if ever or went else "didnt_move"
                    break
    finally:
        steer(None)
    if outcome == "walked":
        # The time ran out before a second of history said anything: a short walk, or
        # one that stopped near the end. Its last moments say whether it still moves.
        total = float(np.dot(ground, unit))
        end = history[-1][0] if history else 0.0
        old = [g0 for t1, g0 in history if end - t1 >= 0.4]
        recent = total - float(np.dot(old[-1], unit)) if old else total
        if total < 0.5 * size:
            outcome = "didnt_move"
        elif recent < 0.2 * size:
            outcome = "blocked"
    if not known and outcome in ("blocked", "didnt_move"):
        # Only the scroll was watched: the view stopping isn't proof the character did.
        outcome = "unsure"
    if log is not None:
        log(f"keep_walking {direction}: {outcome} after {len(history)} frames, ground "
            f"{(ground / size).round(1).tolist()}, last changes {[round(c, 3) for c in popup]}")
    steps = ground / size
    # A step or so sideways or backwards is the tracker's wobble (a follower, the walking
    # animation), not where it went.
    wobble = (np.abs(steps) < 1.5) & ((np.abs(unit) < 0.1) | (np.sign(steps) != np.sign(unit)))
    steps[wobble] = 0.0
    return Went(outcome, steps, float(np.hypot(*view)) / w,
                time.monotonic() - t0 - paused,
                at if outcome != "new_area" and known else None)


class _Stopped(Exception):
    pass


async def find_me(grab: Callable[[], Awaitable[Image.Image]],
                  steer: Callable[[float | None], None],
                  keep_going: Callable[[], Awaitable[bool]] | None = None,
                  log: Callable[[str], None] | None = None
                  ) -> tuple[Look, np.ndarray, tuple[int, int]] | None:
    """Which character is the player's: tap the d-pad one way and back, and see what
    starts and stops moving with it. Someone else on screen (another person with red
    hair, a mother at the bedside) walks on their own time, and a party member following
    catches up after the d-pad is let go. None if nothing moved with it (a text box, a
    cutscene)."""
    rgb, prev = small(await grab())
    h, w = prev.shape
    bw, bh = round(w / 30), round(h / 11)
    rest = np.zeros((h, w), np.float32)
    rests = 0
    best: tuple[float, float, float] | None = None

    last = np.zeros((h, w), np.float32)

    async def watch(angle: float | None, secs: float) -> np.ndarray:
        nonlocal rgb, prev, last
        if keep_going is not None and not await keep_going():
            raise _Stopped
        steer(angle)
        total = np.zeros((h, w), np.float32)
        frames = 0
        end = time.monotonic() + secs
        while time.monotonic() < end:
            await asyncio.sleep(0.03)
            if keep_going is not None:
                t = time.monotonic()
                # The player talking: let go, and don't count what moved meanwhile.
                if not await keep_going() or time.monotonic() - t > 0.1:
                    raise _Stopped
            rgb, g = small(await grab())
            sx, sy, _ = phase_shift(prev, g)
            last = motion(prev, g, (sx, sy))
            total += last
            frames += 1
            prev = g
        return total / max(frames, 1)

    try:
        rest += await watch(None, 0.25)
        rests += 1
        for there, back in ((270, 90), (0, 180)):
            moved = [await watch(there, 0.3)]
            rest += await watch(None, 0.3)
            moved.append(await watch(back, 0.3))
            now = last  # where it moved last, so where it is
            rest += await watch(None, 0.3)
            rests += 2
            still = _box_means(rest / rests, bw, bh)
            # What moved both ways (or the one way it could, up against a wall), and not
            # while the d-pad was let go.
            scores = [_box_means(m, bw, bh) - 3 * still for m in moved]
            going = [s for s in scores if s.max() > 6]
            if not going:
                continue
            score = np.minimum.reduce(going)
            iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
            # Over the way it walked, there and back: where it is now is the end of it.
            x, y, _ = spot(now, (ix + bw / 2, iy + bh / 2), max(bw, bh), (bw, bh))
            best = (x, y, float(score[iy, ix]))
            break
    except _Stopped:
        return None
    finally:
        steer(None)
    if best is None or best[2] <= 6:
        if log is not None:
            log("find_me: nothing moved with the d-pad")
        return None
    x, y, how = best
    look = Look(rgb, (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2))
    if log is not None:
        log(f"find_me: the character is at {x:.0f},{y:.0f} (moved {how:.0f})")
    if not look.usable:
        return None
    return look, np.array([x, y]), (bw, bh)


@dataclass
class Trail:
    """Where walks went in the current area, from where it was entered, in steps."""

    at: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lines: list[str] = field(default_factory=list)
    used: float = field(default_factory=time.monotonic)

    def add(self, what: str, outcome: str, steps: np.ndarray | None) -> None:
        # Long forgotten, or somewhere new: the old map is no use here.
        if time.monotonic() - self.used > 600:
            self.clear()
        self.used = time.monotonic()
        if steps is not None:
            self.at = self.at + steps
        if outcome == "new_area":
            self.clear(f"entered this area by {what}")
            return
        self.lines.append(f"{what}: {outcome}; now {self.where()}")
        del self.lines[:-14]

    def clear(self, first: str = "") -> None:
        self.at = np.zeros(2)
        self.lines = [first] if first else []
        self.used = time.monotonic()

    def where(self) -> str:
        if np.hypot(*self.at) < 0.5:
            return "where this area began"
        return f"{words(self.at)} from where this area began"

    def notes(self) -> str:
        if not self.lines or time.monotonic() - self.used > 600:
            return ""
        return ("Your walks in this area so far, oldest first (positions are rough, in "
                "character steps; ignore this if you've since gone somewhere else):\n"
                + "\n".join(f"- {line}" for line in self.lines))

