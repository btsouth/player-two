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

from .walk import Look, changed, motion, new_place, phase_shift, small, spot

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
    # "stopped", "didnt_move"
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
            "walked": f"walked: still going {direction}, {went}{scrolled}",
            "blocked": f"blocked: something stops you going {direction}, after you {went}"
                       f"{scrolled}",
            "new_area": f"new_area: the screen changed to a new place after you {went}",
            "interrupted": f"interrupted: something came up on screen after you {went}; "
                           "look before going on",
            "stopped": f"stopped by the player after you {went}",
            "didnt_move": f"didnt_move: holding {direction} didn't move your character; a "
                          "text box or menu may be up, or it's a wall",
        }[self.outcome] + f" ({self.seconds:.1f} s)"


async def keep_walking(grab: Callable[[], Awaitable[Image.Image]],
                       steer: Callable[[float | None], None], direction: str,
                       seconds: float = 3.0,
                       keep_going: Callable[[], Awaitable[bool]] | None = None,
                       me: tuple[Look, np.ndarray, tuple[int, int]] | None = None,
                       log: Callable[[str], None] | None = None) -> Went:
    """Hold `direction` for up to `seconds`, stopping early at a wall, a new place or a
    popup. `me` is the character as last seen by a walk (its colours, where, its size),
    to follow it; without it, it's the moving thing nearest the middle of the screen,
    where most 2D games keep the character."""
    angle = ANGLES[direction]
    unit = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
    t0 = time.monotonic()
    rgb, prev = small(await grab())
    h, w = prev.shape
    if me is not None:
        look, at, (bw, bh) = me
        at = np.array(at, float)
    else:
        # About the size of a 2D game's sprite (Crono is a thirtieth of the screen wide),
        # somewhere on screen: found once it moves.
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
            if new_place(prev, g):
                outcome = "new_area"
                break
            mask = (int(at[0] - bw), int(at[1] - bh), int(at[0] + bw), int(at[1] + bh))
            sx, sy, _ = phase_shift(prev, g, mask)
            scroll = np.array([sx, sy])
            # While the screen scrolls, the camera is following the character: that's how
            # far it went (followers and passers-by would only add noise). On a still
            # screen (a small room, the edge of the map) the character crosses it.
            follow = bool(np.hypot(sx, sy) > 1)
            # Where it is: by its colours if a walk has seen it, else by what moves
            # differently from the scenery, near where it was (anywhere, at first).
            radius = int(2 * max(bw, bh)) if known else w // 2
            moved_px = motion(prev, g, (sx, sy))
            if look is not None:
                x, y, score = look.find(rgb, (at[0], at[1]), radius, moved_px)
                found = score > 0.3
            else:
                x, y, score = spot(moved_px, (at[0], at[1]), radius, (bw, bh))
                found = score > 8
            now = np.array([x, y]) if found else at
            if follow:
                ground -= scroll
            elif known:
                ground += now - at
            known = known or found
            at = now
            view += scroll
            at = now
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
    if log is not None:
        log(f"keep_walking {direction}: {outcome} after {len(history)} frames, ground "
            f"{(ground / size).round(1).tolist()}, last changes {[round(c, 3) for c in popup]}")
    return Went(outcome, ground / size, float(np.hypot(*view)) / w,
                time.monotonic() - t0 - paused, at if outcome != "new_area" else None)


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

