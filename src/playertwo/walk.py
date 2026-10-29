"""Walking to something on screen: Muse Spark points, local code steers.

A model call per step is far too slow to walk with (a second of walking, then two of
thinking, and it loses track). Instead Muse Spark looks once: where the player's
character is, where the target is, and a waypoint or two around anything in the way.
Then plain image math does the walking, about ten times a second: it follows the
character's sprite, follows the screen as it scrolls (so the target is followed too),
and turns the stick toward the target until the character is next to it. It asks Muse
Spark again when it gets stuck or loses the character, and stops when the screen changes
to a new place (a door, stairs).

It is for 2D games, where up on the stick or d-pad is up on the screen.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .muse import Muse, image_part

WIDTH = 480  # frames are tracked at this width


def gray(img: Image.Image, width: int = WIDTH) -> np.ndarray:
    h = max(1, round(img.height * width / img.width))
    return np.asarray(img.convert("L").resize((width, h), Image.BILINEAR), dtype=np.float32)


def small(img: Image.Image, width: int = WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """The frame at tracking size: colour (for the character) and gray (for scrolling)."""
    h = max(1, round(img.height * width / img.width))
    rgb = img.convert("RGB").resize((width, h), Image.BILINEAR)
    return np.asarray(rgb), np.asarray(rgb.convert("L"), dtype=np.float32)


def _bins(rgb: np.ndarray) -> np.ndarray:
    q = rgb.astype(np.int32) >> 5  # 8 levels a channel, 512 colours
    return q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]


class Look:
    """What the character looks like, as colours: the ones it has and its surroundings
    mostly don't. Unlike a picture of the sprite, that holds when it turns to walk the
    other way or swings its arms."""

    def __init__(self, rgb: np.ndarray, box: tuple[float, float, float, float]):
        x1, y1, x2, y2 = box
        self.w = max(4, round(x2 - x1))
        self.h = max(4, round(y2 - y1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        b = _bins(rgb)
        # The middle of the box is the character; a ring around it is the ground.
        inner = self._cut(b, cx, cy, 0.7 * self.w, 0.8 * self.h)
        self.fg = np.bincount(inner.ravel(), minlength=512).astype(np.float64)
        self.bg = np.zeros(512)
        self.weights = np.zeros(512)
        self.base = 1.0
        self.ground(rgb, (cx, cy), b)
        _, _, m = self._best(b, cx, cy, 0, 0)
        self.base = max(m, 1e-3)

    @staticmethod
    def _cut(b, cx, cy, w, h):
        H, W = b.shape
        x0, y0 = max(0, round(cx - w / 2)), max(0, round(cy - h / 2))
        return b[y0:min(H, round(cy + h / 2)), x0:min(W, round(cx + w / 2))]

    def ground(self, rgb, at, b=None) -> None:
        """Learn the colours around the character, where it is now."""
        b = _bins(rgb) if b is None else b
        cx, cy = at
        outer = self._cut(b, cx, cy, 3 * self.w, 2 * self.h)
        mid = self._cut(b, cx, cy, self.w, self.h)
        bg = np.bincount(outer.ravel(), minlength=512) - np.bincount(mid.ravel(), minlength=512)
        bg = np.maximum(bg, 0).astype(np.float64)
        self.bg = bg if not self.bg.any() else 0.5 * self.bg + 0.5 * bg
        pf = self.fg / max(self.fg.sum(), 1)
        pb = self.bg / max(self.bg.sum(), 1)
        self.weights = np.where(self.fg >= 3, pf / (pf + pb + 1e-9), 0.0)

    def _best(self, b, cx, cy, rx, ry, moved=None):
        H, W = b.shape
        x0 = max(0, round(cx - self.w / 2 - rx))
        y0 = max(0, round(cy - self.h / 2 - ry))
        x1 = min(W, round(cx + self.w / 2 + rx))
        y1 = min(H, round(cy + self.h / 2 + ry))
        if x1 - x0 < self.w or y1 - y0 < self.h:
            return cx, cy, 0.0
        h, w = self.h, self.w
        sums = _box_means(self.weights[b[y0:y1, x0:x1]], w, h)
        rank = sums
        if moved is not None:
            # Walking, things of its colours that stand still (a stall, a flag) aren't it.
            mot = _box_means(moved[y0:y1, x0:x1], w, h)
            if mot.max() > 4:
                rank = sums * (0.4 + 0.6 * mot / mot.max())
        ys, xs = np.mgrid[0:sums.shape[0], 0:sums.shape[1]]
        px, py = x0 + xs + w / 2, y0 + ys + h / 2
        # Of equally good spots, the one nearest where it should be.
        far = ((px - cx) / max(rx, 1)) ** 2 + ((py - cy) / max(ry, 1)) ** 2
        iy, ix = np.unravel_index(int(np.argmax(rank - 0.15 * rank.max() * far)), sums.shape)
        return float(px[iy, ix]), float(py[iy, ix]), float(sums[iy, ix])

    def find(self, rgb: np.ndarray, center: tuple[float, float], radius: int,
             moved: np.ndarray | None = None) -> tuple[float, float, float]:
        """The character's middle near `center`, and how sure (1: as clear as at first).
        `moved` (see motion) favours what is moving, while the character walks."""
        x, y, m = self._best(_bins(rgb), center[0], center[1], radius, radius, moved)
        return x, y, m / self.base


def _edges(x: np.ndarray) -> np.ndarray:
    """Detail only: the image minus a blurred copy, so texture, not big areas of colour,
    decides how it moved."""
    k = 4
    pad = np.pad(x, k, mode="edge")
    c = pad.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * k + 1
    blur = (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)
    return x - blur[: x.shape[0], : x.shape[1]]


def phase_shift(a: np.ndarray, b: np.ndarray, mask: tuple[int, int, int, int] | None = None,
                limit: int = 24) -> tuple[float, float, float]:
    """How far the picture moved from a to b, in pixels (x right, y down), and how sure.

    `mask` (x0, y0, x1, y1) is left out of both, so the walking character doesn't count
    as the scenery moving. Only moves up to `limit` pixels are considered: between two
    frames a game scrolls only a little, and big jumps are false matches.
    """
    a, b = _edges(a), _edges(b)
    if mask is not None:
        x0, y0, x1, y1 = mask
        a[max(0, y0):y1, max(0, x0):x1] = 0
        b[max(0, y0):y1, max(0, x0):x1] = 0
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    c = np.fft.fftshift(np.fft.ifft2(r).real)
    h, w = c.shape
    cy, cx = h // 2, w // 2
    near = c[cy - limit:cy + limit + 1, cx - limit:cx + limit + 1]
    y, x = np.unravel_index(int(np.argmax(near)), near.shape)

    def sub(m1: float, m0: float, p1: float) -> float:
        """Where the true peak is between three samples (a parabola through them)."""
        d = m1 - 2 * m0 + p1
        return 0.0 if d >= 0 else 0.5 * (m1 - p1) / d

    fy = sub(near[y - 1, x], near[y, x], near[y + 1, x]) if 0 < y < near.shape[0] - 1 else 0
    fx = sub(near[y, x - 1], near[y, x], near[y, x + 1]) if 0 < x < near.shape[1] - 1 else 0
    return float(x - limit + fx), float(y - limit + fy), float(near.max())


def motion(a: np.ndarray, b: np.ndarray, scroll: tuple[float, float]) -> np.ndarray:
    """How much each pixel changed from a to b other than by the scenery scrolling: the
    walking character lights up, whether the screen follows it or it walks across a
    still one, and so do other people moving."""
    ix, iy = round(scroll[0]), round(scroll[1])
    moved = np.abs(b - np.roll(a, (iy, ix), axis=(0, 1)))
    # What rolled in from the far edge is no picture at all.
    if ix > 0:
        moved[:, :ix] = 0
    elif ix < 0:
        moved[:, ix:] = 0
    if iy > 0:
        moved[:iy] = 0
    elif iy < 0:
        moved[iy:] = 0
    return moved


def _box_means(x: np.ndarray, w: int, h: int) -> np.ndarray:
    c = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]) / (w * h)


def spot(moved: np.ndarray, center: tuple[float, float], radius: int, size: tuple[int, int]
         ) -> tuple[float, float, float]:
    """The middle of the most changed `size` box near `center` (see motion), and how
    much it changed (gray levels, on average)."""
    w, h = size
    H, W = moved.shape
    x0 = max(0, round(center[0] - w / 2 - radius))
    y0 = max(0, round(center[1] - h / 2 - radius))
    x1 = min(W, round(center[0] + w / 2 + radius))
    y1 = min(H, round(center[1] + h / 2 + radius))
    if x1 - x0 < w or y1 - y0 < h:
        return center[0], center[1], 0.0
    sums = _box_means(moved[y0:y1, x0:x1], w, h)
    ys, xs = np.mgrid[0:sums.shape[0], 0:sums.shape[1]]
    px, py = x0 + xs + w / 2, y0 + ys + h / 2
    # Others move too (people walking, dancers): the nearest to where it should be wins.
    far = ((px - center[0]) ** 2 + (py - center[1]) ** 2) / max(radius, 1) ** 2
    iy2, ix2 = np.unravel_index(int(np.argmax(sums * (1 - 0.6 * np.minimum(far, 1)))),
                                sums.shape)
    return float(px[iy2, ix2]), float(py[iy2, ix2]), float(sums[iy2, ix2])


def new_place(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the picture changed to a different place (a fade, a door), not just
    scrolled: compared as blurry thumbnails, where a few pixels of scrolling hardly count."""
    def tiny(x: np.ndarray) -> np.ndarray:
        h, w = x.shape
        k = max(1, w // 40)
        x = x[: h - h % k, : w - w % k]
        return x.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    return float(np.abs(tiny(a) - tiny(b)).mean()) > 40


def compass(dx: float, dy: float) -> float:
    """Screen direction as a stick angle: 0 up, 90 right."""
    return math.degrees(math.atan2(dx, -dy)) % 360


def wrap(a: float) -> float:
    return (a + 180) % 360 - 180


LOCATE = """\
You help a player who can't see the screen walk their character around a game. Find, on \
this game screen, the character the player controls{me}, and "{target}".{context} Answer \
as JSON on a 0 to 1000 grid of the image (x right, y down):
{{"player": [x1, y1, x2, y2] or null,   // a tight box around the player's character
  "target": [x1, y1, x2, y2] or null,   // a tight box around the target
  "via": [[x, y], ...],                 // 0 to 3 points to walk through first, only if \
walls, furniture or people block the straight line; [] otherwise
  "note": "a few words: what's in the way, or why something wasn't found"}}
Use null when you can't see it: don't guess. If the target is off screen, set target to \
null and say in note which way it probably is."""


@dataclass
class Located:
    player: tuple[float, float, float, float] | None  # box, in tracked (WIDTH) pixels
    target: tuple[float, float, float, float] | None
    via: list[tuple[float, float]]
    note: str


def _box(v, sx: float, sy: float):
    try:
        x1, y1, x2, y2 = (float(n) for n in v[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


async def locate(muse: Muse, img: Image.Image, target: str, context: str = "",
                 me: str = "") -> Located:
    """Ask Muse Spark where the player and the target are, in tracked (WIDTH) pixels."""
    sx = WIDTH / 1000
    sy = (img.height * WIDTH / img.width) / 1000
    ctx = f" Earlier: {context}." if context else ""
    msgs = [{"role": "user", "content": [
        image_part(img, 1280),
        {"type": "text", "text": LOCATE.format(target=target, context=ctx,
                                               me=f" ({me})" if me else "")}]}]
    data: dict = {}
    for _ in range(2):  # a reply cut short isn't JSON: ask once more
        try:
            data = await muse.json(msgs, effort="minimal", max_tokens=3000)
            break
        except ValueError:
            continue
    if not isinstance(data, dict):
        data = {}
    via = []
    for p in data.get("via") or []:
        try:
            via.append((float(p[0]) * sx, float(p[1]) * sy))
        except (TypeError, ValueError, IndexError):
            continue
    return Located(_box(data.get("player"), sx, sy), _box(data.get("target"), sx, sy),
                   via[:3], str(data.get("note") or ""))


@dataclass
class Walked:
    # "arrived", "new_area", "blocked", "lost", "not_found", "stopped", "timeout",
    # "didnt_move"
    outcome: str
    detail: str
    seconds: float

    def text(self) -> str:
        return f"{self.outcome}: {self.detail} ({self.seconds:.1f} s)"


class Walker:
    """Walks the player's character to a target on screen. See the module docstring."""

    def __init__(self, muse: Muse, grab: Callable[[], Awaitable[Image.Image]],
                 steer: Callable[[float | None], None],
                 keep_going: Callable[[], Awaitable[bool]] | None = None,
                 log: Callable[[str], None] | None = None):
        self.muse = muse
        self.grab = grab
        self.steer = steer          # stick angle (0 up, 90 right), or None to let go
        self.keep_going = keep_going
        self.log = log or (lambda _m: None)
        # The character as last seen (its colours, where, its box size), and as seen now:
        # the model sometimes picks someone else, and this one has been walked already.
        self.memory: tuple[Look, np.ndarray, tuple[int, int]] | None = None
        self._now: tuple[Look, np.ndarray, tuple[int, int]] | None = None
        self.me = ""

    async def _frame(self) -> tuple[np.ndarray, np.ndarray]:
        return small(await self.grab())

    async def walk(self, target: str, me: str = "", timeout: float = 25.0) -> Walked:
        """Walk to `target`; `me` says which character is the player's, if it's unclear."""
        t0 = time.monotonic()
        self.me = me
        # How far it went over the ground, so the model can keep a map in its head.
        self.travel = np.zeros(2)
        self.size = 0.0
        self._now = None

        def done(outcome: str, detail: str) -> Walked:
            self.steer(None)
            self.memory = self._now if outcome != "new_area" else None
            if self.size and np.hypot(*self.travel) > self.size:
                detail += f"; walked {self._words(self.travel, self.size)} in all"
            return Walked(outcome, detail, time.monotonic() - t0)

        img = await self.grab()
        loc = await locate(self.muse, img, target, me=me)
        self.log(f"located {loc}")
        if self.memory is not None:
            look, at, (mw, mh) = self.memory
            x, y, score = look.find(small(img)[0], (at[0], at[1]), 3 * max(mw, mh))
            p = loc.player
            if score > 0.85 and (p is None or np.hypot((p[0] + p[2]) / 2 - x,
                                                       (p[1] + p[3]) / 2 - y) > max(mw, mh)):
                self.log(f"the character as last seen is at {x:.0f},{y:.0f}, not {p}")
                loc.player = (x - mw / 2, y - mh / 2, x + mw / 2, y + mh / 2)
        if loc.player is None:
            return done("lost", "I can't tell which character is yours on this screen. "
                        + loc.note)
        if loc.target is None:
            return done("not_found", f"I can't see {target} on screen. {loc.note}".strip())
        try:
            return await self._walk(target, img, loc, t0, timeout, done)
        finally:
            self.steer(None)

    async def _walk(self, target, img, loc, t0, timeout, done, stuck_turns=0) -> Walked:
        rgb, g = small(img)
        w = g.shape[1]
        px1, py1, px2, py2 = loc.player
        size = max(6.0, min(px2 - px1, py2 - py1, 0.25 * w))
        player = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
        self.size = size
        # The model's box is only roughly right, and a small sprite is mostly background:
        # the character is pinned down once it moves (see spot), then followed by colour.
        bw, bh = max(4, round(px2 - px1)), max(4, round(py2 - py1))
        look: Look | None = None
        unsure = 0  # frames in a row without a clear sight of the character
        began = time.monotonic()
        seen_moving = False
        tx1, ty1, tx2, ty2 = loc.target
        goal = np.array([(tx1 + tx2) / 2, (ty1 + ty2) / 2])
        tsize = max(tx2 - tx1, ty2 - ty1)
        # Lined up with it, side by side or one above the other: characters talk to and
        # open things only straight ahead, in the four directions.
        line_x = max(0.35 * size, 0.35 * (tx2 - tx1))
        line_y = max(0.35 * size, 0.35 * (ty2 - ty1))
        # Close enough: touching distance for the two sprites.
        reach = 0.6 * (size + min(tsize, 3 * size))
        points = [np.array(p) for p in loc.via] + [goal]
        prev = g
        last_rel = time.monotonic()
        slide = 0
        steps = 0
        moves: list[np.ndarray] = []  # the last few moves over the ground
        best = (float("inf"), time.monotonic())  # closest to the aim so far, and when
        stick: float | None = None
        while True:
            if time.monotonic() - t0 > timeout:
                return done("timeout", f"still {self._words(points[-1] - player, size)} "
                                       f"from {target} after {timeout:.0f} s")
            if self.keep_going is not None and not await self.keep_going():
                return done("stopped", "stopped by the player")
            off = points[0] - player
            dist = float(np.hypot(*off))
            if dist < best[0] - 0.3 * size:
                best = (dist, time.monotonic())
            elif time.monotonic() - best[1] > 2.5:
                # Moving, but not getting closer (sliding along a long wall).
                best = (dist, time.monotonic())
                stuck_turns += 1
                if stuck_turns > 2:
                    return done("blocked", f"something is in the way, "
                                f"{self._words(points[-1] - player, size)} from {target}")
                loc2 = await locate(self.muse, await self.grab(), target,
                                    "the straight way is blocked; give a way around", self.me)
                self.log(f"relocated (no progress) {loc2}")
                if loc2.player is not None and loc2.target is not None:
                    return await self._restart(target, loc2, t0, timeout, done, stuck_turns)
            want = None
            if len(points) == 1 and dist <= 1.6 * reach:
                dx, dy = off
                if abs(dx) >= abs(dy):
                    lined, face, across = abs(dy) <= line_y, (90 if dx > 0 else 270), (0, dy)
                else:
                    lined, face, across = abs(dx) <= line_x, (180 if dy > 0 else 0), (dx, 0)
                if lined and dist <= reach:
                    # Face it, so a press of a talks to it or opens it.
                    self.steer(face)
                    await asyncio.sleep(0.08)
                    return done("arrived", f"next to {target}, facing it")
                if not lined and slide == 0:
                    want = compass(*across)  # step sideways to line up
            elif len(points) > 1 and dist <= max(size, reach * 0.6):
                points.pop(0)
                continue
            if want is None:
                want = (compass(*off) + (0, 45, -45, 90, -90)[slide]) % 360
            # Eight ways, like a d-pad: most 2D games walk no other way. Keep the last
            # one until the aim is well past the halfway line, so it doesn't flicker.
            if stick is None or abs(wrap(want - stick)) > 30:
                stick = (round(want / 45) * 45) % 360
                moves = []  # moves the other way say nothing about this one
            self.steer(stick)
            await asyncio.sleep(0.01)
            rgb, g = await self._frame()
            steps += 1
            if new_place(prev, g):
                return done("new_area", f"the screen changed while walking to {target}; "
                                        "probably a door or a new area")
            pad = int(size)
            sx, sy, _q = phase_shift(prev, g, (int(player[0]) - pad, int(player[1]) - pad,
                                               int(player[0]) + pad, int(player[1]) + pad))
            scroll = np.array([sx, sy])
            for p in points:
                p += scroll
            moved = motion(prev, g, (sx, sy))
            if look is not None:
                nx, ny, score = look.find(rgb, (player[0], player[1]), int(size), moved)
                velocity = np.array([nx, ny]) - player
                fast = float(np.hypot(*scroll))
                # Moving just like the scenery: that's the ground, not the character.
                if score < 0.5 or (fast > 2 and np.hypot(*(velocity - scroll)) < 0.3 * fast):
                    look = None
            if look is None:
                unsure += 1
                x, y, how = spot(moved, (player[0], player[1]), int(2 * size), (bw, bh))
                prev = g
                if how > 6:
                    seen_moving = True
                    velocity = np.array([x, y]) - player
                    player = np.array([x, y])
                    look = Look(rgb, (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2))
                    unsure = 0
                elif not seen_moving:
                    if time.monotonic() - began > 1.2:
                        if stuck_turns == 0:
                            return done("didnt_move", "your character didn't move. A text "
                                        "box or a cutscene is probably on: press a to move "
                                        "it along")
                        # Up against something after asking again: go by the model's box.
                        look = Look(rgb, (player[0] - bw / 2, player[1] - bh / 2,
                                          player[0] + bw / 2, player[1] + bh / 2))
                        seen_moving = True
                elif unsure > 20:
                    stuck_turns += 1
                    loc2 = None
                    if stuck_turns <= 2:
                        loc2 = await locate(self.muse, await self.grab(), target,
                                            "lost track of the character", self.me)
                        self.log(f"relocated (lost) {loc2}")
                    if loc2 is None or loc2.player is None or loc2.target is None:
                        return done("lost", "I lost track of your character. "
                                    + (loc2.note if loc2 else ""))
                    return await self._restart(target, loc2, t0, timeout, done, stuck_turns)
                continue
            unsure = 0
            player = player + velocity
            self._now = (look, player.copy(), (bw, bh))
            if steps % 10 == 0:
                look.ground(rgb, player)  # the ground changes as it walks
            prev = g
            # How the character moved over the ground, over the last few frames: one
            # frame's is a few pixels of wobble from the walking animation.
            moves = [*moves[-3:], velocity - scroll]
            self.travel += velocity - scroll
            rel = np.mean(moves, axis=0) if len(moves) > 1 else np.zeros(2)
            if float(np.hypot(*rel)) > 0.6:
                last_rel = time.monotonic()
                if slide:
                    slide = 0  # moving again: aim straight
            elif time.monotonic() - last_rel > 0.5:
                slide += 1
                last_rel = time.monotonic()
                if slide > 4:
                    slide = 0
                    stuck_turns += 1
                    if stuck_turns > 2:
                        return done("blocked", f"something is in the way, "
                                    f"{self._words(points[-1] - player, size)} from {target}")
                    loc2 = await locate(self.muse, await self.grab(), target,
                                        "the straight way is blocked; give a way around", self.me)
                    self.log(f"relocated (blocked) {loc2}")
                    if loc2.player is not None and loc2.target is not None:
                        return await self._restart(target, loc2, t0, timeout, done,
                                                   stuck_turns)

    async def _restart(self, target, loc, t0, timeout, done, stuck_turns=0):
        img = await self.grab()
        return await self._walk(target, img, loc, t0, timeout, done, stuck_turns)

    @staticmethod
    def _words(v: np.ndarray, size: float) -> str:
        """Roughly how far and which way, in character widths."""
        steps = max(1, round(float(np.hypot(*v)) / max(size, 1)))
        d = compass(*v)
        side = ["up", "up and right", "right", "down and right", "down", "down and left",
                "left", "up and left"][int(((d + 22.5) % 360) // 45)]
        return f"about {steps} steps {side}"
