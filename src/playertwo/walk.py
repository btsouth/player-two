"""Walking to something on screen: Muse Spark points, local code steers.

A model call per step is far too slow to walk with (a second of walking, then two of
thinking, and it loses track). Instead Muse Spark looks once: where the player's
character is, where the target is, and a waypoint or two around anything in the way.
Then plain image math does the walking, about ten times a second: it follows the
character's sprite, follows the screen as it scrolls (so the target is followed too),
and turns the stick toward the target until the character is next to it. Up against
something, it finds its own way around by bumping (see nav). It asks Muse Spark again
when that runs out or it loses the character, and stops when the screen changes to a
new place (a door, stairs).

It is for 2D games, where up on the stick or d-pad is up on the screen.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from PIL import Image

from .muse import Muse, image_part
from .nav import Detour

WIDTH = 480  # frames are tracked at this width
BUMPS = 20  # bumps a walk plans its own way around before asking the model
WAY_AROUND = "Something's in the way. Finding a way around."


def gray(img: Image.Image, width: int = WIDTH) -> np.ndarray:
    h = max(1, round(img.height * width / img.width))
    return np.asarray(img.convert("L").resize((width, h), Image.BILINEAR), dtype=np.float32)


_last_small: tuple[Image.Image, int, tuple[np.ndarray, np.ndarray]] | None = None


def small(img: Image.Image, width: int = WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """The frame at tracking size: colour (for the character) and gray (for scrolling).
    The same frame asked for again (the map and a walk both look at it) isn't resized
    twice; the arrays are read only."""
    global _last_small
    last = _last_small
    if last is not None and last[0] is img and last[1] == width:
        return last[2]
    h = max(1, round(img.height * width / img.width))
    rgb = img.convert("RGB").resize((width, h), Image.BILINEAR)
    out = np.asarray(rgb), np.asarray(rgb.convert("L"), dtype=np.float32)
    out[1].setflags(write=False)
    _last_small = (img, width, out)
    return out


def _bins(rgb: np.ndarray) -> np.ndarray:
    q = rgb.astype(np.int32) >> 5  # 8 levels a channel, 512 colours
    return q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]


class Look:
    """What the character looks like, as colours: the ones it has and its surroundings
    mostly don't. Unlike a picture of the sprite, that holds when it turns to walk the
    other way or swings its arms."""

    def __init__(self, rgb: np.ndarray, box: tuple[float, float, float, float],
                 bins: np.ndarray | None = None):
        x1, y1, x2, y2 = box
        self.w = max(4, round(x2 - x1))
        self.h = max(4, round(y2 - y1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        b = _bins(rgb) if bins is None else bins
        # The middle of the box is the character; a ring around it is the ground.
        inner = self._cut(b, cx, cy, 0.7 * self.w, 0.8 * self.h)
        self.fg = np.bincount(inner.ravel(), minlength=512).astype(np.float64)
        self.bg = np.zeros(512)
        self.weights = np.zeros(512)
        self.base = 1.0
        self.ground(rgb, (cx, cy), b)
        _, _, m = self._best(b, cx, cy, 0, 0)
        self.base = max(m, 1e-3)
        # Too little of its own colour to follow (a box that's mostly background), or
        # colours the ground all around has too (a box on cobblestones, not on the
        # character): followed, it would slide along the ground with the scroll. On
        # recorded Chrono Trigger frames a sprite's surroundings score a tenth of it or
        # less, half of them, and ground half or more; a look-alike beside it (a party
        # member, a guard) is only one spot of the surroundings.
        self.around = self._around(b, cx, cy) / self.base
        self.usable = m > 0.05 and self.around < 0.35

    def _around(self, b, cx, cy) -> float:
        """The median score of boxes nearby that don't overlap this one."""
        win = _window(b.shape, cx, cy, self.w, self.h, 2 * self.w, 2 * self.h)
        if win is None:
            return 0.0
        x0, y0, x1, y1 = win
        sums = _box_means(self.weights[b[y0:y1, x0:x1]], self.w, self.h)
        ys, xs = np.mgrid[0:sums.shape[0], 0:sums.shape[1]]
        apart = (np.abs(x0 + xs + self.w / 2 - cx) >= self.w) | \
                (np.abs(y0 + ys + self.h / 2 - cy) >= self.h)
        return float(np.median(sums[apart])) if apart.any() else 0.0

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

    def _best(self, b, cx, cy, rx, ry, moved=None, ahead=None):
        win = _window(b.shape, cx, cy, self.w, self.h, rx, ry)
        if win is None:
            return cx, cy, 0.0
        x0, y0, x1, y1 = win
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
        dx, dy = (px - cx) / max(rx, 1), (py - cy) / max(ry, 1)
        far = dx ** 2 + dy ** 2
        if ahead is not None:
            # Behind it, a party member following looks the same: count that twice as far.
            back = np.minimum(dx * ahead[0] + dy * ahead[1], 0)
            far += 3 * back ** 2
        iy, ix = np.unravel_index(int(np.argmax(rank - 0.15 * rank.max() * far)), sums.shape)
        return float(px[iy, ix]), float(py[iy, ix]), float(sums[iy, ix])

    def find(self, rgb: np.ndarray, center: tuple[float, float], radius: int,
             moved: np.ndarray | None = None, ahead: tuple[float, float] | None = None
             ) -> tuple[float, float, float]:
        """The character's middle near `center`, and how sure (1: as clear as at first).
        `moved` (see motion) favours what is moving, while the character walks, and
        `ahead` (the way it walks, a unit vector) what is in front."""
        x, y, m = self._best(_bins(rgb), center[0], center[1], radius, radius, moved, ahead)
        return x, y, m / self.base


def pin(rgb: np.ndarray, at: tuple[float, float], size: tuple[int, int], reach: int = 6
        ) -> tuple[Look, np.ndarray] | None:
    """The character's colours, from a spot found roughly (by what moved): that box if
    it stands out from its surroundings, else the one a few pixels either way that
    stands out most (a box half on the ground learns the ground's colours too), or
    None."""
    bw, bh = size
    b = _bins(rgb)
    look = Look(rgb, (at[0] - bw / 2, at[1] - bh / 2, at[0] + bw / 2, at[1] + bh / 2), b)
    if look.usable:
        return look, np.array(at, float)
    best: tuple[float, Look, np.ndarray] | None = None
    for dy in range(-reach, reach + 1, 3):
        for dx in range(-reach, reach + 1, 3):
            x, y = at[0] + dx, at[1] + dy
            look = Look(rgb, (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2), b)
            if look.usable and (best is None or look.around < best[0]):
                best = (look.around, look, np.array([x, y], float))
    return None if best is None else (best[1], best[2])


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
                limit: int = 64) -> tuple[float, float, float]:
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
    # Weighted toward coarse detail: a move of a pixel and a half otherwise splits the
    # peak in two, and a long straight wall (the same at every shift along it) wins.
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    weight = np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.15 ** 2)).astype(np.float32)
    r *= weight
    c = np.fft.fftshift(np.fft.ifft2(r).real) / float(weight.mean())
    h, w = c.shape
    cy, cx = h // 2, w // 2
    # Far enough for a slow capture of a running character (Crono walks about 6 px a
    # frame at 26 a second, 40 at 4); on recorded frames a wider search found no false
    # matches at full speed either.
    limit = max(1, min(limit, cy - 1, cx - 1))
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


def _window(shape, cx: float, cy: float, w: int, h: int, rx: int, ry: int):
    """The part of a frame to search for a w x h box around (cx, cy), within rx, ry,
    moved inside the frame at its edges. None if the frame is smaller than the box."""
    H, W = shape
    if w > W or h > H:
        return None
    x0 = max(0, min(round(cx - w / 2 - rx), W - w))
    y0 = max(0, min(round(cy - h / 2 - ry), H - h))
    x1 = min(W, max(round(cx + w / 2 + rx), x0 + w))
    y1 = min(H, max(round(cy + h / 2 + ry), y0 + h))
    return x0, y0, x1, y1


def _blur(x: np.ndarray, k: int = 2) -> np.ndarray:
    pad = np.pad(x, k, mode="edge")
    c = np.pad(pad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = 2 * k + 1
    return ((c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n))[: x.shape[0], : x.shape[1]]


def changed(a: np.ndarray, b: np.ndarray, scroll: tuple[float, float]) -> float:
    """How much of the screen changed from a to b beyond scrolling, 0 to 1. Compared
    slightly blurred, so a scroll measured a pixel off (a half-pixel step) doesn't make
    fine detail everywhere look changed; a text box or a menu coming up still does."""
    return float((motion(_blur(a), _blur(b), scroll) > 40).mean())


def _box_means(x: np.ndarray, w: int, h: int) -> np.ndarray:
    c = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]) / (w * h)


def spot(moved: np.ndarray, center: tuple[float, float], radius: int, size: tuple[int, int]
         ) -> tuple[float, float, float]:
    """The middle of the most changed `size` box near `center` (see motion), and how
    much it changed (gray levels, on average)."""
    w, h = size
    win = _window(moved.shape, center[0], center[1], w, h, radius, radius)
    if win is None:
        return center[0], center[1], 0.0
    x0, y0, x1, y1 = win
    sums = _box_means(moved[y0:y1, x0:x1], w, h)
    ys, xs = np.mgrid[0:sums.shape[0], 0:sums.shape[1]]
    px, py = x0 + xs + w / 2, y0 + ys + h / 2
    # Others move too (people walking, dancers): the nearest to where it should be wins.
    far = ((px - center[0]) ** 2 + (py - center[1]) ** 2) / max(radius, 1) ** 2
    iy2, ix2 = np.unravel_index(int(np.argmax(sums * (1 - 0.6 * np.minimum(far, 1)))),
                                sums.shape)
    return float(px[iy2, ix2]), float(py[iy2, ix2]), float(sums[iy2, ix2])


def _tiny(x: np.ndarray, k: int) -> np.ndarray:
    h, w = x.shape
    x = x[: h - h % k, : w - w % k]
    return x.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def new_place(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the picture changed to a different place at once (a cut, a door), not just
    scrolled: compared as blurry thumbnails, where a few pixels of scrolling hardly count.
    A fade over several frames changes each one too little: see Transition."""
    k = max(1, a.shape[1] // 40)
    return float(np.abs(_tiny(a, k) - _tiny(b, k)).mean()) > 40


def cells(a: np.ndarray, b: np.ndarray, shift: tuple[float, float] = (0.0, 0.0),
          rows: int = 8, cols: int = 1) -> np.ndarray:
    """How different b is from a scrolled by `shift` (x right, y down), as blurry
    thumbnails, in each of rows x cols parts of the screen (mean gray levels): the
    parts say whether a change is all over (a fade, another place) or only in some
    (a text box, a menu, a far-off layer moving at its own speed)."""
    h, w = a.shape
    ix = max(-w // 2, min(round(shift[0]), w // 2))
    iy = max(-h // 2, min(round(shift[1]), h // 2))
    # b[y, x] is a[y - iy, x - ix]: the part both show.
    a = a[max(-iy, 0):h - max(iy, 0), max(-ix, 0):w - max(ix, 0)]
    b = b[max(iy, 0):h - max(-iy, 0), max(ix, 0):w - max(-ix, 0)]
    k = max(1, w // 40)
    d = np.abs(_tiny(a, k) - _tiny(b, k))
    th, tw = d.shape
    ys = np.linspace(0, th, min(rows, th) + 1).astype(int)
    xs = np.linspace(0, tw, min(cols, tw) + 1).astype(int)
    return np.array([[d[y0:y1, x0:x1].mean() for x0, x1 in pairwise(xs)]
                     for y0, y1 in pairwise(ys)])


def blank(g: np.ndarray) -> bool:
    """A dark or nearly flat picture: the middle of a fade, a screen between places."""
    m, sd = float(g.mean()), float(g.std())
    return sd < 5 or (m < 15 and sd < 10)


class Transition:
    """Notices the picture going to another place, at once or gradually (a fade out and
    in over many frames, each changing too little for new_place), so that a walk stops
    counting the scroll and the character's steps (phase_shift on dark frames is noise)
    and says where it ended up only once the picture settles.

    see() each frame, with the scroll phase_shift measured from the last one. A change
    counts when the picture goes dark or flat, or differs over most of the screen from a
    frame about AGE seconds before (allowing for the scroll since): a text box or a menu
    covers less, walking changes little. Then it waits for the picture to settle and
    compares it with the one before the change: the same place (a flash, a false alarm)
    or a new one."""

    AGE = 0.4      # seconds back the frame compared with is
    CHANGE = 20    # gray levels most of the thumbnail must change by
    CALM = 4       # and change less than, over CALM_S, to have settled
    CALM_S = 0.12
    DARK_S = 1.0   # a dark screen calm this long is a dark place, not a fade

    def __init__(self) -> None:
        self._hist: list[tuple[float, np.ndarray, np.ndarray]] = []  # time, frame, scroll
        self._scroll = np.zeros(2)
        self.before: np.ndarray | None = None  # the last frame before the change
        self.since: float | None = None        # when the change began, or None
        self._calm: list[tuple[float, np.ndarray]] = []
        self.state = "steady"

    @property
    def changing(self) -> bool:
        return self.since is not None

    def see(self, g: np.ndarray, scroll: tuple[float, float] = (0.0, 0.0),
            now: float | None = None) -> str:
        """"steady" (scroll and steps count), "changing" (they don't), then once
        settled "same" (back to steady) or "new" (another place; steady from here)."""
        now = time.monotonic() if now is None else now
        if self.since is None:
            if self._hist and self._hist[-1][1].shape != g.shape:
                self._begin(now)
                return self._settle(g, now)
            self._scroll = self._scroll + np.asarray(scroll, float)
            if self._hist:
                old = [h for h in self._hist if now - h[0] >= self.AGE]
                _, ref, s0 = old[-1] if old else self._hist[0]
                last = self._hist[-1][1]
                if (blank(g) and not blank(last)) or new_place(last, g) or \
                        self._moved_on(ref, g, self._scroll - s0):
                    self._begin(now)
                    return self._settle(g, now)
            self._hist = [h for h in self._hist[-40:] if now - h[0] < self.AGE + 0.3]
            self._hist.append((now, g, self._scroll.copy()))
            self.state = "steady"
            return self.state
        return self._settle(g, now)

    def _moved_on(self, ref: np.ndarray, g: np.ndarray, shift) -> bool:
        """Whether most of the screen changed since `ref`, beyond the scroll. The scroll
        added up frame by frame goes wrong once the view moves further between frames
        than phase_shift looks (a slow capture, a running character), so a change is
        checked again with the two frames lined up directly, searching further."""
        if not self._spread(ref, g, shift):
            return False
        sx, sy, _ = phase_shift(ref, g, limit=min(80, g.shape[0] // 2 - 2))
        return self._spread(ref, g, (sx, sy))

    def _spread(self, a: np.ndarray, b: np.ndarray, shift) -> bool:
        if a.shape != b.shape:
            return True
        # Over three quarters of the screen, so a far-off layer moving at its own speed
        # (half of it, say) doesn't count.
        return float(np.quantile(cells(a, b, shift, 4, 4), 0.25)) > self.CHANGE

    def _begin(self, now: float) -> None:
        self.before = self._hist[-1][1] if self._hist else None
        self.since = now
        self._calm = []

    def _settle(self, g: np.ndarray, now: float) -> str:
        self._calm = [c for c in self._calm if now - c[0] < 1.0]
        old = [c for c in self._calm if now - c[0] >= self.CALM_S]
        self._calm.append((now, g))
        dark = blank(g) and self.since is not None and now - self.since < self.DARK_S
        if dark or not old or old[-1][1].shape != g.shape or \
                float(np.median(cells(old[-1][1], g, (0, 0), 4, 4))) >= self.CALM:
            self.state = "changing"
            return self.state
        same = False
        if self.before is not None and self.before.shape == g.shape:
            h = g.shape[0]
            sx, sy, _ = phase_shift(self.before, g, limit=min(60, h // 2 - 2))
            # A cut that changed the picture outright is somewhere else, even where both
            # places share dark margins that match.
            same = not self._spread(self.before, g, (sx, sy)) and not new_place(self.before, g)
        self.since, self.before, self._calm = None, None, []
        self._hist = [(now, g, self._scroll.copy())]
        self.state = "same" if same else "new"
        return self.state


async def settle(grab: Callable[[], Awaitable[Image.Image]], scene: Transition,
                 keep_going: Callable[[], Awaitable[bool]] | None = None, wait: float = 3.0
                 ) -> tuple[str, np.ndarray, np.ndarray]:
    """Once a change began (see Transition), frames until the picture settles: "same"
    or "new" with the last frame (rgb, gray), "new" too if it is still changing after
    `wait` seconds (a screen left black), or "stopped" if the player stopped it."""
    t0 = time.monotonic()
    while True:
        if keep_going is not None:
            t = time.monotonic()
            go = await keep_going()
            t0 += time.monotonic() - t
            if not go:
                return "stopped", *small(await grab())
        await asyncio.sleep(0.04)
        rgb, g = small(await grab())
        state = scene.see(g)
        if state in ("same", "new"):
            return state, rgb, g
        if time.monotonic() - t0 > wait:
            return "new", rgb, g


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
    x1, x2 = max(0.0, min(x1, 1000.0)), max(0.0, min(x2, 1000.0))
    y1, y2 = max(0.0, min(y1, 1000.0)), max(0.0, min(y2, 1000.0))
    if x2 - x1 < 2 or y2 - y1 < 2:
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


# Targets that are someone: SAM 3.1 finds people as "person" far more often than by a
# description ("the man in the tent" finds nothing, "person" finds him).
PEOPLE = re.compile(
    r"\b(man|men|woman|women|girl|boy|kid|child|person|people|guy|lady|guard|soldier|"
    r"merchant|shopkeeper|keeper|vendor|seller|king|queen|knight|mother|mom|father|dad|"
    r"old|friend|him|her|them|someone|somebody|villager|chef|maid|priest|elder)\b",
    re.IGNORECASE)

Boxes = Callable[[Image.Image, str], Awaitable[list[tuple[float, float, float, float]]]]


def snap(target: tuple[float, float, float, float],
         boxes: list[tuple[float, float, float, float]]
         ) -> tuple[float, float, float, float] | None:
    """The exact box (SAM's) for a rough one (the model's): the nearest whose middle is
    within a box and a half of it, or None. All in tracked pixels."""
    x1, y1, x2, y2 = target
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    reach = max(1.5 * max(x2 - x1, y2 - y1), 0.06 * WIDTH)
    best, far = None, reach
    for b in boxes:
        d = math.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
        # Not a whole building or the ground: about the size of what was meant.
        if d < far and (b[2] - b[0]) * (b[3] - b[1]) < 9 * max(1.0, (x2 - x1) * (y2 - y1)):
            best, far = b, d
    return best


@dataclass
class Walked:
    # "arrived", "new_area", "blocked", "lost", "not_found", "stopped", "timeout",
    # "didnt_move", "interrupted"
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
                 log: Callable[[str], None] | None = None,
                 say: Callable[[str], Awaitable[None]] | None = None,
                 exact: Boxes | None = None):
        self.muse = muse
        self.grab = grab
        self.steer = steer          # stick angle (0 up, 90 right), or None to let go
        self.keep_going = keep_going
        self.log = log or (lambda _m: None)
        self.say = say  # a few words out loud when a walk takes longer than it should
        # Exact boxes for a phrase (SAM 3.1), 0 to 1 of the screen: the model's boxes are
        # often a sprite off, and the walk stops and faces by them.
        self.exact = exact
        # The character as last seen (its colours, where, its box size), and as seen now:
        # the model sometimes picks someone else, and this one has been walked already.
        self.memory: tuple[Look, np.ndarray, tuple[int, int]] | None = None
        self._now: tuple[Look, np.ndarray, tuple[int, int]] | None = None
        self.me = ""
        self._where: Callable[[], tuple[float, float]] | None = None
        # When "finding a way around" was last said: once is enough for a player
        # whose next few walks bump about the same room.
        self._said_at: dict[str, float] = {}  # when each line was last said
        self._talk: asyncio.Task | None = None

    async def _frame(self) -> tuple[np.ndarray, np.ndarray]:
        return small(await self.grab())

    def _start(self) -> None:
        # How far it went over the ground, so the model can keep a map in its head.
        self.travel = np.zeros(2)
        self.size = 0.0
        self._now = None
        self.paused = 0.0  # seconds held while the player talked: not the walk's time
        self._bumps = 0
        self._planning: tuple[float, float] | None = None  # began (time, paused then)

    def _done(self, t0: float):
        def done(outcome: str, detail: str) -> Walked:
            self.steer(None)
            # A walk that never saw the character (the target not found) leaves what was
            # known; a new place forgets it.
            self.memory = (self._now or self.memory) if outcome != "new_area" else None
            self._now = None  # where it is between walks is memory's, once checked
            if self.size and np.hypot(*self.travel) > self.size:
                detail += f"; walked {self._words(self.travel, self.size)} in all"
            return Walked(outcome, detail, time.monotonic() - t0)
        return done

    async def walk_toward(self, where: Callable[[], tuple[float, float] | None], label: str,
                          timeout: float = 25.0) -> Walked:
        """Walk to a point that may be off screen, from the area map: `where` says where
        it is on screen now (tracked pixels). Needs the character known (memory)."""
        t0 = time.monotonic()
        self._start()
        self._where = where
        done = self._done(t0)
        try:
            if self.memory is None:
                return done("lost", "I don't know which character is yours yet")
            img = await self.grab()
            loc = self._here(where)
            if loc.target is None:
                return done("lost", loc.note)
            try:
                return await self._walk(label, img, loc, t0, timeout, done)
            finally:
                self.steer(None)
        finally:
            self._where = None

    def _here(self, where: Callable[[], tuple[float, float] | None]) -> Located:
        _look, at, (bw, bh) = self._now or self.memory  # type: ignore[misc]
        me = (at[0] - bw / 2, at[1] - bh / 2, at[0] + bw / 2, at[1] + bh / 2)
        p = where()
        if p is None:
            # The map it was on started over: through a door, or lost track of the view.
            return Located(me, None, [], "the area map started over, so that spot is gone "
                           "(a new area?): look at area_map again")
        x, y = p
        return Located(me, (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2), [], "")

    async def walk(self, target: str, me: str = "", timeout: float = 25.0) -> Walked:
        """Walk to `target`; `me` says which character is the player's, if it's unclear."""
        t0 = time.monotonic()
        self._where = None
        self.me = me
        self._start()
        done = self._done(t0)

        img = await self.grab()
        loc = await self._locate(img, target)
        self.log(f"located {loc}")
        if self.memory is not None:
            look, at, (mw, mh) = self.memory
            rgb = small(img)[0]
            x, y, score = look.find(rgb, (at[0], at[1]), 3 * max(mw, mh))
            p = loc.player
            theirs = 0.0
            if p is not None:
                theirs = look.find(rgb, ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2),
                                   max(mw, mh) // 2)[2]
            # Only when the model's pick doesn't look like the character walked before.
            if score > 0.85 and theirs < 0.5:
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
        # Each look by the model (and its way points) gets its own bumps to plan with.
        self._bumps, self._planning = 0, None
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
        drift = np.zeros(2)  # its walking, a frame's worth over the ground
        hid = 0.0  # when it went out of sight behind something, or 0
        seen = player.copy()  # where it was last really seen (moved with the scroll)
        scene = Transition()
        scene.see(g)
        best = (float("inf"), time.monotonic())  # closest to the aim so far, and when
        stick: float | None = None
        popped: list[float] = []  # how much of the screen changed, last few frames
        # The way around what it bumped into, once it has (see nav), and how long to
        # keep finding it before asking the model.
        detour: Detour | None = None
        budget = max(4.0, min(12.0, timeout / 3))
        while True:
            if time.monotonic() - t0 - self.paused > timeout:
                return done("timeout", f"still {self._words(points[-1] - player, size)} "
                                       f"from {target} after {timeout:.0f} s")
            if self.keep_going is not None:
                t_wait = time.monotonic()
                if not await self.keep_going():
                    return done("stopped", "stopped by the player")
                gap = time.monotonic() - t_wait
                if gap > 0.25:
                    # Held while the player talked: that's not time spent walking.
                    self.paused += gap
                    best = (best[0], best[1] + gap)
                    last_rel += gap
                    began += gap
                    hid = hid + gap if hid else 0.0
                    _, prev = await self._frame()
            # Going around something: the next corner is the aim for now.
            turn = None
            if detour is not None:
                left = len(detour.corners)
                turn = detour.corner(player)
                if len(detour.corners) != left:
                    best = (float("inf"), time.monotonic())  # progress counts from here
            aim = detour.grid.middle(detour.corners[0][0]) if turn is not None else points[0]
            off = aim - player
            dist = float(np.hypot(*off))
            if dist < best[0] - 0.3 * size:
                best = (dist, time.monotonic())
                self._planning = None  # getting closer: the planning clock starts over
            elif time.monotonic() - best[1] > 2.0:
                # Moving, but not getting closer (sliding along a long wall).
                best = (dist, time.monotonic())
                way = self._reroute(detour, player, points[0], stick, size, budget, False)
                if way is not None:
                    detour, last_rel, slide = way, time.monotonic(), 0
                    continue
                stuck_turns += 1
                if stuck_turns > 1:
                    return done("blocked", "something is in the way, maybe out of reach "
                                f"from here, {self._words(points[-1] - player, size)} from "
                                f"{target}")
                img2, loc2 = await self._again(target, "the straight way is blocked; give a "
                                               "way around", WAY_AROUND)
                self.log(f"relocated (no progress) {loc2}")
                if loc2.player is not None and loc2.target is not None:
                    return await self._walk(target, img2, loc2, t0, timeout, done, stuck_turns)
                if self._where is not None:
                    return done("lost", loc2.note)
                # No better idea: go on from here, timed from now, not from before asking.
                best, last_rel, prev, stick = (dist, time.monotonic()), time.monotonic(), \
                    small(img2)[1], None
                detour = None
            want = turn
            if turn is None and len(points) == 1 and dist <= 1.6 * reach:
                dx, dy = off
                if abs(dx) >= abs(dy):
                    lined, face, across = abs(dy) <= line_y, (90 if dx > 0 else 270), (0, dy)
                else:
                    lined, face, across = abs(dx) <= line_x, (180 if dy > 0 else 0), (dx, 0)
                if lined and dist <= reach and not hid and look is not None and unsure == 0:
                    # Face it, so a press of a talks to it or opens it. Only on a fresh
                    # sighting: out of sight or lost, where it is is a guess, and it may
                    # be stuck short.
                    self.steer(face)
                    await asyncio.sleep(0.08)
                    return done("arrived", f"next to {target}, facing it")
                if not lined and slide == 0:
                    want = compass(*across)  # step sideways to line up
            elif turn is None and len(points) > 1 and dist <= max(size, reach * 0.6):
                points.pop(0)
                best = (float("inf"), time.monotonic())  # progress counts from here
                continue
            if want is None:
                want = detour.toward(player, points[0]) if detour is not None else (
                    (compass(*off) + (0, 45, -45, 90, -90)[slide]) % 360)
            if not seen_moving:
                # Not moving yet: maybe up against a wall that way, so try the others.
                want = (want + (0, 90, -90, 180)[min(3, int((time.monotonic() - began) / 0.4))]
                        ) % 360
            # Eight ways, like a d-pad: most 2D games walk no other way. Keep the last
            # one until the aim is well past the halfway line, so it doesn't flicker.
            if stick is None or abs(wrap(want - stick)) > 30:
                new = (round(want / 45) * 45) % 360
                # It walks the new way now, at the same pace.
                t = math.radians(new - (stick or 0))
                drift = np.array([math.cos(t) * drift[0] - math.sin(t) * drift[1],
                                  math.sin(t) * drift[0] + math.cos(t) * drift[1]])
                stick = new
                moves = []  # moves the other way say nothing about this one
                last_rel = time.monotonic()  # nor is a turn being stuck
            self.steer(stick)
            await asyncio.sleep(0.01)
            rgb, g = await self._frame()
            steps += 1
            pad = int(size)
            sx, sy, _q = phase_shift(prev, g, (int(player[0]) - pad, int(player[1]) - pad,
                                               int(player[0]) + pad, int(player[1]) + pad))
            if scene.see(g, (sx, sy)) != "steady":
                # A door or stairs fading out (or a flash): let go, and see where it
                # settles. Scroll measured on dark frames is noise.
                self.steer(None)
                t_fade = time.monotonic()
                state, rgb, g = await settle(self.grab, scene, self.keep_going)
                if state == "new":
                    return done("new_area", f"the screen changed while walking to {target}; "
                                            "probably a door or a new area")
                if state == "stopped":
                    return done("stopped", "stopped by the player")
                gap = time.monotonic() - t_fade
                self.paused += gap
                best = (best[0], best[1] + gap)
                last_rel += gap
                began += gap
                hid = hid + gap if hid else 0.0
                prev, popped, stick = g, [], None
                continue
            scroll = np.array([sx, sy])
            for p in points:
                p += scroll
            seen = seen + scroll
            if detour is not None:
                detour.shift(scroll)
            moved = motion(prev, g, (sx, sy))
            # Walking changes a few percent of the screen; a text box, a menu or a battle
            # coming up changes far more, and stops the character, so the view stops too.
            # While it scrolls, layers moving at their own speed (clouds, a far-off
            # background) change much of the picture and mean nothing.
            popped = [*popped[-2:], 0.0 if np.hypot(sx, sy) > 1 else changed(prev, g, (sx, sy))]
            if sum(popped) > 0.15:
                return done("interrupted", "something came up on screen while walking (a "
                            "text box, a menu or a battle): look before pressing anything")
            if look is not None:
                # Looked for where it should be by now, and in front rather than behind:
                # a party member following a step behind looks just like it.
                guess = player + scroll + drift
                a = math.radians(stick)
                ahead = drift / np.hypot(*drift) if np.hypot(*drift) > 0.6 else (
                    math.sin(a), -math.cos(a))
                # Out of sight, it may be further off than guessed: look wider.
                r = int(2 * size) if hid else int(size)
                nx, ny, score = look.find(rgb, (guess[0], guess[1]), r, moved,
                                          (ahead[0], ahead[1]))
                velocity = np.array([nx, ny]) - player
                fast = float(np.hypot(*scroll))
                # Moving just like the scenery: that's the ground, not the character.
                if fast > 2 and np.hypot(*(velocity - scroll)) < 0.3 * fast and score >= 0.5:
                    look = None
                elif score < 0.5 and np.hypot(*drift) > 0.6 and (
                        not hid or time.monotonic() - hid < 2):
                    # Out of sight behind something (a tree, a roof) while walking: it
                    # keeps going the way it was, and shows up again a little later.
                    hid = hid or time.monotonic()
                    player = guess
                    self._now = (look, seen.copy(), (bw, bh))  # a sighting, not a guess
                    self.travel += drift
                    last_rel = time.monotonic()
                    prev = g
                    continue
                elif score < 0.5:
                    look = None
                    if hid:
                        # It didn't come out where it should have (stuck behind something
                        # out of sight): go back to where it was last seen.
                        self.travel += seen - player
                        player = seen.copy()
                hid = 0.0
            if look is None:
                unsure += 1
                x, y, how = spot(moved, (player[0], player[1]), int(2 * size), (bw, bh))
                prev = g
                if how > 6:
                    seen_moving = True
                    # A box that doesn't stand out from its surroundings is on the ground,
                    # not the character: none, and look again next frame.
                    look = Look(rgb, (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2))
                    if not look.usable:
                        look = None
                    velocity = np.array([x, y]) - player
                    player = np.array([x, y])
                    unsure = 0
                elif not seen_moving:
                    if time.monotonic() - began > 1.6:
                        if stuck_turns == 0:
                            return done("didnt_move", "your character didn't move any way. "
                                        "A text box or a cutscene is probably on: press a to "
                                        "move it along")
                        # Up against something after asking again: go by the model's box.
                        look = Look(rgb, (player[0] - bw / 2, player[1] - bh / 2,
                                          player[0] + bw / 2, player[1] + bh / 2))
                        if not look.usable:
                            look = None
                        seen_moving = True
                elif unsure > 20:
                    stuck_turns += 1
                    loc2 = None
                    if stuck_turns <= 2:
                        img2, loc2 = await self._again(target, "lost track of the character",
                                                       "I lost sight of you. Looking again.")
                        self.log(f"relocated (lost) {loc2}")
                    if loc2 is None or loc2.player is None or loc2.target is None:
                        return done("lost", "I lost track of your character. "
                                    + (loc2.note if loc2 else ""))
                    return await self._walk(target, img2, loc2, t0, timeout, done, stuck_turns)
                continue
            unsure = 0
            player = player + velocity
            seen = player.copy()
            self._now = (look, player.copy(), (bw, bh))
            if steps % 10 == 0:
                look.ground(rgb, player)  # the ground changes as it walks
            prev = g
            # How the character moved over the ground, over the last few frames: one
            # frame's is a few pixels of wobble from the walking animation.
            moves = [*moves[-3:], velocity - scroll]
            self.travel += velocity - scroll
            rel = np.mean(moves, axis=0) if len(moves) > 1 else np.zeros(2)
            if len(moves) > 1:
                drift = rel
            if float(np.hypot(*rel)) > 0.6:
                last_rel = time.monotonic()
                if slide:
                    slide = 0  # moving again: aim straight
            elif time.monotonic() - last_rel > (0.35 if detour is not None else 0.5):
                last_rel = time.monotonic()
                # Up against something: mark it, and plan a way around from here.
                way = self._reroute(detour, player, points[0], stick, size, budget)
                if way is not None:
                    detour, slide, best = way, 0, (float("inf"), time.monotonic())
                    continue
                slide += 1 if detour is None else 5  # no way around found: ask
                if slide > 4:
                    slide = 0
                    stuck_turns += 1
                    if stuck_turns > 1:
                        return done("blocked", "something is in the way, maybe out of "
                                    "reach from here, "
                                    f"{self._words(points[-1] - player, size)} from {target}")
                    img2, loc2 = await self._again(target, "the straight way is blocked; give "
                                                   "a way around", WAY_AROUND)
                    self.log(f"relocated (blocked) {loc2}")
                    if loc2.player is not None and loc2.target is not None:
                        return await self._walk(target, img2, loc2, t0, timeout, done,
                                                stuck_turns)
                    if self._where is not None:
                        return done("lost", loc2.note)
                    best, last_rel, prev, stick = (best[0], time.monotonic()), \
                        time.monotonic(), small(img2)[1], None
                    detour = None

    def _reroute(self, detour: Detour | None, player, aim, stick: float | None,
                 size: float, budget: float, bumped: bool = True) -> Detour | None:
        """Stuck (bumped, or not getting closer): plan the way around from here, or
        None when there is none, or after enough bumps or seconds that the model
        should look instead."""
        now = time.monotonic()
        if self._planning is None:
            self._planning = (now, self.paused)
        began, held = self._planning
        if self._bumps >= BUMPS or now - began - (self.paused - held) > budget:
            return None
        detour = detour or Detour(size, player)
        self._bumps += 1
        ahead = detour.bump(player, stick) if bumped else None
        found = detour.plan(player, aim)
        self.log(f"bump {self._bumps} going {stick}: blocked {ahead}, "
                 f"{len(detour.grid.blocked)} known; "
                 + (f"around by {[c for c, _ in detour.corners]}" if found else "no way"))
        if not found:
            return None
        if self._bumps == 3:  # not for a step aside, but a walk that takes a while
            self._tell(WAY_AROUND)
        return detour

    def _tell(self, words: str) -> Awaitable[None] | None:
        """Say why it's taking longer, unless that was just said."""
        if self.say is None or time.monotonic() - self._said_at.get(words, -math.inf) < 60:
            return None
        self._said_at[words] = time.monotonic()
        self._talk = asyncio.ensure_future(self.say(words))
        # Spoken while walking on: a voice that fails mustn't end the walk.
        self._talk.add_done_callback(lambda t: t.cancelled() or t.exception())
        return self._talk

    async def _again(self, target: str, why: str, words: str
                     ) -> tuple[Image.Image, Located]:
        """Let go and ask the model again, about the screen as it is once stopped. The
        first time in a walk, also tell the player why it's taking longer."""
        self.steer(None)
        await asyncio.sleep(0.15)
        img = await self.grab()
        if self._where is not None and (self._now or self.memory) is not None:
            # A point from the map: it's where the map says; the model can't see it.
            return img, self._here(self._where)
        jobs = [self._locate(img, target, why)]
        talk = self._tell(words)
        if talk is not None:
            jobs.append(talk)
        return img, (await asyncio.gather(*jobs))[0]

    async def _locate(self, img: Image.Image, target: str, context: str = "") -> Located:
        """The model's look at the screen, with the target's box made exact by SAM 3.1
        when it finds the same thing. Both run at once, and SAM gets little longer than
        the model took, so it adds no wait; without a match the model's box stands."""
        exact = asyncio.create_task(self._exact(img, target)) if self.exact else None
        try:
            loc = await locate(self.muse, img, target, context, self.me)
            if exact is None or loc.target is None:
                return loc
            try:
                boxes = await asyncio.wait_for(asyncio.shield(exact), 0.6)
            except TimeoutError:
                self.log("SAM 3.1 took too long; going by the model's box")
                return loc
            sx = WIDTH
            sy = img.height * WIDTH / img.width
            better = snap(loc.target, [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
                                       for b in boxes])
            if better is not None:
                self.log(f"SAM 3.1 placed {target} at {[round(v) for v in better]} "
                         f"(the model said {[round(v) for v in loc.target]})")
                loc.target = better
            return loc
        finally:
            if exact is not None and not exact.done():
                exact.cancel()

    async def _exact(self, img: Image.Image, target: str
                     ) -> list[tuple[float, float, float, float]]:
        assert self.exact is not None
        phrase = re.sub(r"^(the|a|an|my|that|this|some)\s+", "", target.strip(),
                        flags=re.IGNORECASE)
        phrases = [phrase] + (["person"] if PEOPLE.search(target) else [])
        try:
            found = await asyncio.gather(*(self.exact(img, p) for p in phrases))
        except Exception as exc:  # noqa: BLE001 - an extra, never a reason a walk fails
            self.log(f"SAM 3.1 failed ({type(exc).__name__}); going by the model's box")
            return []
        return [b for boxes in found for b in boxes]

    @staticmethod
    def _words(v: np.ndarray, size: float) -> str:
        """Roughly how far and which way, in character widths."""
        steps = max(1, round(float(np.hypot(*v)) / max(size, 1)))
        d = compass(*v)
        side = ["up", "up and right", "right", "down and right", "down", "down and left",
                "left", "up and left"][int(((d + 22.5) % 360) // 45)]
        return f"about {steps} steps {side}"
