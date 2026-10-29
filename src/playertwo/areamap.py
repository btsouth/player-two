"""A picture of the whole area, stitched from the frames grabbed while walking.

The model only ever sees one screen, so an exit, a gap in a fence or someone walked past
are gone once they scroll off. AreaMap lays every frame where the camera was (from the
scroll walk.phase_shift measures, checked against what is already on the map) and keeps,
for each spot, the colour seen there most often. Sprites walking about, the character the
camera follows and water rippling only pass over a spot for a few frames, so the ground
under them wins and they don't smear; what moves differently from the scroll is also left
out as it is seen. image() draws it for a vision model: what was seen, dark grey where
nothing was yet, the screen now outlined, the way the character came and named places.

Map coordinates are pixels at tracking size (walk.WIDTH wide frames), x right, y down,
with (0, 0) the top left of the first frame of the area.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .walk import motion, new_place, phase_shift

UNSEEN = (64, 64, 64)  # never seen yet: a flat grey no game draws over a whole region
CAP = 8          # how many sightings a colour can bank: a sprite that stood still this long
                 # takes as many frames of ground to be replaced once it leaves
STILL_CAP = 3    # the same, from frames where the view stood still: a party standing about
                 # for seconds is seen the same each frame, which proves nothing more
TAKE = 3         # frames running a new colour must be seen, still, to replace the old
AGREE = 24       # how far apart (gray levels, plus half the colour difference) still
                 # counts as the same colour: half-pixel steps change it a little
MOVED = 24       # gray levels a pixel changes by, beyond the scroll, to count as moving
SEARCH = 2       # pixels each way the placement is checked against the map
STEP = 3         # every STEP-th pixel each way is compared when checking
GROW = 256       # the canvas grows by at least this much at a time


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * np.float32(0.299) + rgb[..., 1] * np.float32(0.587) \
        + rgb[..., 2] * np.float32(0.114)


def _soft(g: np.ndarray) -> np.ndarray:
    """A 3 x 3 box blur."""
    p = np.pad(g, 1, mode="edge")
    rows = p[:-2] + p[1:-1] + p[2:]
    return (rows[:, :-2] + rows[:, 1:-1] + rows[:, 2:]) * np.float32(1 / 9)


class AreaMap:
    """The area seen so far. Feed it every frame with add() (or track(), which also
    measures the scroll and notices a new place), and reset() it on a new area."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the area: the next frame starts a new map."""
        self._rgb = np.zeros((0, 0, 3), np.uint8)   # the colour each spot is
        self._n = np.zeros((0, 0), np.uint8)        # how sure (0: never seen)
        self._g = np.zeros((0, 0), np.float32)      # _rgb in gray, for matching
        self._u = np.zeros((0, 0), np.float32)      # and its colour: red - green
        self._v = np.zeros((0, 0), np.float32)      # blue - green
        self._cg = np.zeros((0, 0), np.float32)     # a different colour seen there lately
        self._cs = np.zeros((0, 0), np.uint8)       # how many frames running it was
        self._origin = np.zeros(2, int)             # map coordinates of _rgb[0, 0]
        self.camera = np.zeros(2)                   # map coordinates of the screen's top left
        self.view = (0, 0)                          # screen size, w x h
        self.path: list[tuple[float, float]] = []   # where the character went, map coordinates
        self.frames = 0
        self.corrections: list[tuple[float, float]] = []  # what checking the map moved, per frame
        self._prev: np.ndarray | None = None        # last frame in gray
        self._soft: np.ndarray | None = None        # the same, blurred a little
        self._fit: tuple[float, float, float] | None = None  # last image(): x0, y0, scale

    # Placing frames

    def track(self, rgb: np.ndarray, gray: np.ndarray | None = None,
              scroll: tuple[float, float] | None = None,
              mask: tuple[int, int, int, int] | None = None) -> bool:
        """add() for a caller that has only the frames: measures the scroll from the
        last frame (phase_shift, leaving out `mask`, the character) unless given, and
        starts a new map when the picture changes to a new place (walk.new_place).
        True when it started a new map."""
        g = _gray(rgb.astype(np.float32)) if gray is None else gray
        fresh = False
        if self._prev is not None and new_place(self._prev, g):
            self.reset()
            fresh = True
        if scroll is None:
            scroll = (0.0, 0.0) if self._prev is None else phase_shift(self._prev, g, mask)[:2]
        self.add(rgb, scroll, g)
        return fresh

    def add(self, rgb: np.ndarray, scroll: tuple[float, float],
            gray: np.ndarray | None = None) -> np.ndarray:
        """Lay one frame (H x W x 3 uint8 RGB, at tracking size) on the map. `scroll` is
        how far the picture moved since the last frame (walk.phase_shift: x right, y down),
        so the camera moved the other way. Returns where the screen's top left is now."""
        h, w = rgb.shape[:2]
        g = _gray(rgb.astype(np.float32)) if gray is None else np.asarray(gray, np.float32)
        if self.frames == 0:
            self.view = (w, h)
            self.camera = np.zeros(2)
        else:
            guess = self.camera - np.asarray(scroll, float)
            self.camera = self._check(g, guess)
            self.corrections.append(tuple(self.camera - guess))
        soft = _soft(g)
        moving = self._moving(soft, scroll) if self._soft is not None and \
            self._soft.shape == g.shape else None
        parked = self.frames > 0 and float(np.hypot(*scroll)) < 0.5
        self._vote(rgb, g, moving, STILL_CAP if parked else CAP)
        self._prev, self._soft = g, soft
        self.frames += 1
        return self.camera.copy()

    def _check(self, g: np.ndarray, guess: np.ndarray) -> np.ndarray:
        """Where the frame fits the map best within SEARCH pixels of `guess`: scroll
        errors would otherwise add up as the character walks."""
        h, w = g.shape
        r = SEARCH
        ix, iy = round(guess[0]), round(guess[1])
        self._fit_canvas(ix - r, iy - r, ix + w + r, iy + h + r)
        x0, y0 = ix - r - self._origin[0], iy - r - self._origin[1]
        mg = self._g[y0:y0 + h + 2 * r, x0:x0 + w + 2 * r]
        sure = (self._n[y0:y0 + h + 2 * r, x0:x0 + w + 2 * r] >= 2).astype(np.float32)
        # The frame, sampled sparsely, away from its edges so every shift stays on it.
        f = g[r:h - r:STEP, r:w - r:STEP]
        fh, fw = f.shape
        scores = np.full((2 * r + 1, 2 * r + 1), np.inf)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                ys, xs = slice(dy + r, dy + r + fh * STEP, STEP), slice(dx + r, dx + r + fw * STEP, STEP)
                wt = sure[ys, xs]
                total = float(wt.sum())
                if total < 0.25 * f.size:
                    continue
                scores[dy, dx] = float((np.abs(mg[ys, xs] - f) * wt).sum()) / total
        if not np.isfinite(scores).any():
            return guess
        by, bx = np.unravel_index(int(np.argmin(scores)), scores.shape)

        def sub(m1: float, m0: float, p1: float) -> float:
            if not (np.isfinite(m1) and np.isfinite(p1)):
                return 0.0
            d = m1 - 2 * m0 + p1
            return 0.0 if d <= 0 else 0.5 * (m1 - p1) / d

        fy = sub(scores[by - 1, bx], scores[by, bx], scores[by + 1, bx]) if 0 < by < 2 * r else 0
        fx = sub(scores[by, bx - 1], scores[by, bx], scores[by, bx + 1]) if 0 < bx < 2 * r else 0
        # Flat ground fits everywhere alike: then the scroll is the better guess.
        here = scores[r, r]
        if not np.isfinite(here) or here - scores[by, bx] < 0.5:
            return guess
        return np.array([ix + bx - r + fx, iy + by - r + fy], float)

    def _moving(self, soft: np.ndarray, scroll) -> np.ndarray:
        """Pixels that changed other than by the scroll (walk.motion), a pixel around them
        too. Compared slightly blurred, so fine detail a half-pixel step blurs isn't
        taken for something moving."""
        moved = motion(self._soft, soft, scroll) > MOVED
        grown = moved.copy()
        grown[1:] |= moved[:-1]
        grown[:-1] |= moved[1:]
        wide = grown.copy()
        wide[:, 1:] |= grown[:, :-1]
        wide[:, :-1] |= grown[:, 1:]
        return wide

    def _fit_canvas(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Grow the canvas to hold map coordinates [x0, x1) x [y0, y1)."""
        H, W = self._n.shape
        ox, oy = self._origin
        if W and x0 >= ox and y0 >= oy and x1 <= ox + W and y1 <= oy + H:
            return
        if not W:
            nx0, ny0, nx1, ny1 = x0 - GROW, y0 - GROW, x1 + GROW, y1 + GROW
        else:
            nx0 = ox if x0 >= ox else min(x0, ox - GROW)
            ny0 = oy if y0 >= oy else min(y0, oy - GROW)
            nx1 = ox + W if x1 <= ox + W else max(x1, ox + W + GROW)
            ny1 = oy + H if y1 <= oy + H else max(y1, oy + H + GROW)
        rgb = np.zeros((ny1 - ny0, nx1 - nx0, 3), np.uint8)
        n = np.zeros(rgb.shape[:2], np.uint8)
        g = np.zeros(rgb.shape[:2], np.float32)
        u = np.zeros(rgb.shape[:2], np.float32)
        v = np.zeros(rgb.shape[:2], np.float32)
        cg = np.zeros(rgb.shape[:2], np.float32)
        cs = np.zeros(rgb.shape[:2], np.uint8)
        if W:
            sy, sx = oy - ny0, ox - nx0
            rgb[sy:sy + H, sx:sx + W] = self._rgb
            n[sy:sy + H, sx:sx + W] = self._n
            g[sy:sy + H, sx:sx + W] = self._g
            u[sy:sy + H, sx:sx + W] = self._u
            v[sy:sy + H, sx:sx + W] = self._v
            cg[sy:sy + H, sx:sx + W] = self._cg
            cs[sy:sy + H, sx:sx + W] = self._cs
        self._rgb, self._n, self._g, self._u, self._v = rgb, n, g, u, v
        self._cg, self._cs = cg, cs
        self._origin = np.array([nx0, ny0])

    def _vote(self, rgb: np.ndarray, g: np.ndarray, moving: np.ndarray | None,
              cap: int) -> None:
        """Each spot keeps the colour it was seen as most (a running majority vote):
        agreeing adds a sighting (up to `cap`), disagreeing takes one away, and once
        they are spent a new colour seen steadily takes the spot."""
        h, w = g.shape
        ix, iy = round(self.camera[0]), round(self.camera[1])
        self._fit_canvas(ix, iy, ix + w, iy + h)
        y0, x0 = iy - self._origin[1], ix - self._origin[0]
        m = self._rgb[y0:y0 + h, x0:x0 + w]
        n = self._n[y0:y0 + h, x0:x0 + w]
        mg = self._g[y0:y0 + h, x0:x0 + w]
        mu = self._u[y0:y0 + h, x0:x0 + w]
        mv = self._v[y0:y0 + h, x0:x0 + w]
        cg = self._cg[y0:y0 + h, x0:x0 + w]
        cs = self._cs[y0:y0 + h, x0:x0 + w]
        f = rgb.astype(np.float32)
        u = f[..., 0] - f[..., 1]
        v = f[..., 2] - f[..., 1]
        # Gray alone would take a red shirt for brown ground of the same brightness.
        same = np.abs(mg - g) + 0.5 * (np.abs(mu - u) + np.abs(mv - v)) < AGREE
        empty = n == 0
        still = ~empty if moving is None else ~empty & ~moving
        differ = ~same & ~empty
        # One more where it agrees (up to cap, but never less than it was), one less
        # (down to one) where it doesn't.
        more = np.maximum(np.minimum(n + np.uint8(1), np.uint8(cap)), n)
        nn = np.where(same, np.where(still, more, n),
                      np.where(differ, np.maximum(n, np.uint8(2)) - np.uint8(1), n))
        # A colour that disagrees replaces it only once its sightings are spent and the
        # new one was seen the same TAKE frames running without moving: a sprite moving
        # with the camera (a follower) passes over a spot in fewer, even where the
        # pixel-by-pixel motion check misses it (plain colour moving over itself).
        fresh = differ & still
        like = np.abs(cg - g) < AGREE / 2
        np.copyto(cg, g, where=fresh & ~like)
        cs[...] = np.where(fresh, np.where(like, np.minimum(cs, np.uint8(254)) + np.uint8(1),
                                           np.uint8(1)), np.uint8(0))
        take = empty | (fresh & (nn == 1) & (cs >= TAKE))
        nn[take] = 1
        cs[take] = 0
        n[...] = nn
        m[take] = rgb[take]
        mg[take] = g[take]
        mu[take] = u[take]
        mv[take] = v[take]

    # Positions

    def to_map(self, screen_xy) -> np.ndarray:
        """A point on the screen now, in map coordinates."""
        return self.camera + np.asarray(screen_xy, float)

    def to_screen(self, map_xy) -> np.ndarray:
        """A point on the map, on the screen now (may be off it)."""
        return np.asarray(map_xy, float) - self.camera

    def on_screen(self, map_xy) -> bool:
        x, y = self.to_screen(map_xy)
        return 0 <= x < self.view[0] and 0 <= y < self.view[1]

    def character_at(self, screen_xy) -> np.ndarray:
        """Note where the character is now (on the screen), for the path. Returns the
        spot in map coordinates."""
        p = self.to_map(screen_xy)
        if not self.path or np.hypot(*(p - self.path[-1])) >= 1:
            self.path.append((float(p[0]), float(p[1])))
        return p

    def bounds(self) -> tuple[int, int, int, int]:
        """The part of the map seen, x0, y0, x1, y1 in map coordinates."""
        seen = self._n > 0
        if not seen.any():
            return 0, 0, 0, 0
        rows, cols = np.flatnonzero(seen.any(axis=1)), np.flatnonzero(seen.any(axis=0))
        ox, oy = self._origin
        return (int(cols[0] + ox), int(rows[0] + oy), int(cols[-1] + 1 + ox),
                int(rows[-1] + 1 + oy))

    def seen(self, map_xy) -> bool:
        """Whether that spot has been seen."""
        x, y = (round(v) for v in np.asarray(map_xy, float) - self._origin)
        H, W = self._n.shape
        return 0 <= x < W and 0 <= y < H and bool(self._n[y, x])

    # Pictures

    def picture(self) -> np.ndarray:
        """The seen part of the map at full size, RGB, unseen spots UNSEEN."""
        x0, y0, x1, y1 = self.bounds()
        ox, oy = self._origin
        crop = self._rgb[y0 - oy:y1 - oy, x0 - ox:x1 - ox].copy()
        crop[self._n[y0 - oy:y1 - oy, x0 - ox:x1 - ox] == 0] = UNSEEN
        return crop

    def image(self, path: bool = True, marks: dict[str, tuple[float, float]] | None = None,
              size: int = 1024, view: bool = True) -> Image.Image:
        """The map to show a vision model: scaled so the long side is about `size`, the
        screen now outlined in yellow, the character's path in cyan (a dot where it is
        now) and `marks` (label: map x, y) as labelled red dots. Unseen is flat dark
        grey. from_image() turns a point on it back into map coordinates."""
        x0, y0, x1, _ = self.bounds()
        if x1 <= x0:
            return Image.new("RGB", (size, size * 9 // 16), UNSEEN)
        pic = Image.fromarray(self.picture())
        s = size / max(pic.width, pic.height)
        out = pic.resize((max(1, round(pic.width * s)), max(1, round(pic.height * s))),
                         Image.NEAREST if s >= 1 else Image.BILINEAR)
        self._fit = (x0, y0, s)
        d = ImageDraw.Draw(out)

        def at(p) -> tuple[float, float]:
            return (p[0] - x0) * s, (p[1] - y0) * s

        lw = max(2, round(size / 400))
        if view and self.frames:
            a = at(self.camera)
            b = at(self.camera + np.array(self.view))
            d.rectangle((*a, *b), outline=(255, 230, 0), width=lw)
        if path and self.path:
            pts = [at(p) for p in self.path]
            if len(pts) > 1:
                d.line(pts, fill=(0, 0, 0), width=lw + 2)
                d.line(pts, fill=(0, 230, 255), width=lw)
            r = 2 * lw + 2
            sx, sy = pts[0]
            d.rectangle((sx - r, sy - r, sx + r, sy + r), outline=(0, 0, 0), fill=(0, 230, 255))
            ex, ey = pts[-1]
            d.ellipse((ex - r - 1, ey - r - 1, ex + r + 1, ey + r + 1), outline=(0, 0, 0),
                      fill=(0, 230, 255), width=2)
        if marks:
            font = ImageFont.load_default(size=max(12, round(size / 60)))
            r = 2 * lw + 2
            for label, p in marks.items():
                mx, my = at(p)
                d.ellipse((mx - r, my - r, mx + r, my + r), fill=(255, 40, 40),
                          outline=(0, 0, 0), width=2)
                tx, ty = mx + r + 3, my - r
                box = d.textbbox((tx, ty), label, font=font)
                d.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 0, 0))
                d.text((tx, ty), label, fill=(255, 255, 255), font=font)
        return out

    def from_image(self, xy) -> np.ndarray:
        """A point on the last image() (say, one a model pointed at), in map coordinates."""
        if self._fit is None:
            raise ValueError("no image drawn yet")
        x0, y0, s = self._fit
        return np.array([x0 + xy[0] / s, y0 + xy[1] / s])
