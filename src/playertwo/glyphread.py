"""Reading a game's own pixel font and cursor without a model: fast, local, game-agnostic.

A game profile (see ctsteam.py) gives the canvas size, a learned glyph library and cursor
templates; this module does the work:

- `text_mask` finds text pixels: bright cores inside the dark outline pixel fonts draw;
- `text_lines` and `glyphs_in_line` cut a mask into lines and glyphs;
- `GlyphLib` names each glyph by its nearest neighbour in the library;
- `runs_in` reads a box into runs of words (list columns are wide gaps apart);
- `find_template` finds a cursor (a pointing hand) by masked template matching.

Coordinates are on the profile's canvas (every frame is scaled to it first), so a window
of any size reads the same.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def to_canvas(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A BGR frame scaled to the canvas size (w, h)."""
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def bgr(image) -> np.ndarray:
    """A PIL image or an array as a BGR array."""
    if isinstance(image, np.ndarray):
        return image
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def text_mask(crop: np.ndarray, thresh: float | None = None, local: bool = False) -> np.ndarray:
    """Text pixels: bright, in small components, each wrapped in the font's dark outline."""
    b, g, r = cv2.split(crop)
    v = cv2.max(cv2.max(b, g), r).astype(np.float32)
    if thresh is None:
        top = float(np.percentile(v, 99.7))
        thresh = max(70.0, 0.40 * top)  # grey text ~140; a pause dims everything
    m = v > thresh
    if local:
        # Against the local background (bright metal menus, dark battle windows).
        small = cv2.resize(v.astype(np.uint8), None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.medianBlur(small, 11), (v.shape[1], v.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        m &= v > bg.astype(np.float32) + 55
    out = np.zeros(m.shape, bool)
    _keep_components(v, m.astype(np.uint8), out, split=True)
    return out


def _keep_components(v, m, out, split):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return
    w, h, a = stats[:, 2], stats[:, 3], stats[:, 4]
    cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float32)
    core = np.bincount(lab.ravel(), weights=v.ravel(), minlength=n) / np.maximum(cnt, 1)
    # The ring: pixels within 2 px of a component, given to it by a max filter.
    labd = cv2.dilate(lab.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)
    ring = (lab == 0) & (labd > 0)
    rl = labd[ring]
    dark = v[ring] < 0.6 * core[rl]
    rc = np.bincount(rl, minlength=n).astype(np.float32)
    rd = np.bincount(rl, weights=dark.astype(np.float32), minlength=n) / np.maximum(rc, 1)
    ok = (w <= 60) & (h <= 48) & (a >= 8) & (rd >= 0.45)
    wide = (w >= 44) & (h >= 24) & (w <= 60) & (h <= 48)
    ok &= ~wide
    ok[0] = False
    out |= ok[lab]
    if split:
        for i in np.where(wide & (core <= 232))[0]:
            # Two coloured glyphs fused through a bright outline (yellow digits): cut higher.
            x, y, ww, hh = stats[i, :4]
            x0, y0 = max(0, x - 4), max(0, y - 4)
            sub = lab[y0:y + hh + 4, x0:x + ww + 4] == i
            vs = v[y0:y + hh + 4, x0:x + ww + 4]
            m2 = ((vs > 0.8 * core[i]) & sub).astype(np.uint8)
            o2 = np.zeros(m2.shape, bool)
            _keep_components(vs, m2, o2, split=False)
            out[y0:y + hh + 4, x0:x + ww + 4] |= o2


@dataclass
class Glyph:
    x0: int
    x1: int
    y0: int
    y1: int
    bmp: np.ndarray  # GH x GW, the glyph over the whole line height


def text_lines(m: np.ndarray, min_h: int = 18) -> list[tuple[int, int]]:
    prof = m.sum(axis=1)

    def bands(lo, hi, floor):
        out, start = [], None
        for i in range(lo, hi + 1):
            on = i < hi and prof[i] > floor
            if on and start is None:
                start = i
            elif not on and start is not None:
                out.append((start, i))
                start = None
        return out

    out = []
    for a, b in bands(0, len(prof), 2):
        if b - a > 48:
            # Clutter bridged two rows of text: cut at the sparse rows.
            peak = prof[a:b].max()
            out += list(bands(a, b, max(2, 0.12 * peak)))
        else:
            out.append((a, b))
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a - merged[-1][1] < 8 and b - merged[-1][0] <= 48:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_h]


GW, GH = 10, 14  # the normalised glyph bitmap
CAP, DESC = 33, 9  # canvas rows above and below the baseline a glyph bitmap covers
FONT_PX = 5.33  # canvas pixels per font pixel


def glyphs_in_line(m: np.ndarray, y0: int, y1: int, min_w: int = 3) -> list[Glyph]:
    """Glyphs as column runs, cut by components where a run is wider than any glyph (a
    leaning slash touches its neighbours' columns)."""
    band = np.ascontiguousarray(m[y0:y1]).astype(np.uint8)
    _n, lab, st, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    cols = band.any(axis=0)
    spans, start = [], None
    for i, c in enumerate([*list(cols), False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            spans.append((start, i))
            start = None
    groups: list[list] = []
    for a, b in spans:
        ids = [int(i) for i in np.unique(lab[:, a:b]) if i and st[i, 4] >= 4]
        if not ids:
            continue
        if b - a <= 38:
            groups.append([a, b, ids])
            continue
        comps = sorted(([int(st[i, 0]), int(st[i, 0] + st[i, 2]), [i]] for i in ids),
                       key=lambda c: c[0])
        sub: list[list] = []
        for c in comps:
            if sub:
                g = sub[-1]
                ov = min(g[1], c[1]) - max(g[0], c[0])
                if ov >= 0.5 * min(g[1] - g[0], c[1] - c[0]):
                    g[0], g[1] = min(g[0], c[0]), max(g[1], c[1])
                    g[2] += c[2]
                    continue
            sub.append(c)
        groups += sub
    groups = [g for g in groups if g[1] - g[0] >= min_w]
    if not groups:
        return []
    masks, bottoms = [], []
    for a, b, ids in groups:
        gm = np.isin(lab[:, a:b], ids)
        ys = np.where(gm.any(axis=1))[0]
        masks.append((gm, ys))
        bottoms.append(ys[-1] + 1)
    base = int(np.median(bottoms))  # most glyphs sit on the baseline
    out = []
    pad_m = 100
    for (a, b, _), (gm, ys) in zip(groups, masks, strict=True):
        pad = np.zeros((2 * pad_m + gm.shape[0], b - a), bool)
        pad[pad_m:pad_m + gm.shape[0]] = gm
        top = pad_m + base - CAP
        sub = pad[top:top + CAP + DESC]
        bmp = cv2.resize(sub.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
        out.append(Glyph(a, b, y0 + int(ys[0]), y0 + int(ys[-1]) + 1, bmp))
    return out


def feature(g: Glyph) -> np.ndarray:
    return np.concatenate([g.bmp.ravel(), [(g.x1 - g.x0) / FONT_PX * 0.5]]).astype(np.float32)


class GlyphLib:
    """Labelled glyph features; a glyph is named by its nearest one (L1)."""

    def __init__(self, X: np.ndarray | None = None, y: list[str] | None = None):
        self.X = np.zeros((0, GW * GH + 1), np.float32) if X is None else X.astype(np.float32)
        self.y = list(y or [])

    @classmethod
    def load(cls, path: Path) -> GlyphLib:
        d = np.load(path)
        return cls(d["X"], [str(c) for c in d["y"]])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, X=self.X.astype(np.float16), y=np.array(self.y))

    def add(self, g: Glyph, ch: str) -> None:
        self.X = np.vstack([self.X, feature(g)[None]])
        self.y.append(ch)

    def read(self, gl: list[Glyph], space_px: int = 14) -> tuple[str, float]:
        """The text of a run of glyphs, and the worst glyph's distance."""
        if not gl or not len(self.y):
            return "", 0.0
        fm = np.array([feature(g) for g in gl], np.float32)
        dist, idx = cv2.batchDistance(fm, np.ascontiguousarray(self.X, np.float32), cv2.CV_32F,
                                      normType=cv2.NORM_L1, K=1)  # nearest by L1, in C
        s, worst = "", 0.0
        for k, g in enumerate(gl):
            if k and g.x0 - gl[k - 1].x1 >= space_px:
                s += " "
            s += self.y[int(idx[k, 0])]
            worst = max(worst, float(dist[k, 0]))
        # A "u" whose thin bottom didn't make the mask comes apart as "i]".
        return re.sub(r"(?<=[A-Za-z])i\](?=[a-z])", "u", s), worst


def colour(crop: np.ndarray, m: np.ndarray) -> str:
    """The text colour of a run: yellow, green, cyan, grey or white."""
    if not m.any():
        return ""
    b, g, r = (float(crop[..., i][m].mean()) for i in range(3))
    top = max(b, g, r)
    if r > 0.8 * top and g > 0.8 * top and b < 0.55 * top:
        return "yellow"
    if g >= top * 0.99 and r < 0.75 * top and b < 0.75 * top:
        return "green"
    if b >= 0.9 * top and g > 0.85 * top and r < 0.7 * top:
        return "cyan"
    return "grey" if top < 190 else "white"


@dataclass
class Run:
    """Words on one line, with no wide gap inside."""
    text: str
    x0: int
    x1: int
    y: int
    colour: str
    worst: float


def runs_in(lib: GlyphLib, img: np.ndarray, box: tuple[int, int, int, int],
            blank: list[tuple[int, int, int, int]] = (), local: bool = False,
            gap: int = 60) -> list[list[Run]]:
    """The text in a box, line by line, each line split into runs at gaps wider than
    `gap`. `blank` boxes (x, y, w, h), such as cursors, are left out."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    m = text_mask(crop, local=local)
    for hx, hy, hw, hh in blank:
        m[max(0, hy - y0 - 4):max(0, hy - y0 + hh + 4),
          max(0, hx - x0 - 4):max(0, hx - x0 + hw + 4)] = False
    lines = []
    for a, b in text_lines(m):
        gl = glyphs_in_line(m, a, b)
        if not gl:
            continue
        groups, cur = [], [gl[0]]
        for g in gl[1:]:
            if g.x0 - cur[-1].x1 > gap:
                groups.append(cur)
                cur = [g]
            else:
                cur.append(g)
        groups.append(cur)
        row = []
        for r in groups:
            if len(r) <= 1 and (r[0].x1 - r[0].x0) < 12:
                continue  # a speck
            t, worst = lib.read(r)
            sub = crop[a:b, r[0].x0:r[-1].x1]
            row.append(Run(t, x0 + r[0].x0, x0 + r[-1].x1, y0 + (a + b) // 2,
                           colour(sub, m[a:b, r[0].x0:r[-1].x1]), round(worst, 1)))
        if row:
            lines.append(row)
    return lines


def snap(s: str, vocab, cutoff: float = 0.75) -> str:
    """The vocabulary word closest to what was read, or the reading itself."""
    s2 = s.lstrip("*").strip()
    m = difflib.get_close_matches(s2, list(vocab), n=1, cutoff=cutoff)
    return m[0] if m else s2


@dataclass
class Template:
    """A cursor picture and its mask, cut at `quarter` scale (canvas / 4)."""
    image: np.ndarray
    mask: np.ndarray
    name: str


def find_template(img: np.ndarray, templates: list[Template], canvas: tuple[int, int],
                  thresh: float = 0.68, roi: tuple[int, int, int, int] | None = None,
                  most: int = 6) -> list[tuple[int, int, int, int, float, str]]:
    """Where templates match, best first: [(x, y, w, h, score, name)] on the canvas.
    Matching runs at a quarter of the canvas on the brightest channel."""
    cw, ch = canvas
    q = cv2.resize(img, (cw // 4, ch // 4), interpolation=cv2.INTER_AREA).max(axis=2)
    ox = oy = 0
    if roi:
        x0, y0, x1, y1 = (int(c / 4) for c in roi)
        q = q[y0:y1, x0:x1]
        ox, oy = x0, y0
    out = []
    for t in templates:
        if q.shape[0] < t.image.shape[0] or q.shape[1] < t.image.shape[1]:
            continue
        r = cv2.matchTemplate(q, t.image, cv2.TM_CCOEFF_NORMED, mask=t.mask)
        r = np.nan_to_num(r, nan=0, posinf=0, neginf=0)
        for _ in range(most):
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx < thresh:
                break
            x, y = loc
            out.append(((x + ox) * 4, (y + oy) * 4, t.image.shape[1] * 4,
                        t.image.shape[0] * 4, round(float(mx), 3), t.name))
            r[max(0, y - 6):y + 6, max(0, x - 8):x + 8] = 0
    out.sort(key=lambda h: -h[4])
    keep: list = []
    for h in out:
        if all(abs(h[0] - k[0]) > 40 or abs(h[1] - k[1]) > 30 for k in keep):
            keep.append(h)
    return keep


def scaled(image: np.ndarray, mask: np.ndarray, scale: float, name: str) -> Template:
    """A template resized by `scale` (the same cursor drawn bigger)."""
    t = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = (cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) > 0)
    return Template(t, m.astype(np.uint8) * 255, name)
