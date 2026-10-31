"""Inside places (towns, houses, castles, forests): where the player is and the way to a
named way out, from pictures of each place surveyed beforehand.

A vision model walking a room it can only see a screen of can't find a way out that's
off screen, and loses track of which door it came in by. A sighted friend who knows the
game knows the whole place. So does this: scripts/survey_rooms.py walks every place in the
emulator (the game's memory says where the character and the view are), notes which
tiles can be walked and where each way out goes, and stitches the whole place into one
picture. Here the screen is found on those pictures by plain image matching (the screen
is a window onto the place), the character on it by its look, the way to the chosen way
out planned on the walkable tiles (through other places when it's further), and walking
checks where it is several times a second. Pictures only: it works on any copy of the
game, the emulator's memory is only for making the survey. The Steam port's redrawn
screens are found on the same pictures (see STEAM_VIEW).

<atlas folder>/rooms/:
    loc-002.json, loc-002.png   one place: its tiles, ways out and picture (survey_rooms.py)
    names.json                  {"names": {"2": "Crono's house, upstairs"}, "worlds": [496]}
    player-down.png ...         the player's character facing each way, and player.json:
                                {"anchor": {"down": [x, y]}} where its feet are on each
"""

from __future__ import annotations

import asyncio
import heapq
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .atlas import ATLASES, compass, slug

TILE = 16  # field maps are made of 16-pixel tiles
NATIVE = (256, 224)  # the game's screen, in its own pixels
# Where the view keeps the character (its feet), from the view's top left, while the
# view follows it; at a place's edges the view stops and the character walks on.
FOLLOW = (128, 138)
EDGE = 8  # a margin off the screen's edges, for matching
# The Steam port of Chrono Trigger draws the same places redrawn and smoothed, filling a
# wide window: 304x191 of the game's pixels in any 16:9 window (as on the world map, see
# view-scales.json), a place narrower than that in the middle of a dark red fill. Its
# view is matched in colour on the pictures padded with black (a view can hang off a
# small place), accepted when it stands out from anywhere else.
STEAM_VIEW = (304, 191)
STEAM_FILL = (24, 8, 8)  # RGB of the fill round a small place
STEAM_PAD = 80
STEAM_CUT = (12, 8, 40, 14)  # off the view's left, top, right and bottom (Steam's toasts)
STEAM_SCORE, STEAM_MARGIN = 0.7, 0.1
SEEN_LEAST = 0.6  # of a Steam view on what the survey saw, to count a match there
STEAM_NEAR = 48  # how far from the last view the quick search looks, in game pixels
# Where Steam's view keeps the character's feet while it follows (the game's own view
# shifted 24 left for the wider picture, 18 down for the shorter one). At a place's
# edges it stops: across at the place's own sides, or centred on a place narrower than
# it; down from the game's own top to STEAM_LOW below the game's own bottom.
STEAM_FOLLOW = (152, 120)
STEAM_LOW = 31
STEAM_TAP = 0.1  # Steam misses a 60 ms tap; 100 ms is one step
STEAM_LEAD = 0.2  # input lag and the look's own time: turns start this much early (s)
# Things named in a place that are ways out: the survey's own ways out stand for them (a
# spot is only where its name was put on the picture, and can't be walked through).
WAY_OUT = re.compile(r"\b(exit|entrance|door|doorway|stairs?|staircase|way out|gate)\b", re.IGNORECASE)


@dataclass
class Exit:
    frm: tuple[int, int]  # the last tile inside (row, col)
    dir: tuple[int, int]  # the way to walk from it to go out (dx, dy)
    to: int  # where it goes (the game's location number)
    at: tuple[float, float] = (0.0, 0.0)  # where you come out there
    label: str = ""  # what to call it: where it goes ("outside, by Guardia Castle")


@dataclass
class Room:
    loc: int
    name: str
    image: np.ndarray  # BGR, the whole place
    walk: set[tuple[int, int]]
    wall: set[tuple[int, int]]
    exits: list[Exit]
    cams: list[tuple[int, int]]
    learned: set[tuple[int, int]] = field(default_factory=set)  # found shut while walking
    small: np.ndarray | None = None
    spots: dict[str, tuple[float, float]] = field(default_factory=dict)  # things in it

    def cam_range(self) -> tuple[int, int, int, int]:
        xs = [c[0] for c in self.cams] or [0]
        ys = [c[1] for c in self.cams] or [0]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Here:
    loc: int
    cam: tuple[int, int]  # the view's top left on the place's picture
    x: float  # the character's feet, in the place's pixels
    y: float
    score: float
    sure: bool = True  # the character itself was found (not just where the view keeps it)
    steam: bool = False  # seen in the Steam port's wide window (cam is its view's top left)

    @property
    def tile(self) -> tuple[int, int]:
        return int(self.y) // TILE, int(self.x) // TILE


def view(frame: Image.Image | np.ndarray) -> np.ndarray:
    """The game's picture at its own size, BGR: the middle 4:3 of a wide window, scaled
    down (RetroArch keeps the picture in the middle)."""
    import cv2

    rgb = np.asarray(frame.convert("RGB")) if isinstance(frame, Image.Image) else frame
    h, w = rgb.shape[:2]
    if (w, h) != NATIVE:
        gw = min(w, round(h * 4 / 3))
        x0 = (w - gw) // 2
        rgb = cv2.resize(rgb[:, x0:x0 + gw], NATIVE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def steam_view(rgb: np.ndarray) -> np.ndarray:
    """The Steam port's whole window at the game's own pixels (STEAM_VIEW), BGR, its
    fill round a small place painted black (as the pictures are where nothing was)."""
    import cv2

    if rgb.shape[1] > 4 * STEAM_VIEW[0]:  # quicker: the fill is the same colour at any size
        rgb = cv2.resize(rgb, (2 * STEAM_VIEW[0], 2 * STEAM_VIEW[1]), interpolation=cv2.INTER_AREA)
    fill = np.abs(rgb.astype(np.int16) - np.array(STEAM_FILL, np.int16)).sum(2) < 18
    rgb = rgb.copy()
    rgb[fill] = 0
    return cv2.resize(rgb, STEAM_VIEW, interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()


def steam_range(room: Room) -> tuple[float, float, float, float]:
    """Where Steam's view can be on a place's picture (its top left): least and most
    across, then down. The survey's cams are the game's own (256 wide, 224 high)."""
    x0, y0, x1, y1 = room.cam_range()
    w = STEAM_VIEW[0]
    lo_x, hi_x = x0, x1 + NATIVE[0] - w
    if hi_x < lo_x:  # narrower than the view: in the middle of it
        lo_x = hi_x = (x0 + x1 + NATIVE[0] - w) / 2
    lo_y, hi_y = y0, y1 + STEAM_LOW
    return lo_x, lo_y, hi_x, hi_y


class Seen:
    """A picture to match a view on, counting only the pixels the survey saw: the
    normalised correlation (as cv2.TM_CCOEFF_NORMED) over the view's pixels that fall on
    seen ones, at least SEEN_LEAST of them."""

    def __init__(self, img: np.ndarray, seen: np.ndarray):
        m = seen.astype(np.float32)
        self.m = np.repeat(m[:, :, None], img.shape[2], axis=2) if img.ndim == 3 else m
        self.i = img * self.m
        self._sums: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _box(self, th: int, tw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Over every window th x tw: how many seen pixels, their sum, their squares'."""
        import cv2

        if (th, tw) not in self._sums:
            def box(a: np.ndarray) -> np.ndarray:
                a = a.sum(2) if a.ndim == 3 else a
                s = cv2.integral(a.astype(np.float64))
                return (s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]).astype(np.float32)

            self._sums[(th, tw)] = (box(self.m), box(self.i), box(self.i * self.i))
        return self._sums[(th, tw)]

    def match(self, t: np.ndarray, x0: int = 0, y0: int = 0, w: int | None = None,
              h: int | None = None) -> np.ndarray | None:
        """The match of view t at each top left from (x0, y0), w and h further on (all
        of the picture without them); None if the window is off the picture."""
        import cv2

        th, tw = t.shape[:2]
        H, W = self.i.shape[:2]
        x1 = W if w is None else min(W, x0 + tw + w)
        y1 = H if h is None else min(H, y0 + th + h)
        if y1 - y0 < th or x1 - x0 < tw:
            return None
        n, si, sii = (a[y0:y1 - th + 1, x0:x1 - tw + 1] for a in self._box(th, tw))
        m, i = self.m[y0:y1, x0:x1], self.i[y0:y1, x0:x1]
        st = cv2.matchTemplate(m, t, cv2.TM_CCORR)
        stt = cv2.matchTemplate(m, t * t, cv2.TM_CCORR)
        sit = cv2.matchTemplate(i, t, cv2.TM_CCORR)
        k = np.maximum(n, 1)
        var_i = np.maximum(sii - si * si / k, 1e-3)
        var_t = np.maximum(stt - st * st / k, 1e-3)
        r = (sit - si * st / k) / np.sqrt(var_i * var_t)
        r[n < SEEN_LEAST * t.size] = -1
        return r


class Rooms:
    def __init__(self, folder: Path, atlas=None):
        """atlas: the game's world maps (atlas.Atlas), to name a way out onto one by the
        place it comes out by."""
        import cv2

        self.folder = folder
        meta = {}
        if (folder / "names.json").exists():
            meta = json.loads((folder / "names.json").read_text())
        self.names: dict[int, str] = {int(k): v for k, v in meta.get("names", {}).items()}
        self.worlds: set[int] = set(meta.get("worlds", []))
        self.rooms: dict[int, Room] = {}
        for f in sorted(folder.glob("loc-*.json")):
            d = json.loads(f.read_text())
            img = cv2.imread(str(f.with_suffix(".png")), cv2.IMREAD_COLOR)
            if img is None or img.shape[0] < NATIVE[1] or img.shape[1] < NATIVE[0]:
                continue
            exits = [Exit(tuple(e["from"]), tuple(e["dir"]), e["to"], tuple(e.get("at", (0, 0))))
                     for e in d.get("exits", []) if e["to"] != d["loc"]]
            room = Room(d["loc"], d.get("name") or self.names.get(d["loc"], ""), img,
                        {tuple(t) for t in d["walk"]}, {tuple(t) for t in d.get("wall", [])},
                        exits, [tuple(c) for c in d.get("cams", [])],
                        spots={k: tuple(v) for k, v in d.get("spots", {}).items()
                               if not WAY_OUT.search(k)})
            room.small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                                    interpolation=cv2.INTER_AREA)
            self.rooms[room.loc] = room
            if room.name and room.loc not in self.names:
                self.names[room.loc] = room.name
        for room in self.rooms.values():
            for e in room.exits:
                e.label = self._label(room, e, atlas)
        self.sprites: list[tuple[np.ndarray, np.ndarray, tuple[float, float]]] = []
        self._colours: np.ndarray | None = None
        self.dark = False  # the last screen looked at was black (a fade between places)
        self._steam_pics: dict[int, tuple[Seen, Seen]] | None = None
        self._steam_sizes: set[tuple[int, int]] = set()  # window sizes seen as Steam's
        if (folder / "player.json").exists():
            anchors = json.loads((folder / "player.json").read_text()).get("anchor", {})
            for way, (ax, ay) in anchors.items():
                p = folder / f"player-{way}.png"
                if p.exists():
                    rgba = np.asarray(Image.open(p).convert("RGBA"))
                    bgr = np.ascontiguousarray(rgba[:, :, 2::-1])
                    mask = np.ascontiguousarray((rgba[:, :, 3] > 0).astype(np.uint8) * 255)
                    self.sprites.append((bgr, mask, (ax, ay)))

    def _label(self, room: Room, e: Exit, atlas) -> str:
        """A way out onto a world map is named by the place it comes out by; when a
        place has ways out at more than one spot (a forest's two ends), by the nearest
        place other than itself, so each end has a name of its own."""
        if e.to in self.worlds and atlas is not None:
            m = next((m for m in atlas.maps if getattr(m, "world", None) == e.to), None)
            if m is not None and m.places and any(e.at):
                spots = {o.at for o in room.exits if o.to == e.to and any(o.at)}
                if all(math.dist(o, e.at) < 16 for o in spots):
                    return f"outside, by {m.nearest(*e.at)[0]}"
                own = room.name.lower()
                others = [(math.dist(p, e.at), n) for n, p in m.places.items()
                          if not own or (n.lower() not in own and own not in n.lower())]
                if others:
                    return f"outside, toward {min(others)[1]}"
        return self.name(e.to)

    @staticmethod
    def folder_for(game: str, folder: Path = ATLASES) -> Path | None:
        from .atlas import Atlas

        d = Atlas.folder_for(game, folder)
        if d is None:
            d = folder / slug(game) if game else None
        if d is None or not (d / "rooms").is_dir():
            return None
        return d / "rooms"

    @classmethod
    def for_game(cls, game: str, folder: Path = ATLASES) -> Rooms | None:
        from .atlas import Atlas

        d = cls.folder_for(game, folder)
        if d is None:
            return None
        try:
            atlas = Atlas(d.parent) if (d.parent / "atlas.json").exists() else None
        except (OSError, ValueError, KeyError):
            atlas = None
        rooms = cls(d, atlas)
        return rooms if rooms.rooms else None

    def name(self, loc: int) -> str:
        if loc in self.worlds:
            return self.names.get(loc, "outside (the world map)")
        return self.names.get(loc) or f"place {loc:#x}"

    # -- where ---------------------------------------------------------------------
    def locate(self, frame, near: Here | None = None) -> Here | None:
        """Which surveyed place the screen shows and where the character is in it, or
        None (the world map, a menu, a battle, a fade, a place not surveyed)."""
        import cv2

        rgb = np.asarray(frame.convert("RGB")) if isinstance(frame, Image.Image) else frame
        if self.is_steam(rgb):
            return self._locate_steam(rgb, near)
        v = view(rgb)
        self.dark = float(v.std()) < 6 or float(v.mean()) < 12
        if self.dark:  # a black or faded screen
            return None
        t = v[EDGE:200, EDGE:NATIVE[0] - EDGE]
        if near is not None and near.loc in self.rooms:
            room = self.rooms[near.loc]
            x0, y0 = max(0, near.cam[0] - 40), max(0, near.cam[1] - 40)
            win = room.image[y0:y0 + t.shape[0] + 2 * EDGE + 80,
                             x0:x0 + t.shape[1] + 2 * EDGE + 80]
            if win.shape[0] >= t.shape[0] and win.shape[1] >= t.shape[1]:
                r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv2.minMaxLoc(r)
                if score > 0.7:
                    return self._player(v, room, (x0 + x - EDGE, y0 + y - EDGE), score, near)
            # Further round it before anywhere else: a forest's trees look alike all
            # over, and a match far off would make the walk jump back and forth.
            x0, y0 = max(0, near.cam[0] - 120), max(0, near.cam[1] - 120)
            win = room.image[y0:y0 + t.shape[0] + 2 * EDGE + 240,
                             x0:x0 + t.shape[1] + 2 * EDGE + 240]
            if win.shape[0] >= t.shape[0] and win.shape[1] >= t.shape[1]:
                r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv2.minMaxLoc(r)
                if score > 0.6:
                    return self._player(v, room, (x0 + x - EDGE, y0 + y - EDGE), score, near)
        # Everywhere, at half size first, then exactly around the best.
        ts = cv2.resize(t, (t.shape[1] // 2, t.shape[0] // 2), interpolation=cv2.INTER_AREA)
        scores = []
        for room in self.rooms.values():
            s = room.small
            if s is None or s.shape[0] < ts.shape[0] or s.shape[1] < ts.shape[1]:
                continue
            r = cv2.matchTemplate(s, ts, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(r)
            scores.append((float(score), room.loc, (2 * x, 2 * y)))
        if not scores:
            return None
        scores.sort(reverse=True)
        best = None
        for _score, loc, (x, y) in scores[:3]:
            room = self.rooms[loc]
            x0, y0 = max(0, x - 4), max(0, y - 4)
            win = room.image[y0:y0 + t.shape[0] + 8, x0:x0 + t.shape[1] + 8]
            if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, s2, _, (dx, dy) = cv2.minMaxLoc(r)
            if best is None or s2 > best[0]:
                best = (float(s2), loc, (x0 + dx - EDGE, y0 + dy - EDGE))
        if best is None or best[0] < 0.6:
            return None
        return self._player(v, self.rooms[best[1]], best[2], best[0], near)

    def _player(self, v: np.ndarray, room: Room, cam: tuple[int, int], score: float,
                near: Here | None = None) -> Here:
        """Where the character is, on a view found at `cam`: what the screen draws over
        the bare place is the people; the player's is the one with its look (the
        colours of the cut-outs in player-*.png) nearest where it was, or where the view
        keeps it while the view follows (not at the place's edges)."""
        import cv2

        x0, y0, x1, y1 = room.cam_range()
        fx, fy = cam[0] + FOLLOW[0], cam[1] + FOLLOW[1]
        follows = (x0 < cam[0] < x1, y0 < cam[1] < y1)
        h, w = v.shape[:2]
        bg = room.image[cam[1]:cam[1] + h, cam[0]:cam[0] + w]
        if bg.shape != v.shape:
            return Here(room.loc, cam, fx, fy, score, all(follows))
        drawn = (np.abs(v.astype(np.int16) - bg.astype(np.int16)).sum(2) > 60)
        drawn[200:] = False  # the field ends there; a notice may be drawn below
        drawn = cv2.morphologyEx(drawn.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((5, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(drawn)
        if near is not None and near.loc == room.loc:
            prior = (near.x - cam[0], near.y - cam[1])
        else:
            prior = (FOLLOW[0], FOLLOW[1])
        best = None
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if not (6 <= bw <= 48 and 12 <= bh <= 56 and area >= 40):
                continue
            mask = labels[by:by + bh, bx:bx + bw] == i
            mine = self._mine(v[by:by + bh, bx:bx + bw], mask)
            k = int(mine.sum())
            if k < 25:
                continue
            ys, xs = np.nonzero(mine)
            # The feet: under the player's own colours (someone may stand beside it).
            feet = (bx + float(xs.mean()), by + int(ys.max()) + 1)
            cost = math.dist(feet, prior) + 80 * (1 - min(1.0, k / 150))
            if best is None or cost < best[0]:
                best = (cost, feet)
        if best is not None:
            return Here(room.loc, cam, cam[0] + best[1][0], cam[1] + best[1][1], score, True)
        if near is not None and near.loc == room.loc and not all(follows):
            # Not seen (behind something drawn in front): where it was, where the view
            # doesn't say.
            return Here(room.loc, cam, fx if follows[0] else near.x,
                        fy if follows[1] else near.y, score, False)
        return Here(room.loc, cam, fx, fy, score, all(follows))

    def _mine(self, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Which drawn pixels of a patch have the player's colours (the colours of the
        cut-outs in player-*.png); all of them with no cut-outs."""
        if not self.sprites:
            return mask
        if self._colours is None:
            px = np.concatenate([b[m > 0] for b, m, _ in self.sprites]).astype(np.int32)
            q = px // 32
            keys, counts = np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], return_counts=True)
            self._colours = np.array(sorted(keys[counts >= max(3, len(px) // 400)]))
        q = patch.astype(np.int32) // 32
        k = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
        return mask & np.isin(k, self._colours)

    # -- the Steam port ------------------------------------------------------------------
    def is_steam(self, rgb: np.ndarray) -> bool:
        """Whether a screen is the Steam port's (the window filled) rather than an
        emulator's 4:3 picture with black bars (atlas.Atlas._boxed's test): the sides
        show the place or Steam's fill. Sides all black (a dark place, a fade) say
        nothing: as the last clear screen of that size said."""
        h, w = rgb.shape[:2]
        if (w, h) == NATIVE:
            return False
        x0 = (w - min(w, round(h * 4 / 3))) // 2
        if x0 < 4:
            return False
        sides = np.concatenate([rgb[::4, :x0:4], rgb[::4, w - x0::4]], axis=1)
        sides = sides.reshape(-1, 3).astype(np.int16)
        fill = np.abs(sides - np.array(STEAM_FILL, np.int16)).sum(1) < 18
        lit = (sides.max(1) >= 24) & ~fill
        if float(fill.mean()) > 0.2 or float(lit.mean()) > 0.05:
            self._steam_sizes.add((w, h))
            return True
        return (w, h) in self._steam_sizes

    @staticmethod
    def _steam_prep(bgr: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), 1.0)

    def _steam_pictures(self) -> dict[int, tuple[Seen, Seen]]:
        """Each place's picture padded with black and softened as Steam's view is
        matched (full size and half), made on the first Steam screen."""
        import cv2

        if self._steam_pics is None:
            pics = {}
            for loc, room in self.rooms.items():
                img = cv2.copyMakeBorder(room.image, STEAM_PAD, STEAM_PAD, STEAM_PAD,
                                         STEAM_PAD, cv2.BORDER_CONSTANT, value=0)
                seen = cv2.erode((img.max(2) > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
                p = self._steam_prep(img)
                pics[loc] = (Seen(p, seen), Seen(cv2.pyrDown(p), seen[::2, ::2]))
            self._steam_pics = pics
        return self._steam_pics

    def _locate_steam(self, rgb: np.ndarray, near: Here | None) -> Here | None:
        """locate() for the Steam port: around the last view first, else every place at
        half size, then exactly around the best three; the best is kept if it scores
        STEAM_SCORE and beats anywhere else (that place further off, the other places)
        by STEAM_MARGIN. Only what the survey saw counts (Steam's view can go further
        than the game's own: off a place's sides, past where the game's view stops)."""
        import cv2

        v = steam_view(rgb)
        self.dark = float(v.std()) < 6
        if self.dark:
            return None
        b, a, r_, d_ = STEAM_CUT
        t = self._steam_prep(v)[a:STEAM_VIEW[1] - d_, b:STEAM_VIEW[0] - r_]
        pics = self._steam_pictures()
        if near is not None and near.steam and near.loc in pics:
            x0 = max(0, near.cam[0] + STEAM_PAD + b - STEAM_NEAR)
            y0 = max(0, near.cam[1] + STEAM_PAD + a - STEAM_NEAR)
            r = pics[near.loc][0].match(t, x0, y0, 2 * STEAM_NEAR, 2 * STEAM_NEAR)
            if r is not None:
                _, score, _, (x, y) = cv2.minMaxLoc(r)
                if score >= STEAM_SCORE:
                    cam = (x0 + x - b - STEAM_PAD, y0 + y - a - STEAM_PAD)
                    return self._player_steam(v, self.rooms[near.loc], cam, float(score), near)
        half = cv2.pyrDown(t)
        cands = []
        maps = {}
        for loc, (_, H) in pics.items():
            r = H.match(half)
            if r is None:
                continue
            _, s, _, (x, y) = cv2.minMaxLoc(r)
            cands.append((float(s), loc, 2 * x, 2 * y))
            maps[loc] = r
        cands.sort(reverse=True)
        fine = []
        for _s, loc, x, y in cands[:3]:
            x0, y0 = max(0, x - 6), max(0, y - 6)
            r = pics[loc][0].match(t, x0, y0, 12, 12)
            if r is None:
                continue
            _, s2, _, (dx, dy) = cv2.minMaxLoc(r)
            fine.append((float(s2), loc, x0 + dx, y0 + dy))
        if not fine:
            return None
        fine.sort(reverse=True)
        score, loc, x, y = fine[0]
        if score < STEAM_SCORE:
            return None
        # Anywhere else in that place: at half size (as good or better than exactly,
        # being softer), off the best by more than 24 pixels.
        r = maps[loc]
        r[max(0, y // 2 - 12):y // 2 + 12, max(0, x // 2 - 12):x // 2 + 12] = -1
        other = max([float(r.max())] + [f[0] for f in fine[1:] if f[1] != loc])
        if score - other < STEAM_MARGIN:
            return None
        cam = (x - b - STEAM_PAD, y - a - STEAM_PAD)
        return self._player_steam(v, self.rooms[loc], cam, score, near)

    def _steam_sprites(self) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
        """The player's cut-outs softened as Steam draws them, a step to the left
        mirrored from one to the right when there's none of its own."""
        import cv2

        if getattr(self, "_ssprites", None) is None:
            got: dict[str, tuple[np.ndarray, np.ndarray, tuple[float, float]]] = {}
            if (self.folder / "player.json").exists():
                anchors = json.loads((self.folder / "player.json").read_text()).get("anchor", {})
                for way, (ax, ay) in anchors.items():
                    p = self.folder / f"player-{way}.png"
                    if p.exists():
                        rgba = np.asarray(Image.open(p).convert("RGBA"))
                        got[way] = (np.ascontiguousarray(rgba[:, :, 2::-1]),
                                    np.ascontiguousarray((rgba[:, :, 3] > 0).astype(np.uint8)),
                                    (ax, ay))
            for way, (bgr, mask, (ax, ay)) in list(got.items()):
                if way.startswith("right") and way.replace("right", "left") not in got:
                    got[way.replace("right", "left")] = (
                        np.ascontiguousarray(bgr[:, ::-1]), np.ascontiguousarray(mask[:, ::-1]),
                        (bgr.shape[1] - ax, ay))
            self._ssprites = [(cv2.GaussianBlur(b.astype(np.float32), (0, 0), 0.7), m,
                               (round(ax), round(ay))) for b, m, (ax, ay) in got.values()]
        return self._ssprites

    def _steam_feet(self, v: np.ndarray, box: tuple[float, float, float, float] | None = None,
                    n: int = 4) -> list[tuple[float, tuple[int, int]]]:
        """Where the player's cut-outs fit Steam's view best, as (score, feet), the
        best few at least a body apart; feet only within `box` (x0, y0, x1, y1) when
        given, which is quicker."""
        import cv2

        h, w = v.shape[:2]
        ox = oy = 0
        if box is not None:
            ox, oy = max(0, int(box[0]) - 24), max(0, int(box[1]) - 48)
            v = v[oy:max(oy, int(box[3]) + 8), ox:max(ox, int(box[2]) + 24)]
        vh, vw = v.shape[:2]
        vf = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 0.7)
        best = np.full((vh, vw), -1, np.float32)
        for spr, mask, (ax, ay) in self._steam_sprites():
            if spr.shape[0] > vh or spr.shape[1] > vw:
                continue
            r = cv2.matchTemplate(vf, spr, cv2.TM_CCOEFF_NORMED, mask=mask)
            r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
            sub = best[ay:ay + r.shape[0], ax:ax + r.shape[1]]
            np.maximum(sub, r[:sub.shape[0], :sub.shape[1]], out=sub)
        # Steam's toasts, bottom right
        best[max(0, h - STEAM_CUT[3] - oy):, max(0, w - STEAM_CUT[2] - ox):] = -1
        out = []
        for _ in range(n):
            _, s, _, (x, y) = cv2.minMaxLoc(best)
            if s <= 0:
                break
            out.append((float(s), (x + ox, y + oy)))
            best[max(0, y - 20):y + 20, max(0, x - 12):x + 12] = -1
        return out

    def _player_steam(self, v: np.ndarray, room: Room, cam: tuple[int, int], score: float,
                      near: Here | None = None) -> Here:
        """_player() on Steam's view: Steam draws people as the game does, softened,
        but not in the game's colours, so the player is found by its shape (the
        cut-outs, softened alike): a clear match, or a weaker one where it was last."""
        x0, y0, x1, y1 = steam_range(room)
        fx, fy = cam[0] + STEAM_FOLLOW[0], cam[1] + STEAM_FOLLOW[1]
        follows = (x0 + 1 < cam[0] < x1 - 1, y0 + 1 < cam[1] < y1 - 1)
        same = near is not None and near.loc == room.loc
        prior = (near.x - cam[0], near.y - cam[1]) if same else STEAM_FOLLOW
        best = None
        # Around where it was when it was seen, or where the view keeps it while it
        # follows: quicker, and no one else is taken for it.
        w, h = STEAM_VIEW
        if same and near.sure and near.steam:
            box = (prior[0] - STEAM_NEAR, prior[1] - STEAM_NEAR,
                   prior[0] + STEAM_NEAR, prior[1] + STEAM_NEAR)
        else:
            sx, sy = STEAM_FOLLOW
            box = (sx - 24 if follows[0] else 0, sy - 28 if follows[1] else 0,
                   sx + 24 if follows[0] else w, sy + 28 if follows[1] else h)
        for s, feet in self._steam_feet(v, box):
            far = math.dist(feet, prior)
            if s < 0.6 and not (s >= 0.4 and far <= 24):
                continue
            if (follows[0] and abs(feet[0] - STEAM_FOLLOW[0]) > 24) or (
                    follows[1] and abs(feet[1] - STEAM_FOLLOW[1]) > 28):
                continue  # not where the view keeps it
            cost = far + 100 * (1 - s)
            if best is None or cost < best[0]:
                best = (cost, feet)
        if best is not None:
            return Here(room.loc, cam, cam[0] + best[1][0], cam[1] + best[1][1], score, True,
                        True)
        if same and not all(follows):
            return Here(room.loc, cam, fx if follows[0] else near.x,
                        fy if follows[1] else near.y, score, False, True)
        return Here(room.loc, cam, fx, fy, score, all(follows), True)

    # -- the way ---------------------------------------------------------------------
    def hops(self, start: int, goal: int | set[int]) -> list[int] | None:
        """Places to go through, start to goal, by the ways out (fewest first)."""
        goals = goal if isinstance(goal, set) else {goal}
        came = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if u in goals:
                out = []
                while u is not None:
                    out.append(u)
                    u = came[u]
                return out[::-1]
            if u not in self.rooms:
                continue
            for e in self.rooms[u].exits:
                if e.to not in came:
                    came[e.to] = u
                    q.append(e.to)
        return None

    def reachable(self, start: int) -> dict[str, tuple[int, int]]:
        """Everywhere the ways out lead from start, by name: (location, how many places
        away). A world map is one place for each side it's come out on."""
        dist = {start: 0}
        q = deque([start])
        out: dict[str, tuple[int, int]] = {}
        while q:
            u = q.popleft()
            for e in self.rooms[u].exits if u in self.rooms else []:
                if e.to == start:
                    continue
                if e.label not in out or out[e.label][1] > dist[u] + 1:
                    out[e.label] = (e.to, dist[u] + 1)
                if e.to not in dist:
                    dist[e.to] = dist[u] + 1
                    q.append(e.to)
        outs = [v for v in out.values() if v[0] in self.worlds]
        if len(outs) > 1:  # outside with no side named: the nearest way
            out[NEAREST_OUT] = min(outs, key=lambda v: v[1])
        return out

    def route(self, room: Room, start: tuple[int, int], goals: set[tuple[int, int]]
              ) -> list[tuple[int, int]] | None:
        """Tiles from start to any goal tile over the place's walkable tiles (A*)."""
        def shut(t):
            if t == start or t in goals:
                return False
            return t in room.learned or t not in room.walk

        pq = [(0.0, start)]
        came = {start: None}
        cost = {start: 0.0}
        while pq:
            _, u = heapq.heappop(pq)
            if u in goals:
                path = []
                while u is not None:
                    path.append(u)
                    u = came[u]
                return path[::-1]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not (dr or dc):
                        continue
                    v = (u[0] + dr, u[1] + dc)
                    if shut(v):
                        continue
                    if dr and dc and (shut((u[0] + dr, u[1])) or shut((u[0], u[1] + dc))):
                        continue  # no cutting corners
                    # Keep off walls: the character is wider than a point.
                    side = sum((v[0] + a, v[1] + b) not in room.walk
                               for a, b in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                    c = cost[u] + (1.4 if dr and dc else 1.0) + 0.3 * side
                    if c < cost.get(v, 1e9):
                        cost[v] = c
                        came[v] = u
                        h = min(abs(v[0] - g[0]) + abs(v[1] - g[1]) for g in goals)
                        heapq.heappush(pq, (c + h, v))
        return None


def centre(t: tuple[int, int]) -> tuple[float, float]:
    """Where to stand on a tile: across, its middle; down, low on it (the feet come to
    rest at a tile's bottom edge when the one above is shut)."""
    return t[1] * TILE + TILE / 2, t[0] * TILE + 13


def angle_to(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, -dy)) % 360


def lead(here: Here, trail) -> tuple[float, float]:
    """Where the character will be by the time a turn takes on Steam: walked on at
    its speed over the last half second for STEAM_LEAD, at most a tile. Where it is
    on the emulator (no lag to speak of). trail: (when, place, x, y), oldest first."""
    if not here.steam:
        return here.x, here.y
    now = time.monotonic()
    recent = [p for p in trail if p[1] == here.loc and now - p[0] <= 0.5]
    if not recent or now - recent[0][0] < 0.1:
        return here.x, here.y
    dt = now - recent[0][0]
    vx, vy = (here.x - recent[0][2]) / dt, (here.y - recent[0][3]) / dt
    dx, dy = vx * STEAM_LEAD, vy * STEAM_LEAD
    k = min(1.0, TILE / max(1e-6, math.hypot(dx, dy)))
    return here.x + dx * k, here.y + dy * k


async def walk_out(rooms: Rooms, goal: int, shoot, steer, keep_going, log,
                   limit: float = 90.0, start: Here | None = None, label: str = "",
                   spot: tuple[float, float] | None = None) -> str:
    """Walk to the place `goal` (a location number; a world map for "outside") through
    the ways out, place by place, with the screen alone: where the character is comes
    from the pictures several times a second. label: which way out into it, by its name
    (a world map's sides). spot: a place in `goal` to walk to (in its pixels), not a
    way into it."""
    t0 = time.monotonic()
    here = start
    misses = 0
    gone_by: Exit | None = None  # the way out last walked into
    heading: Exit | None = None  # the way out being walked to
    target = label or rooms.name(goal)
    stalls = 0
    gone_t = said = stuck_t = 0.0
    faded = False  # the screen went black since the place was last seen
    dud: set = set()  # ways out walked into that didn't go through
    trail: deque = deque()  # (when, place, x, y) over the last seconds
    best: tuple[int, int, float] = (-1, 10**6, 0.0)  # place, fewest tiles to go, since when

    async def look() -> Here | None:
        return await asyncio.to_thread(rooms.locate, await shoot(), here)

    try:
        while time.monotonic() - t0 < limit:
            if not await keep_going():
                return "stopped by the player"
            now = await look()
            if now is None:
                misses += 1
                faded = faded or rooms.dark
                way = gone_by
                if way is None and here is not None and here.loc in rooms.rooms:
                    # Went through a way out before reaching its tile (they're often
                    # wider): the one it was walking to (where it was last seen can be
                    # a glimpse off), else the nearest if it's close. Into another
                    # surveyed place instead, it's seen there after the fade.
                    exits = rooms.rooms[here.loc].exits
                    if heading in exits:
                        way = heading
                    elif exits:
                        e = min(exits, key=lambda e: math.dist(centre(e.frm), (here.x, here.y)))
                        if math.dist(centre(e.frm), (here.x, here.y)) <= 2.5 * TILE:
                            way = e
                if misses >= 3 and not faded:
                    # Not a fade: something over the place (a text box, a scene).
                    steer(None)
                    if misses >= 5:
                        return ("interrupted: something came up on screen (a text box or a "
                                "scene?): read it, press a, then go_to_exit again")
                    await asyncio.sleep(0.15)
                    continue
                if way is not None and misses >= 3:
                    steer(None)
                    if way.to == goal and (goal in rooms.worlds or goal not in rooms.rooms):
                        # Out of the surveyed places, where it was going: wait the fade.
                        await asyncio.sleep(1.0)
                        if await look() is None:
                            return (f"arrived: went out to {target} "
                                    f"({time.monotonic() - t0:.0f} s)")
                    if way.to not in rooms.rooms and misses > 12:
                        return (f"new_area: went out to {rooms.name(way.to)} on the way to "
                                f"{target}; look")
                    await asyncio.sleep(0.3)
                    if misses > 20:
                        return "lost: went through a way out and don't know this place"
                    continue
                if misses > 25:
                    steer(None)
                    return ("lost: the screen isn't a place I know (a menu, a text box or "
                            "a scene?): look")
                await asyncio.sleep(0.1)
                continue
            faded = False
            misses = 0
            if here is not None and now.loc != here.loc:
                log(f"rooms: now in {rooms.name(now.loc)}")
                gone_by = None
                stalls = 0
                if now.loc == goal and spot is None:
                    steer(None)
                    return f"arrived: in {target} ({time.monotonic() - t0:.0f} s)"
            here = now
            room = rooms.rooms[here.loc]
            # Where it will be when a turn starts: on Steam, input lags and the look
            # takes a while, so it's walked on a little by then (turns start early).
            px, py = lead(here, trail)
            tap = STEAM_TAP if here.steam else 0.06
            if here.loc == goal and spot is not None:
                # The last of it: to the walkable tile nearest the spot, facing it.
                tile = here.tile
                if tile not in room.walk:
                    tile = min(room.walk, key=lambda t: abs(t[0] - tile[0]) + abs(t[1] - tile[1]))
                end = min(room.walk, key=lambda t: math.dist(centre(t), spot))
                if math.dist((px, py), centre(end)) < 10 or tile == end:
                    steer(angle_to(spot[0] - here.x, spot[1] - here.y))
                    await asyncio.sleep(max(0.08, tap))
                    steer(None)
                    return f"arrived: at {target} ({time.monotonic() - t0:.0f} s)"
                path = rooms.route(room, tile, {end})
                if path is None:
                    steer(None)
                    return f"blocked: no way to {target} from here"
                aim = centre(path[1]) if len(path) > 1 else centre(end)
                for t in path[1:4]:
                    if math.dist((px, py), centre(t)) > 12:
                        aim = centre(t)
                        break
                steer(angle_to(aim[0] - px, aim[1] - py))
                if here.steam:
                    trail.append((time.monotonic(), here.loc, here.x, here.y))
                await asyncio.sleep(0.02)
                continue
            if here.loc == goal:
                return f"already in {target}"
            hops = rooms.hops(here.loc, goal)
            if hops is None or len(hops) < 2:
                steer(None)
                return f"no way from {rooms.name(here.loc)} to {target} that I know of"
            ways = [e for e in room.exits if e.to == hops[1]]
            if len(hops) == 2 and label and any(e.label == label for e in ways):
                ways = [e for e in ways if e.label == label]
            if all((e.frm, e.dir) in dud for e in ways):
                dud.clear()
            ways = [e for e in ways if (e.frm, e.dir) not in dud]
            if gone_by is not None:
                # Stepping through a way out: keep walking it until the screen changes.
                steer(angle_to(*gone_by.dir))
                await asyncio.sleep(0.05)
                if time.monotonic() - gone_t > 2.5:
                    # Didn't go through from here: another tile of it, or another way.
                    dud.add((gone_by.frm, gone_by.dir))
                    log(f"rooms: the way out at {gone_by.frm} didn't open")
                    gone_by = None
                continue
            tile = here.tile
            if tile not in room.walk:
                # Between tiles or off what was surveyed: the nearest walked tile.
                tile = min(room.walk, key=lambda t: abs(t[0] - tile[0]) + abs(t[1] - tile[1]))
            goals = {e.frm for e in ways}
            path = rooms.route(room, tile, goals)
            if path is not None:
                heading = next((e for e in ways if e.frm == path[-1]), None)
            if path is None and room.learned:
                # Shut off by what was found shut on the way (people move): try again.
                room.learned.clear()
                path = rooms.route(room, tile, goals)
            if path is None:
                steer(None)
                return (f"blocked: no way through to {rooms.name(hops[1])} from here in "
                        f"{rooms.name(here.loc)}")
            at_end = len(path) == 1 or (len(path) == 2 and
                                        math.dist((px, py), centre(path[-1])) < 10)
            if at_end:
                e = min((e for e in ways if e.frm == path[-1]),
                        key=lambda e: 0, default=ways[0])
                cx, cy = centre(e.frm)
                # Line up across the way first, then walk through it.
                off = (cx - here.x) if e.dir[1] else (cy - here.y)
                if abs(off) > 5:
                    a = angle_to(off, 0) if e.dir[1] else angle_to(0, off)
                    steer(a)
                    await asyncio.sleep(tap)
                    steer(None)
                    if here.steam:  # till the step shows
                        await asyncio.sleep(STEAM_LEAD)
                        trail.clear()
                    continue
                gone_by, gone_t = e, time.monotonic()
                log(f"rooms: through the way out to {rooms.name(e.to)} at {e.frm}")
                steer(angle_to(*e.dir))
                continue
            # Aim a little way along the path (several tiles when it runs straight).
            aim = centre(path[1])
            for t in path[1:4]:
                c = centre(t)
                if math.dist((px, py), c) > 12:
                    aim = c
                    break
            a = angle_to(aim[0] - px, aim[1] - py)
            steer(a)
            if time.monotonic() - said > 1.0:
                said = time.monotonic()
                log(f"rooms: at {here.x:.0f},{here.y:.0f} {tile}{'' if here.sure else '?'} "
                    f"in {rooms.name(here.loc)}, {len(path) - 1} tiles to go, heading {a:.0f}")
            # Stuck: not off this spot for a while: the tile it's pushing into is shut
            # after all (walls aren't all on tile edges; people stand about).
            now_t = time.monotonic()
            if here.loc != best[0] or len(path) - 1 < best[1]:
                best = (here.loc, len(path) - 1, now_t)
            elif now_t - best[2] > 6.0 and len(path) > 1:
                # No nearer in a while, though it wobbles about (pushing into a corner
                # the survey thought open, or round someone): that way is shut.
                stalls += 1
                if path[1] not in goals:
                    room.learned.add(path[1])
                log(f"rooms: no nearer for 6 s at {tile} in {rooms.name(here.loc)}, shut "
                    f"{path[1]} ({stalls})")
                steer((a + (90 if stalls % 2 else -90)) % 360)
                await asyncio.sleep(0.3)
                steer(None)
                best = (here.loc, 10**6, now_t)
                if stalls > 12:
                    return (f"blocked: can't get through to {rooms.name(hops[1])} from "
                            f"{tile} in {rooms.name(here.loc)}")
                continue
            trail.append((now_t, here.loc, here.x, here.y))
            while trail and now_t - trail[0][0] > 2.0:
                trail.popleft()
            if now_t - stuck_t > 2.0 and now_t - trail[0][0] > 1.6 and all(
                    p[1] == here.loc and math.dist(p[2:], (here.x, here.y)) < 5 for p in trail):
                stalls += 1
                rad = math.radians(a)
                ahead = (int((here.y - 12 * math.cos(rad)) // TILE),
                         int((here.x + 12 * math.sin(rad)) // TILE))
                if ahead in (here.tile, tile) and len(path) > 1:
                    ahead = path[1]  # (the tile it stands on, as the survey has it)
                if ahead not in goals:
                    room.learned.add(ahead)
                log(f"rooms: stuck at {tile} in {rooms.name(here.loc)}, shut {ahead} "
                    f"({stalls})")
                # A step aside, one way then the other: round someone in the way.
                steer((a + (90 if stalls % 2 else -90)) % 360)
                await asyncio.sleep(0.3)
                steer(None)
                stuck_t = now_t
                trail.clear()
                if stalls > 12:
                    return (f"blocked: can't get through to {rooms.name(hops[1])} from "
                            f"{tile} in {rooms.name(here.loc)}")
            await asyncio.sleep(0.02)
    finally:
        steer(None)
    where = rooms.name(here.loc) if here else "somewhere unknown"
    return f"timeout: still in {where}, on the way to {target}"


WORDS_OUT = {"out", "outside", "exit", "leave", "outdoors", "world"}
NEAREST_OUT = "outside, the nearest way out"


def pick_words(said: str, choices: dict[str, str]) -> str | None:
    """An obvious pick without a model: 'outside' when there's one way out to it, or a
    place named word for word."""
    q = set(re.findall(r"[a-z0-9']+", said.lower().replace("'s", "")))
    if q & WORDS_OUT and not q - WORDS_OUT - {"take", "me", "go", "get", "the", "way",
                                              "of", "here", "to", "a", "let", "us", "lets"}:
        outs = [k for k, v in choices.items() if "world map" in v or v.startswith("outside")]
        if len(outs) == 1:
            return outs[0]
        if NEAREST_OUT in choices:
            return NEAREST_OUT
    for k, v in choices.items():
        if v.lower() == said.lower().strip(" .!?"):
            return k
    return None


PICK_EXIT = """The player is inside {here} in {game} and says: "{said}".
The places they can walk to from here, by how many ways out away:
{places}
Which one do they mean? "outside", "the way out", "leave" mean the world map unless
they name a place (with more than one way outside and none of them named or meant, "outside,
the nearest way out"). "Upstairs" or "downstairs" is the floor of the same building. JSON:
{{"place": "the number before the name", "why": "a few words"}} or {{"place": null,
"why": "..."}} if none fits."""


async def pick_exit(muse, game: str, said: str, here: str,
                    choices: dict[str, tuple[int, int]]) -> tuple[str | None, str]:
    """The place the player means, among those reachable (by name: location, how far):
    (its name, why), or (None, why none fits)."""
    hit = pick_words(said, {k: k for k in choices})
    if hit is not None:
        return hit, "named"
    names = sorted(choices, key=lambda k: choices[k][1])
    lines = "\n".join(f"{i + 1}: {k} ({choices[k][1]} away)" for i, k in enumerate(names))
    try:
        data = await muse.json([{"role": "user", "content": PICK_EXIT.format(
            game=game or "the game", here=here, said=said, places=lines)}],
            effort="minimal", max_tokens=600)
    except (ValueError, OSError) as exc:
        return None, f"couldn't choose ({exc})"
    try:
        i = int(str(data.get("place")))
    except ValueError:
        i = 0
    why = str(data.get("why") or "")
    if not 1 <= i <= len(names):
        return None, why or "no place matches"
    return names[i - 1], why


def describe(rooms: Rooms, here: Here) -> str:
    """Where the player is in a place, said plainly, with the ways out by direction."""
    room = rooms.rooms[here.loc]
    parts = []
    for to in sorted({e.to for e in room.exits}):
        e = min((e for e in room.exits if e.to == to),
                key=lambda e: math.dist(centre(e.frm), (here.x, here.y)))
        c = centre(e.frm)
        steps = math.dist(c, (here.x, here.y)) / TILE
        parts.append(f"{rooms.name(to)}: {steps:.0f} steps "
                     f"{compass(c[0] - here.x, c[1] - here.y)}")
    return (f"in {rooms.name(here.loc)}. Ways out: " + "; ".join(parts)) if parts else \
        f"in {rooms.name(here.loc)}; no ways out surveyed"
