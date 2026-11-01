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
    atmos.json                  {"lines": {"2": "A snug wooden bedroom, sunlit..."}}: what a
                                place is like in a sentence, said the first time it's entered
                                (scripts/atmos_rooms.py drafts them from the pictures)
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
# By the edges: Steam's own screens of places not surveyed reach 0.35 at best.
STEAM_EDGES = (0.55, 0.15)
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
class Person:
    """Someone (or something drawn, like a save point) standing in a place, found by the
    survey with the game's memory and named from their look."""
    name: str
    kind: str  # "person", "animal", "save point", "chest" or "thing"
    at: tuple[float, float]  # their feet, in the place's pixels
    moves: bool = False  # walks about: may not be just there
    look: str = ""  # their cut-out (people/<look>, the survey's), to find them on screen
    # Other names they go by ("Melchior", "swordsmith", "old man") and their line when
    # talked to: what the player may call them by (people.resolve).
    aka: tuple[str, ...] = ()
    said: str = ""


@dataclass
class Target:
    """Somewhere to go: a thing in a place (where its name was put on the picture) or
    someone standing there (walked up to and faced, to talk to them)."""
    loc: int
    at: tuple[float, float]
    person: bool = False  # someone or something standing there: walked up to, faced
    name: str = ""
    kind: str = ""  # the survey's kind for it (a save point is stood on, not talked to)


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
    moving: bool = False  # its ground never stands still (drifting mist): found by its edges
    people: list[Person] = field(default_factory=list)
    # Tiles beside someone who stands still: walked past only when there's no other
    # way (their feet's box reaches into them; a slanting step there catches on it).
    crowd: set[tuple[int, int]] = field(default_factory=set)

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


def steam_edges(bgr: np.ndarray) -> np.ndarray:
    """The shapes in a picture: how steeply its softened brightness changes."""
    import cv2

    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


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
        self.feel: dict[int, str] = {}
        if (folder / "atmos.json").exists():
            lines = json.loads((folder / "atmos.json").read_text()).get("lines", {})
            self.feel = {int(k): " ".join(str(v).split()) for k, v in lines.items()
                         if str(v).strip()}
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
                               if not WAY_OUT.search(k)}, moving=bool(d.get("moving")),
                        people=[Person(p["name"], p.get("kind") or "person",
                                       tuple(p["at"]), bool(p.get("moves")),
                                       p.get("look") or "", tuple(p.get("aka") or ()),
                                       str(p.get("said") or ""))
                                for p in d.get("people", []) if p.get("name")
                                and p.get("kind") != "none"])
            room.small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                                    interpolation=cv2.INTER_AREA)
            for p in room.people:
                if not p.moves and p.kind in ("person", "animal"):
                    pr, pc = int(p.at[1]) // TILE, int(p.at[0]) // TILE
                    room.crowd |= {(pr + 1, pc), (pr - 1, pc), (pr, pc + 1), (pr, pc - 1)}
            self.rooms[room.loc] = room
            if room.name and room.loc not in self.names:
                self.names[room.loc] = room.name
        self._name_alike(atlas)
        for room in self.rooms.values():
            for e in room.exits:
                e.label = self._label(room, e, atlas)
        self.sprites: list[tuple[np.ndarray, np.ndarray, tuple[float, float]]] = []
        self._colours: np.ndarray | None = None
        self.dark = False  # the last screen looked at was black (a fade between places)
        # The world map last seen (its location number): of two places that look the
        # same in two eras, the one that leads out onto it.
        self.world_hint: int | None = None
        # Where on it (its pixels): of two places alike in one era (two towns' elders'
        # houses), the one whose door is nearest.
        self.world_spot: tuple[float, float] | None = None
        self._steam_pics: dict[int, dict[str, tuple[Seen, Seen]]] | None = None
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
                    return f"outside, by {plain_place(m.nearest(*e.at)[0])}"
                own = room.name.lower()
                others = [(math.dist(p, e.at), n) for n, p in m.places.items()
                          if not own or (n.lower() not in own and own not in n.lower()
                                         and plain_place(n).lower() != own)]
                if others:
                    return f"outside, toward {plain_place(min(others)[1])}"
        return self.name(e.to)

    def _name_alike(self, atlas) -> None:
        """Places the game names alike ("Residence" five times, "Market" three) are
        called by where they are, in plain words: the world map's name for the place
        whose door leads in ("Truce Residence (west)": "the west house in Truce"), when
        that tells them apart."""
        if atlas is None:
            return
        by_name: dict[str, list[int]] = {}
        for loc, n in self.names.items():
            if loc not in self.worlds:
                by_name.setdefault(n, []).append(loc)
        alike = {loc for locs in by_name.values() if len(locs) > 1 for loc in locs}
        if not alike:
            return
        found: dict[int, str] = {}
        for m in getattr(atlas, "maps", []):
            doors = getattr(m, "doors", None)
            if not doors or not hasattr(m, "entrance"):
                continue
            for place in m.places:
                for c in m.entrance(place) or ():
                    loc = doors.get(c)
                    if loc in alike and loc not in found:
                        found[loc] = place
        for n, locs in by_name.items():
            named = {loc: plain_place(found[loc]) for loc in locs if loc in found}
            if len(set(named.values())) < 2 and len(named) < len(locs):
                continue  # the map doesn't tell them apart either
            for loc, full in named.items():
                if full.lower() != n.lower():
                    self.names[loc] = full
                    if loc in self.rooms:
                        self.rooms[loc].name = full

    @staticmethod
    def folder_for(game: str, folder: Path = ATLASES) -> Path | None:
        """The game's rooms/ in its atlas folder; PLAYER_TWO_ROOMS names another folder
        of rooms instead (a survey being tried out), for whichever game is played."""
        import os

        from .atlas import Atlas

        own = os.environ.get("PLAYER_TWO_ROOMS")
        if own and folder == ATLASES and Path(own).expanduser().is_dir():
            return Path(own).expanduser()
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
        a = Atlas.folder_for(game, folder) or d.parent
        try:
            atlas = Atlas(a) if (a / "atlas.json").exists() else None
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
        if any(r.moving for r in self.rooms.values()):
            # Places whose ground drifts (the Mountain of Woe's mist) by their edges,
            # first: in colour they'd match anywhere about as well.
            here = self._locate_moving(v, near)
            if here is not None:
                return here
        if near is not None and near.loc in self.rooms and not self.rooms[near.loc].moving:
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
            if room.moving:
                continue
            s = room.small
            if s is None or s.shape[0] < ts.shape[0] or s.shape[1] < ts.shape[1]:
                continue
            r = cv2.matchTemplate(s, ts, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(r)
            scores.append((float(score), room.loc, (2 * x, 2 * y)))
        if not scores:
            return None
        scores.sort(reverse=True)
        fine = []
        for _score, loc, (x, y) in scores[:3]:
            room = self.rooms[loc]
            x0, y0 = max(0, x - 4), max(0, y - 4)
            win = room.image[y0:y0 + t.shape[0] + 8, x0:x0 + t.shape[1] + 8]
            if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, s2, _, (dx, dy) = cv2.minMaxLoc(r)
            fine.append((float(s2), loc, (x0 + dx - EDGE, y0 + dy - EDGE)))
        if not fine:
            return None
        top = max(f[0] for f in fine)
        # One place in two eras matches as well as the other: the likelier era's.
        best = min((f for f in fine if f[0] >= top - 0.02),
                   key=lambda f: (self.prefer(f[1], near), -f[0]))
        if best[0] < 0.6:
            return None
        return self._player(v, self.rooms[best[1]], best[2], best[0], near)

    def _locate_moving(self, v: np.ndarray, near: Here | None) -> Here | None:
        """locate() on the places whose ground drifts, by the edges, only what the survey
        saw counting: around the last view, else all of those places."""
        import cv2

        if getattr(self, "_moving_pics", None) is None:
            self._moving_pics = {}
            for loc, room in self.rooms.items():
                if room.moving:
                    seen = cv2.erode((room.image.max(2) > 0).astype(np.uint8),
                                     np.ones((9, 9), np.uint8))
                    e = steam_edges(room.image)
                    self._moving_pics[loc] = (Seen(e, seen), Seen(cv2.pyrDown(e), seen[::2, ::2]))
        t = steam_edges(v)[EDGE:200, EDGE:NATIVE[0] - EDGE]
        low, margin = STEAM_EDGES
        if near is not None and near.loc in self._moving_pics:
            x0, y0 = max(0, near.cam[0] - 40), max(0, near.cam[1] - 40)
            r = self._moving_pics[near.loc][0].match(t, x0 + EDGE, y0 + EDGE, 80, 80)
            if r is not None:
                _, score, _, (x, y) = cv2.minMaxLoc(r)
                if score >= low:
                    return self._player_steam(v[:200], self.rooms[near.loc], (x0 + x, y0 + y),
                                              float(score), near, steam=False)
        hit = self._steam_search(t, self._moving_pics, low, margin)
        if hit is None:
            return None
        score, loc, x, y = hit
        return self._player_steam(v[:200], self.rooms[loc], (x - EDGE, y - EDGE), score, near,
                                  steam=False)

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
        # By its shape first: the player's cut-outs fit it far better than anyone
        # else (0.85 and up against 0.45), even pressed up against someone, where the
        # colours of the two run together.
        if near is not None and near.loc == room.loc and near.sure:
            px, py = near.x - cam[0], near.y - cam[1]
            box = (px - STEAM_NEAR, py - STEAM_NEAR, px + STEAM_NEAR, py + STEAM_NEAR)
        else:
            box = None
        fit = self._steam_feet(v[:200], box, 3, standing(room, cam)) if self.sprites else []
        # Feet on ground the survey walked (not someone behind a counter).
        fit = [f for f in fit if on_ground(room, cam[0] + f[1][0], cam[1] + f[1][1])][:2]
        if fit:
            s0, (fx_, fy_) = fit[0]
            second = fit[1][0] if len(fit) > 1 else 0.0
            prior = (near.x - cam[0], near.y - cam[1]) if box is not None else FOLLOW
            # A clear fit, or a fair one standing out from the next (facing up fits
            # least, ~0.55), or a fair one where it was.
            # Where it was, fainter still: half hidden by those following it.
            if s0 >= 0.65 or (s0 >= 0.45 and s0 - second >= 0.08) or (
                    s0 >= (0.38 if box is not None else 0.45)
                    and math.dist((fx_, fy_), prior) <= 24):
                return Here(room.loc, cam, cam[0] + fx_, cam[1] + fy_, score, True)
        bg = room.image[cam[1]:cam[1] + h, cam[0]:cam[0] + w]
        if bg.shape != v.shape:
            return Here(room.loc, cam, fx, fy, score, all(follows))
        drawn = drawn_over(v, room.image, cam) > 60
        drawn[200:] = False  # the field ends there; a notice may be drawn below
        drawn = cv2.morphologyEx(drawn.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((5, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(drawn)
        if near is not None and near.loc == room.loc:
            prior = (near.x - cam[0], near.y - cam[1])
        else:
            prior = (FOLLOW[0], FOLLOW[1])
        tracked = near is not None and near.loc == room.loc and near.sure
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
            if not on_ground(room, cam[0] + feet[0], cam[1] + feet[1]):
                continue  # someone behind a counter, more likely
            if tracked and math.dist(feet, prior) > 64:
                continue  # further than it walks between two looks: someone else
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

    def _steam_pictures(self) -> dict[int, dict[str, tuple[Seen, Seen]]]:
        """Each place's picture padded with black, made ready to match Steam's view on
        (full size and half): softened colour, and its edges (the shapes: places whose
        look Steam changed, the Mountain of Woe's mist and tint). Made on the first
        Steam screen."""
        import cv2

        if self._steam_pics is None:
            pics = {}
            for loc, room in self.rooms.items():
                img = cv2.copyMakeBorder(room.image, STEAM_PAD, STEAM_PAD, STEAM_PAD,
                                         STEAM_PAD, cv2.BORDER_CONSTANT, value=0)
                lit = (img.max(2) > 0).astype(np.uint8)
                seen = cv2.erode(lit, np.ones((5, 5), np.uint8))
                inner = cv2.erode(lit, np.ones((9, 9), np.uint8))  # off the seen's edge
                p, e = self._steam_prep(img), steam_edges(img)
                pics[loc] = {"colour": (Seen(p, seen), Seen(cv2.pyrDown(p), seen[::2, ::2])),
                             "edges": (Seen(e, inner), Seen(cv2.pyrDown(e), inner[::2, ::2]))}
            self._steam_pics = pics
        return self._steam_pics

    def boxed(self, frame, here: Here) -> bool:
        """Whether a text box (or a menu's panel) is over the place: a band across the
        screen's top or bottom where the place doesn't show, all the way across."""
        room = self.rooms.get(here.loc)
        if room is None:
            return False
        rgb = np.asarray(frame.convert("RGB")) if isinstance(frame, Image.Image) else frame
        v = steam_view(rgb) if here.steam else view(rgb)
        h, w = v.shape[:2]
        known = under(room.image, here.cam, h, w)[1:h + 1, 1:w + 1].sum(2) > 30
        cols = known.sum(0) > 0.5 * h  # columns the survey saw most of
        if cols.sum() < 0.4 * w:
            return False
        # (Where the ground drifts, mist differs all over: only a box far off it.)
        e = drawn_over(v, room.image, here.cam)[:, cols] > (90 if room.moving else 45)
        band = e.mean(1) > (0.8 if room.moving else 0.7)
        run, ends = 0, []
        for i, b in enumerate(band):
            run = run + 1 if b else 0
            if run:
                ends.append((run, i))
        if not ends:
            return False
        run, last = max(ends)
        first = last - run + 1
        return run >= 0.15 * h and (first <= 0.15 * h or last >= 0.85 * h)

    def _locate_steam(self, rgb: np.ndarray, near: Here | None) -> Here | None:
        """locate() for the Steam port: around the last view first, else every place;
        in colour, else by the edges (the shapes) when Steam changed a place's look.
        Only what the survey saw counts (Steam's view can go further than the game's
        own: off a place's sides, past where the game's view stops)."""
        import cv2

        v = steam_view(rgb)
        self.dark = float(v.std()) < 6
        if self.dark:
            return None
        b, a, r_, d_ = STEAM_CUT
        ts = {"colour": self._steam_prep(v)[a:STEAM_VIEW[1] - d_, b:STEAM_VIEW[0] - r_],
              "edges": steam_edges(v)[a:STEAM_VIEW[1] - d_, b:STEAM_VIEW[0] - r_]}
        least = {"colour": (STEAM_SCORE, STEAM_MARGIN), "edges": STEAM_EDGES}
        pics = self._steam_pictures()
        if near is not None and near.steam and near.loc in pics:
            x0 = max(0, near.cam[0] + STEAM_PAD + b - STEAM_NEAR)
            y0 = max(0, near.cam[1] + STEAM_PAD + a - STEAM_NEAR)
            for kind, t in ts.items():
                r = pics[near.loc][kind][0].match(t, x0, y0, 2 * STEAM_NEAR, 2 * STEAM_NEAR)
                if r is None:
                    continue
                _, score, _, (x, y) = cv2.minMaxLoc(r)
                if score >= least[kind][0]:
                    cam = (x0 + x - b - STEAM_PAD, y0 + y - a - STEAM_PAD)
                    return self._player_steam(v, self.rooms[near.loc], cam, float(score), near)
        for kind, t in ts.items():
            some = {loc: p[kind] for loc, p in pics.items()}
            hit = self._steam_search(t, some, *least[kind],
                                     twins=lambda a, b, some=some, t=t: twins(some, a, b, t),
                                     prefer=lambda loc: self.prefer(loc, near))
            if hit is not None:
                score, loc, x, y = hit
                cam = (x - b - STEAM_PAD, y - a - STEAM_PAD)
                return self._player_steam(v, self.rooms[loc], cam, score, near)
        return None

    def arriving(self, e: Exit, steam: bool) -> Here | None:
        """Where the view will likely be on coming out of a way out into a surveyed place
        (where the character comes out, the view following it as far as it goes), for
        the first look there to start from."""
        room = self.rooms.get(e.to)
        if room is None or not any(e.at):
            return None
        if steam:
            lo_x, lo_y, hi_x, hi_y = steam_range(room)
            fx, fy = STEAM_FOLLOW
        else:
            lo_x, lo_y, hi_x, hi_y = room.cam_range()
            fx, fy = FOLLOW
        cam = (int(min(max(e.at[0] - fx, lo_x), hi_x)), int(min(max(e.at[1] - fy, lo_y), hi_y)))
        return Here(e.to, cam, e.at[0], e.at[1], 0.0, False, steam)

    def prefer(self, loc: int, near: Here | None) -> float:
        """Which of two places that look the same (a forest in two eras) it more likely
        is, least first: the one it was just in, then one whose ways out lead onto the
        world map last seen (world_hint), its door nearest where it was last seen on it
        (world_spot) first, then the rest."""
        if near is not None and near.loc == loc:
            return 0
        room = self.rooms.get(loc)
        if room is not None and self.world_hint is not None and self.hops(
                loc, self.world_hint) is not None and len(self.hops(loc, self.world_hint)) <= 3:
            if self.world_spot is None:
                return 1
            doors = [e.at for e in room.exits if e.to == self.world_hint and e.at]
            far = min((math.dist(a, self.world_spot) for a in doors), default=5000.0)
            return 1 + min(far, 5000.0) / 10000
        return 2

    @staticmethod
    def _steam_search(t: np.ndarray, pics: dict[int, tuple[Seen, Seen]], low: float,
                      margin: float, twins=None, prefer=None
                      ) -> tuple[float, int, int, int] | None:
        """Every place at half size, then exactly around the best three: the best (score,
        place, x, y on its padded picture) if it scores `low` and beats anywhere else
        (that place further off, the other places) by `margin`. twins(a, b): whether two
        hits (score, place, x, y) are the same picture in two places (eras): they don't
        count against each other, and prefer(place) (least first) picks between them."""
        import cv2

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
        if score < low:
            return None
        same = [f for f in fine[1:] if f[1] != loc and twins is not None
                and score - f[0] < margin and twins(fine[0], f)]
        # Anywhere else in that place: at half size (as good or better than exactly,
        # being softer), off the best by more than 24 pixels.
        r = maps[loc]
        r[max(0, y // 2 - 12):y // 2 + 12, max(0, x // 2 - 12):x // 2 + 12] = -1
        other = max([float(r.max())] + [f[0] for f in fine[1:] if f[1] != loc
                                        and f not in same])
        if score - other < margin:
            return None
        if same and prefer is not None:
            return min([fine[0], *same], key=lambda f: (prefer(f[1]), -f[0]))
        return score, loc, x, y

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

    def seen_now(self, frame, here: Here, who: Person) -> float | None:
        """Where someone who walks about is now across the screen (their x, in the
        place's pixels), by their cut-out (either way round) near where the survey saw
        them; None when they aren't clearly there."""
        import cv2

        f = self.folder / "people" / who.look
        if not who.look or not f.exists():
            return None
        if getattr(self, "_looks", None) is None:
            self._looks: dict[str, list] = {}
        if who.look not in self._looks:
            rgba = np.asarray(Image.open(f).convert("RGBA"))
            bgr = cv2.GaussianBlur(np.ascontiguousarray(rgba[:, :, 2::-1]).astype(np.float32),
                                   (0, 0), 0.7)
            m = np.ascontiguousarray((rgba[:, :, 3] > 0).astype(np.uint8))
            self._looks[who.look] = [(bgr, m), (np.ascontiguousarray(bgr[:, ::-1]),
                                                np.ascontiguousarray(m[:, ::-1]))]
        rgb = np.asarray(frame.convert("RGB")) if isinstance(frame, Image.Image) else frame
        v = steam_view(rgb) if here.steam else view(rgb)
        sx, sy = who.at[0] - here.cam[0], who.at[1] - here.cam[1]
        h, w = v.shape[:2]
        x0, x1 = max(0, int(sx) - 88), min(w, int(sx) + 88)
        y0, y1 = max(0, int(sy) - 60), min(h, int(sy) + 20)
        win = cv2.GaussianBlur(v[y0:y1, x0:x1].astype(np.float32), (0, 0), 0.7)
        # Not the player (and whoever follows close behind).
        px, py = here.x - here.cam[0] - x0, here.y - here.cam[1] - y0
        best, at = 0.5, None
        for spr, m in self._looks[who.look]:
            if spr.shape[0] > win.shape[0] or spr.shape[1] > win.shape[1]:
                continue
            r = cv2.matchTemplate(win, spr, cv2.TM_CCOEFF_NORMED, mask=m)
            r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
            fy = np.arange(r.shape[0])[:, None] + spr.shape[0]  # their feet there
            fx = np.arange(r.shape[1])[None, :] + spr.shape[1] / 2
            r[(np.abs(fy - py) < 12) & (np.abs(fx - px) < 12)] = -1
            _, sc, _, (x, _y) = cv2.minMaxLoc(r)
            if sc > best:
                best, at = sc, x0 + x + spr.shape[1] / 2 + here.cam[0]
        return at

    def _steam_feet(self, v: np.ndarray, box: tuple[float, float, float, float] | None = None,
                    n: int = 4, skip: list[tuple[float, float]] = ()
                    ) -> list[tuple[float, tuple[int, int]]]:
        """Where the player's cut-outs fit Steam's view best, as (score, feet), the
        best few at least a body apart; feet only within `box` (x0, y0, x1, y1) when
        given, which is quicker; none at `skip` (where people who don't walk about
        stand: someone dressed like the player, seen from behind, fits as well)."""
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
        for sx, sy in skip:
            lx, ly = round(sx) - ox, round(sy) - oy
            if -6 <= lx < vw + 6 and -8 <= ly < vh + 8:
                best[max(0, ly - 8):max(0, ly + 9), max(0, lx - 6):max(0, lx + 7)] = -1
        out = []
        for _ in range(n):
            _, s, _, (x, y) = cv2.minMaxLoc(best)
            if s <= 0:
                break
            out.append((float(s), (x + ox, y + oy)))
            best[max(0, y - 20):y + 20, max(0, x - 12):x + 12] = -1
        return out

    def _player_steam(self, v: np.ndarray, room: Room, cam: tuple[int, int], score: float,
                      near: Here | None = None, steam: bool = True) -> Here:
        """_player() on Steam's view: Steam draws people as the game does, softened,
        but not in the game's colours, so the player is found by its shape (the
        cut-outs, softened alike): a clear match, or a weaker one where it was last.
        steam=False: on the emulator's own view, for places whose ground drifts (what's
        drawn over the bare place is everywhere there)."""
        follow = STEAM_FOLLOW if steam else FOLLOW
        x0, y0, x1, y1 = steam_range(room) if steam else room.cam_range()
        fx, fy = cam[0] + follow[0], cam[1] + follow[1]
        moving = (x0 + 1 < cam[0] < x1 - 1, y0 + 1 < cam[1] < y1 - 1)
        # Surely following, well inside (where Steam's view stops is known only to a
        # tile or so): the player is looked for only near where it keeps it.
        follows = (x0 + TILE < cam[0] < x1 - TILE, y0 + TILE < cam[1] < y1 - TILE)
        same = near is not None and near.loc == room.loc
        prior = (near.x - cam[0], near.y - cam[1]) if same else follow
        best = None
        # Around where it was when it was seen, or where the view keeps it while it
        # follows: quicker, and no one else is taken for it.
        h, w = v.shape[:2]
        if same and near.sure and near.steam == steam:
            box = (prior[0] - STEAM_NEAR, prior[1] - STEAM_NEAR,
                   prior[0] + STEAM_NEAR, prior[1] + STEAM_NEAR)
        else:
            sx, sy = follow
            box = (sx - 24 if follows[0] else 0, sy - 28 if follows[1] else 0,
                   sx + 24 if follows[0] else w, sy + 28 if follows[1] else h)
        for s, feet in self._steam_feet(v, box, skip=standing(room, cam)):
            far = math.dist(feet, prior)
            if s < 0.6 and not (s >= 0.4 and far <= 24):
                continue
            if (follows[0] and abs(feet[0] - follow[0]) > 24) or (
                    follows[1] and abs(feet[1] - follow[1]) > 28):
                continue  # not where the view keeps it
            # Feet on ground the survey walked, rather than off it.
            tile = (int(cam[1] + feet[1]) // TILE, int(cam[0] + feet[0]) // TILE)
            cost = far + 100 * (1 - s) + (0 if tile in room.walk else 40)
            if best is None or cost < best[0]:
                best = (cost, feet)
        if best is not None:
            return Here(room.loc, cam, cam[0] + best[1][0], cam[1] + best[1][1], score, True,
                        steam)
        if same and not all(moving):
            return Here(room.loc, cam, fx if moving[0] else near.x,
                        fy if moving[1] else near.y, score, False, steam)
        return Here(room.loc, cam, fx, fy, score, all(follows), steam)

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

    def targets(self, start: int, here: Here | None = None, reach: int = 2
                ) -> dict[str, Target]:
        """Things and people to go to, by name: in this place ("the innkeeper (in
        here)"), and in places up to `reach` ways out away ("the mayor (Mayor's Manor,
        upstairs)"), not on world maps."""
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= reach or u not in self.rooms:
                continue
            for e in self.rooms[u].exits:
                if e.to not in dist and e.to not in self.worlds:
                    dist[e.to] = dist[u] + 1
                    q.append(e.to)
        out: dict[str, Target] = {}
        for loc in sorted(dist, key=dist.get):
            room = self.rooms.get(loc)
            if room is None:
                continue
            where = "in here" if loc == start else self.name(loc)
            for p in room.people:
                key = f"{p.name} ({where})"
                if key not in out:  # walked up to and faced (stood on: a save point)
                    out[key] = Target(loc, p.at, p.kind != "save point", p.name, p.kind)
            for k, v in room.spots.items():
                key = f"{k} ({where})"
                if key not in out:
                    out[key] = Target(loc, v, False, k, "spot")
        return out

    def options(self, here: Here) -> tuple[dict[str, tuple[int, int]], dict[str, Target],
                                           dict[str, str]]:
        """Everywhere the player can be taken from here, by name: (location, how many
        ways out away); which of those are things or people (Targets); and which way
        each is from the player, in words."""
        choices = self.reachable(here.loc)
        targets = {k: t for k, t in self.targets(here.loc).items() if k not in choices}
        for k, t in targets.items():
            hops = self.hops(here.loc, t.loc)
            choices[k] = (t.loc, len(hops) - 1 if hops else 0)
        room = self.rooms[here.loc]
        hints: dict[str, str] = {}
        for k, (loc, _n) in choices.items():
            ways = [e for e in room.exits if e.label == k]
            if k in targets:
                hints[k] = self.hint(here, loc, targets[k].at)
            elif ways:
                c = min((centre(e.frm) for e in ways),
                        key=lambda c: math.dist(c, (here.x, here.y)))
                hints[k] = (f"{math.dist(c, (here.x, here.y)) / TILE:.0f} steps "
                            f"{compass(c[0] - here.x, c[1] - here.y)}")
            else:
                hints[k] = self.hint(here, loc)
        return choices, targets, hints

    def hint(self, here: Here, loc: int, at: tuple[float, float] | None = None) -> str:
        """Which way and how far something is from the player, in words: "6 steps
        north" in this place; "through <way>" for another place."""
        room = self.rooms.get(here.loc)
        if room is None:
            return ""
        if loc == here.loc and at is not None:
            steps = math.dist(at, (here.x, here.y)) / TILE
            if steps < 1:
                return "right here"
            return f"{steps:.0f} steps {compass(at[0] - here.x, at[1] - here.y)}"
        hops = self.hops(here.loc, loc)
        if not hops or len(hops) < 2:
            return ""
        ways = [e for e in room.exits if e.to == hops[1]]
        if not ways:
            return ""
        e = min(ways, key=lambda e: math.dist(centre(e.frm), (here.x, here.y)))
        c = centre(e.frm)
        steps = math.dist(c, (here.x, here.y)) / TILE
        return (f"way out {steps:.0f} steps {compass(c[0] - here.x, c[1] - here.y)}"
                + (f", then {len(hops) - 2} more" if len(hops) > 2 else ""))

    def stand_by(self, room: Room, at: tuple[float, float], person: bool
                 ) -> dict[tuple[int, int], tuple[int, int]]:
        """Where to stand to reach something, and the way to face it: (dx, dy). Someone
        is talked to from a tile straight beside them, or across a counter (two tiles
        off, with no ground between); a thing from the nearest walked tile."""
        out: dict[tuple[int, int], tuple[int, int]] = {}
        for tier in self.stand_tiers(room, at, person):
            out.update(tier)
        return out

    def stand_tiers(self, room: Room, at: tuple[float, float], person: bool
                    ) -> list[dict[tuple[int, int], tuple[int, int]]]:
        """stand_by's places, best first: for someone, right beside them; then across a
        counter in front of them (below: shopkeepers face the room); then across
        anything else."""
        tr, tc = int(at[1]) // TILE, int(at[0]) // TILE
        out: dict[tuple[int, int], tuple[int, int]] = {}
        if person:
            beside: dict[tuple[int, int], tuple[int, int]] = {}
            front: dict[tuple[int, int], tuple[int, int]] = {}
            across: dict[tuple[int, int], tuple[int, int]] = {}
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                one = (tr + dy, tc + dx)
                two = (tr + 2 * dy, tc + 2 * dx)
                if one in room.walk and one not in room.learned:
                    beside[one] = (-dx, -dy)
                elif two in room.walk and two not in room.learned:
                    (front if dy == 1 else across)[two] = (-dx, -dy)
            tiers = [t for t in (beside, front, across) if t]
            if tiers:
                return tiers
        near = sorted(room.walk, key=lambda t: math.dist(centre(t), at))
        if not near:
            return out
        # The nearest few (any side of a counter: whichever the way reaches first).
        least = math.dist(centre(near[0]), at)
        for end in near[:4]:
            if math.dist(centre(end), at) > least + TILE / 2:
                break
            cx, cy = centre(end)
            dx, dy = at[0] - cx, at[1] - cy
            out[end] = (0, 1 if dy > 0 else -1) if abs(dy) >= abs(dx) else \
                (1 if dx > 0 else -1, 0)
        return [out] if out else []

    def route(self, room: Room, start: tuple[int, int], goals: set[tuple[int, int]]
              ) -> list[tuple[int, int]] | None:
        """Tiles from start to any goal tile over the place's walkable tiles (A*)."""
        def shut(t):
            if t == start or t in goals:
                return False
            return t in room.learned or t not in room.walk

        # Other ways out are kept clear of: a step onto one can take you through it.
        doors = {e.frm for e in room.exits} - goals

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
                    c = cost[u] + (1.4 if dr and dc else 1.0) + 0.3 * side + \
                        (6.0 if v in doors else 0.0) + (3.0 if v in room.crowd else 0.0)
                    if c < cost.get(v, 1e9):
                        cost[v] = c
                        came[v] = u
                        h = min(abs(v[0] - g[0]) + abs(v[1] - g[1]) for g in goals)
                        heapq.heappush(pq, (c + h, v))
        return None


def twins(pics: dict[int, tuple[Seen, Seen]], a, b, t: np.ndarray) -> bool:
    """Whether two hits (score, place, x, y) on the places' pictures show the same
    picture: one place surveyed in two eras (Guardia Forest in 1000 and 600 A.D.)."""
    h, w = t.shape[:2]
    _, la, xa, ya = a
    _, lb, xb, yb = b
    pa = pics[la][0].i[ya:ya + h, xa:xa + w].astype(np.float32)
    pb = pics[lb][0].i[yb:yb + h, xb:xb + w].astype(np.float32)
    if pa.shape != pb.shape or pa.size == 0:
        return False
    pa, pb = pa.ravel() - pa.mean(), pb.ravel() - pb.mean()
    d = float(np.sqrt((pa @ pa) * (pb @ pb)))
    return d > 0 and float(pa @ pb) / d >= 0.9


def came_up(before, after) -> bool:
    """Whether something came up on screen between two screens (a text box, a shop):
    much of it changed, the character standing still."""
    a, b = (np.asarray((f if isinstance(f, Image.Image) else Image.fromarray(f))
                       .convert("L").resize((160, 90))).astype(np.int16)
            for f in (before, after))
    return float((np.abs(a - b) > 24).mean()) > 0.06


def drawn_over(v: np.ndarray, image: np.ndarray, cam: tuple[int, int]) -> np.ndarray:
    """How far each pixel of a view found at `cam` is from the bare place under it (the
    sum of its colours' differences), the nearest of the place's pixels a pixel either
    way: a screen captured scaled is off by a pixel here and there, which on stone walls
    would differ everywhere; people drawn over it differ from all round them."""
    h, w = v.shape[:2]
    pad = under(image, cam, h, w)
    vi = v.astype(np.int16)
    out = None
    for dy in range(3):
        for dx in range(3):
            e = np.abs(vi - pad[dy:dy + h, dx:dx + w]).sum(2)
            out = e if out is None else np.minimum(out, e)
    return out


def under(image: np.ndarray, cam: tuple[int, int], h: int, w: int) -> np.ndarray:
    """The place's picture under a view at `cam`, a pixel more all round (edges
    repeated off the picture)."""
    x0, y0 = cam[0] - 1, cam[1] - 1
    H, W = image.shape[:2]
    ys = np.clip(np.arange(y0, y0 + h + 2), 0, H - 1)
    xs = np.clip(np.arange(x0, x0 + w + 2), 0, W - 1)
    return image[ys[:, None], xs[None, :]].astype(np.int16)


def in_the_way(room: Room, here: Here, ahead: tuple[int, int],
               goals: set[tuple[int, int]]) -> Person | None:
    """Someone the survey saw standing where the way is shut (by the tile pushed into,
    or beside the character), to say who's in the way."""
    c = centre(ahead)
    for p in room.people:
        if p.kind in ("person", "animal") and (
                math.dist(p.at, c) < 24 or math.dist(p.at, (here.x, here.y)) < 26):
            return p
    return None


async def wait_for_way(room: Room, blocker: Person | None, waits: int, steer, log) -> str:
    """No way left but through what was found shut: wait a moment for whoever is in the
    way to move, and look again (the shut tiles forgotten). After a few waits: what to
    tell the player ("" while still waiting)."""
    steer(None)
    who = blocker.name if blocker is not None else "someone"
    if waits >= 3:
        return f"{who} is standing in"
    log(f"rooms: the way is shut ({who} in the way?): waiting a moment")
    await asyncio.sleep(2.0)
    room.learned.clear()
    return ""


async def step_aside(rooms: Rooms, room: Room, here: Here, a: float, n: int,
                     goals: set[tuple[int, int]], steer) -> None:
    """A short step to one side of the way it's heading (one side then the other, by
    `n`), round someone in the way: not onto another way out (it could go through), and
    not at all when both sides are."""
    doors = {e.frm for e in room.exits} - goals
    for turn in ((90, -90) if n % 2 else (-90, 90)):
        b = (a + turn) % 360
        rad = math.radians(b)
        side = (int((here.y - 14 * math.cos(rad)) // TILE),
                int((here.x + 14 * math.sin(rad)) // TILE))
        if side not in doors:
            steer(b)
            await asyncio.sleep(0.1 if n <= 2 else 0.3)  # a nudge first (lining up)
            steer(None)
            return


def centre(t: tuple[int, int]) -> tuple[float, float]:
    """Where to stand on a tile: across, its middle; down, low on it (the feet come to
    rest at a tile's bottom edge when the one above is shut)."""
    return t[1] * TILE + TILE / 2, t[0] * TILE + 13


def on_ground(room: Room, x: float, y: float) -> bool:
    """Whether feet at x, y are on (or at the edge of) a tile the survey walked."""
    return not room.walk or any((int(y + dy) // TILE, int(x + dx) // TILE) in room.walk
                                for dx in (-4, 0, 4) for dy in (-3, 0, 3))


def standing(room: Room, cam: tuple[int, int]) -> list[tuple[float, float]]:
    """Where the people who don't walk about stand, in a view at `cam` (not save points
    and the like: those are walked over)."""
    return [(p.at[0] - cam[0], p.at[1] - cam[1]) for p in room.people
            if not p.moves and p.kind in ("person", "animal")]


def line_up(t: tuple[int, int], at: tuple[float, float], face: tuple[int, int]) -> float:
    """Where across the way it faces to stand on tile t to talk to someone at `at`: in
    line with them (the game answers only within about 10 px of straight on), kept on
    the tile."""
    if face[1]:  # facing up or down: an x
        lo = t[1] * TILE
        return min(max(at[0], lo + 3), lo + TILE - 3)
    lo = t[0] * TILE
    return min(max(at[1], lo + 3), lo + TILE - 3)


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


FIGHT_STARTED = ("interrupted: a fight started on the way (the party's HP box is up): win "
                 "or run it with the battle tools, then go on")


BOX_UP = ("interrupted: a text box (or a menu) is up: read it to the player, press a to go "
          "on, then go_to_exit again")


def over_place(image) -> str:
    """What the game drew over the place on a Steam frame, by its windows and font: "fight"
    (the party's HP box), "box" (a text box, a choice or a menu), or "" (nothing)."""
    from . import ctscreen

    if image is None:
        return ""
    kind = ctscreen.look(image).kind
    return "fight" if kind == "battle" else "box" if kind in ("text", "choice", "menu") else ""


def fighting(image) -> bool:
    """Whether a Steam frame shows a fight (the party's HP box, a battle list or glove)."""
    return over_place(image) == "fight"


async def walk_out(rooms: Rooms, goal: int, shoot, steer, keep_going, log,
                   limit: float = 90.0, start: Here | None = None, label: str = "",
                   spot: tuple[float, float] | Target | None = None, press=None,
                   talk: bool = False) -> str:
    """Walk to the place `goal` (a location number; a world map for "outside") through
    the ways out, place by place, with the screen alone: where the character is comes
    from the pictures several times a second. label: which way out into it, by its name
    (a world map's sides). spot: a place in `goal` to walk to (in its pixels), or a
    Target there (someone is walked up to and faced), not a way into it. talk: press
    a (`press`) facing them, to talk (or to open, search, use)."""
    t0 = time.monotonic()
    person = isinstance(spot, Target) and spot.person
    # A save point is stood on. Pressing a there does nothing in Chrono Trigger (live on
    # Steam: a on the Mountain of Woe's sparkle brought nothing up, a step away it
    # climbed the chain beside it, 232 s and no save): saving is the menu's Save.
    saving = isinstance(spot, Target) and spot.kind == "save point"
    at = (spot.at if isinstance(spot, Target) else spot) if spot is not None else None
    # Someone who walks about (behind a counter): found on screen at the end, a was
    # pressed when they come across from the player.
    who = next((p for p in rooms.rooms[spot.loc].people if p.at == spot.at and p.moves
                and p.look), None) if person and spot.loc in rooms.rooms else None
    pressed = waited = 0
    settled = False  # stopped on a tile to stand on, at the end
    lines = 0  # small steps taken to line up with someone, at the end
    blocker: Person | None = None  # someone the survey saw standing in the way
    waits = 0  # times waited for the way to clear
    here = start
    misses = 0
    gone_by: Exit | None = None  # the way out last walked into
    heading: Exit | None = None  # the way out being walked to
    target = (spot.name if isinstance(spot, Target) and spot.name else
              label or rooms.name(goal))
    stalls = 0
    gone_t = said = stuck_t = 0.0
    faded = False  # the screen went black since the place was last seen
    dud: set = set()  # ways out walked into that didn't go through
    trail: deque = deque()  # (when, place, x, y) over the last seconds
    best: tuple[int, int, float] = (-1, 10**6, 0.0)  # place, fewest tiles to go, since when

    coming: Here | None = None  # where a way out just taken comes out, to look there first

    shot = None  # the screen last looked at
    boxes = 0  # looks in a row with a text box (or a menu) over the place
    unboxed: set[int] = set()  # places where the survey's look stands for a box (Steam)

    async def answered(before) -> bool:
        """Whether pressing a brought something up (a text box can take a moment)."""
        if came_up(before, await shoot()):
            return True
        await asyncio.sleep(0.5)
        return came_up(before, await shoot())

    async def look() -> Here | None:
        nonlocal shot
        shot = await shoot()
        return await asyncio.to_thread(rooms.locate, shot, coming or here)

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
                    if coming is None and here is not None:
                        coming = rooms.arriving(way, here.steam)
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
            if now.loc not in unboxed and await asyncio.to_thread(rooms.boxed, shot, now):
                # The place shows, but a text box is over part of it (a scene, someone
                # who stopped the player): walking can't go on till it's read.
                steer(None)
                boxes += 1
                if boxes >= 3:
                    if now.steam and not await asyncio.to_thread(over_place, shot):
                        # Steam drew part of this place unlike the survey (the fair's
                        # balustrade: "a text box is up" four times, none there). The
                        # game's windows decide; a box that does come up stops the
                        # walk as being stuck.
                        unboxed.add(now.loc)
                        boxes = 0
                        continue
                    return BOX_UP
                await asyncio.sleep(0.15)
                continue
            boxes = 0
            faded = False
            misses = 0
            coming = None
            if here is not None and now.loc != here.loc:
                log(f"rooms: now in {rooms.name(now.loc)}")
                gone_by = None
                stalls = 0
                if now.loc == goal and at is None:
                    steer(None)
                    return f"arrived: in {target} ({time.monotonic() - t0:.0f} s)"
            here = now
            room = rooms.rooms[here.loc]
            # Where it will be when a turn starts: on Steam, input lags and the look
            # takes a while, so it's walked on a little by then (turns start early).
            px, py = lead(here, trail)
            tap = STEAM_TAP if here.steam else 0.06
            if here.loc == goal and at is not None:
                # The last of it: to a tile to stand on by it, facing it.
                tiers = rooms.stand_tiers(room, at, person)
                tile = here.tile
                if tile not in room.walk:
                    tile = min(room.walk, key=lambda t: abs(t[0] - tile[0]) + abs(t[1] - tile[1]))
                if not tiers:
                    steer(None)
                    return f"blocked: nowhere to stand by {target}"
                # The best places to stand that the way reaches.
                path, ends = None, tiers[0]
                for tier in tiers:
                    path = rooms.route(room, tile, set(tier))
                    if path is not None:
                        ends = tier
                        break
                if tile in ends and not settled:
                    # Stop first and look again: still walking, it would slide on past
                    # while it lines up (the look was from a moment before).
                    settled = True
                    steer(None)
                    await asyncio.sleep(STEAM_LEAD if here.steam else 0.15)
                    trail.clear()
                    continue
                if tile in ends and saving:
                    steer(None)
                    return SAVE_POINT.format(secs=f"{time.monotonic() - t0:.0f}")
                if tile in ends:
                    fx, fy = ends[tile]
                    aim, seen = at, None
                    if who is not None and fy and shot is not None:
                        seen = await asyncio.to_thread(rooms.seen_now, shot, here, who)
                        if seen is not None:
                            aim = (seen, at[1])
                    # Someone is talked to straight on: lined up across the way it faces.
                    off = line_up(tile, aim, (fx, fy)) - (here.x if fy else here.y)
                    if person and abs(off) > (3 if pressed else 5) and lines < 8:
                        lines += 1
                        steer(angle_to(off, 0) if fy else angle_to(0, off))
                        # (on the emulator, half a step when it's close)
                        await asyncio.sleep(tap / 2 if abs(off) < 8 and not here.steam else tap)
                        steer(None)
                        if here.steam:  # till the step shows
                            await asyncio.sleep(STEAM_LEAD)
                            trail.clear()
                        continue
                    steer(angle_to(fx, fy))
                    await asyncio.sleep(max(0.08, tap))
                    steer(None)
                    if talk and press is not None and who is not None:
                        if seen is not None and abs(seen - here.x) > 9 and waited < 24:
                            waited += 1  # not across from the player yet: wait for them
                            await asyncio.sleep(0.2)
                            continue
                        await asyncio.sleep(STEAM_LEAD if here.steam else 0.1)
                        before = await shoot()
                        press("a")
                        await asyncio.sleep(0.6)
                        if await answered(before):
                            return (f"arrived: at {target}, facing them, and pressed a to "
                                    f"talk ({time.monotonic() - t0:.0f} s): read their line "
                                    "to the player word for word; a question in it (rest? "
                                    "buy?) is the player's to answer")
                        pressed += 1
                        log(f"rooms: nothing came up ({pressed}): {target} walks about; "
                            "lining up again")
                        if pressed >= 10:  # about 10 s: they come by now and then
                            return (f"arrived: at {target} ({time.monotonic() - t0:.0f} s), "
                                    "but they walk about and pressing a as they came by "
                                    "brought nothing up: look")
                        lines = 0
                        continue
                    if talk and press is not None:
                        # Pressed straight on, then a step nearer (someone without a
                        # counter between answers only from right beside them), then a
                        # little to each side: a shop's counter answers within ~10 px of
                        # where the game has it, which isn't always where they're drawn.
                        base = line_up(tile, at, (fx, fy))
                        for k, side in enumerate((0, 0, -5, 5, -10, 10)):
                            if k == 1:
                                log("rooms: nothing came up: a step nearer")
                                steer(angle_to(fx, fy))
                                await asyncio.sleep(tap)
                                steer(None)
                            elif k > 1:
                                log(f"rooms: nothing came up: {abs(side)} px "
                                    f"{'back' if side < 0 else 'on'} along the way across")
                                for _ in range(4):
                                    cur = await look()
                                    if cur is None or cur.loc != here.loc:
                                        break
                                    d = base + side - (cur.x if fy else cur.y)
                                    if abs(d) <= 2:
                                        break
                                    steer(angle_to(d, 0) if fy else angle_to(0, d))
                                    await asyncio.sleep(tap / 2 if abs(d) < 8 and not
                                                        here.steam else tap)
                                    steer(None)
                                    await asyncio.sleep(STEAM_LEAD if here.steam else 0.05)
                                steer(angle_to(fx, fy))
                                await asyncio.sleep(max(0.08, tap))
                                steer(None)
                            await asyncio.sleep(STEAM_LEAD if here.steam else 0.1)
                            before = await shoot()
                            press("a")
                            await asyncio.sleep(0.6)
                            if await answered(before):
                                return (f"arrived: at {target}, facing them, and pressed a "
                                        f"to talk ({time.monotonic() - t0:.0f} s): read their "
                                        "line to the player word for word; a question in it "
                                        "(rest? buy?) is the player's to answer")
                        return (f"arrived: at {target} ({time.monotonic() - t0:.0f} s), "
                                "facing them, but pressing a brought nothing up: look")
                    return f"arrived: at {target} ({time.monotonic() - t0:.0f} s)"
                settled = False
                goals = set(ends)
                if path is None and room.learned:
                    if (said_wait := await wait_for_way(room, blocker, waits, steer, log)):
                        return f"blocked: {said_wait} the way to {target}"
                    waits += 1
                    continue
                if path is None:
                    steer(None)
                    return f"blocked: no way to {target} from here"
                toward = target
            else:
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
                    # Shut off by what was found shut on the way (people move): a moment
                    # for them to, then try again.
                    if (said_wait := await wait_for_way(room, blocker, waits, steer, log)):
                        return f"blocked: {said_wait} the way to {rooms.name(hops[1])}"
                    waits += 1
                    continue
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
                toward = rooms.name(hops[1])
            # Aim a little way along the path (several tiles when it runs straight).
            aim = centre(path[1]) if len(path) > 1 else centre(path[-1])
            for t in path[1:4]:
                c = centre(t)
                if math.dist((px, py), c) > 12:
                    aim = c
                    break
            last = here.loc == goal and at is not None and len(path) == 2
            if last:  # onto where to stand: its middle (clear of whoever's beside it)
                aim = centre(path[1])
            a = angle_to(aim[0] - px, aim[1] - py)
            steer(a)
            if last:
                # A short step at a time: walking on, it would slide past while it looks.
                await asyncio.sleep(tap)
                steer(None)
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
                over = await asyncio.to_thread(over_place, shot) if here.steam else ""
                if over:
                    steer(None)
                    return FIGHT_STARTED if over == "fight" else BOX_UP
                stalls += 1
                if path[1] not in goals:
                    room.learned.add(path[1])
                blocker = in_the_way(room, here, path[1], goals) or blocker
                log(f"rooms: no nearer for 6 s at {tile} in {rooms.name(here.loc)}, shut "
                    f"{path[1]} ({stalls}){f' by {blocker.name}' if blocker else ''}")
                await step_aside(rooms, room, here, a, stalls, goals, steer)
                best = (here.loc, 10**6, now_t)
                if stalls > 12:
                    return (f"blocked: can't get through to {toward} from "
                            f"{tile} in {rooms.name(here.loc)}")
                continue
            trail.append((now_t, here.loc, here.x, here.y))
            while trail and now_t - trail[0][0] > 2.0:
                trail.popleft()
            if now_t - stuck_t > 2.0 and now_t - trail[0][0] > 1.6 and all(
                    p[1] == here.loc and math.dist(p[2:], (here.x, here.y)) < 5 for p in trail):
                # A fight on the field looks like the place, the party standing still
                # (live: 13 "stuck" steps in a Guardia Forest fight, then "blocked").
                over = await asyncio.to_thread(over_place, shot) if here.steam else ""
                if over:
                    steer(None)
                    return FIGHT_STARTED if over == "fight" else BOX_UP
                stalls += 1
                rad = math.radians(a)
                ahead = (int((here.y - 12 * math.cos(rad)) // TILE),
                         int((here.x + 12 * math.sin(rad)) // TILE))
                if ahead in (here.tile, tile) and len(path) > 1:
                    ahead = path[1]  # (the tile it stands on, as the survey has it)
                if ahead not in goals:
                    room.learned.add(ahead)
                blocker = in_the_way(room, here, ahead, goals) or blocker
                log(f"rooms: stuck at {tile} in {rooms.name(here.loc)}, shut {ahead} "
                    f"({stalls}){f' by {blocker.name}' if blocker else ''}")
                # A step aside, one way then the other: round someone in the way.
                await step_aside(rooms, room, here, a, stalls, goals, steer)
                stuck_t = now_t
                trail.clear()
                if stalls > 12:
                    return (f"blocked: can't get through to {toward} from "
                            f"{tile} in {rooms.name(here.loc)}")
            await asyncio.sleep(0.02)
    finally:
        steer(None)
    where = rooms.name(here.loc) if here else "somewhere unknown"
    return f"timeout: still in {where}, on the way to {target}"


SAVE_POINT = ("arrived: standing on the save point ({secs} s). Don't press a here: a save "
              "point doesn't answer a in this game, and a by something beside it uses that "
              "instead (a chain is climbed). To save, the menu: y opens it, and its Save is "
              "lit while standing on a save point; then a slot, picked by the player. Only "
              "if they asked to save; else tell them we're on it and how to save")
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
    for w in ("upstairs", "downstairs"):  # "go upstairs": the one floor called that
        if w in q and not q - {w, "go", "take", "me", "back", "please", "the", "let",
                               "us", "lets", "head", "walk"}:
            hits = [k for k, v in choices.items() if w in v.lower() and "(" not in v]
            if len(hits) == 1:
                return hits[0]
    # Someone or something by its name ("talk to the innkeeper"): when just one fits.
    want = bare(re.sub(r"^\s*(please\s+)?((take|bring|get) me|go|walk|run|head|talk|"
                       r"speak|chat)\s+(over\s+)?(to|with)\s+", "", said.lower()))
    hits = [k for k, v in choices.items() if bare(v.split(" (")[0]) == want]
    return hits[0] if len(hits) == 1 and want else None


def bare(name: str) -> str:
    """A name without "the" or "a" in front or a full stop after."""
    return re.sub(r"^(the|a|an|my)\s+", "", name.lower().strip(" .!?"))


PICK_EXIT = """The player is inside {here} in {game} and says: "{said}".
Where they can go from here: ways out to other places (by how many ways out away), and
things and people in this place or the places next to it (which place in brackets), with
which way each is from the player:
{places}
Which one do they mean? "outside", "the way out", "leave" mean the world map unless
they name a place (with more than one way outside and none of them named or meant, "outside,
the nearest way out"). "Upstairs" or "downstairs" is the floor of the same building; "the
north door" or "the door on the left" is the way out that way from the player. "The
shop" or "buy something" in a shop is its shopkeeper or counter; "the inn" or "rest"
in an inn, its innkeeper; "save" or "the save point", the save point. Someone named or
described ("the old man", "the guard", "Mom") is the person who fits best. JSON:
{{"place": "the number before the name", "talk": true if they want to talk to, ask, buy
from or use it (not just go there), "why": "a few words"}} or {{"place": null, "why":
"..."}} if none fits."""


async def pick_exit(muse, game: str, said: str, here: str,
                    choices: dict[str, tuple[int, int]], hints: dict[str, str] | None = None
                    ) -> tuple[str | None, str]:
    """The place (or thing, or person) the player means, among those reachable (by name:
    location, how far), hints saying which way each is: (its name, why), or (None, why
    none fits). why starts with "talk:" when they want to talk to it."""
    hit = pick_words(said, {k: k for k in choices})
    if hit is not None:
        return hit, "named"
    names = sorted(choices, key=lambda k: choices[k][1])
    hints = hints or {}
    lines = "\n".join(f"{i + 1}: {k} ({choices[k][1]} away"
                      f"{'; ' + hints[k] if hints.get(k) else ''})"
                      for i, k in enumerate(names))
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
    return names[i - 1], ("talk: " if data.get("talk") is True else "") + why


TALK = re.compile(r"\b(talk|speak|chat|ask|tell|buy|sell|shop|rest|sleep|stay|"
                  r"save(?! point)|open|search|check|read|examine|use)\b", re.IGNORECASE)


PLACE_KINDS = {"residence": "house"}


def plain_place(name: str) -> str:
    """A world map's name for a place, the way it's said: "Truce Residence (west)" is
    "the west house in Truce", "Medina Residence" "the house in Medina"; others as they
    are ("Truce Market", "Last Village Hut (west)")."""
    m = re.fullmatch(r"(?P<town>[A-Z][\w']*) (?P<kind>\w+)(?: \((?P<side>[a-z -]+)\))?",
                     name.strip())
    if m is None or m.group("kind").lower() not in PLACE_KINDS:
        return name
    kind = PLACE_KINDS[m.group("kind").lower()]
    side = f"{m.group('side')} " if m.group("side") else ""
    return f"the {side}{kind} in {m.group('town')}"


def describe(rooms: Rooms, here: Here) -> str:
    """Where the player is in a place, said plainly: the ways out, and the people and
    things in it, by direction."""
    room = rooms.rooms[here.loc]
    parts = []
    for lab in sorted({e.label or rooms.name(e.to) for e in room.exits}):
        e = min((e for e in room.exits if (e.label or rooms.name(e.to)) == lab),
                key=lambda e: math.dist(centre(e.frm), (here.x, here.y)))
        c = centre(e.frm)
        steps = math.dist(c, (here.x, here.y)) / TILE
        parts.append(f"{lab}: {steps:.0f} steps {compass(c[0] - here.x, c[1] - here.y)}")
    out = (f"in {rooms.name(here.loc)}. Ways out: " + "; ".join(parts)) if parts else \
        f"in {rooms.name(here.loc)}; no ways out surveyed"
    near = [f"{p.name} ({rooms.hint(here, here.loc, p.at)}"
            f"{', walks about' if p.moves else ''})" for p in room.people]
    near += [f"{k} ({rooms.hint(here, here.loc, v)})" for k, v in room.spots.items()]
    if near:
        out += ". Here: " + "; ".join(near)
    return out
