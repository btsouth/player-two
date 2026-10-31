"""Where the player is on a game's maps, and the way to a named place, from map images.

Telling world-map icons apart and knowing a game's geography is where a vision model
fails (it takes a village for the castle; it puts the fair west of town when it's north).
A sighted friend would look at a map. So does this: an atlas is a game's map images
(fan-made rips of each world map, with the places named) and where each place is on
them, kept in ~/.config/player-two/atlas/<game>/atlas.json, made once per game. The
player's view is found on a map by plain image matching (the screen is a window onto
the map), the way there is planned on the map's ground, and walking checks where it is
several times a second.

atlas.json:
    {"native": [256, 224],              # the game's own screen size, in map pixels
     "maps": [{"name": "1000 A.D.", "image": "w1000.png", "height": 1024,
               "places": {"Leene Square": [422, 216], ...},
               "open": [[x0, y0, x1, y1], ...]}]}  # bridges and stairs, walkable
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CONFIG_DIR

ATLASES = CONFIG_DIR / "atlas"
CELL = 8  # route planning grid, in map pixels (a world-map step is about this)


def slug(name: str) -> str:
    words = re.findall(r"[a-z0-9']+", re.sub(r"\(.*?\)", "", name.lower()))
    return "-".join(w.replace("'", "") for w in words)


@dataclass
class Spot:
    map: str
    x: float
    y: float
    score: float  # how well the screen matched there, 0 to 1
    margin: float  # over the best match elsewhere on that map


@dataclass
class WorldMap:
    name: str
    image: np.ndarray  # BGR, the map only (no legend)
    places: dict[str, tuple[float, float]]
    blocked: np.ndarray = field(default=None)  # type: ignore[assignment]  # per CELL
    learned: set[tuple[int, int]] = field(default_factory=set)  # cells found shut

    def __post_init__(self) -> None:
        if self.blocked is None:
            self.blocked = ground(self.image)

    def nearest(self, x: float, y: float) -> tuple[str, float, str]:
        """The named place nearest a point: its name, distance in map pixels, and the
        compass way from the point to it."""
        best = min(self.places.items(), key=lambda kv: math.dist(kv[1], (x, y)))
        name, (px, py) = best
        return name, math.dist((px, py), (x, y)), compass(px - x, py - y)


def compass(dx: float, dy: float) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "here"
    a = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    return ["north", "northeast", "east", "southeast", "south", "southwest", "west",
            "northwest"][int((a + 22.5) // 45) % 8]


def ground(img: np.ndarray) -> np.ndarray:
    """Which CELL squares can't be walked: water, rock faces and thick forest, from the
    map's colours. Rough; walking corrects it (see WorldMap.learned)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    water = (h > 95) & (h < 130) & (s > 80)
    rock = (h < 20) & (s > 90) & (v > 60) & (v < 200)
    # Forest and grass are the same green; forest is speckled with dark shade (water
    # is never that dark).
    shade = v < 50
    gh, gw = img.shape[0] // CELL, img.shape[1] // CELL

    def cell(m: np.ndarray) -> np.ndarray:
        return m[:gh * CELL, :gw * CELL].reshape(gh, CELL, gw, CELL).mean(axis=(1, 3))

    return (cell(water) > 0.3) | (cell(rock) > 0.35) | (cell(shade) > 0.12)


class Atlas:
    def __init__(self, folder: Path):
        import cv2

        spec = json.loads((folder / "atlas.json").read_text())
        self.native = tuple(spec.get("native", [256, 224]))
        self.maps: list[WorldMap] = []
        for m in spec["maps"]:
            img = cv2.imread(str(folder / m["image"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = img[: m.get("height", img.shape[0])]
            wm = WorldMap(m["name"], img,
                          {k: tuple(v) for k, v in m.get("places", {}).items()})
            for x0, y0, x1, y1 in m.get("open", []):  # bridges, stairs: walkable
                wm.blocked[int(y0) // CELL:int(y1) // CELL + 1,
                           int(x0) // CELL:int(x1) // CELL + 1] = False
            self.maps.append(wm)

    @classmethod
    def for_game(cls, game: str, folder: Path = ATLASES) -> Atlas | None:
        if not folder.is_dir():
            return None
        name = slug(game)
        for d in sorted(p for p in folder.iterdir() if (p / "atlas.json").exists()):
            if d.name == name or all(w in name.split("-") for w in d.name.split("-")):
                return cls(d)
        return None

    def view(self, frame: Image.Image) -> np.ndarray:
        """The game's picture at its own size, BGR: the screen cropped to the game (the
        middle 4:3 of a wide window) and scaled down."""
        import cv2

        rgb = np.asarray(frame.convert("RGB"))
        h, w = rgb.shape[:2]
        gw = min(w, round(h * 4 / 3))  # RetroArch keeps the 4:3 picture in the middle
        x0 = (w - gw) // 2
        game = rgb[:, x0:x0 + gw]
        small = cv2.resize(game, self.native, interpolation=cv2.INTER_AREA)
        return small[:, :, ::-1].copy()

    def locate(self, frame: Image.Image, near: Spot | None = None) -> Spot | None:
        """Where the middle of the player's view is on a map, or None when the screen
        isn't any of the maps (inside a place, a menu, a battle). `near` (the last spot)
        searches around it first, which is quicker."""
        import cv2

        v = self.view(frame)
        nw, nh = self.native
        t = v[16:nh - 24, 16:nw - 16]  # off the edges (the era box sits bottom left)
        if near is not None:  # a step from the last spot: search only around it
            m = self.map(near.map)
            x0 = max(0, int(near.x - nw / 2 + 16 - 48))
            y0 = max(0, int(near.y - nh / 2 + 16 - 48))
            win = m.image[y0:y0 + t.shape[0] + 96, x0:x0 + t.shape[1] + 96]
            if win.shape[0] >= t.shape[0] and win.shape[1] >= t.shape[1]:
                r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv2.minMaxLoc(r)
                if score > 0.65:
                    return Spot(m.name, x0 + x - 16 + nw / 2, y0 + y - 16 + nh / 2,
                                float(score), 1.0)
        best: Spot | None = None
        maps = self.maps
        if near is not None:
            maps = sorted(self.maps, key=lambda m: m.name != near.map)
        for m in maps:
            r = cv2.matchTemplate(m.image, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(r)
            r2 = r.copy()
            r2[max(0, y - 24):y + 24, max(0, x - 24):x + 24] = -1
            margin = float(score - r2.max())
            spot = Spot(m.name, x - 16 + nw / 2, y - 16 + nh / 2, float(score), margin)
            if best is None or spot.score > best.score:
                best = spot
            if near is not None and m.name == near.map and score > 0.7:
                break
        if best is None or best.score < 0.6:
            return None
        return best

    def map(self, name: str) -> WorldMap:
        return next(m for m in self.maps if m.name == name)

    def find_place(self, query: str) -> tuple[WorldMap, str] | None:
        """The place a request means: exact names first, then most shared words."""
        q = set(re.findall(r"[a-z0-9]+", query.lower().replace("'s", "")))
        best, score = None, 0.0
        for m in self.maps:
            for name in m.places:
                n = set(re.findall(r"[a-z0-9]+", name.lower().replace("'s", "")))
                s = len(q & n) / len(n) if n else 0
                if s > score:
                    best, score = (m, name), s
        return best if score >= 0.5 else None

    def route(self, m: WorldMap, start: tuple[float, float], goal: tuple[float, float]
              ) -> list[tuple[float, float]] | None:
        """Way points from start to goal on the map's walkable cells (A*), or None."""
        gh, gw = m.blocked.shape
        s = (int(start[1]) // CELL, int(start[0]) // CELL)
        g = (int(goal[1]) // CELL, int(goal[0]) // CELL)

        def shut(c: tuple[int, int]) -> bool:
            if not (0 <= c[0] < gh and 0 <= c[1] < gw):
                return True
            if c == s:
                return False  # wherever the player stands is walkable
            if c in m.learned:
                return True  # walking found it shut, whatever the map looks like
            if abs(c[0] - g[0]) <= 1 and abs(c[1] - g[1]) <= 1:
                return False  # the place itself (its icon) can be walked onto
            return bool(m.blocked[c])

        def edge(c: tuple[int, int]) -> float:
            """Extra cost next to shut ground, so routes keep off walls and corners
            (the character is wider than a point)."""
            return 0.6 * sum(shut((c[0] + dy, c[1] + dx)) for dy in (-1, 0, 1)
                             for dx in (-1, 0, 1) if dy or dx) / 8

        pq: list = [(0.0, s)]
        came: dict = {s: None}
        cost = {s: 0.0}
        while pq:
            _, u = heapq.heappop(pq)
            if u == g:
                path = []
                while u is not None:
                    path.append(((u[1] + 0.5) * CELL, (u[0] + 0.5) * CELL))
                    u = came[u]
                return simplify(path[::-1])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    v = (u[0] + dy, u[1] + dx)
                    if (dy or dx) and not shut(v):
                        if dy and dx and (shut((u[0] + dy, u[1])) or shut((u[0], u[1] + dx))):
                            continue  # no cutting corners
                        c = cost[u] + (1.42 if dy and dx else 1.0) + edge(v) * 4
                        if c < cost.get(v, 1e18):
                            cost[v], came[v] = c, u
                            heapq.heappush(pq, (c + math.dist(v, g), v))
        return None


# The character stands this far above the middle of the view on a world map.
RISE = 12


async def walk_to(atlas: Atlas, m: WorldMap, name: str, shoot, steer, press, keep_going,
                  log, limit: float = 90.0) -> str:
    """Walk to a named place on a world map: where the view is on the map (image
    matching, several times a second), a route over walkable ground (A*) and steering
    along it; a spot that stops the character is marked shut and the route planned
    again. Arrived when the screen stops being the map at the place.

    shoot: async, the game's picture. steer(angle or None), press(button): the
    controls. keep_going: async, False when the player stopped it. log(text)."""
    import asyncio
    import time

    goal = m.places[name]
    spot = atlas.locate(await shoot())
    if spot is None:
        return ("not on a world map now (inside somewhere, a menu or a fade): go out "
                "to the world map first")
    if spot.map != m.name:
        return f"{name} is on the {m.name} map; you're on the {spot.map} map"

    def here(s: Spot) -> tuple[float, float]:
        return (s.x, s.y - RISE)

    p = here(spot)
    path = atlas.route(m, p, goal)
    if path is None:
        return (f"no way to {name} over open ground from here "
                f"({compass(goal[0] - p[0], goal[1] - p[1])}); it may be through another "
                "place (a forest, a cave), or by boat")
    t0 = time.monotonic()
    i, last, last_t, stalls, misses, presses, angle = 0, p, t0, 0, 0, 0, 0.0
    logged, trail = t0, deque([(t0, p)])

    async def gone(secs: float) -> bool:
        """Whether the screen stops being the map within secs (went in somewhere)."""
        end = time.monotonic() + secs
        while time.monotonic() < end:
            await asyncio.sleep(0.25)
            if atlas.locate(await shoot(), near=spot) is None:
                await asyncio.sleep(0.3)
                if atlas.locate(await shoot(), near=spot) is None:
                    return True
        return False

    async def slide(angle: float) -> bool:
        nonlocal spot
        for turn in (45, -45, 90, -90):
            steer((angle + turn) % 360)
            await asyncio.sleep(0.3)
            now = atlas.locate(await shoot(), near=spot)
            if now is not None and math.dist(here(now), here(spot)) > 3:
                spot = now
                return True
        steer(None)
        return False
    try:
        while time.monotonic() - t0 < limit:
            if not await keep_going():
                return "stopped by the player"
            now = atlas.locate(await shoot(), near=spot)
            if now is None:
                misses += 1
                if misses < 3:  # a cloud or a blink; a place fades for longer
                    continue
                steer(None)
                if math.dist(last, goal) <= 3 * CELL:
                    return f"arrived: went into {name} ({time.monotonic() - t0:.0f} s)"
                return (f"new_area: went into somewhere else, by {m.nearest(*last)[0]}, "
                        f"before reaching {name}; look")
            misses = 0
            spot, p = now, here(now)
            if math.dist(p, goal) <= CELL * 0.75:
                # On the place's icon: games go in with a button (Chrono Trigger) or
                # by themselves.
                steer(None)
                press("a")
                if await gone(1.5):
                    return f"arrived: went into {name} ({time.monotonic() - t0:.0f} s)"
                presses += 1
                if presses >= 3:
                    return (f"at {name} on the map, but it doesn't open from here "
                            "(maybe not yet in the story)")
            while i < len(path) - 1 and math.dist(p, path[i]) < CELL * 0.75:
                i += 1
            tx, ty = goal if i >= len(path) - 1 else path[i]
            angle = math.degrees(math.atan2(tx - p[0], -(ty - p[1]))) % 360
            steer(angle)
            t = time.monotonic()
            if t - logged > 2:
                logged = t
                log(f"go_to_place: at {p[0]:.0f},{p[1]:.0f}, heading {angle:.0f} to "
                    f"{tx:.0f},{ty:.0f}")
            trail.append((t, p))
            while t - trail[0][0] > 0.8:
                trail.popleft()
            if max(math.dist(p, q) for _, q in trail) > 5 or t - last_t < 0.8:
                last = p  # still getting somewhere (the match jitters a pixel or two)
            else:
                # Stuck: slide along whatever it is first (a step to either side of
                # the way), then mark the way ahead shut and plan again.
                if await slide(angle):
                    last_t = time.monotonic()
                    trail.clear()
                    trail.append((last_t, here(spot)))
                    continue
                stalls += 1
                rad = math.radians(angle)
                ahead = (int((p[1] - 7 * math.cos(rad)) // CELL),
                         int((p[0] + 7 * math.sin(rad)) // CELL))
                cur = (int(p[1]) // CELL, int(p[0]) // CELL)
                g = (int(goal[1]) // CELL, int(goal[0]) // CELL)
                if ahead not in (cur, g):
                    m.learned.add(ahead)
                log(f"go_to_place: stuck at {p[0]:.0f},{p[1]:.0f} heading "
                    f"{angle:.0f}, shut {ahead} ({stalls})")
                path = atlas.route(m, p, goal)
                i, last_t = 0, time.monotonic()
                trail.clear()
                trail.append((last_t, p))
                if path is None or stalls > 30:
                    return (f"blocked: no way through to {name} from here; "
                            f"{m.nearest(*p)[0]} is nearest")
            await asyncio.sleep(0.02)
    finally:
        steer(None)
    return (f"timeout: about {math.dist(p, goal) / CELL:.0f} steps "
            f"{compass(goal[0] - p[0], goal[1] - p[1])} of {name} still")


def simplify(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Way points where the route turns, every few cells at most."""
    if len(path) < 3:
        return path
    out = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:], strict=False):
        d1 = (np.sign(b[0] - a[0]), np.sign(b[1] - a[1]))
        d2 = (np.sign(c[0] - b[0]), np.sign(c[1] - b[1]))
        if d1 != d2 or math.dist(out[-1], b) > 6 * CELL:
            out.append(b)
    out.append(path[-1])
    return out
