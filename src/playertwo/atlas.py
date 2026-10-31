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
    # From walking the map (scripts/survey.py): cells known walkable or shut, and the
    # cells of each place's icon, by the game's own number for it.
    walk: set[tuple[int, int]] = field(default_factory=set)
    wall: set[tuple[int, int]] = field(default_factory=set)
    exits: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    # Which exits were tried and where they went: {exit: {"location": n}}, and the
    # world map's own location number.
    entries: dict[str, dict] = field(default_factory=dict)
    world: int | None = None
    doors: dict[tuple[int, int], int] = field(default_factory=dict)  # tile -> location
    names: dict[str, str] = field(default_factory=dict)  # location -> the game's name

    def __post_init__(self) -> None:
        if self.blocked is None:
            self.blocked = ground(self.image)

    @property
    def surveyed(self) -> bool:
        return bool(self.walk)

    def entrance(self, name: str) -> set[tuple[int, int]] | None:
        """The cells of a place's icon (standing on one and pressing the button goes
        in), when the survey found it: the icon nearest where the place is named."""
        if name not in self.places or not self.exits:
            return None
        px, py = self.places[name]

        def far(cells: set[tuple[int, int]]) -> float:
            return min(math.dist(((c[1] + 0.5) * CELL, (c[0] + 0.5) * CELL), (px, py))
                       for c in cells)

        # Icons are several tiles, and only some of them open: the doors are the tiles
        # found to go in somewhere, grouped by where they go, each with the name the
        # game shows on it.
        groups = self.door_groups()
        if not groups:
            groups = [(cells, self.names.get(e, "")) for e, cells in self.exits.items()]
        want = words(name)
        named = [g for g in groups if g[1] and (words(g[1]) <= want or want <= words(g[1]))]
        best = min(named or groups, key=lambda g: far(g[0]))[0]
        return best if far(best) <= 3 * CELL else None

    def door_groups(self) -> list[tuple[set[tuple[int, int]], str]]:
        """Each place's door tiles (from the survey) and the game's name for it."""
        by_loc: dict[int, set[tuple[int, int]]] = {}
        for c, loc in self.doors.items():
            if loc != self.world:
                by_loc.setdefault(loc, set()).add(c)
        return [(cells, self.names.get(str(loc), "")) for loc, cells in by_loc.items()]

    def nearest(self, x: float, y: float) -> tuple[str, float, str]:
        """The named place nearest a point: its name, distance in map pixels, and the
        compass way from the point to it."""
        best = min(self.places.items(), key=lambda kv: math.dist(kv[1], (x, y)))
        name, (px, py) = best
        return name, math.dist((px, py), (x, y)), compass(px - x, py - y)


def words(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", name.lower().replace("'s", "")))


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
            survey = folder / f"survey-{slug(m['name'])}.json"
            if survey.exists():
                sv = json.loads(survey.read_text())
                wm.walk = {tuple(c) for c in sv.get("walk", [])}
                wm.wall = {tuple(c) for c in sv.get("wall", [])}
                wm.exits = {k: {tuple(c) for c in v} for k, v in sv.get("exits", {}).items()}
                wm.entries = sv.get("entries", {})
                wm.world = sv.get("world")
                wm.doors = {(r, c): loc for r, c, loc in sv.get("doors", [])}
                wm.names = sv.get("names", {})
                # Places the game names that the map picture didn't: add them where
                # their door is, when the name is the only one of its kind.
                labels = [label for _, label in wm.door_groups() if label]
                for cells, label in wm.door_groups():
                    if label and labels.count(label) == 1 and not any(
                            words(label) <= words(n) for n in wm.places):
                        r, c = min(cells)
                        wm.places[label] = ((c + 0.5) * CELL, (r + 0.5) * CELL)
            self.maps.append(wm)

    @staticmethod
    def folder_for(game: str, folder: Path = ATLASES) -> Path | None:
        """The game's atlas folder, without loading it."""
        if not game or not folder.is_dir():
            return None
        name = slug(game)
        for d in sorted(p for p in folder.iterdir() if (p / "atlas.json").exists()):
            if d.name == name or all(w in name.split("-") for w in d.name.split("-")):
                return d
        return None

    @classmethod
    def for_game(cls, game: str, folder: Path = ATLASES) -> Atlas | None:
        d = cls.folder_for(game, folder)
        return cls(d) if d is not None else None

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

    def route(self, m: WorldMap, start: tuple[float, float], goal: tuple[float, float],
              goals: set[tuple[int, int]] | None = None) -> list[tuple[float, float]] | None:
        """Way points from start to goal (or to any of the `goals` cells) on the map's
        walkable cells (A*), or None. On a surveyed map the way points are where the
        character comes to rest in each cell (its corner)."""
        gh, gw = m.blocked.shape
        s = (int(start[1]) // CELL, int(start[0]) // CELL)
        g = (int(goal[1]) // CELL, int(goal[0]) // CELL)
        targets = goals or {g}
        surveyed = m.surveyed

        def shut(c: tuple[int, int]) -> bool:
            if not (0 <= c[0] < gh and 0 <= c[1] < gw):
                return True
            if c == s:
                return False  # wherever the player stands is walkable
            if c in m.learned:
                return True  # walking found it shut, whatever the map looks like
            if c in targets:
                return False
            if surveyed:
                if c in m.walk:
                    return False
                if c in m.wall:
                    return True
            elif abs(c[0] - g[0]) <= 1 and abs(c[1] - g[1]) <= 1:
                return False  # the place itself (its icon) can be walked onto
            return bool(m.blocked[c])

        def edge(c: tuple[int, int]) -> float:
            """Extra cost next to shut ground, so routes keep off walls and corners
            (the character is wider than a point); and off ground no one has walked."""
            if surveyed:
                return 0.0 if c in m.walk or c in targets else 3.0
            return 2.4 * sum(shut((c[0] + dy, c[1] + dx)) for dy in (-1, 0, 1)
                             for dx in (-1, 0, 1) if dy or dx) / 8

        def spot(c: tuple[int, int]) -> tuple[float, float]:
            if surveyed:
                return (c[1] * CELL, c[0] * CELL)
            return ((c[1] + 0.5) * CELL, (c[0] + 0.5) * CELL)

        def h(c: tuple[int, int]) -> float:
            return min(math.dist(c, t) for t in targets) if len(targets) < 40 else \
                math.dist(c, g)

        pq: list = [(0.0, s)]
        came: dict = {s: None}
        cost = {s: 0.0}
        while pq:
            _, u = heapq.heappop(pq)
            if u in targets:
                path = []
                while u is not None:
                    path.append(spot(u))
                    u = came[u]
                path = path[::-1]
                return path if surveyed else simplify(path)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    v = (u[0] + dy, u[1] + dx)
                    if (dy or dx) and not shut(v):
                        if dy and dx and (shut((u[0] + dy, u[1])) or shut((u[0], u[1] + dx))):
                            continue  # no cutting corners
                        c = cost[u] + (1.42 if dy and dx else 1.0) + edge(v)
                        if c < cost.get(v, 1e18):
                            cost[v], came[v] = c, u
                            heapq.heappush(pq, (c + h(v), v))
        return None


# The character stands this far above the middle of the view on a world map (from
# Chrono Trigger's memory: its position is the view's middle, a pixel up).
RISE = 1


def heading(dx: float, dy: float, dead: float = 2.0) -> float | None:
    """The d-pad way (an angle, 0 up) toward an offset, each axis only when it's more
    than `dead` pixels off: straight lines and no wobble on a d-pad."""
    x = 1 if dx > dead else -1 if dx < -dead else 0
    y = 1 if dy > dead else -1 if dy < -dead else 0
    if not x and not y:
        return None
    return math.degrees(math.atan2(x, -y)) % 360


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
    if spot is not None and spot.margin < 0.05:
        spot = None  # it matches a few places about as well: not really the map
    if spot is None:
        return ("not on a world map now (inside somewhere, a menu or a fade): go out "
                "to the world map first")
    if spot.map != m.name:
        return f"{name} is on the {m.name} map; you're on the {spot.map} map"

    def here(s: Spot) -> tuple[float, float]:
        return (s.x, s.y - RISE)

    p = here(spot)
    door = m.entrance(name)  # the icon's cells, on a surveyed map

    def cell(q: tuple[float, float]) -> tuple[int, int]:
        return int(q[1]) // CELL, int(q[0]) // CELL

    def resting(q: tuple[float, float]) -> bool:
        """On a tile: the character comes to rest on multiples of CELL pixels."""
        return all(min(v % CELL, CELL - v % CELL) <= 2 for v in q)

    def at_door(q: tuple[float, float]) -> bool:
        if door is not None:
            return resting(q) and cell((q[0] + 2, q[1] + 2)) in door
        return math.dist(q, goal) <= CELL * 0.75

    path = atlas.route(m, p, goal, door)
    if path is None:
        return (f"no way to {name} over open ground from here "
                f"({compass(goal[0] - p[0], goal[1] - p[1])}); it may be through another "
                "place (a forest, a cave), or by boat")
    t0 = time.monotonic()
    i, last, last_t, stalls, misses, presses, angle = 0, p, t0, 0, 0, 0, 0.0
    logged, trail = t0, deque([(t0, p)])
    cur: tuple[tuple[int, int] | None, float] = (None, t0)  # the tile and since when

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
                near = (min(math.dist(cell(last), c) for c in door) <= 1.5 if door
                        else math.dist(last, goal) <= 3 * CELL)
                if near:
                    return f"arrived: went into {name} ({time.monotonic() - t0:.0f} s)"
                return (f"new_area: went into somewhere else, by {m.nearest(*last)[0]}, "
                        f"before reaching {name}; look")
            misses = 0
            spot, p = now, here(now)
            if at_door(p):
                # On the place's icon: games go in with a button (Chrono Trigger) or
                # by themselves.
                steer(None)
                press("a")
                if await gone(3.0):
                    return f"arrived: went into {name} ({time.monotonic() - t0:.0f} s)"
                presses += 1
                if presses >= 3:
                    return (f"at {name} on the map, but it doesn't open from here "
                            "(maybe not yet in the story)")
            if m.surveyed:
                # Tile by tile: hold the way along a straight run and let go a little
                # before the turn, so the step that's under way ends on the turn's
                # tile; from a standstill one tap is one step.
                cells = [cell(q) for q in path]
                c = cell((p[0] + 2, p[1] + 2))
                if c not in cells:
                    path = atlas.route(m, p, goal, door)
                    if path is None:
                        return (f"blocked: no way through to {name} from here; "
                                f"{m.nearest(*p)[0]} is nearest")
                    cells = [cell(q) for q in path]
                    c = cells[0]
                k = cells.index(c)
                if c != cur[0]:
                    cur = (c, time.monotonic())
                elif k < len(cells) - 1 and time.monotonic() - cur[1] > 1.5:
                    # Not off this tile in a while: the next one is shut after all.
                    stalls += 1
                    m.learned.add(cells[k + 1])
                    log(f"go_to_place: stuck at {p[0]:.0f},{p[1]:.0f}, shut "
                        f"{cells[k + 1]} ({stalls})")
                    steer(None)
                    path = atlas.route(m, p, goal, door)
                    cur = (c, time.monotonic())
                    if path is None or stalls > 20:
                        return (f"blocked: no way through to {name} from here; "
                                f"{m.nearest(*p)[0]} is nearest")
                    continue
                if k < len(cells) - 1:
                    d = (cells[k + 1][0] - c[0], cells[k + 1][1] - c[1])
                    j = k + 1
                    while j + 1 < len(cells) and (cells[j + 1][0] - cells[j][0],
                                                  cells[j + 1][1] - cells[j][1]) == d:
                        j += 1
                    tx, ty = path[j]
                    angle = math.degrees(math.atan2(d[1], -d[0])) % 360
                    if max(abs(tx - p[0]), abs(ty - p[1])) > CELL + 2:
                        steer(angle)
                    else:
                        steer(None)  # the step under way ends on the turn's tile
                        await asyncio.sleep(0.12)
                        now = atlas.locate(await shoot(), near=spot)
                        if now is not None and resting(here(now)) and \
                                cell((here(now)[0] + 2, here(now)[1] + 2)) == c:
                            steer(angle)  # standing short of it: one step
                            await asyncio.sleep(0.06)
                            steer(None)
                            await asyncio.sleep(0.12)
                    last, last_t = p, time.monotonic()
                    trail.clear()
                    trail.append((last_t, p))
                    t = time.monotonic()
                    if t - logged > 2:
                        logged = t
                        log(f"go_to_place: at {p[0]:.0f},{p[1]:.0f}, heading {angle:.0f}")
                    await asyncio.sleep(0.01)
                    continue
                tx, ty = path[-1]  # the end of the way: line up on it
            while i < len(path) - 1 and math.dist(p, path[i]) < CELL * 0.75:
                i += 1
            if not m.surveyed:
                tx, ty = path[i] if door is not None or i < len(path) - 1 else goal
            way = heading(tx - p[0], ty - p[1])
            if way is None and i < len(path) - 1:
                i += 1
                continue
            angle = way if way is not None else angle
            steer(way)
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
                path = atlas.route(m, p, goal, door)
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
