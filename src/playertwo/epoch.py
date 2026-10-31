"""Flying Chrono Trigger's Epoch: across the sea to a place walking can't reach, and to
another era by its time gauge.

The Epoch is parked somewhere on a world map. Standing on its tile and pressing a boards
it; it then flies over anything, about twice walking speed, and when let go it stops on
the next 8-pixel tile like walking does. a lands it, but only on some ground (not water,
not a town or a house): a landing that doesn't take leaves it flying. y opens the time
gauge: up and down move the needle one era at a time (the era you're in is skipped),
a goes. It arrives flying at the same spot on that era's map.

Everything here works from the screen, like atlas.walk_to: where the view is on the map
(Atlas.locate), whether it's flying (how far one short hold goes), and where the parked
Epoch is (where it was landed last, kept beside the notes; else its sprite on screen).
Which ground takes a landing is learned and kept in the atlas folder (epoch.json), from
scripts/epoch_survey.py and from every landing tried.

    epoch.json: {"maps": {"1000 A.D.": {"land": [[row, col], ...], "noland": [...]}},
                 "sprites": [{"file": "epoch-0.png", "dx": -10, "dy": -12}]}
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import math
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from .atlas import CELL, RISE, Atlas, Spot, WorldMap, compass, slug, walk_to

EPOCH = "the Epoch"
# The time gauge, top to bottom: down goes further back in time.
GAUGE = ("End of Time", "2300 A.D.", "1999 A.D.", "1000 A.D.", "600 A.D.", "12,000 B.C.",
         "65,000,000 B.C.")
# What players call each era.
ERA_WORDS = {
    "End of Time": ("end of time", "end of the time", "eot", "gaspar", "old man", "lamppost"),
    "2300 A.D.": ("2300", "future", "domes", "post apocalyptic", "post-apocalyptic"),
    "1999 A.D.": ("1999", "day of lavos", "apocalypse"),
    "1000 A.D.": ("1000", "present", "crono's time", "guardia now", "home era"),
    "600 A.D.": ("600", "middle ages", "medieval", "frog's time", "frog's era"),
    "12,000 B.C.": ("12000", "12,000", "12 000", "dark ages", "zeal", "antiquity",
                    "ice age"),
    "65,000,000 B.C.": ("65", "prehistor", "dinosaur", "ayla", "reptite", "stone age",
                        "65000000", "65,000,000"),
}
FLY = 0.35  # seconds of a hold that tells flying (48 px or more) from walking (24 to 32)
FLY_PX = 40


def era_of(map_name: str) -> str | None:
    """The gauge's era a world map is in ("12,000 B.C. Dark Ages" is 12,000 B.C.)."""
    for e in GAUGE:
        if map_name.startswith(e):
            return e
    return None


def era_named(text: str) -> str | None:
    """The era a player means: "600 AD", "the middle ages", "go to the future"."""
    t = " " + " ".join(text.lower().replace(".", "").split()) + " "
    t = re.sub(r"(\d) (\d{3})", r"\1\2", t)
    for e in GAUGE:
        if e.lower().replace(".", "") in t:
            return e
    for e, words in ERA_WORDS.items():
        for w in words:
            if w.isdigit():
                if re.search(rf"(?<![\d,]){w}(?![\d])", t.replace(",", "")):
                    return e
            elif w in t:
                return e
    return None


def era_only(text: str) -> str | None:
    """The era when that's all the text names ("take me to 600 AD", "the future"), not
    a place in one ("Truce Inn in 600 AD")."""
    era = era_named(text)
    if era is None:
        return None
    t = " " + " ".join(text.lower().replace(".", " ").replace(",", "").split()) + " "
    for w in (*ERA_WORDS[era], era.lower().replace(".", " ").replace(",", "")):
        t = t.replace(w.replace(",", ""), " ")
    rest = set(re.findall(r"[a-z]+", t)) - {
        "a", "d", "b", "c", "ad", "bc", "the", "era", "time", "to", "in", "back", "age",
        "ages", "period", "year", "years", "of", "go", "take", "me", "us", "please", "now"}
    return era if not rest else None


# --- what's known about the ground: where the Epoch lands -----------------------------

class Hangar:
    """What's been learned about landing, per map, kept in the atlas folder."""

    def __init__(self, folder: Path | None):
        self.path = folder / "epoch.json" if folder is not None else None
        self.data: dict = {"maps": {}}
        if self.path is not None:
            with contextlib.suppress(OSError, ValueError):
                self.data = json.loads(self.path.read_text())
        self.data.setdefault("maps", {})

    def _cells(self, m: str, kind: str) -> set[tuple[int, int]]:
        return {tuple(c) for c in self.data["maps"].get(m, {}).get(kind, [])}

    def land(self, m: str) -> set[tuple[int, int]]:
        return self._cells(m, "land")

    def noland(self, m: str) -> set[tuple[int, int]]:
        return self._cells(m, "noland")

    def learn(self, m: str, c: tuple[int, int], ok: bool) -> None:
        if self.path is not None:  # someone else may have learned more meanwhile
            with contextlib.suppress(OSError, ValueError):
                self.data = json.loads(self.path.read_text())
                self.data.setdefault("maps", {})
        d = self.data["maps"].setdefault(m, {})
        good, bad = self._cells(m, "land"), self._cells(m, "noland")
        (good if ok else bad).add(tuple(c))
        (bad if ok else good).discard(tuple(c))
        d["land"], d["noland"] = sorted(good), sorted(bad)
        self.save()

    def save(self) -> None:
        if self.path is not None:
            with contextlib.suppress(OSError):
                self.path.write_text(json.dumps(self.data))


def landing_spots(m: WorldMap, door: set[tuple[int, int]], hangar: Hangar,
                  limit: int = 12, depth: int = 40) -> list[tuple[int, int]]:
    """Tiles to land on for a place's door: surveyed ground that walks to the door,
    the nearest on foot first, ones known to take a landing before untried ones, never
    one known not to; off the icons (towns and houses don't take one)."""
    good, bad = hangar.land(m.name), hangar.noland(m.name)
    icons = {(r + dr, c + dc) for cells in m.exits.values() for r, c in cells
             for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    icons |= {(r + dr, c + dc) for (r, c) in m.doors for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    seen = {c: 0 for c in door}
    q = deque(door)
    out: list[tuple[int, tuple[int, int]]] = []
    while q:
        c = q.popleft()
        d = seen[c]
        if c not in icons and c not in bad and c in m.walk:
            out.append((d, c))
        if d >= depth:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (c[0] + dr, c[1] + dc)
            if n not in seen and n in m.walk and n not in m.learned:
                seen[n] = d + 1
                q.append(n)
    # Known good within a few steps of the nearest untried beats the untried.
    first = out[0][0] if out else 0
    out.sort(key=lambda dc: (dc[0] - (8 if dc[1] in good else 0) if dc[0] <= first + 12
                             else dc[0], dc[0]))
    spaced: list[tuple[int, int]] = []
    for _, c in out:  # not all side by side: a strip of unlandable ground is common
        if c in good or all(max(abs(c[0] - s[0]), abs(c[1] - s[1])) >= 2 for s in spaced):
            spaced.append(c)
        if len(spaced) >= limit:
            break
    return spaced


def reachable(m: WorldMap, start: tuple[float, float], door: set[tuple[int, int]]) -> bool:
    """Whether surveyed ground joins where the player stands to a door (on foot)."""
    s = (int(start[1]) // CELL, int(start[0]) // CELL)
    seen, q = {s}, deque([s])
    while q:
        c = q.popleft()
        if c in door:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (c[0] + dr, c[1] + dc)
                if n not in seen and (n in m.walk or n in door) and n not in m.learned:
                    seen.add(n)
                    q.append(n)
    return False


# --- where the Epoch is parked ----------------------------------------------------------

def _memory(folder: Path | None, game: str) -> Path | None:
    if folder is None:
        return None
    return folder / f"{re.sub(r'[^a-z0-9]+', '-', game.lower()).strip('-') or 'game'}.epoch.json"


def parked(folder: Path | None, game: str) -> tuple[str, float, float] | None:
    """Where the Epoch was last landed (map, x, y), or None."""
    p = _memory(folder, game)
    if p is None:
        return None
    try:
        d = json.loads(p.read_text())
        return str(d["map"]), float(d["x"]), float(d["y"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def left_flying(folder: Path | None, game: str) -> bool:
    """Whether the Epoch was last left in the air (the player is in it)."""
    p = _memory(folder, game)
    try:
        return bool(p is not None and json.loads(p.read_text()).get("flying"))
    except (OSError, ValueError, AttributeError):
        return False


def park(folder: Path | None, game: str, map_name: str | None, x: float = 0,
         y: float = 0, flying: bool = False) -> None:
    """Keep where the Epoch was landed (None: not known any more)."""
    p = _memory(folder, game)
    if p is None:
        return
    with contextlib.suppress(OSError):
        p.parent.mkdir(parents=True, exist_ok=True)
        if map_name is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(json.dumps({"map": map_name, "x": x, "y": y, "flying": flying,
                                     "t": time.time()}))


class Sprites:
    """The parked Epoch's look on the game's own picture (from scripts/epoch_survey.py
    --sprite): masked templates, and where its tile is from each template's corner."""

    def __init__(self, folder: Path | None):
        import cv2

        self.items: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        if folder is None:
            return
        try:
            spec = json.loads((folder / "epoch.json").read_text()).get("sprites", [])
        except (OSError, ValueError):
            spec = []
        for s in spec:
            img = cv2.imread(str(folder / s["file"]), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue
            mask = (img[:, :, 3] > 0).astype(np.uint8)
            mask = np.repeat(mask[:, :, None], 3, axis=2)
            self.items.append((img[:, :, :3].copy(), mask, int(s["dx"]), int(s["dy"])))

    def find(self, view: np.ndarray) -> tuple[float, float, float] | None:
        """The parked Epoch's tile on the view (x, y in the view's pixels) and how well
        it matched (0 best), or None."""
        import cv2

        best = None
        for img, mask, dx, dy in self.items:
            if view.shape[0] < img.shape[0] or view.shape[1] < img.shape[1]:
                continue
            r = cv2.matchTemplate(view, img, cv2.TM_SQDIFF, mask=mask)
            r = r / max(1, int(mask.sum()))
            _, _, (x, y), _ = cv2.minMaxLoc(r)
            v = float(r[y, x])
            if best is None or v < best[2]:
                best = (x + dx, y + dy, v)
        if best is None or best[2] > 900:  # about 30 levels off per channel
            return None
        return best


# --- flying ------------------------------------------------------------------------------

class _OnTile(Exception):
    """walk_to reached the tile it was sent to (it pressed a): stop it there."""


@dataclasses.dataclass
class Pilot:
    """Flies the Epoch with the same hands and eyes as atlas.walk_to: shoot() the
    game's picture, steer(angle or None), press(button), keep_going() False when the
    player stopped it, log(text). `notes` is where the parked spot is kept; `find`
    (optional, async (image) -> (x, y) as fractions of the picture, or None) finds the
    parked Epoch on a picture when its sprite isn't known (the Steam port's art)."""

    atlas: Atlas
    game: str
    shoot: object
    steer: object
    press: object
    keep_going: object
    log: object
    notes: Path | None = None
    find: object = None
    spot: Spot | None = None

    def __post_init__(self) -> None:
        folder = getattr(self.atlas, "folder", None)
        self.hangar = Hangar(folder)
        self.sprites = Sprites(folder)
        self._vel = (0.0, 0.0)  # map pixels a second, from the last fixes
        self._fix = 0.0  # when the last spot was seen

    # eyes
    async def where(self, near: bool = True) -> Spot | None:
        img = await self.shoot()
        t = time.monotonic()
        guess = None
        if near and self.spot is not None:
            # Where it should be by now, at the speed it was going (a slow frame in
            # flight is 50 px or more).
            vx, vy = self._vel if t - self._fix < 1.0 else (0.0, 0.0)
            dt = min(1.0, t - self._fix)
            guess = dataclasses.replace(self.spot, x=self.spot.x + vx * dt,
                                        y=self.spot.y + vy * dt)
        s = await asyncio.to_thread(self._locate, img, guess)
        if s is not None and s.margin < 0.05:
            s = None
        if s is not None:
            if self.spot is not None and s.map == self.spot.map and 0.02 < t - self._fix < 1:
                dt = t - self._fix
                vx, vy = (s.x - self.spot.x) / dt, (s.y - self.spot.y) / dt
                if math.hypot(vx, vy) < 400:
                    self._vel = (0.5 * self._vel[0] + 0.5 * vx, 0.5 * self._vel[1] + 0.5 * vy)
            else:
                self._vel = (0.0, 0.0)
            self.spot, self._fix = s, t
        return s

    def _locate(self, img, near: Spot | None) -> Spot | None:
        """Atlas.locate, but around the last spot it takes a weaker match that still
        stands out: over the sea, with clouds and the Epoch over the middle, the match
        drops under locate's bar and every miss would cost a search of every map."""
        import cv2

        if near is not None:
            rgb = np.asarray(img.convert("RGB"))
            if self.atlas._boxed(rgb):
                v = self.atlas._view(rgb)
                nw, nh = self.atlas.native
                t = v[16:nh - 24, 16:nw - 16]
                m = self.atlas.map(near.map)
                x0 = max(0, int(near.x - nw / 2 + 16 - 64))
                y0 = max(0, int(near.y - nh / 2 + 16 - 64))
                win = m.image[y0:y0 + t.shape[0] + 128, x0:x0 + t.shape[1] + 128]
                if win.shape[0] >= t.shape[0] and win.shape[1] >= t.shape[1]:
                    r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                    _, score, _, (x, y) = cv2.minMaxLoc(r)
                    r[max(0, y - 8):y + 8, max(0, x - 8):x + 8] = -1
                    margin = score - float(r.max())
                    if score > 0.45 and margin > 0.1:
                        return Spot(m.name, x0 + x - 16 + nw / 2, y0 + y - 16 + nh / 2,
                                    float(score), 1.0)
        return self.atlas.locate(img, near)

    @staticmethod
    def here(s: Spot) -> tuple[float, float]:
        return s.x, s.y - RISE

    async def settle(self, secs: float = 0.3) -> Spot | None:
        self.steer(None)
        await asyncio.sleep(secs)
        return await self.where()

    async def tap(self, angle: float, secs: float) -> None:
        self.steer(angle)
        await asyncio.sleep(secs)
        self.steer(None)

    async def moved(self, angle: float, secs: float = FLY) -> tuple[float, Spot | None]:
        """How far one hold goes (flying goes about twice as far as walking)."""
        a = await self.where()
        if a is None:
            return 0.0, None
        self.steer(angle)
        await asyncio.sleep(secs)
        self.steer(None)
        await asyncio.sleep(0.35)
        b = await self.where()
        if b is None or b.map != a.map:
            return 0.0, b
        return math.dist(self.here(a), self.here(b)), b

    # the Epoch
    async def find_epoch(self, m: WorldMap, s: Spot) -> tuple[float, float] | None:
        """Where the parked Epoch's tile is on this map (map pixels), or None."""
        p = self.here(s)
        img = await self.shoot()
        view = self.atlas.view(img)
        nw, nh = self.atlas.native
        if self.sprites.items and view.shape[1] == nw:
            hit = await asyncio.to_thread(self.sprites.find, view)
            if hit is not None:
                x = p[0] + hit[0] - nw / 2
                y = p[1] + hit[1] - (nh / 2 - RISE)
                return round(x / CELL) * CELL, round(y / CELL) * CELL
        mem = parked(self.notes, self.game)
        if mem is not None and mem[0] == m.name and not left_flying(self.notes, self.game):
            return mem[1], mem[2]
        if self.find is not None:
            hit = await self.find(img)  # where on the picture, as fractions of it
            if hit is not None:
                w, h = img.size
                if self.atlas._boxed(np.asarray(img.convert("RGB"))):
                    gw = min(w, round(h * 4 / 3))
                    xv = (hit[0] * w - (w - gw) / 2) / gw * nw
                    yv = hit[1] * nh
                    top = s.y - nh / 2
                else:  # the window is the view; it sits FULL_DROP above the spot
                    vw, vh = view.shape[1], view.shape[0]
                    xv, yv = hit[0] * vw, hit[1] * vh
                    nw = vw
                    top = s.y - self.atlas.FULL_DROP - vh / 2
                x = s.x - nw / 2 + xv
                y = top + yv
                return round(x / CELL) * CELL, round(y / CELL) * CELL
        return None

    async def flying(self, angle: float) -> bool | None:
        """Whether it's flying, by a short hold that way (None: lost the map)."""
        d, s = await self.moved(angle)
        if s is None:
            return None
        self.log(f"epoch: a {FLY:.2f} s hold went {d:.0f} px")
        return d >= FLY_PX

    async def board(self, m: WorldMap, toward: tuple[float, float]) -> str:
        """Walk onto the parked Epoch and get in: "" when flying, else why not."""
        s = await self.where(near=False)
        if s is None:
            return "not on a world map now: go out to the world map first"
        at = await self.find_epoch(m, s)
        if at is None:
            return ("no Epoch on screen here (it's parked where it was last landed): "
                    "go to it first, or walk")
        self.log(f"epoch: parked at {at[0]:.0f},{at[1]:.0f}")
        p = self.here(s)
        if math.dist(p, at) > 4:
            went = await self._walk_onto(m, at)
            if went:
                return f"couldn't reach the Epoch: {went}"
        for _ in range(2):
            self.press("a")
            await asyncio.sleep(3.0)
            s = await self.where()
            if s is None:
                await asyncio.sleep(1.0)
                s = await self.where()
            if s is None:
                return "something came up while boarding the Epoch (the screen isn't the map)"
            angle = _angle(self.here(s), toward)
            up = await self.flying(angle)
            if up:
                park(self.notes, self.game, None)
                return ""
            if up is None:
                return "lost the map while boarding the Epoch"
            # Not flying: off its tile by a step (the hold moved us), or it didn't see a.
            s = await self.where()
            if s is not None and math.dist(self.here(s), at) > 4:
                went = await self._walk_onto(m, at)
                if went:
                    return f"couldn't reach the Epoch: {went}"
        return "pressed a on the Epoch's spot but it didn't take off"

    async def _walk_onto(self, m: WorldMap, at: tuple[float, float]) -> str:
        """Walk to a tile with walk_to (it routes on the survey), stopping on it."""
        tmp = dataclasses.replace(m, places={**m.places, EPOCH: at}, exits={}, doors={})

        def press(button: str) -> None:
            if button == "a":
                raise _OnTile
            self.press(button)

        try:
            out = await walk_to(self.atlas, tmp, EPOCH, self.shoot, self.steer, press,
                                self.keep_going, self.log, 40.0)
        except _OnTile:
            return ""
        finally:
            self.steer(None)
        return out

    async def fly_to(self, goal: tuple[float, float], limit: float = 45.0) -> bool:
        """Fly to a tile (map pixels): hold the way, let go a little before it (it
        stops on the next tile), then step onto it."""
        t0 = time.monotonic()
        last: tuple[float, tuple[float, float]] | None = None
        speed, misses, fixes, said = 120.0, 0, 0, t0
        try:
            while time.monotonic() - t0 < limit:
                if not await self.keep_going():
                    return False
                s = await self.where()
                now = time.monotonic()
                if s is None:
                    misses += 1
                    if misses > 12:  # the view lost for long: stop and look again
                        self.steer(None)
                        await asyncio.sleep(0.4)
                        s = await self.where(near=False)
                        if s is None:
                            return False
                        misses = 0
                    else:
                        await asyncio.sleep(0.03)
                        continue
                misses = 0
                p = self.here(s)
                if now - said > 2:
                    said = now
                    self.log(f"epoch: flying at {p[0]:.0f},{p[1]:.0f} ({s.score:.2f}) to "
                             f"{goal[0]:.0f},{goal[1]:.0f}")
                if last is not None and now - last[0] > 0.02:
                    v = math.dist(p, last[1]) / (now - last[0])
                    if v > 40:
                        speed = 0.7 * speed + 0.3 * v
                last = (now, p)
                dx, dy = goal[0] - p[0], goal[1] - p[1]
                if abs(dx) <= 2 and abs(dy) <= 2:
                    s = await self.settle(0.3)
                    if s is not None:
                        p = self.here(s)
                        if abs(goal[0] - p[0]) <= 2 and abs(goal[1] - p[1]) <= 2:
                            return True
                    last = None
                    continue
                lead = CELL + speed * 0.12  # it runs on to the next tile, and we see late
                sx = (dx > lead) - (dx < -lead)
                sy = (dy > lead) - (dy < -lead)
                if sx or sy:
                    self.steer(math.degrees(math.atan2(sx, -sy)) % 360)
                    await asyncio.sleep(0.02)
                    continue
                # Close on both: let it stop on a tile, then a step at a time.
                s = await self.settle(0.35)
                last = None
                if s is None:
                    continue
                p = self.here(s)
                dx, dy = goal[0] - p[0], goal[1] - p[1]
                if abs(dx) <= 2 and abs(dy) <= 2:
                    return True
                fixes += 1
                if fixes > 12:
                    self.log(f"epoch: couldn't settle on {goal}, at {p}")
                    return False
                tx = (dx > 2) - (dx < -2)
                ty = (dy > 2) - (dy < -2)
                far = max(abs(dx), abs(dy))
                one = 0.05 if s.tap < 0.08 else s.tap  # a tile's tap (the port's is longer)
                await self.tap(math.degrees(math.atan2(tx, -ty)) % 360,
                               one if far <= CELL + 2 else 2 * one)
                await asyncio.sleep(0.25)
        finally:
            self.steer(None)
        return False

    async def land(self, m: WorldMap, door: set[tuple[int, int]], name: str,
                   limit: int = 12) -> str:
        """Land as near the door as the ground lets: "" when down, else why not."""
        spots = landing_spots(m, door, self.hangar, limit=limit)
        if not spots:
            return f"no surveyed ground near {name} to land on"
        for c in spots:
            goal = (c[1] * CELL, c[0] * CELL)
            self.log(f"epoch: landing at {goal[0]:.0f},{goal[1]:.0f} for {name}")
            if not await self.fly_to(goal):
                s = await self.where(near=False)
                if s is None:
                    return "lost the map while flying"
                continue
            self.press("a")
            await asyncio.sleep(3.0)
            s = await self.where()
            if s is None:
                await asyncio.sleep(1.0)
                s = await self.where(near=False)
                if s is None:
                    return "the screen stopped being the map after landing"
            # Which way on foot toward the door: the first step of the walk there.
            p = self.here(s)
            path = self.atlas.route(m, p, (0, 0), door)
            nxt = path[1] if path and len(path) > 1 else _middle(door)
            up = await self.flying(_angle(p, nxt))
            if up is None:
                return "lost the map after landing"
            self.hangar.learn(m.name, c, not up)
            if not up:
                park(self.notes, self.game, m.name, *goal)
                return ""
            self.log(f"epoch: {goal[0]:.0f},{goal[1]:.0f} doesn't take a landing")
        return f"found nowhere to land near {name} ({len(spots)} spots tried)"

    async def set_era(self, era: str, check=None) -> str:
        """With the Epoch flying, go to an era by its time gauge: "" when there (a
        press lost on the gauge lands in a neighbouring era: then once more from there).
        check: optional async (image) -> the era the gauge shows, to confirm."""
        why = await self._gauge(era, check)
        if why.startswith("after the time gauge, the screen is the "):
            self.log(f"epoch: {why}; once more")
            why = await self._gauge(era, check)
        return why

    async def _gauge(self, era: str, check=None) -> str:
        s = await self.where(near=False)
        if s is None:
            return "not on a world map"
        cur = era_of(s.map)
        if cur == era:
            return ""
        if cur is None or era not in GAUGE:
            return f"the time gauge doesn't go from {s.map} to {era}"
        self.press("y")
        await asyncio.sleep(2.5)
        if await self.where(near=False) is not None:
            self.press("y")  # the first press can be lost while the Epoch settles
            await asyncio.sleep(2.5)
            if await self.where(near=False) is not None:
                return "the time gauge didn't open (y in the Epoch)"
        n = GAUGE.index(era) - GAUGE.index(cur)
        for _ in range(abs(n)):
            self.press("down" if n > 0 else "up")
            await asyncio.sleep(0.9)  # the needle swings slowly; a press during it is lost
        if check is not None:
            shown = await check(await self.shoot())
            tries = 0
            while shown and shown != era and tries < 6:
                tries += 1
                order = [e for e in GAUGE if e != cur]
                if shown in order:
                    k = GAUGE.index(era) - GAUGE.index(shown)
                    self.press("down" if k > 0 else "up")
                    await asyncio.sleep(0.9)
                shown = await check(await self.shoot())
            if shown and shown != era:
                self.press("b")
                return f"the time gauge stayed on {shown}, not {era}"
        self.press("a")
        await asyncio.sleep(6.0)
        # It comes out at the same spot on that era's map: look there first.
        before = s
        for _ in range(30):
            img = await self.shoot()
            for m in self.atlas.maps:
                if era_of(m.name) == era:
                    near = Spot(m.name, before.x, before.y, 0.0, 0.0, before.tap)
                    hit = await asyncio.to_thread(self._locate, img, near)
                    if hit is not None and era_of(hit.map) == era and hit.margin >= 0.05:
                        self.spot = hit
                        return ""
            await asyncio.sleep(0.5)
        s = await self.where(near=False)
        return (f"after the time gauge, the screen is {'the ' + s.map + ' map' if s else 'not a map'}"
                f", not {era}")

    async def travel(self, m: WorldMap, name: str) -> str:
        """Go to a named place by the Epoch: board it, change era when the place is in
        another, fly near its door, land and walk in."""
        t0 = time.monotonic()
        s = await self.where(near=False)
        if s is None:
            return "not on a world map now (inside somewhere, a menu or a fade): go out first"
        here = self.atlas.map(s.map)
        door = m.entrance(name)
        if door is None:
            return (f"{name} has no surveyed door on the {m.name} map, so there's nowhere "
                    "known to land for it")
        target = _middle(door)
        if not await self._flying_now(here, target if here.name == m.name else None):
            why = await self.board(here, target if here.name == m.name else
                                   (s.x, s.y - 64))
            if why:
                return why
        era = era_of(m.name)
        if era and era_of(here.name) != era:
            why = await self.set_era(era)
            if why:
                return why
            s = await self.where()
            if s is None or s.map != m.name:
                return (f"arrived in {era}, on the {s.map if s else '?'} map, not "
                        f"{m.name}: {name} isn't on this one now")
        why = await self.land(m, door, name)
        if why:
            return why
        self.log(f"epoch: landed, {time.monotonic() - t0:.0f} s; walking in")
        left = max(30.0, 90.0 - (time.monotonic() - t0))
        out = await walk_to(self.atlas, m, name, self.shoot, self.steer, self.press,
                            self.keep_going, self.log, left)
        if out.startswith("arrived"):
            return (f"arrived: flew the Epoch and went into {name} "
                    f"({time.monotonic() - t0:.0f} s)")
        return f"landed the Epoch near {name}, then: {out}"

    async def land_here(self, m: WorldMap) -> str:
        """Land by the nearest places: on ground known to take it first."""
        s = await self.where() or await self.where(near=False)
        if s is None:
            return "not on a world map"
        p = self.here(s)
        groups = sorted(((cells, label) for cells, label in m.door_groups()),
                        key=lambda g: math.dist(_middle(g[0]), p))
        for cells, label in groups[:3]:
            if math.dist(_middle(cells), p) > 600:
                break
            why = await self.land(m, cells, label or "a place", limit=4)
            if not why:
                return ""
            if why.startswith(("lost", "the screen")):
                return why
        return "found nowhere to land near here"

    async def go_to_era(self, era: str, check=None) -> str:
        """Board the Epoch, go to an era by the time gauge and land near where it
        arrives."""
        t0 = time.monotonic()
        s = await self.where(near=False)
        if s is None:
            return "not on a world map now (inside somewhere, a menu or a fade): go out first"
        here = self.atlas.map(s.map)
        if era_of(here.name) == era:
            return f"already in {era} (the {here.name} map)"
        if not await self._flying_now(here, None):
            why = await self.board(here, (s.x, s.y - 64))
            if why:
                return why
        why = await self.set_era(era, check)
        if why:
            return why
        s = await self.where() or await self.where(near=False)
        m = self.atlas.map(s.map) if s is not None else None
        if m is None:
            return f"went to {era}, but the screen isn't a map now"
        if not m.surveyed:
            park(self.notes, self.game, m.name, *self.here(s), flying=True)
            return (f"arrived in {era} ({m.name} map), still flying the Epoch: no surveyed "
                    "ground here to land on")
        why = await self.land_here(m)
        if why:
            s = await self.where(near=False)
            if s is not None:
                park(self.notes, self.game, m.name, *self.here(s), flying=True)
            return f"arrived in {era} ({m.name} map), still flying the Epoch: {why}"
        near = m.nearest(*self.here(self.spot))[0] if self.spot and m.places else ""
        return (f"arrived: in {era} ({m.name} map), landed"
                f"{' near ' + near if near else ''} ({time.monotonic() - t0:.0f} s)")

    async def _flying_now(self, m: WorldMap, toward: tuple[float, float] | None) -> bool:
        """Already in the Epoch? Not when it's parked somewhere else in view; else a
        short hold says (flying goes about twice as far), toward where it's going."""
        s = await self.where(near=False)
        if s is None:
            return False
        at = await self.find_epoch(m, s)
        if at is not None and math.dist(self.here(s), at) > 4:
            return False  # parked over there: on foot
        angle = _angle(self.here(s), toward) if toward is not None else 0.0
        return bool(await self.flying(angle))


def _middle(cells: set[tuple[int, int]]) -> tuple[float, float]:
    r = sum(c[0] for c in cells) / len(cells)
    c = sum(c[1] for c in cells) / len(cells)
    return c * CELL, r * CELL


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """The d-pad way from a to b: one of eight."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    sx = (dx > 3) - (dx < -3)
    sy = (dy > 3) - (dy < -3)
    if not sx and not sy:
        return 0.0
    return math.degrees(math.atan2(sx, -sy)) % 360


def epoch_ok(atlas: Atlas) -> bool:
    """Whether this game's atlas has the Epoch (scripts/epoch_survey.py made epoch.json)."""
    folder = getattr(atlas, "folder", None)
    return folder is not None and (folder / "epoch.json").exists()


def needs_epoch(walked: str) -> bool:
    """Whether walk_to's answer means walking can't get there (another continent)."""
    return walked.startswith(("no way to", "blocked: no way"))


__all__ = ["EPOCH", "GAUGE", "Hangar", "Pilot", "Sprites", "compass", "epoch_ok", "era_named",
           "era_of", "landing_spots", "needs_epoch", "park", "parked", "reachable", "slug"]
