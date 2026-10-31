"""Survey the places inside a game's world map (towns, houses, castles, forests): which
tiles can be walked, where each way out goes, and a full picture of the place stitched
from the game's own frames, by walking it in the emulator with the game's memory saying
where the character and the camera are. src/playertwo/rooms.py finds the player on the
pictures and walks the ground to a named way out.

    python scripts/survey_rooms.py scripts/tasks/ct_rooms.toml --door 0x1 [--follow 0x2]
        [--box 11] [--speed 2] [--minutes 8] [--diagonal]
    python scripts/survey_rooms.py scripts/tasks/ct_rooms.toml --state STATE --follow 0x188
    python scripts/survey_rooms.py scripts/tasks/ct_rooms.toml --names     # Muse, no box
    python scripts/survey_rooms.py scripts/tasks/ct_rooms.toml --sprites   # the player's look

--door LOC enters the place whose world-map door goes to LOC (from the world map's survey,
scripts/survey.py) from the [rooms] start state, surveys it, then follows its ways out to
the places in --follow (or every place, with --follow all), entering each from the state
saved on arriving there. Runs in a hidden omabox desktop (scripts/emu_tasks.py's Box) with
the sprites switched off in the emulator, so the pictures are the ground alone.

[game.ram] in the task file names the memory: field_x, field_y (the objects' places, in
1/16 pixels), player (which object the player walks), cam_x, cam_y (the view's top left, in 1/32 pixels) and location. Writes
~/.config/player-two/atlas/<game>/rooms/loc-<hex>.json beside loc-<hex>.png:

    {"loc": 2, "name": "Crono's house, upstairs", "tile": 16,
     "size": [w, h],                       # the picture, in the game's pixels
     "walk": [[row, col], ...], "wall": [[row, col], ...],
     "exits": [{"from": [row, col], "dir": [dx, dy], "to": 1, "at": [x, y]}, ...],
     "cams": [[x, y], ...]}                # where the view has been: the picture's parts
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emu_tasks import EMU, Box
from survey import Game

from playertwo.atlas import ATLASES, slug

TILE = 16  # field maps are made of 16-pixel tiles
SPEED = 2.0
DIRS = {(0, -1): ["Up"], (1, 0): ["Right"], (0, 1): ["Down"], (-1, 0): ["Left"]}
# With --diagonal, the ways between too: ground only walked slantwise (the Mountain of
# Woe's chain bridges) is tried as well, after the four straight ways.
DIAGONALS = {(1, -1): ["Up", "Right"], (1, 1): ["Down", "Right"], (-1, 1): ["Down", "Left"],
             (-1, -1): ["Up", "Left"]}
TOP = 8  # the emulator's raw frame has 8 blank lines above the picture
HIGH = 208  # and the game draws a field 211 lines high: the rest of the 224 is black
# With --moving-ground: places whose ground never stands still (the Mountain of Woe's
# drifting mist) keep a frame that doesn't settle if the view didn't move while it was
# taken; the picture's middle value over several such frames is the ground.
MOVING = False


def secs(s: float) -> float:
    return max(0.03, s / SPEED)


class Between(RuntimeError):
    """Not in a place just now (a fade between two)."""


class Battle(RuntimeError):
    """A fight started (an enemy touched)."""


class Field(Game):
    """The game on a field map (inside a place): where the character and the view are."""

    def read(self, off: int, n: int) -> bytes:
        for i in range(12):
            if i >= 5:
                time.sleep(0.25)
            r = self.box.cmd(f"READ_CORE_RAM {off:x} {n}", wait=0.5).split()
            if len(r) == n + 2:
                return bytes(int(x, 16) for x in r[2:])
        raise RuntimeError(f"memory read failed at {off:x}")

    def pos(self) -> tuple[float, float]:
        """The leading character: the objects' x and y are two tables, and which object
        the player walks is a byte of its own (twice the object's number)."""
        i = self.read(self.ram["player"], 1)[0]
        x, y = self.ram["field_x"] + i, self.ram["field_y"] + i
        b = self.read(x, y - x + 2)
        p = ((b[0] | b[1] << 8) / 16, (b[y - x] | b[y - x + 1] << 8) / 16)
        if p[0] >= 0xF00 or p[1] >= 0xF00:  # no such object: between places
            raise Between()
        if p == (0.0, 0.0) and "battle" in self.ram and self.read(self.ram["battle"], 1)[0]:
            raise Battle()
        return p

    def flee(self, log) -> None:
        """Run from a fight (hold L and R), or fight it out (attack) if that fails."""
        self.hold(["q", "e"])  # L and R in the emulator's keys
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < 25:
                time.sleep(0.3)
                if not self.read(self.ram["battle"], 1)[0]:
                    log("ran from a fight")
                    return
        finally:
            self.hold([])
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            self.press("Return", 0.08)
            time.sleep(0.25)
            if not self.read(self.ram["battle"], 1)[0]:
                log("won a fight")
                return
        raise RuntimeError("stuck in a fight")

    def cam(self) -> tuple[int, int]:
        b = self.read(self.ram["cam_x"], self.ram["cam_y"] - self.ram["cam_x"] + 2)
        n = self.ram["cam_y"] - self.ram["cam_x"]
        return (b[0] | b[1] << 8) // 32, (b[n] | b[n + 1] << 8) // 32

    def scroll(self) -> tuple[int, int]:
        """Where the view really is, to the pixel: cam() is the view rounded down to 8
        pixels (or held at a map's edge), and the picture glides up to 7 pixels past it.
        The background's own scroll registers say by how much: their low 3 bits (each
        place adds its own multiple of 8 to them, so only those bits agree)."""
        c = self.cam()
        if "scroll_x" not in self.ram:
            return c
        sx = self.read(self.ram["scroll_x"], 1)[0]
        sy = self.read(self.ram["scroll_y"], 1)[0]
        return c[0] // 8 * 8 + sx % 8, c[1] // 8 * 8 + sy % 8


def tile_of(p: tuple[float, float]) -> tuple[int, int]:
    return int(p[1]) // TILE, int(p[0]) // TILE


def middle(t: tuple[int, int]) -> tuple[float, float]:
    """Where to stand on a tile: across, its middle; down, low on it (the feet come to
    rest at a tile's bottom edge when the one above is shut)."""
    return t[1] * TILE + TILE / 2, t[0] * TILE + 13


class Shots:
    """The game's raw frames (RetroArch's SCREENSHOT: the emulator's own pixels), each
    at a known view position, pasted into one picture of the whole place. Kept on disk
    too (`keep`), so the picture can be made again without walking the place."""

    def __init__(self, box: Box | None, keep: Path | None = None):
        self.box = box
        self.keep = keep
        self.dir = box.home / ".config/retroarch/screenshots" if box else None
        self.frames: dict[tuple[int, int], list[np.ndarray]] = collections.defaultdict(list)

    @classmethod
    def load(cls, keep: Path) -> Shots:
        from PIL import Image

        shots = cls(None, keep)
        for f in sorted(keep.glob("*.png")):
            cx, cy, _ = (int(v) for v in f.stem.split("_"))
            shots.frames[(cx, cy)].append(np.asarray(Image.open(f).convert("RGB")))
        return shots

    def grab(self) -> np.ndarray | None:
        from PIL import Image

        self.dir.mkdir(parents=True, exist_ok=True)
        before = set(self.dir.glob("*.png"))
        self.box.cmd("SCREENSHOT", wait=0.3)
        for _ in range(60):
            new = set(self.dir.glob("*.png")) - before
            if new:
                f = new.pop()
                a = None
                for _ in range(10):  # until it's all written
                    time.sleep(0.05)
                    try:
                        with Image.open(f) as im:
                            a = np.asarray(im.convert("RGB")).copy()
                        break
                    except (OSError, ValueError):
                        continue
                f.unlink(missing_ok=True)
                return a
            time.sleep(0.02)
        return None

    def take(self, cam) -> None:
        """A frame of the view, if it's standing still: `cam` reads where the view is,
        before and after (the view glides on a little after the character stops)."""
        from PIL import Image

        c = cam()
        if len(self.frames.get(c, [])) >= 2:
            return
        time.sleep(secs(0.3))
        if cam() != c:
            return
        a = self.grab()
        for _ in range(4):  # the picture still gliding: two frames the same, or none
            b = self.grab()
            if a is None or b is None or a.shape != b.shape:
                return
            if float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < 1.5:
                break
            a = b
        else:
            if not MOVING:
                return
        if a.shape[0] < TOP + 224 or cam() != c:
            return
        f = b[TOP:TOP + HIGH]
        self.frames[c].append(f)
        if self.keep is not None:
            self.keep.mkdir(parents=True, exist_ok=True)
            Image.fromarray(f).save(self.keep / f"{c[0]}_{c[1]}_{len(self.frames[c])}.png")

    def _median(self, frames: list[tuple[tuple[int, int], np.ndarray]], size) -> np.ndarray:
        w, h = size
        layers = 5
        stack = np.full((layers, h, w, 3), np.nan, np.float32)
        count = np.zeros((h, w), np.int16)
        for (cx, cy), f in frames:
            region = count[cy:cy + HIGH, cx:cx + 256]
            for k in range(layers):
                m = region == k
                if m.any():
                    stack[k, cy:cy + HIGH, cx:cx + 256][m] = f[m]
            count[cy:cy + HIGH, cx:cx + 256] += 1
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pixels no frame saw
            pic = np.nanmedian(stack, axis=0)
        return np.nan_to_num(pic, nan=0.0)

    @staticmethod
    def _register(frames, pic: np.ndarray, reach: int = 8):
        """Each frame where it best fits the picture of all of them, within `reach`
        pixels of where the game's memory put it (the view can still be gliding when
        the memory has settled)."""
        import cv2

        out = []
        h, w = pic.shape[:2]
        p32 = pic.astype(np.float32)
        for (cx, cy), f in frames:
            t = f[16:HIGH - 16, 16:240].astype(np.float32)
            x0, y0 = max(0, cx + 16 - reach), max(0, cy + 16 - reach)
            win = p32[y0:min(h, cy + HIGH - 16 + reach), x0:min(w, cx + 240 + reach)]
            if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                out.append(((cx, cy), f))
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_SQDIFF)
            _, _, (dx, dy), _ = cv2.minMaxLoc(r)
            out.append(((x0 + dx - 16, y0 + dy - 16), f))
        # Where the frames agree with the memory most often is where the memory is
        # right: the picture keeps the game's own coordinates.
        shifts = collections.Counter((a[0] - c[0], a[1] - c[1])
                                     for (c, _), (a, _) in zip(frames, out, strict=True))
        sx, sy = shifts.most_common(1)[0][0]
        return [((max(0, x - sx), max(0, y - sy)), f) for (x, y), f in out]

    def picture(self) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
        """The place's picture: each pixel the middle value over the frames that saw it
        (water and torches move; a frame caught mid-fade is outvoted), made twice: the
        second time without frames that disagree with the first (put in the wrong place)."""
        if not self.frames:
            return None
        frames = [(c, f) for c, fs in sorted(self.frames.items()) for f in fs]
        if not frames:
            return None
        size = (max(c[0] for c, _ in frames) + 256 + 8, max(c[1] for c, _ in frames) + 224 + 8)
        pic = self._median(frames, size)
        if os.environ.get("ROOMS_REGISTER") == "1":
            frames = self._register(frames, pic)
            pic = self._median(frames, size)
        errs = [float(np.abs(f.astype(np.float32) - pic[cy:cy + HIGH, cx:cx + 256]).mean())
                for (cx, cy), f in frames]
        limit = max(10.0, 2.5 * float(np.median(errs)))
        good = [fr for fr, e in zip(frames, errs, strict=True) if e <= limit]
        if len(good) < len(frames):
            pic = self._median(good, size)
        return pic.astype(np.uint8), sorted({c for c, _ in good})


class RoomSurvey:
    def __init__(self, g: Field, loc: int, state: Path, log):
        self.g, self.loc, self.state, self.log = g, loc, state, log
        self.walk: set[tuple[int, int]] = set()
        self.wall: collections.Counter = collections.Counter()
        self.tried: set = set()
        self.failed: collections.Counter = collections.Counter()
        self.doorways: set[tuple[int, int]] = set()  # tiles a way out was taken from
        self.exits: list[dict] = []
        self.exit_keys: set = set()
        # (where a way out leads, where it comes out) -> a state there
        self.arrivals: dict[tuple[int, tuple], Path] = {}
        self.worlds: set[int] = set()  # world maps' location numbers
        self.shots = Shots(g.box, Path(str(state.parent)) / "frames" / f"{loc:03x}")
        self.last_cam: tuple[int, int] | None = None
        self.out: Path | None = None  # saved to as it goes
        self.ram_battle = g.ram.get("battle", 0)

    # -- moving --------------------------------------------------------------------
    def snap(self) -> None:
        """A frame of the view here, once it has stopped moving."""
        c = self.g.scroll()
        time.sleep(secs(0.12))
        if self.g.scroll() == c and self.g.location() == self.loc and c != self.last_cam:
            self.shots.take(self.g.scroll)
            self.last_cam = c

    def ray(self, d: tuple[int, int], tiles: int = 12) -> str:
        """Hold one way until stopped, `tiles` tiles on, or out of the place: 'moved',
        'blocked' or 'left'."""
        g = self.g
        start = last = g.pos()
        self.walk.add(tile_of(start))
        g.hold(DIRS[d])
        moved_t = time.monotonic()
        out = "moved"
        try:
            while True:
                time.sleep(0.005)
                try:
                    p = g.pos()
                except Between:
                    out = "left"
                    break
                if p != last:
                    if math.dist(p, last) > 2 * TILE:  # moved by the game, not walking
                        out = "left" if g.location() != self.loc else "jumped"
                        break
                    self.line(last, p)
                    last, moved_t = p, time.monotonic()
                    if math.dist(p, start) > tiles * TILE:
                        break
                elif time.monotonic() - moved_t > secs(0.18):
                    out = "blocked"
                    break
                if g.location() != self.loc:
                    out = "left"
                    break
        finally:
            g.hold([])
        if out != "left":
            time.sleep(secs(0.1))
            if g.location() != self.loc:
                out = "left"
        if out == "left":
            self.went_out(tile_of(last), d)
            return out
        if out == "blocked":
            ahead = (tile_of(last)[0] + d[1], tile_of(last)[1] + d[0])
            if ahead not in self.walk:
                self.wall[ahead] += 1
        return out

    def line(self, a, b) -> None:
        n = max(1, int(math.dist(a, b) // 4))
        for i in range(n + 1):
            t = tile_of((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n))
            self.walk.add(t)

    def went_out(self, frm: tuple[int, int], d: tuple[int, int]) -> None:
        """Walked out of the place: note where to and keep a state there, then load the
        place again and walk back."""
        g = self.g
        time.sleep(secs(1.5))
        at: tuple[float, float] = (0.0, 0.0)
        for _ in range(30):  # the fade and the new place's start
            to = g.location()
            try:
                p1 = g.pos()
                time.sleep(secs(0.25))
                if g.location() == to and g.pos() == p1:
                    at = p1
                    break
            except Between:
                time.sleep(secs(0.25))
        to = g.location()
        if to in self.worlds:  # where on the world map: its own memory
            time.sleep(secs(0.5))
            at = Game.pos(g)
        key = (frm, d)
        if key not in self.exit_keys and to != self.loc:
            self.exit_keys.add(key)
            self.exits.append({"from": list(frm), "dir": list(d), "to": to,
                               "at": [round(at[0], 1), round(at[1], 1)]})
            self.log(f"way out at {frm} going {d} -> location {to:#x} at {at}")
            # A state there for each place it leads to, and for each end of a big one
            # (a forest's two sides on the world map).
            same = [a for (t, a) in self.arrivals if t == to]
            if all(math.dist(a, at) > 64 for a in same):
                n = f"-{len(same)}" if same else ""
                self.arrivals[(to, tuple(at))] = save_state(
                    g.box, self.state.parent / f"arrive-{self.loc:x}-{to:x}{n}.state")
        self.walk.discard((frm[0] + d[1], frm[1] + d[0]))
        # The tile it left from is (partly) the way out itself: nothing more to try from
        # it; carry on from the nearest other one.
        self.doorways.add(frm)
        self.reload()
        came = self.bfs(tile_of(self.g.pos()))
        back = [t for t in came if t not in self.doorways]
        if back:
            self.walk_to(min(back, key=lambda t: abs(t[0] - frm[0]) + abs(t[1] - frm[1])))

    def reload(self) -> None:
        self.g.hold([])
        self.g.box.load(self.state)
        time.sleep(secs(0.5))
        if self.g.location() != self.loc:
            raise RuntimeError(f"the state for {self.loc:#x} loaded somewhere else")

    def centre(self, t: tuple[int, int]) -> bool:
        """Stand in the middle of a tile (rays then run along its middle), or as near as
        the walls let it (they aren't all on the tiles' edges): False if off the tile."""
        g = self.g
        tx, ty = middle(t)
        t0 = time.monotonic()
        last, moved = None, t0
        while time.monotonic() - t0 < secs(2.5):
            try:
                p = g.pos()
            except Between:
                g.hold([])
                self.reload()
                return False
            dx, dy = tx - p[0], ty - p[1]
            if abs(dx) <= 3.5 and abs(dy) <= 3.5:
                g.hold([])
                return True
            if p != last:
                last, moved = p, time.monotonic()
            elif time.monotonic() - moved > secs(0.4):
                g.hold([])
                return tile_of(p) == t  # pressed against something, on the tile
            keys = ((["Right"] if dx > 3.5 else ["Left"] if dx < -3.5 else [])
                    + (["Down"] if dy > 3.5 else ["Up"] if dy < -3.5 else []))
            if max(abs(dx), abs(dy)) < 8:  # close: a tap
                g.hold(keys)
                time.sleep(0.02)
                g.hold([])
                time.sleep(secs(0.05))
            else:
                g.hold(keys)
                time.sleep(0.01)
            if g.location() != self.loc:
                g.hold([])
                return False
        g.hold([])
        return tile_of(g.pos()) == t

    def go(self, path: list[tuple[int, int]]) -> bool:
        """Walk tile to tile along known ground; False if something stopped it. Going out
        of the place on the way (a way out next to the path) is noted like a ray's."""
        g = self.g
        for t in path[1:]:
            tx, ty = middle(t)
            t0 = moved = time.monotonic()
            last = g.pos()
            keys: list[str] = []
            while True:
                try:
                    p = g.pos()
                except Between:
                    p = None
                if p is None or g.location() != self.loc:
                    g.hold([])
                    d = ((1 if "Right" in keys else -1 if "Left" in keys else 0),
                         (1 if "Down" in keys else -1 if "Up" in keys else 0))
                    if d != (0, 0) and d in DIRS:
                        self.went_out(tile_of(last), d)
                    else:
                        self.reload()
                    return False
                dx, dy = tx - p[0], ty - p[1]
                if abs(dx) <= 5 and abs(dy) <= 4:
                    break
                keys = ((["Right"] if dx > 4 else ["Left"] if dx < -4 else [])
                        + (["Down"] if dy > 3.5 else ["Up"] if dy < -3.5 else []))
                g.hold(keys)
                if p != last:
                    last, moved = p, time.monotonic()
                if time.monotonic() - moved > secs(0.5) or time.monotonic() - t0 > secs(4):
                    g.hold([])
                    if tile_of(p) == t:
                        break  # on it, pressed against something: near enough
                    # Walked once, but not from here now (a person in the way, or a
                    # step only one way): plan around it after a second time.
                    self.failed[(tile_of(p), t)] += 1
                    return False
                time.sleep(0.008)
        g.hold([])
        return self.centre(path[-1])

    def neighbours(self, t):
        for d in DIRS:
            n = (t[0] + d[1], t[1] + d[0])
            if self.failed[(t, n)] < 2:
                yield n, d

    def bfs(self, start):
        came = {start: None}
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v, _ in self.neighbours(u):
                if v in self.walk and v not in came:
                    came[v] = u
                    q.append(v)
        return came

    def path_to(self, target) -> list | None:
        came = self.bfs(tile_of(self.g.pos()))
        if target not in came:
            return None
        path, u = [], target
        while u is not None:
            path.append(u)
            u = came[u]
        return path[::-1]

    def walk_to(self, target) -> bool:
        path = self.path_to(target)
        return path is not None and self.go(path)

    def unknown(self, t):
        if t in self.doorways:
            return []
        return [d for d in DIRS if (t, d) not in self.tried
                and (n := (t[0] + d[1], t[1] + d[0])) not in self.walk
                and self.wall[n] < 2]

    # -- the survey ------------------------------------------------------------------
    def explore(self, minutes: float) -> None:
        end = time.monotonic() + minutes * 60
        here = tile_of(self.g.pos())
        self.walk.add(here)
        self.snap()
        rays = jumps = still = 0
        last = None
        beat = time.monotonic()
        while time.monotonic() < end:
            p = self.g.pos()
            still = still + 1 if p == last else 0
            last = p
            if time.monotonic() - beat > 60:
                beat = time.monotonic()
                self.log(f"at {tile_of(p)}, {rays} rays, {len(self.walk)} walkable")
            if still >= 6:
                # Not moved for several tries: the game has taken over (a scene, a text
                # box, a fight about to start).
                still = 0
                self.recover()
            try:
                got = self.step(rays, jumps)
            except Battle:
                self.g.hold([])
                self.g.flee(self.log)
                time.sleep(secs(1.0))
                continue
            except Between:
                # Caught between places (a way out taken while walking back): start
                # again from the place's state.
                self.g.hold([])
                time.sleep(secs(1.5))
                self.reload()
                continue
            if got is None:
                break
            rays, jumps = got

    def recover(self) -> None:
        g = self.g
        p = g.pos()
        self.log(f"not moving at {tile_of(p)}: a scene or a fight? pressing on")
        for _ in range(4):
            g.press("Return", 0.08)
            time.sleep(secs(0.4))
        if self.ram_battle and g.read(self.ram_battle, 1)[0]:
            g.flee(self.log)
        for k in ("Up", "Down", "Left", "Right"):
            g.press(k, 0.15)
            time.sleep(secs(0.2))
            if g.pos() != p:
                return
        self.log("still not moving: loading the place again")
        self.walk.discard(tile_of(p))
        self.wall[tile_of(p)] += 2  # don't come back here
        self.reload()

    def step(self, rays: int, jumps: int) -> tuple[int, int] | None:
        """One ray from here, or a walk to the nearest tile with ways not yet tried;
        None when there's nothing left."""
        here = tile_of(self.g.pos())
        self.walk.add(here)
        dirs = self.unknown(here)
        if dirs:
            d = dirs[0]
            self.tried.add((here, d))
            self.centre(here)
            out = self.ray(d)
            rays += 1
            if out == "jumped":
                jumps += 1
                self.log(f"moved by the game at {here} (an event?): loading again")
                self.reload()
                if jumps > 8:
                    raise RuntimeError("the game keeps taking over")
                self.walk_to(here)
            elif out != "left":
                self.snap()
            if rays % 40 == 0:
                if self.out is not None:
                    self.save(self.out, keep=True)
                self.log(f"{rays} rays, {len(self.walk)} walkable, "
                         f"{sum(1 for v in self.wall.values() if v)} shut, "
                         f"{len(self.exits)} ways out, {len(self.shots.frames)} views")
            return rays, jumps
        came = self.bfs(here)
        goals = [t for t in came if self.unknown(t)]
        if not goals:
            self.log("nothing left to explore")
            return None
        dist = {}
        for t in goals:
            n, u = 0, t
            while came[u] is not None:
                u, n = came[u], n + 1
            dist[t] = n
        goal = min(goals, key=dist.get)
        path, u = [], goal
        while u is not None:
            path.append(u)
            u = came[u]
        if not self.go(path[::-1]):
            for d in DIRS:
                self.tried.add((goal, d))
            self.log(f"couldn't reach {goal}, stopped at {tile_of(self.g.pos())}")
        return rays, jumps

    def views(self, minutes: float = 6.0) -> None:
        """Make sure the picture covers the whole place: go to the nearest tile whose
        view hasn't been pictured yet (the view follows the character), picture it, and
        so on, from wherever it gets to."""
        end = time.monotonic() + minutes * 60
        skip: set[tuple[int, int]] = set()
        self.reload()

        def covered(t) -> bool:
            x, y = middle(t)
            return any(abs(cx + 128 - x) < 72 and abs(cy + 138 - y) < 64
                       for cx, cy in [c for c, f in self.shots.frames.items() if f])

        while time.monotonic() < end:
            try:
                if self.g.location() != self.loc:
                    self.reload()
                here = tile_of(self.g.pos())
                came = self.bfs(here)
                todo = [t for t in came if t not in skip and not covered(t)]
                if not todo:
                    break
                dist = {}
                for t in todo:
                    n, u = 0, t
                    while came[u] is not None:
                        u, n = came[u], n + 1
                    dist[t] = n
                goal = min(todo, key=dist.get)
                path, u = [], goal
                while u is not None:
                    path.append(u)
                    u = came[u]
                self.go(path[::-1])
                self.last_cam = None
                self.snap()
                if not covered(goal):
                    skip.add(goal)
            except Battle:
                self.g.hold([])
                self.g.flee(self.log)
            except Between:
                self.g.hold([])
                time.sleep(secs(1.5))
                self.reload()
        self.log(f"pictured {sum(1 for f in self.shots.frames.values() if f)} views")

    def save(self, out: Path, name: str = "", keep: bool = False) -> None:
        """keep: as it goes (the picture only at the end: it takes a while)."""
        from PIL import Image

        made = None if keep else self.shots.picture()
        data = {"loc": self.loc, "name": name, "tile": TILE,
                "walk": sorted(self.walk),
                "wall": sorted(t for t, n in self.wall.items() if n and t not in self.walk),
                "exits": self.exits}
        if made is not None:
            pic, cams = made
            Image.fromarray(pic).save(out.with_suffix(".png"))
            data["size"] = [pic.shape[1], pic.shape[0]]
            data["cams"] = sorted(cams)
        old = json.loads(out.read_text()) if out.exists() else {}
        for k in ("name", "spots", "size", "cams"):  # kept: names given since, the picture
            if old.get(k) and not data.get(k):
                data[k] = old[k]
        out.write_text(json.dumps(data))


def deep(s: RoomSurvey, dest: Path) -> Path | None:
    """Stand on the tile furthest from every way out and save a state there: a start
    for trips out of the place that aren't one step."""
    starts = {tuple(e["from"]) for e in s.exits}
    if not starts:
        return None
    dist = {t: 0 for t in starts if t in s.walk}
    q = collections.deque(dist)
    while q:
        u = q.popleft()
        for v, _ in s.neighbours(u):
            if v in s.walk and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    s.reload()
    here = s.bfs(tile_of(s.g.pos()))
    for t in sorted(dist, key=dist.get, reverse=True)[:8]:
        if t in here and s.walk_to(t):
            s.log(f"furthest from the ways out: {t}, {dist[t]} steps")
            return save_state(s.g.box, dest)
    return None


def save_state(box: Box, dest: Path) -> Path:
    box.cmd("SAVE_STATE", wait=0.3)
    time.sleep(0.8)
    made = box.home / ".config/retroarch/states" / box.game["core_name"] / \
        f"{box.rom.stem}.state"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(made, dest)
    return dest


def boot(box: Box, speed: float, sprites: bool = False) -> None:
    """The emulator in a box, at `speed`, drawing the ground only (no sprites: people
    walking about would end up in the pictures), its raw frames for screenshots."""
    opt = EMU / f"rooms-core-{box.port}.opt"
    opt.write_text('snes9x_overscan = "disabled"\n'
                   f'snes9x_layer_5 = "{"enabled" if sprites else "disabled"}"\n')
    box.up(f'fastforward_ratio = {speed}\nvideo_gpu_screenshot = "false"\n'
           'video_font_enable = "false"\nglobal_core_options = "true"\n'
           f'core_options_path = "{opt}"\n')
    if speed > 1:
        box.cmd("FAST_FORWARD", wait=0.2)


def enter_door(g: Field, world_survey: Path, state: Path, loc: int, log) -> bool:
    """From a world-map state, walk to the door that goes to `loc` and go in."""
    import survey as sv

    d = json.loads(world_survey.read_text())
    g.box.load(state)
    s = sv.Survey(g, (8192, 8192), d["world"], log)
    s.walk = {tuple(c) for c in d["walk"]}
    doors = [(r, c) for r, c, v in d["doors"] if v == loc]
    if not doors:
        return False
    s.walk |= set(doors)
    wg = Game(g.box, g.ram)  # world-map memory: x, y
    s.g = wg
    sv.SPEED = SPEED
    came = s.bfs(sv.cell_of(wg.pos()))
    def path_to(c):
        out, u = [], c
        while u is not None:
            out.append(u)
            u = came[u]
        return out[::-1]

    # The whole way at once, then a cell at a time from each side of each door tile
    # (one of them goes in: a diagonal step can catch on a corner by the door).
    tries = [(door, path_to(door), False) for door in doors if door in came]
    tries += [(door, [*path_to(n), door], True) for door in doors
              for n in ((door[0] + 1, door[1]), (door[0] - 1, door[1]),
                        (door[0], door[1] + 1), (door[0], door[1] - 1)) if n in came]
    for door, path, stepwise in tries:
        g.box.load(state)
        if stepwise:
            for c in path[1:]:  # from where it really is each time
                s.go([sv.cell_of(wg.pos()), c])
        else:
            s.go(path)
        for _ in range(2):
            wg.press("Return", 0.12)
            time.sleep(max(secs(2.5), 0.8))
            if g.location() == loc:
                return True
        log(f"the door at {door} didn't open (stopped at {sv.cell_of(wg.pos())})")
    return False


NAME_PROMPT = """This is the whole of one place inside {game} (a room, a building's floor, a
town or a forest), stitched from the game's screens with the people left out (black is
what wasn't seen). {known}
Ways out lead to: {exits}.
1. Name it the way a player would say it, short, with the building or town it's part of
("Crono's house, upstairs", "Leene Square, the bell", "Guardia Castle, throne room").
2. List up to 8 things in it a player might ask to go to (a counter, a bed, stairs, a
bell, a fountain, a tent's door, a chest, a sign), each with where its middle is in the
picture: x from 0 (left edge) to 1000 (right edge), y from 0 (top) to 1000 (bottom).
JSON: {{"name": "...", "looks": "what's in it, a few words",
"spots": [{{"name": "the counter", "x": 0, "y": 0}}]}}"""


def name_rooms(folder: Path, game: str, known: dict[str, str]) -> dict[int, dict]:
    """Each surveyed place's name (the world map's for a place entered from it, Muse's
    reading of the picture for the rest: rooms, floors) and the things in it to go to,
    in the picture's pixels: {loc: {"name": ..., "spots": {name: [x, y]}}}."""
    import asyncio

    from PIL import Image

    from playertwo.config import Settings
    from playertwo.muse import Muse, image_part

    muse = Muse(Settings.load())
    rooms = {int(f.stem.split("-")[1], 16): json.loads(f.read_text())
             for f in sorted(folder.glob("loc-*.json"))}

    def label(loc: int) -> str:
        if str(loc) in known:
            return known[str(loc)]
        if rooms.get(loc, {}).get("name"):
            return rooms[loc]["name"]
        return f"place {loc:#x}"

    async def one(loc: int, r: dict) -> tuple[int, dict | None]:
        img = Image.open(folder / f"loc-{loc:03x}.png")
        froms = [label(k) for k, o in rooms.items() if any(e["to"] == loc
                                                           for e in o["exits"])]
        hint = f"The world map calls it {known[str(loc)]!r}." if str(loc) in known else \
            (f"It's reached from: {', '.join(froms)}." if froms else "")
        exits = ", ".join(sorted({label(e["to"]) for e in r["exits"]})) or "none found"
        msgs = [{"role": "user", "content": [
            image_part(img, 1024),
            {"type": "text", "text": NAME_PROMPT.format(game=game, known=hint, exits=exits)}]}]
        try:
            out = await muse.json(msgs, effort="low", max_tokens=3000)
        except ValueError:
            return loc, None
        spots = {}
        for sp in out.get("spots") or []:
            try:
                spots[str(sp["name"])] = [round(float(sp["x"]) / 1000 * img.width),
                                          round(float(sp["y"]) / 1000 * img.height)]
            except (KeyError, TypeError, ValueError):
                continue
        return loc, {"name": out.get("name"), "spots": spots}

    async def run():
        return await asyncio.gather(*(one(k, r) for k, r in rooms.items()))

    return {k: v for k, v in asyncio.run(run()) if v}


def sprites(box: Box, g: Field, folder: Path, state: Path, log) -> None:
    """The player's character as it looks facing each way, standing and in mid-step,
    cut from frames with the sprites on against the place's picture: rooms.py finds the
    character with these when the view stops following it (at a place's edges)."""
    from PIL import Image

    loc = g.location()
    pic = np.asarray(Image.open(folder / f"loc-{loc:03x}.png").convert("RGB")).astype(int)
    shots = Shots(box)
    out: dict[str, list[float]] = {}

    def cut(name: str) -> None:
        p0, cam0 = g.pos(), g.cam()
        a = shots.grab()
        p1, cam1 = g.pos(), g.cam()
        if a is None or math.dist(p0, p1) > 6 or cam0 != cam1:
            return  # the frame is between the two readings: only near enough if slow
        x, y = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        cx, cy = cam0
        f = a[TOP:TOP + 224].astype(int)
        sx, sy = int(x - cx), int(y - cy)
        x0, y0 = max(0, sx - 16), max(0, sy - 40)
        crop = f[y0:sy + 8, x0:sx + 16]
        ground = pic[cy + y0:cy + sy + 8, cx + x0:cx + sx + 16]
        if crop.shape != ground.shape:
            return
        mask = np.abs(crop - ground).sum(2) > 24
        ys, xs = np.nonzero(mask)
        if len(ys) < 40:
            log(f"no sprite found for {name}")
            return
        b = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        img = np.zeros((b[3] - b[1], b[2] - b[0], 4), np.uint8)
        img[..., :3] = crop[b[1]:b[3], b[0]:b[2]]
        img[..., 3] = mask[b[1]:b[3], b[0]:b[2]] * 255
        Image.fromarray(img).save(folder / f"player-{name}.png")
        # Where the game's position (the feet) is, from the cut's top left.
        out[name] = [round(float(sx - (x0 + b[0])), 1), round(float(sy - (y0 + b[1])), 1)]
        log(f"player {name}: {img.shape[1]}x{img.shape[0]}, feet at {out[name]}")

    for name, key in (("down", "Down"), ("up", "Up"), ("left", "Left"), ("right", "Right")):
        box.load(state)
        g.press(key, 0.03)
        time.sleep(0.5)
        cut(name)
        g.hold([key])  # in mid-step: two moments of the walk
        time.sleep(0.25)
        cut(f"{name}-step1")
        time.sleep(0.13)
        cut(f"{name}-step2")
        g.hold([])
    (folder / "player.json").write_text(json.dumps({"anchor": out}))


def main() -> None:
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1)  # kill -USR1: where it is, on stderr
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--door", help="survey the place whose world-map door goes here (hex)")
    ap.add_argument("--state", help="or survey the place this state stands in")
    ap.add_argument("--follow", default="", help="places to go on to through ways out: "
                    "hex numbers, comma separated, or 'all'")
    ap.add_argument("--max", type=int, default=12, help="at most this many places")
    ap.add_argument("--box", type=int, default=11)
    ap.add_argument("--minutes", type=float, default=8, help="per place")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--names", action="store_true", help="name the surveyed places (Muse)")
    ap.add_argument("--sprites", action="store_true",
                    help="cut the player's sprite from the --state's place")
    ap.add_argument("--deep", action="store_true",
                    help="only save a state far from the ways out of the --state's place")
    ap.add_argument("--repaint", action="store_true",
                    help="only make the pictures again from the frames kept (no box)")
    ap.add_argument("--new-frames", action="store_true",
                    help="with --resume: picture the place again from scratch")
    ap.add_argument("--resume", action="store_true",
                    help="carry on from what an earlier survey of a place found")
    ap.add_argument("--keep-up", action="store_true")
    ap.add_argument("--moving-ground", action="store_true",
                    help="keep frames whose ground never settles (drifting mist)")
    ap.add_argument("--diagonal", action="store_true",
                    help="try the four slanting ways too (chain bridges, slopes)")
    args = ap.parse_args()
    spec = tomllib.loads(args.tasks.read_text())
    gspec = spec["game"]
    rspec = spec.get("rooms", {})
    folder = ATLASES / slug(gspec["name"]) / "rooms"
    folder.mkdir(parents=True, exist_ok=True)
    known = {str(int(k, 16)): v for k, v in rspec.get("names", {}).items()}
    if args.repaint:
        frames = Path(gspec["states"]).expanduser() / "rooms" / "frames"
        for d in sorted(frames.iterdir()):
            out = folder / f"loc-{d.name}.json"
            if not out.exists():
                continue
            shots = Shots.load(d)
            made = shots.picture()
            if made is None:
                continue
            from PIL import Image

            pic, cams = made
            Image.fromarray(pic).save(out.with_suffix(".png"))
            data = json.loads(out.read_text())
            data.update(size=[pic.shape[1], pic.shape[0]], cams=cams)
            out.write_text(json.dumps(data))
            n = sum(len(v) for v in shots.frames.values())
            print(f"{d.name}: {len(cams)} views kept of {len(shots.frames)} ({n} frames)")
        return
    if args.names:
        world = ATLASES / slug(gspec["name"]) / rspec["world_survey"]
        known = {**json.loads(world.read_text()).get("names", {}), **known}
        names = name_rooms(folder, gspec["name"], known)
        every = dict(known)
        for f in folder.glob("loc-*.json"):
            d = json.loads(f.read_text())
            got = names.get(d["loc"]) or {}
            n = known.get(str(d["loc"])) or got.get("name")
            if n:
                d["name"] = n
                every[str(d["loc"])] = n
            if got.get("spots"):
                d["spots"] = got["spots"]
            f.write_text(json.dumps(d))
            print(f"{d['loc']:#5x} {d.get('name')}: {', '.join(d.get('spots', {}))}")
        worlds = [int(str(w), 16) for w in rspec.get("worlds", [])]
        (folder / "names.json").write_text(json.dumps({"names": every, "worlds": worlds},
                                                      indent=1))
        return
    global SPEED, MOVING
    SPEED = args.speed
    MOVING = args.moving_ground
    if args.diagonal:
        DIRS.update(DIAGONALS)
    ram = {k: int(v, 16) if isinstance(v, str) else v for k, v in gspec["ram"].items()}
    box = Box(args.box, gspec)
    states = Path(gspec["states"]).expanduser() / "rooms"
    states.mkdir(parents=True, exist_ok=True)
    if args.keep_up and "PLAYING" in box.cmd("GET_STATUS"):
        box.home = Path(subprocess.run(["omabox", "path", "--home", "-b", box.name],
                                       check=False, capture_output=True,
                                       text=True).stdout.split()[-1])
    else:
        boot(box, args.speed, sprites=args.sprites)
    g = Field(box, ram)
    for k in ("Up", "Down", "Left", "Right", "q", "e", "Return"):
        g.key(k, "up")  # a run stopped mid-walk can leave keys down
    t0 = time.monotonic()

    def log(text: str) -> None:
        print(f"{time.monotonic() - t0:7.1f} {text}", flush=True)

    if args.sprites:
        box.load(Path(args.state).expanduser())
        sprites(box, g, folder, Path(args.state), log)
        return
    if args.deep:
        st = Path(args.state).expanduser()
        box.load(st)
        loc = g.location()
        s = RoomSurvey(g, loc, st, log)
        d = json.loads((folder / f"loc-{loc:03x}.json").read_text())
        s.walk = {tuple(t) for t in d["walk"]}
        s.exits = d["exits"]
        print(deep(s, states / f"deep-{loc:x}.state"))
        return
    if args.door:
        loc = int(args.door, 16)
        world_state = Path(gspec["states"]).expanduser() / f"{rspec['start']}.state"
        if not enter_door(g, ATLASES / slug(gspec["name"]) / rspec["world_survey"],
                          world_state, loc, log):
            sys.exit(f"couldn't go in at the door to {loc:#x}")
        time.sleep(secs(1.0))
        first = save_state(box, states / f"door-{loc:x}.state")
    else:
        box.load(Path(args.state).expanduser())
        first = Path(args.state).expanduser()
    follow = set() if not args.follow else (
        {"all"} if args.follow == "all" else {int(x, 16) for x in args.follow.split(",")})
    world = {int(str(w), 16) if isinstance(w, str) else w for w in rspec.get("worlds", [])}
    queue = collections.deque([first])
    done: set[int] = set()
    while queue:
        st = queue.popleft()
        box.load(st)
        time.sleep(secs(0.5))
        loc = g.location()
        if loc in done or loc in world or len(done) >= args.max:
            continue
        done.add(loc)
        log(f"surveying location {loc:#x} from {st.name}")
        s = RoomSurvey(g, loc, st, log)
        s.worlds = world
        out = folder / f"loc-{loc:03x}.json"
        s.out = out
        if args.resume and s.shots.keep.is_dir() and not args.new_frames:
            kept = Shots.load(s.shots.keep)
            s.shots.frames.update(kept.frames)
        elif s.shots.keep.is_dir():
            shutil.rmtree(s.shots.keep)  # a new survey: new frames
        if args.resume and out.exists():
            old = json.loads(out.read_text())
            s.walk = {tuple(t) for t in old["walk"]}
            s.wall.update({tuple(t): 2 for t in old.get("wall", [])})
            s.exits = old.get("exits", [])
            s.exit_keys = {(tuple(e["from"]), tuple(e["dir"])) for e in s.exits}
            s.doorways = {tuple(e["from"]) for e in s.exits}
            for e in s.exits:  # the states saved going through them before
                f = states / f"arrive-{loc:x}-{e['to']:x}.state"
                if f.exists() and all(t != e["to"] for t, _ in s.arrivals):
                    s.arrivals[(e["to"], tuple(e.get("at", (0, 0))))] = f
            log(f"carrying on from {len(s.walk)} walkable, {len(s.exits)} ways out")
        tries = 0
        while True:
            try:
                s.explore(args.minutes)
                s.views()
                break
            except RuntimeError as exc:
                tries += 1
                log(f"{type(exc).__name__} {exc}: loading the place again ({tries})")
                if tries > 8:
                    break
                if "PLAYING" not in box.cmd("GET_STATUS"):
                    boot(box, args.speed)
                s.reload()
        s.save(out, known.get(str(loc), ""))
        try:
            deep(s, states / f"deep-{loc:x}.state")
        except RuntimeError as exc:
            log(f"no deep state: {exc}")
        log(f"{loc:#x}: {len(s.walk)} walkable, {len(s.exits)} ways out to "
            f"{sorted({hex(e['to']) for e in s.exits})}, {len(s.shots.frames)} views -> {out}")
        for (to, _at), path in s.arrivals.items():
            if to not in done and to not in world and ("all" in follow or to in follow):
                queue.append(path)


if __name__ == "__main__":
    main()
