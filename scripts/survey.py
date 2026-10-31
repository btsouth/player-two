"""Survey a world map in an emulated game: which ground can be walked, and where each
place's entrance is, found by walking it, with the game's memory saying exactly where
the character is. The atlas plans routes on the result instead of guessing ground from
the map picture's colours.

    python scripts/survey.py scripts/tasks/ct_snes.toml --state world-1000-home \\
        --map "1000 A.D." [--box 0] [--minutes 60] [--exits --entries]
    python scripts/survey.py scripts/tasks/ct_snes.toml --state x --map "1000 A.D." --names

Runs in a hidden omabox desktop (scripts/emu_tasks.py's Box). The game's [game.ram] in
the task file names the memory: x, y (the character on the world map, in map pixels),
location (which map or place it's in), exit (which place's icon it's standing on, or
`none`). Writes ~/.config/player-two/atlas/<game>/survey-<map>.json:

    {"cell": 8, "walk": [[row, col], ...], "wall": [[row, col], ...],
     "exits": {"28": [[row, col], ...]},          # icon cells, by the game's exit number
     "doors": [[row, col, location], ...],        # icon cells that go in, and where
     "names": {"1": "Crono's house"}}             # each place's name as the game shows it
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emu_tasks import Box

from playertwo.atlas import ATLASES, CELL, slug

SPEED = 1.0  # how many times normal speed the game runs at


def secs(s: float) -> float:
    """A wait in game time, at the speed it runs."""
    return max(0.03, s / SPEED)


DIRS = {(0, -1): ["Up"], (1, -1): ["Up", "Right"], (1, 0): ["Right"],
        (1, 1): ["Down", "Right"], (0, 1): ["Down"], (-1, 1): ["Down", "Left"],
        (-1, 0): ["Left"], (-1, -1): ["Up", "Left"]}


class Game:
    """The emulator in a box: keys in, memory out."""

    def __init__(self, box: Box, ram: dict):
        self.box, self.ram = box, ram
        self.held: list[str] = []

    def word(self, off: int, n: int = 2) -> int:
        for i in range(12):  # a UDP reply can get lost, or the emulator be busy
            if i >= 5:
                time.sleep(0.25)
            r = self.box.cmd(f"READ_CORE_RAM {off:x} {n}", wait=0.5).split()
            if len(r) == n + 2 and r[1].lower().lstrip("0") == f"{off:x}".lstrip("0"):
                return sum(int(x, 16) << (8 * i) for i, x in enumerate(r[2:]))
        raise RuntimeError(f"memory read failed at {off:x}")

    def pos(self) -> tuple[int, int]:
        return self.word(self.ram["x"]), self.word(self.ram["y"])

    def location(self) -> int:
        return self.word(self.ram["location"]) & self.ram.get("location_mask", 0xFFFF)

    def exit(self) -> int | None:
        v = self.word(self.ram["exit"], 1)
        return None if v == self.ram.get("exit_none", 255) else v

    def key(self, k: str, state: str) -> None:
        lua = f'hl.dsp.send_key_state({{ mods = "", key = "{k}", state = "{state}" }})'
        subprocess.run(["omabox", "-b", self.box.name, "hyprctl", "dispatch", lua],
                       check=False, capture_output=True, timeout=5)

    def hold(self, keys: list[str]) -> None:
        for k in self.held:
            if k not in keys:
                self.key(k, "up")
        for k in keys:
            if k not in self.held:
                self.key(k, "down")
        self.held = list(keys)

    def press(self, k: str, secs: float = 0.1) -> None:
        self.hold([])
        self.key(k, "down")
        time.sleep(secs)
        self.key(k, "up")


def cell_of(p: tuple[float, float]) -> tuple[int, int]:
    return int(p[1]) // CELL, int(p[0]) // CELL


def centre(c: tuple[int, int]) -> tuple[float, float]:
    """Where to stand in a cell: its corner, since the character comes to rest on
    multiples of 8 pixels."""
    return c[1] * CELL, c[0] * CELL


class Survey:
    def __init__(self, game: Game, size: tuple[int, int], world: int, log):
        self.g, self.world, self.log = game, world, log
        self.rows, self.cols = size[1] // CELL, size[0] // CELL
        self.walk: set[tuple[int, int]] = set()
        self.wall: set[tuple[int, int]] = set()
        self.tried: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.exits: dict[int, set[tuple[int, int]]] = collections.defaultdict(set)
        self.left = False  # walked into a place by accident
        self.shots: Path | None = None  # where to keep a picture of each icon
        self.doors: dict[tuple[int, int], int] = {}
        self.test_doors = False
        self.names: dict[str, str] = {}  # location number -> the game's name for it  # icon tile -> location it opens

    def inside(self, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def seen(self, p: tuple[int, int]) -> None:
        c = cell_of(p)
        if self.inside(c):
            self.walk.add(c)
            self.wall.discard(c)

    def line(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        """Every cell between two readings was walked over."""
        n = max(1, int(math.dist(a, b) // 2))
        for i in range(n + 1):
            self.seen((round(a[0] + (b[0] - a[0]) * i / n), round(a[1] + (b[1] - a[1]) * i / n)))

    def note_exit(self, p: tuple[int, int], test: bool = False) -> None:
        e = self.g.exit()
        if e is None:
            return
        c = cell_of(p)
        self.exits[e].add(c)
        if test and c not in self.doors:
            self.try_door(c, e)

    def try_door(self, c: tuple[int, int], e: int) -> None:
        """Press the button standing on an icon tile and see where it goes: the game is
        saved first and loaded after, so the walk carries on from the same tile."""
        g = self.g
        g.box.cmd("SAVE_STATE", wait=0.2)
        time.sleep(0.4)
        shot = None
        if self.shots is not None:  # the game shows the place's name here
            shot = g.box.shot(self.shots / "icon.png")
        g.press("Return", 0.12)
        time.sleep(max(secs(2.5), 0.8))
        loc = g.location()
        self.doors[c] = loc
        if shot is not None and loc != self.world:
            # Named by where it goes: that number stays the same all through the game.
            shot.replace(self.shots / f"loc-{loc:x}.png")
        g.box.cmd("LOAD_STATE", wait=0.2)
        time.sleep(0.6)
        if g.location() != self.world or cell_of(g.pos()) != c:
            raise RuntimeError(f"didn't get back onto {c} after trying its door")
        if loc != self.world:
            self.log(f"door at {c} (icon {e}) goes to location {loc:#x}")

    def ray(self, d: tuple[int, int], cells: int = 14) -> tuple[int, int]:
        """Hold one way until stopped (or `cells` cells on): every cell passed can be
        walked, and the one it stopped at can't."""
        g = self.g
        start = last = g.pos()
        self.seen(start)
        moved_t = time.monotonic()
        g.hold(DIRS[d])
        stopped = False
        try:
            while True:
                time.sleep(0.004)
                p = g.pos()
                if p != last:
                    self.line(last, p)
                    last, moved_t = p, time.monotonic()
                    if p[0] % CELL == 0 and p[1] % CELL == 0:
                        # On a tile exactly: the icon number is for the tile a step
                        # is going to, so only read it between steps.
                        self.note_exit(p)
                    if math.dist(p, start) > cells * CELL:
                        break
                elif time.monotonic() - moved_t > secs(0.15):
                    stopped = True
                    break
        finally:
            g.hold([])
        if g.location() != self.world:
            self.left = True
            return last
        self.note_exit(last)
        if stopped:
            n = math.hypot(*d)
            ahead = cell_of((last[0] + d[0] / n * CELL, last[1] + d[1] / n * CELL))
            if ahead != cell_of(last) and ahead not in self.walk and self.inside(ahead):
                self.wall.add(ahead)
        return last

    def neighbours(self, c):
        for (dx, dy) in DIRS:
            n = (c[0] + dy, c[1] + dx)
            if dx and dy and ((c[0] + dy, c[1]) not in self.walk
                              or (c[0], c[1] + dx) not in self.walk):
                continue
            yield n, (dx, dy)

    def bfs(self, start: tuple[int, int]) -> dict:
        came = {start: None}
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v, _ in self.neighbours(u):
                if v in self.walk and v not in came:
                    came[v] = u
                    q.append(v)
        return came

    def unknown_dirs(self, c):
        return [d for d in DIRS if (c, d) not in self.tried
                and self.inside(n := (c[0] + d[1], c[1] + d[0]))
                and n not in self.walk and n not in self.wall]

    def go(self, path: list[tuple[int, int]]) -> bool:
        """Walk cell to cell along known ground; False if something stopped it."""
        g = self.g
        for c in path[1:]:
            tx, ty = centre(c)
            t0 = moved = time.monotonic()
            last = g.pos()
            while True:
                p = g.pos()
                dx, dy = tx - p[0], ty - p[1]
                if abs(dx) <= 2 and abs(dy) <= 2:
                    break
                keys = ((["Right"] if dx > 2 else ["Left"] if dx < -2 else [])
                        + (["Down"] if dy > 2 else ["Up"] if dy < -2 else []))
                g.hold(keys)
                if p != last:
                    self.line(last, p)
                    last, moved = p, time.monotonic()
                if time.monotonic() - moved > secs(0.4) or time.monotonic() - t0 > secs(4):
                    g.hold([])
                    return False
                time.sleep(0.01)
        g.hold([])
        return True

    def explore(self, minutes: float) -> None:
        end = time.monotonic() + minutes * 60
        here = cell_of(self.g.pos())
        self.walk.add(here)
        rays = 0
        while time.monotonic() < end and not self.left:
            here = cell_of(self.g.pos())
            self.walk.add(here)
            dirs = self.unknown_dirs(here)
            if dirs:
                d = dirs[0]
                self.tried.add((here, d))
                self.ray(d)
                rays += 1
                if rays % 50 == 0:
                    self.log(f"{rays} rays, {len(self.walk)} walkable, {len(self.wall)} "
                             f"shut, {len(self.exits)} exits")
                continue
            came = self.bfs(here)
            goals = [c for c in came if self.unknown_dirs(c)]
            if not goals:
                self.log("nothing left to explore from here")
                break
            goal = min(goals, key=lambda c: (abs(c[0] - here[0]) + abs(c[1] - here[1])))
            # The nearest by steps, not as the crow flies.
            dist = {}
            for c in goals:
                n, u = 0, c
                while came[u] is not None:
                    u, n = came[u], n + 1
                dist[c] = n
            goal = min(goals, key=dist.get)
            path, u = [], goal
            while u is not None:
                path.append(u)
                u = came[u]
            if not self.go(path[::-1]):
                # Couldn't get there along what looked walkable: give up on its ways
                # out rather than loop on it.
                stuck = cell_of(self.g.pos())
                for d in DIRS:
                    self.tried.add((goal, d))
                self.log(f"couldn't reach {goal}, stopped at {stuck}")

    def scan_exits(self) -> None:
        """Stand on every walkable tile once (a depth-first walk over the surveyed
        ground) and note which are place icons."""
        start = cell_of(self.g.pos())
        seen = {start}
        stack = [start]
        self.note_exit(self.g.pos())
        steps = 0
        while stack:
            u = stack[-1]
            nxt = next((v for v, d in self.neighbours(u)
                        if v in self.walk and v not in seen and not (d[0] and d[1])), None)
            if nxt is None:
                stack.pop()
                if stack and cell_of(self.g.pos()) != stack[-1]:
                    self.go([cell_of(self.g.pos()), stack[-1]]) or self.walk_back(stack[-1])
                continue
            seen.add(nxt)
            if self.go([u, nxt]):
                time.sleep(secs(0.05))
                p = self.g.pos()
                if p[0] % CELL == 0 and p[1] % CELL == 0:
                    self.note_exit(p, test=self.test_doors)
                stack.append(nxt)
            steps += 1
            if steps % 200 == 0:
                self.log(f"exit scan: {len(seen)} of {len(self.walk)} tiles, "
                         f"{len(self.exits)} icons")
            if self.g.location() != self.world:
                raise RuntimeError("walked into a place during the exit scan")

    def walk_back(self, target: tuple[int, int]) -> bool:
        came = self.bfs(cell_of(self.g.pos()))
        if target not in came:
            return False
        path, u = [], target
        while u is not None:
            path.append(u)
            u = came[u]
        return self.go(path[::-1])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "cell": CELL, "world": self.world,
            "walk": sorted(self.walk), "wall": sorted(self.wall),
            "exits": {str(k): sorted(v) for k, v in self.exits.items()},
            "doors": [[c[0], c[1], loc] for c, loc in sorted(self.doors.items())],
            "names": self.names}))


NAME_PROMPT = """This is a game's world map. If a place's name is written on the screen (a
label over the map, usually near the top or over the place), what is it, exactly as
written? JSON: {"name": "..."} or {"name": null} if no name is shown."""


def name_icons(folder: Path) -> dict[str, str]:
    """The game's own name for each place, by its location number, read off the screen
    with Muse Spark (the survey pictured each door with its label showing)."""
    import asyncio

    from PIL import Image

    from playertwo.config import Settings
    from playertwo.muse import Muse, image_part

    muse = Muse(Settings.load())

    async def one(f: Path) -> tuple[str, str | None]:
        img = Image.open(f)
        w, h = img.size
        gw = h * 4 // 3
        img = img.crop(((w - gw) // 2, 0, (w + gw) // 2, h))  # the game, not the bars
        msgs = [{"role": "user", "content": [image_part(img, 1024),
                                             {"type": "text", "text": NAME_PROMPT}]}]
        try:
            r = await muse.json(msgs, effort="low", max_tokens=1500)
        except ValueError:
            return f.stem.split("-")[1], None
        return str(int(f.stem.split("-")[1], 16)), r.get("name")

    async def run() -> list:
        return await asyncio.gather(*(one(f) for f in sorted(folder.glob("loc-*.png"))))

    return {e: n for e, n in asyncio.run(run()) if n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--state", required=True, help="state name in the game's states dir")
    ap.add_argument("--map", required=True, help="the atlas map's name")
    ap.add_argument("--box", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--exits", action="store_true",
                    help="stand on every surveyed tile to find the place icons again")
    ap.add_argument("--entries", action="store_true",
                    help="with --exits: try each icon tile's door (where it goes)")
    ap.add_argument("--names", action="store_true",
                    help="only read each icon's name off its picture (Muse), no box")
    ap.add_argument("--speed", type=float, default=4.0, help="emulation speed, 1 = normal")
    ap.add_argument("--keep-up", action="store_true", help="reuse a running box")
    args = ap.parse_args()
    if args.names:
        spec = tomllib.loads(args.tasks.read_text())
        out = ATLASES / slug(spec["game"]["name"]) / f"survey-{slug(args.map)}.json"
        data = json.loads(out.read_text())
        data["names"] = name_icons(out.parent / f"survey-{slug(args.map)}")
        out.write_text(json.dumps(data))
        print(json.dumps(data["names"], indent=1))
        return
    global SPEED
    SPEED = args.speed
    spec = tomllib.loads(args.tasks.read_text())
    gspec = spec["game"]
    ram = {k: int(v, 16) if isinstance(v, str) else v for k, v in gspec["ram"].items()}
    box = Box(args.box, gspec)
    if args.keep_up and "PLAYING" in box.cmd("GET_STATUS"):
        box.home = Path(subprocess.run(["omabox", "path", "--home", "-b", box.name],
                                       check=False, capture_output=True, text=True).stdout.split()[-1])
    else:
        # Fast-forward at a set pace: quicker, but keys still land where they should.
        box.up(f"fastforward_ratio = {args.speed}\n")
    if args.speed > 1:
        box.cmd("FAST_FORWARD", wait=0.2)
    state = Path(gspec["states"]).expanduser() / f"{args.state}.state"
    box.load(state)
    game = Game(box, ram)
    world = game.location()
    atlas_dir = ATLASES / slug(gspec["name"])
    meta = json.loads((atlas_dir / "atlas.json").read_text())
    m = next(x for x in meta["maps"] if x["name"] == args.map)
    from PIL import Image

    w, h = Image.open(atlas_dir / m["image"]).size
    out = atlas_dir / f"survey-{slug(args.map)}.json"
    t0 = time.monotonic()

    def log(text: str) -> None:
        print(f"{time.monotonic() - t0:7.1f} {text}", flush=True)

    s = Survey(game, (w, m.get("height", h)), world, log)
    if out.exists():  # carry on from an earlier survey
        old = json.loads(out.read_text())
        s.walk = {tuple(c) for c in old["walk"]}
        s.wall = {tuple(c) for c in old["wall"]}
        s.exits.update({int(k): {tuple(c) for c in v} for k, v in old["exits"].items()})
        s.doors = {(r, c): loc for r, c, loc in old.get("doors", [])}
        s.names = old.get("names", {})
    def boot() -> None:
        box.up(f"fastforward_ratio = {args.speed}\n")
        if args.speed > 1:
            box.cmd("FAST_FORWARD", wait=0.2)
        game.held = []
        box.load(state)

    phase, restarts = "explore", 0
    try:
        while phase != "done":
            try:
                if phase == "explore":
                    while True:
                        s.explore(args.minutes - (time.monotonic() - t0) / 60)
                        if not s.left or time.monotonic() - t0 > args.minutes * 60:
                            break
                        log("walked into a place: back to the start")
                        s.left = False
                        box.load(state)
                    s.save(out)
                    phase = "scan" if args.exits else "done"
                    if phase == "scan":
                        # Scan again only the ground reachable from here: other parts
                        # of the map (surveyed from other saves) keep their doors.
                        here = s.bfs(cell_of(game.pos()))
                        for e in list(s.exits):
                            s.exits[e] -= set(here)
                            if not s.exits[e]:
                                del s.exits[e]
                        s.doors = {c: v for c, v in s.doors.items() if c not in here}
                        s.test_doors = args.entries
                        s.shots = out.parent / f"survey-{slug(args.map)}"
                        s.shots.mkdir(exist_ok=True)
                if phase == "scan":
                    box.load(state)
                    s.scan_exits()  # doors already tried are skipped on a rerun
                    s.save(out)
                    phase = "done"
            except RuntimeError as exc:
                # The emulator stopped answering or the walk went somewhere: keep what
                # was found, start the game again and carry on.
                restarts += 1
                s.save(out)
                log(f"{exc}: starting the game again ({restarts})")
                if restarts > 6:
                    raise
                boot()
    finally:
        game.hold([])
        s.save(out)
    log(f"done: {len(s.walk)} walkable, {len(s.wall)} shut, {len(s.exits)} exits, "
        f"{len(s.doors)} doors -> {out}")


if __name__ == "__main__":
    main()
