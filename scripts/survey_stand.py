"""Which of a surveyed place's tiles can really be stood on, measured with the game's
memory in an emulator box (scripts/emu_tasks.py's Box).

The room survey (survey_rooms.py) counts a tile walked when the feet go into it at all,
and they go into some that can't be stood on or crossed: 8 px into a fence's row from
above, the last 1/16 px of a tile under a fence, the edge of a stall's side. Its ground
then runs straight through fences, and the walker (rooms.walk_out) bumps into each of
them once a session (live, the fair: 42 s and 50 s for "buy three potions", and the
place's data edited by hand). Here the character is walked from the spot to stand on of
each tile it stands on (rooms.centre: across its middle, low on it) to its neighbours',
a tile at a time, in the four ways; a tile none of its neighbours reach that way (and
not a way out) comes off the walkable ones. Who follows the player doesn't matter: the
party never blocks the leader (the RAM: it walks through them at full speed).

    python scripts/survey_stand.py scripts/tasks/ct_rooms.toml --state rooms/door-5
        [--box 41] [--out DIR] [--minutes 40]

Writes DIR/loc-XXX.json: the place's data with "walk" and "wall" corrected, the survey's
own walk kept as "walk_survey", and "stand_probe" saying what was found (the tiles taken
off, the steps found shut between tiles that can both be stood on, the state, when).
DIR is the rooms folder by default.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emu_tasks import Box
from survey_rooms import Between, Field

from playertwo.atlas import ATLASES
from playertwo.rooms import centre

TILE = 16
KEYS = {(0, -1): "Up", (1, 0): "Right", (0, 1): "Down", (-1, 0): "Left"}


def tile_of(p: tuple[float, float]) -> tuple[int, int]:
    return int(p[1]) // TILE, int(p[0]) // TILE


class Probe:
    def __init__(self, g: Field, data: dict, state: Path, log):
        self.g, self.state, self.log = g, state, log
        self.loc = data["loc"]
        self.walk = {tuple(t) for t in data["walk"]}
        self.doors = {tuple(e["from"]) for e in data.get("exits", [])}
        self.ok: set[tuple[int, int]] = set()  # stood on, at its spot
        self.open: set[tuple] = set()  # (from, to): stepped
        self.shut: set[tuple] = set()  # (from, to): tried, didn't get there
        self.moves = 0

    # -- moving --------------------------------------------------------------------
    def pos(self) -> tuple[float, float]:
        return self.g.pos()

    def stand(self, t: tuple[int, int], tol: float = 1.5) -> bool:
        """Onto tile t's spot to stand on, within tol px, by holds then taps; False if
        something stops it short."""
        g = self.g
        cx, cy = centre(t)
        stuck = 0
        last = None
        for _ in range(60):
            p = self.pos()
            dx, dy = cx - p[0], cy - p[1]
            if abs(dx) <= tol and abs(dy) <= tol:
                return True
            if last is not None and abs(p[0] - last[0]) < 0.3 and abs(p[1] - last[1]) < 0.3:
                stuck += 1
                if stuck >= 4:
                    return False
            else:
                stuck = 0
            last = p
            # the further axis first
            if abs(dx) >= abs(dy):
                key, d = ("Right" if dx > 0 else "Left"), abs(dx)
            else:
                key, d = ("Down" if dy > 0 else "Up"), abs(dy)
            if d > 6:
                g.hold([key])
                t0 = time.monotonic()
                while time.monotonic() - t0 < 0.5:
                    q = self.pos()
                    rem = (cx - q[0]) if key in ("Right", "Left") else (cy - q[1])
                    if abs(rem) <= 3 or (rem > 0) != ((cx - p[0] if key in ("Right", "Left")
                                                       else cy - p[1]) > 0):
                        break
                    time.sleep(0.004)
                g.hold([])
            else:
                g.hold([key])
                time.sleep(0.017)
                g.hold([])
            time.sleep(0.05)
        return False

    def step(self, t: tuple[int, int], n: tuple[int, int]) -> bool:
        """From t's spot (stood on) a tile over to n's, straight: True when the feet get
        to n's spot without being slid aside (round a corner) or stopped short."""
        g = self.g
        d = (n[1] - t[1], n[0] - t[0])
        across = d[0] != 0
        goal = centre(n)[0] if across else centre(n)[1]
        start = self.pos()
        side0 = start[1] if across else start[0]
        self.moves += 1
        g.hold([KEYS[d]])
        t0 = moved = time.monotonic()
        last = start
        try:
            while time.monotonic() - t0 < 0.9:
                try:
                    p = self.pos()
                except Between:
                    return False
                rem = (goal - p[0]) * d[0] if across else (goal - p[1]) * d[1]
                if rem <= 2:
                    break
                if p != last:
                    last, moved = p, time.monotonic()
                elif time.monotonic() - moved > 0.15:
                    break
                time.sleep(0.003)
        finally:
            g.hold([])
        time.sleep(0.05)
        if self.g.location() != self.loc:
            return False
        p = self.pos()
        on = abs((p[0] if across else p[1]) - goal)
        drift = abs((p[1] if across else p[0]) - side0)
        return on <= 4 and drift <= 3 and self.stand(n, 2.5)

    def reload(self) -> None:
        self.g.hold([])
        self.g.box.load(self.state)
        time.sleep(0.5)

    def travel(self, goal: tuple[int, int]) -> bool:
        """To a tile stood on before, a step at a time by steps that went either way (a
        step back the way it came mostly goes); another way round when one doesn't,
        reloading the state when lost (out of the place, or off every tile stood on)."""
        bad: set = set()
        for _attempt in range(4):
            if self.g.location() != self.loc:
                self.reload()
            # Onto a tile stood on before: this one, else one beside it (pressed into a
            # corner, the feet can be where no spot is got to from), else the start.
            here = tile_of(self.pos())
            cands = ([here] if here in self.ok else []) + sorted(
                t for t in self.ok if abs(t[0] - here[0]) + abs(t[1] - here[1]) == 1)
            for c in cands:
                if self.stand(c, 2.5):
                    here = c
                    break
            else:
                self.reload()
                here = tile_of(self.pos())
                if here not in self.ok or not self.stand(here, 2.5):
                    continue
            came = {here: None}
            q = collections.deque([here])
            while q:
                u = q.popleft()
                if u == goal:
                    break
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    v = (u[0] + dr, u[1] + dc)
                    if v not in came and v in self.ok and (u, v) not in bad and (
                            (u, v) in self.open or (v, u) in self.open):
                        came[v] = u
                        q.append(v)
            if goal not in came:
                self.log(f"no known way from {here} to {goal}")
                return False
            path = []
            u = goal
            while u is not None:
                path.append(u)
                u = came[u]
            path.reverse()
            for x, y in itertools.pairwise(path):
                if not self.step(x, y):
                    self.log(f"the step {x} to {y} didn't go (at {self.pos()}), going to {goal}")
                    bad.add((x, y))
                    break
                self.open.add((x, y))
            else:
                return True
        return False

    # -- the probe ---------------------------------------------------------------
    def todo(self, t: tuple[int, int]) -> list[tuple[int, int]]:
        """t's neighbours the survey walked, not stood on yet, not tried from t."""
        return [n for n in ((t[0] - 1, t[1]), (t[0], t[1] + 1), (t[0] + 1, t[1]),
                            (t[0], t[1] - 1))
                if n in self.walk and n not in self.doors and n not in self.ok
                and (t, n) not in self.shut]

    def run(self, minutes: float) -> None:
        p = self.pos()
        start = tile_of(p)
        if start not in self.walk:
            start = min(self.walk, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))
        if not self.stand(start, 2.5):
            raise RuntimeError(f"can't stand on {start} where the state starts")
        self.ok.add(start)
        stack = [start]
        t_end = time.monotonic() + 60 * minutes
        retried: set = set()
        dead: set = set()  # tiles stood on that it couldn't get back to
        while time.monotonic() < t_end:
            if not stack:
                here = tile_of(self.pos())

                def far(t, here=here):
                    return abs(t[0] - here[0]) + abs(t[1] - here[1])

                edge = sorted((t for t in self.ok if t not in dead and self.todo(t)), key=far)
                if edge:
                    if self.travel(edge[0]):
                        stack = [edge[0]]
                    else:
                        self.log(f"couldn't get to {edge[0]}")
                        dead.add(edge[0])
                    continue
                # Once round: a tile tried from one side only, once more (someone who
                # walks about may have been in the way), nearest first.
                tries = collections.Counter(b for _a, b in self.shut)
                again = sorted(((a, b) for a, b in self.shut if b not in self.ok
                                and tries[b] == 1 and (a, b) not in retried
                                and a in self.ok and a not in dead),
                               key=lambda e: far(e[0]))
                if not again:
                    self.log("every tile got to has been tried all round")
                    break
                a, b = again[0]
                retried.add((a, b))
                self.shut.discard((a, b))
                if self.travel(a):
                    stack = [a]
                else:
                    self.shut.add((a, b))
                continue
            t = stack[-1]
            todo = self.todo(t)
            if not todo:
                stack.pop()
                if stack and not self.travel(stack[-1]):
                    self.log(f"couldn't get back to {stack[-1]}: on from elsewhere")
                    stack = []
                continue
            n = todo[0]
            if (tile_of(self.pos()) != t or not self.stand(t, 2.5)) and not self.travel(t):
                dead.add(t)
                stack = []
                continue
            if self.step(t, n):
                self.open.add((t, n))
                self.ok.add(n)
                stack.append(n)
                if len(self.ok) % 25 == 0:
                    self.log(f"{len(self.ok)} tiles stood on, {len(self.shut)} steps shut, "
                             f"{self.moves} moves")
            else:
                self.shut.add((t, n))
                if self.g.location() != self.loc:
                    self.reload()
                if not self.stand(t, 2.5) and not self.travel(t):
                    stack = []

    def result(self) -> dict:
        never = sorted(t for t in self.walk - self.ok - self.doors
                       if any((n, t) in self.shut for n in self.ok))
        unseen = sorted(t for t in self.walk - self.ok - self.doors if t not in never)
        cuts = sorted((a, b) for a, b in self.shut if a in self.ok and b in self.ok)
        return {"stood": len(self.ok), "never": never, "unseen": unseen, "cuts": cuts,
                "moves": self.moves}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--state", required=True, help="a state in the place (under the "
                    "task file's states folder, without .state)")
    ap.add_argument("--box", type=int, default=41)
    ap.add_argument("--out", type=Path, help="where to write loc-XXX.json (the rooms "
                    "folder by default)")
    ap.add_argument("--rooms", type=Path, help="the survey's rooms folder to read")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--up", action="store_true", help="start the box (else it's running)")
    args = ap.parse_args()
    spec = tomllib.loads(args.tasks.read_text())
    game = spec["game"]
    ram = {k: int(v, 16) if isinstance(v, str) else v for k, v in game["ram"].items()}
    state = Path(game["states"]).expanduser() / f"{args.state}.state"
    box = Box(args.box, game)
    if args.up:
        box.up()
    else:
        box.home = Path(box.ob("path", "--home").strip().splitlines()[-1])
    g = Field(box, ram)
    for k in KEYS.values():  # keys a stopped run left held in the box
        g.key(k, "up")
    box.load(state)
    loc = box.location()
    rooms = args.rooms or ATLASES / "chrono-trigger" / "rooms"
    src = rooms / f"loc-{loc:03x}.json"
    data = json.loads(src.read_text())
    t0 = time.monotonic()

    def log(text: str) -> None:
        print(f"{time.monotonic() - t0:7.1f} {text}", flush=True)

    probe = Probe(g, data, state, log)
    try:
        probe.run(args.minutes)
    finally:
        g.hold([])
    res = probe.result()
    log(f"stood on {res['stood']} of {len(probe.walk)} tiles the survey walked; "
        f"{len(res['never'])} can't be stood on, {len(res['unseen'])} not got to, "
        f"{len(res['cuts'])} steps shut between tiles stood on; {res['moves']} moves")
    log(f"can't be stood on: {res['never']}")
    out = data.copy()
    out["walk_survey"] = data.get("walk_survey", data["walk"])
    before = data.get("stand_probe", {})  # an earlier run from elsewhere in the place
    stood = probe.ok | {tuple(t) for t in before.get("stood", [])}
    # (stood on in either run: someone walking about was in the way in the other)
    never = {tuple(t) for t in res["never"]} - stood
    out["walk"] = [list(t) for t in sorted({tuple(t) for t in data["walk"]} - never)]
    out["wall"] = [list(t) for t in sorted({tuple(t) for t in data.get("wall", [])} | never)]
    never |= {tuple(t) for t in before.get("never", [])}
    out["stand_probe"] = {"never": [list(t) for t in sorted(never)],
                          "stood": [list(t) for t in sorted(stood)],
                          "unseen": [list(t) for t in res["unseen"]],
                          "cuts": [[*a, *b] for a, b in res["cuts"]],
                          "state": ", ".join(filter(None, [before.get("state"), args.state])),
                          "when": time.strftime("%Y-%m-%d %H:%M")}
    dest = (args.out or rooms) / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
