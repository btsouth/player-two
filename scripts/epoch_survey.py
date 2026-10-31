"""Survey for the Epoch in Chrono Trigger (SNES), with the game's memory as the judge:
the parked Epoch's look on screen, which ground takes a landing near each place, and
save states on other continents (for scripts/survey.py to walk them).

    python scripts/epoch_survey.py sprite --box 14
    python scripts/epoch_survey.py land --box 14 --state world-1000-epoch --map "1000 A.D."
    python scripts/epoch_survey.py park --box 14 --state world-1000-epoch \\
        --map "1000 A.D." --at 950,400 --save epoch-1000-medina
    python scripts/epoch_survey.py era --box 14 --state world-1000-epoch --era "600 A.D." \\
        --save epoch-600-truce

Runs in a hidden omabox desktop (scripts/emu_tasks.py's Box). Writes epoch.json and the
sprite pictures in ~/.config/player-two/atlas/chrono-trigger/; states go to
~/.cache/player-two/emu/states/ct-snes/.

Memory (offsets into work RAM): the character on a world map 0x283/0x285; the parked
Epoch 0x290/0x292 (its tile, where to stand to board); 0x294 is 160 when there's an
Epoch on this map (0 when not, 227 flying, 243 in the time gauge); 0x28F is 2 while
flying, 0 on foot; 0x1BB7 is the time gauge's era (0 End of Time .. 6 65,000,000 B.C.).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emu_tasks import Box
from survey import Game

from playertwo.atlas import ATLASES, CELL, RISE, Atlas, slug
from playertwo.epoch import GAUGE, Hangar, era_of, landing_spots

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "scripts/tasks/ct_snes.toml"
SPEED = 1.0


class Epoch(Game):
    def epoch(self) -> tuple[int, int] | None:
        if self.word(0x294, 1) == 0:
            return None
        return self.word(0x290), self.word(0x292)

    def flying(self) -> bool:
        return self.word(0x28F, 1) == 2

    def wait(self, secs: float) -> None:
        time.sleep(max(0.05, secs / SPEED))

    def step_to(self, goal: tuple[int, int], limit: float = 60) -> bool:
        """Walk or fly to a tile by memory: hold toward it, let go a tile early, tap."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit / max(1, SPEED / 2):
            x, y = self.pos()
            dx, dy = goal[0] - x, goal[1] - y
            if dx == 0 and dy == 0:
                self.hold([])
                return True
            lead = 12 if SPEED <= 1 else 24
            keys = []
            if dx > lead:
                keys.append("Right")
            elif dx < -lead:
                keys.append("Left")
            if dy > lead:
                keys.append("Down")
            elif dy < -lead:
                keys.append("Up")
            if keys:
                self.hold(keys)
                time.sleep(0.01)
                continue
            self.hold([])
            self.wait(0.3)
            x, y = self.pos()
            dx, dy = goal[0] - x, goal[1] - y
            if dx == 0 and dy == 0:
                return True
            k = ["Right" if dx > 0 else "Left"] if dx else ["Down" if dy > 0 else "Up"]
            p0 = (x, y)
            self.hold(k)
            t = time.monotonic()
            while self.pos() == p0 and time.monotonic() - t < 1:
                time.sleep(0.003)
            self.hold([])
            self.wait(0.3)
            if self.pos() == p0:
                return False  # walking into something
        self.hold([])
        return False

    def board(self) -> bool:
        if self.flying():
            return True
        e = self.epoch()
        if e is None or not self.step_to(e):
            return False
        for _ in range(3):
            self.press("Return")
            self.wait(3.0)
            if self.flying():
                return True
        return False

    def try_land(self) -> bool:
        self.press("Return")
        self.wait(3.0)
        for _ in range(10):
            if self.word(0x294, 1) in (160, 227) or not self.flying():
                break
            self.wait(0.3)
        return not self.flying()


def setup(args) -> tuple[Box, Epoch, dict]:
    global SPEED
    SPEED = args.speed
    gspec = tomllib.loads(TASKS.read_text())["game"]
    ram = {k: int(v, 16) if isinstance(v, str) else v for k, v in gspec["ram"].items()}
    box = Box(args.box, gspec)
    box.up(f"fastforward_ratio = {args.speed}\n")
    if args.speed > 1:
        box.cmd("FAST_FORWARD", wait=0.2)
    return box, Epoch(box, ram), gspec


def states_dir(gspec: dict) -> Path:
    return Path(gspec["states"]).expanduser()


def save_state(box: Box, gspec: dict, name: str) -> Path:
    d = box.home / ".config/retroarch/states" / gspec["core_name"]
    f = d / f"{box.rom.stem}.state"
    before = f.stat().st_mtime if f.exists() else 0
    box.cmd("SAVE_STATE", wait=0.5)
    for _ in range(40):
        if f.exists() and f.stat().st_mtime != before:
            break
        time.sleep(0.25)
    time.sleep(1.0)
    out = states_dir(gspec) / f"{name}.state"
    shutil.copy2(f, out)
    box.shot(out.with_suffix(".png"))
    return out


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def cmd_land(args) -> None:
    """For each place's door on the map: try the landing spots in order till two take."""
    box, g, gspec = setup(args)
    atlas = Atlas.for_game(gspec["name"])
    m = atlas.map(args.map)
    hangar = Hangar(atlas.folder)
    box.load(states_dir(gspec) / f"{args.state}.state")
    if not g.board():
        log("couldn't board the Epoch")
        return
    groups = m.door_groups()
    want = [p.lower() for p in args.places] if args.places else None
    for cells, label in groups:
        place = next((n for n in m.places if m.entrance(n) == cells), label)
        if want and not any(w in (place or "").lower() for w in want):
            continue
        took = 0
        for c in landing_spots(m, cells, hangar, limit=args.tries):
            if c in hangar.land(m.name):
                took += 1
                if took >= 2:
                    break
                continue
            goal = (c[1] * CELL, c[0] * CELL)
            if not g.flying() and not g.board():
                log("lost the Epoch")
                return
            if not g.step_to(goal):
                log(f"{place}: couldn't fly to {goal}")
                continue
            ok = g.try_land()
            hangar.learn(m.name, c, ok)
            log(f"{place}: {goal} {'LAND' if ok else 'no'}")
            if ok:
                took += 1
                if not g.board():
                    log("couldn't board again")
                    return
                if took >= 2:
                    break
        log(f"{place}: {took} landing spots")
    box.down()


def cmd_park(args) -> None:
    """Fly to a spot, land on the nearest ground that takes it, save a state there."""
    box, g, gspec = setup(args)
    atlas = Atlas.for_game(gspec["name"])
    m = atlas.map(args.map)
    hangar = Hangar(atlas.folder)
    box.load(states_dir(gspec) / f"{args.state}.state")
    if not g.board():
        log("couldn't board the Epoch")
        return
    if args.era:
        to_era(g, args.era, gspec)
    x0, y0 = (int(v) for v in args.at.split(","))
    c0 = (y0 // CELL, x0 // CELL)
    order = sorted(((r, c) for r in range(c0[0] - 8, c0[0] + 9)
                    for c in range(c0[1] - 8, c0[1] + 9)),
                   key=lambda rc: abs(rc[0] - c0[0]) + abs(rc[1] - c0[1]))
    tried = 0
    for c in order:
        if c in hangar.noland(m.name) or (m.blocked[c] and c not in m.walk):
            continue
        if tried and abs(c[0] - c0[0]) % 2 and abs(c[1] - c0[1]) % 2:
            continue
        tried += 1
        if tried > 40:
            break
        goal = (c[1] * CELL, c[0] * CELL)
        if not g.step_to(goal):
            continue
        ok = g.try_land()
        hangar.learn(m.name, c, ok)
        log(f"{goal} {'LAND' if ok else 'no'}")
        if ok:
            out = save_state(box, gspec, args.save)
            log(f"saved {out} at {g.pos()}, location {g.location():#x}")
            break
    box.down()


def to_era(g: Epoch, era: str, gspec: dict) -> None:
    cur = {0x1F0: "1000 A.D.", 0x1F1: "600 A.D.", 0x1F2: "2300 A.D.", 0x1F3: "65,000,000 B.C.",
           0x1F4: "12,000 B.C.", 0x1F6: "12,000 B.C."}[g.location()]
    g.press("Tab")
    g.wait(2.5)
    n = GAUGE.index(era) - GAUGE.index(cur)
    for _ in range(abs(n)):
        g.press("Down" if n > 0 else "Up")
        g.wait(1.0)
    log(f"gauge at {g.word(0x1BB7, 1)} for {era}")
    g.press("Return")
    for _ in range(80):
        g.wait(0.5)
        if g.word(0x294, 1) == 227 and g.location() != 0 and g.flying():
            break
    log(f"now at location {g.location():#x}, {g.pos()}")


def cmd_era(args) -> None:
    box, g, gspec = setup(args)
    box.load(states_dir(gspec) / f"{args.state}.state")
    if not g.board():
        log("couldn't board the Epoch")
        return
    to_era(g, args.era, gspec)
    if args.save:
        log(f"saved {save_state(box, gspec, args.save)}")
    box.down()


def cmd_sprite(args) -> None:
    """The parked Epoch on the game's picture, from states with it on screen: the
    pixels that differ from the map picture around its tile (the memory says where)."""
    import cv2
    import numpy as np
    from PIL import Image

    box, g, gspec = setup(args)
    atlas = Atlas.for_game(gspec["name"])
    nw, nh = atlas.native
    sprites = []
    import json
    for i, name in enumerate(args.states):
        box.load(states_dir(gspec) / f"{name}.state")
        time.sleep(1.0)
        e, p = g.epoch(), g.pos()
        shot = Path(f"/tmp/epoch-sprite-{i}.png")
        box.shot(shot)
        img = Image.open(shot)
        s = atlas.locate(img)
        if e is None or s is None:
            log(f"{name}: no Epoch ({e}) or not on the map ({s})")
            continue
        view = atlas.view(img).astype(int)
        m = atlas.map(s.map)
        x0, y0 = round(s.x - nw / 2), round(s.y - nh / 2)
        rip = m.image[y0:y0 + nh, x0:x0 + nw].astype(int)
        xv, yv = e[0] - (s.x - nw / 2), e[1] - (s.y - nh / 2)
        log(f"{name}: player {p}, view {s.x},{s.y}; Epoch {e} at view {xv},{yv}")
        a, b = int(xv) - 14, int(yv) - 20
        crop = view[b:b + 34, a:a + 30]
        diff = np.abs(crop - rip[b:b + 34, a:a + 30]).sum(axis=2)
        # The player's own sprite is drawn in the middle of the view: not the Epoch.
        cx, cy = nw / 2 - a, nh / 2 - b
        yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
        me = (abs(xx - cx) < 9) & (yy > cy - 20) & (yy < cy + 8)
        mask = (diff > 60) & ~me
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3)))
        if mask.sum() < 40:
            log(f"{name}: too few Epoch pixels ({mask.sum()})")
            continue
        out = np.dstack([crop.astype(np.uint8), (mask * 255).astype(np.uint8)])
        f = f"epoch-{len(sprites)}.png"
        cv2.imwrite(str(atlas.folder / f), out)
        sprites.append({"file": f, "dx": int(xv) - a, "dy": int(yv) - b, "from": name})
        log(f"{name}: {mask.sum()} px -> {f}")
    spec = json.loads((atlas.folder / "epoch.json").read_text()) \
        if (atlas.folder / "epoch.json").exists() else {"maps": {}}
    spec["sprites"] = sprites
    (atlas.folder / "epoch.json").write_text(json.dumps(spec))
    box.down()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["land", "park", "era", "sprite"])
    ap.add_argument("--box", type=int, default=14)
    ap.add_argument("--state", default="world-1000-epoch")
    ap.add_argument("--states", nargs="*", default=["world-1000-epoch", "world-600-epoch",
                                                     "world-2300-epoch", "world-12000-late"])
    ap.add_argument("--map", default="1000 A.D.")
    ap.add_argument("--places", nargs="*", help="only these places' doors (words)")
    ap.add_argument("--tries", type=int, default=8, help="landing spots per door at most")
    ap.add_argument("--at", help="x,y to land near (park)")
    ap.add_argument("--era", help="go to this era first (park), or the era to go to")
    ap.add_argument("--save", help="state name to save as")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    {"land": cmd_land, "park": cmd_park, "era": cmd_era, "sprite": cmd_sprite}[args.what](args)


if __name__ == "__main__":
    main()

_ = (ATLASES, RISE, era_of, slug)
