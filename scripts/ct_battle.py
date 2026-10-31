"""Chrono Trigger (SNES) battles in a hidden omabox desktop: find fights and save states at
their start, and read the battle from the game's memory (who's in it, HP and MP, whose
turn), for the battle bench (scripts/tasks/ct_battle.toml).

    python scripts/ct_battle.py hunt -n 16 STATE NAME [--minutes 5]   # walk until a fight
    python scripts/ct_battle.py read -n 16                            # the battle now
    python scripts/ct_battle.py dump -n 16 OUT.bin                    # all 128 KB of WRAM
    python scripts/ct_battle.py load -n 16 STATE                      # load a state

Uses scripts/ct_states.py's box (keys, screenshots, saves) and the RAM in
scripts/ct_ram.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ct_states import CT, STATES, game

import ct_ram


class Battle(CT):
    def read(self, off: int, n: int) -> bytes:
        """`n` bytes of WRAM from `off`, in pieces a UDP reply holds, retried."""
        out = b""
        while len(out) < n:
            k = min(4096, n - len(out))
            for i in range(8):
                got = self.ram(off + len(out), k)
                if len(got) == k:
                    out += got
                    break
                time.sleep(0.2 * (i + 1))
            else:
                raise RuntimeError(f"memory read failed at {off + len(out):x}")
        return out

    def hold(self, keys: list[str], secs: float) -> None:
        from ct_states import BUTTONS
        for st in ("down", "up"):
            for k in keys:
                self.ob("hyprctl", "dispatch",
                        f'hl.dsp.send_key_state({{ mods = "", key = "{BUTTONS.get(k, k)}", '
                        f'state = "{st}" }})')
            if st == "down":
                time.sleep(secs)

    def in_battle(self) -> bool:
        return ct_ram.in_battle(self.read)

    def load_state(self, state: Path) -> None:
        self.load(state)


def objects(box: Battle) -> dict[int, tuple[int, int]]:
    """Field objects (offset into the object tables: 2 x index) at their place in pixels;
    0x197 says which is the player."""
    xs, ys = box.read(0x1800, 0x80), box.read(0x1880, 0x80)
    out = {}
    for o in range(0, 0x80, 2):
        x, y = (xs[o] | xs[o + 1] << 8) >> 4, (ys[o] | ys[o + 1] << 8) >> 4
        if x < 2048 and y < 2048:
            out[o] = (x, y)
    return out


def hunt(box: Battle, state: str, name: str, minutes: float, seed: int) -> bool:
    """Walk about from `state` until a fight starts; save it once the first menu is up."""
    rnd = random.Random(seed)
    if state != "-":  # "-": from where the game is now
        box.load_state(STATES / f"{state}.state")
    start = STATES / f"{state}.state" if state != "-" else None
    if start is None:  # come back here when a walk leaves for a world map
        start = box.save(STATES / f"tmp-hunt-{box.name}.state")
    end = time.monotonic() + minutes * 60
    dirs = ["up", "down", "left", "right"]
    d = rnd.choice(dirs)
    skip: set[int] = set()  # objects it couldn't get to
    target, best, tries = None, 1e9, 0
    while time.monotonic() < end:
        objs = objects(box)
        me = objs.pop(box.read(0x197, 1)[0], None)
        cands = {o: p for o, p in objs.items() if o >= 16 and o not in skip}
        if me and cands and rnd.random() < 0.75:
            if target not in cands:
                target = min(cands, key=lambda o: abs(cands[o][0] - me[0]) +
                             abs(cands[o][1] - me[1]))
                best, tries = 1e9, 0
            dx, dy = cands[target][0] - me[0], cands[target][1] - me[1]
            dist = abs(dx) + abs(dy)
            tries = 0 if dist < best - 4 else tries + 1
            best = min(best, dist)
            if tries > 6:  # not getting closer: something's in the way
                skip.add(target)
                target = None
            if abs(dx) > abs(dy) or (abs(dx) > 8 and rnd.random() < 0.3):
                d = "right" if dx > 0 else "left"
            else:
                d = "down" if dy > 0 else "up"
            box.hold([d], rnd.uniform(0.25, 0.6))
        else:
            if rnd.random() < 0.5:
                d = rnd.choice(dirs)
            box.hold([d], rnd.uniform(0.4, 1.5))
        if box.in_battle():
            break
        if (box.where() or 0) >= 0x1F0:  # out on a world map: start over
            box.load_state(start)
            skip.clear()
            continue
        # A text box or a door: a press moves it on (and opens nothing on its own).
        if rnd.random() < 0.1:
            box.keys(["a"])
    else:
        print("no fight found", flush=True)
        return False
    print(f"fight at location {box.where():#x}", flush=True)
    for _ in range(40):  # the intro, then the first turn's menu
        time.sleep(0.5)
        b = ct_ram.battle(box.read)
        if b and b["enemies"]:
            break
    time.sleep(2)  # they come in, the first gauge fills
    dest = box.save(STATES / f"{name}.state")
    box.shot(STATES / f"{name}.png")
    print(json.dumps(ct_ram.battle(box.read), indent=1), dest, flush=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hunt", "read", "dump", "load"):
        s = sub.add_parser(name)
        s.add_argument("-n", type=int, default=16)
        if name == "hunt":
            s.add_argument("state")
            s.add_argument("name")
            s.add_argument("--minutes", type=float, default=5)
            s.add_argument("--seed", type=int, default=1)
        if name == "dump":
            s.add_argument("out", type=Path)
        if name == "load":
            s.add_argument("state")
    a = p.parse_args()
    box = Battle(a.n, game())
    box.attach()
    if a.cmd == "hunt":
        sys.exit(0 if hunt(box, a.state, a.name, a.minutes, a.seed) else 1)
    elif a.cmd == "read":
        print(json.dumps(ct_ram.battle(box.read), indent=1))
    elif a.cmd == "dump":
        a.out.write_bytes(box.read(0, 0x20000))
    elif a.cmd == "load":
        p = Path(a.state)
        box.load_state(p if p.suffix else STATES / f"{a.state}.state")


if __name__ == "__main__":
    main()
