#!/usr/bin/env python3
"""Monster Hunter Stories 3 bench: the routine's pixel reads against saved frames, with labels I
checked by eye. The frames stay out of the repo (game screens); pass their folder.

  mh_bench.py pixels [--dir ~/.cache/player-two/games-bakeoff/mhs3]
  mh_bench.py muse   [--dir ...]   real Muse reads of the wheel and Status screen, with timing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from PIL import Image

from playertwo import mhstories as m

# The arc in each turn frame of the Tetranadon fights, by eye.
ARCS = {"213912": "red", "213938": "blue", "213955": "red", "214018": "blue", "214042": "none",
        "214105": "blue", "214124": "red", "214511": "red", "214559": "none", "214619": "red",
        "214642": "blue", "214701": "none", "214730": "none", "214753": "blue", "214812": "red",
        "214853": "red"}
# Blue arcs and who they reach, by eye; red ones by whose health then dropped.
TARGETS = {"213938": "Ogden", "214018": "Chirpy", "214105": "Ogden", "214642": "Ogden",
           "214753": "Chirpy", "214124": "Chirpy", "214812": "Chirpy", "214853": "Chirpy",
           "214619": "Canyne"}
# The part each wheel is on, what works on it, broken or not.
WHEEL = {"214853": ("tail", ["slash"], False), "214812": ("tail", ["slash"], False),
         "214559": ("head", ["pierce"], False), "214701": ("head", ["pierce"], True),
         "214642": ("head", ["pierce"], False), "213912": ("legs", ["blunt"], False),
         "214018": ("legs", ["blunt"], True), "214124": ("legs", ["blunt"], True)}
# Monster moves as they play: frame crop, ribbon type.
RIBBONS = {"214609-moves1": "Technical", "214630-moves1": "none", "214651-moves1": "Technical",
           "214713-moves1": "none", "214801-moves1": "Power", "214903-moves1": "none"}
# Ribbons over the party in a turn frame.
PARTY = {"214853": {"Canyne": "Speed", "Zerk": "Speed", "Chirpy": ""},
         "214642": {"Zerk": "Power", "Ogden": "Power", "Canyne": "Speed"},
         "214105": {"Zerk": "Speed", "Ogden": "Speed", "Chirpy": "Technical"}}


class Score:
    def __init__(self) -> None:
        self.ok = self.n = 0

    def check(self, name: str, got, want) -> None:
        self.n += 1
        good = got == want
        self.ok += good
        print(f"{'ok ' if good else 'BAD'} {name}: {got!r}" + ("" if good else f" want {want!r}"))


def full(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def top_only(path: Path) -> Image.Image:
    """A saved top band of a 16:9 frame, back on a full-sized picture."""
    band = full(path)
    pic = Image.new("RGB", (band.width, band.width * 9 // 16))
    pic.paste(band, (0, 0))
    return pic


def pixels(d: Path) -> int:
    s = Score()
    frames = d / "frames"
    for name, want in ARCS.items():
        s.check(f"arc {name}", m.arc_seen(full(frames / f"mh-{name}.jpg"))[0], want)
    for name, want in TARGETS.items():
        s.check(f"arc target {name}", m.arc_seen(full(frames / f"mh-{name}.jpg"))[1], want)
    for name, (part, works, broken) in WHEEL.items():
        img = full(frames / f"mh-{name}.jpg")
        s.check(f"wheel open {name}", m.wheel_has_parts(img), True)
        s.check(f"wheel works {name} {part}", m.wheel_works(img), works)
        s.check(f"wheel broken {name} {part}", m.part_broken(img), broken)
    for name, want in RIBBONS.items():
        s.check(f"monster ribbon {name}", m.monster_ribbon(top_only(frames / f"mh-{name}.jpg")),
                want)
    for name, who in PARTY.items():
        img = full(frames / f"mh-{name}.jpg")
        for w, want in who.items():
            s.check(f"ribbon {name} {w}", m.ribbon(img, w), want)
    ui = d / "ui"
    s.check("skills tab hammer", m.skills_tab(full(ui / "x.jpg")), 0)
    s.check("skills tab sword", m.skills_tab(full(ui / "x1.jpg")), 1)
    s.check("skills tab harp", m.skills_tab(full(ui / "x2.jpg")), 2)
    s.check("skills cursor on plain", m.skill_row(full(ui / "x.jpg")), 0)
    s.check("rage slash greyed", m.skill_greyed(full(ui / "x1.jpg"), 2), True)
    s.check("charge not greyed", m.skill_greyed(full(ui / "x1.jpg"), 1), False)
    s.check("aerial aim greyed", m.skill_greyed(full(ui / "x2.jpg"), 1), True)
    s.check("rapid shot not greyed", m.skill_greyed(full(ui / "x2.jpg"), 2), False)
    s.check("items open y", m.items_open(full(ui / "y.jpg")), True)
    s.check("items open i1", m.items_open(full(ui / "i1.jpg")), True)
    s.check("items not skills", m.items_open(full(ui / "x.jpg")), False)
    s.check("item row herb", m.lit_row(full(ui / "y.jpg"), m.ITEM_ROWS), 0)
    s.check("item row potion", m.lit_row(full(ui / "i1.jpg"), m.ITEM_ROWS), 1)
    s.check("ring on canyne", m.ring_who(full(ui / "i2.jpg")), "Canyne")
    s.check("ring on zerk", m.ring_who(full(ui / "i3.jpg")), "Zerk")
    for name in ("rt", "rt2", "m3"):
        s.check(f"status open {name}", m.status_open(full(ui / f"{name}.jpg")), True)
    for name in ("x", "y", "i2", "now", "a0", "w0"):
        s.check(f"status not {name}", m.status_open(full(ui / f"{name}.jpg")), False)
    s.check("menu up now", m.menu_up(full(ui / "now.jpg")), True)
    for name, want in (("214042", True), ("214730", True), ("214853", False), ("214642", False)):
        s.check(f"message band {name}", m.toast_up(full(frames / f"mh-{name}.jpg")), want)
    want = {"Canyne": 70 / 241, "Zerk": 44 / 197, "Ogden": 103 / 222, "Chirpy": 1.0}
    for who, share in want.items():
        got = m.bar_share(full(ui / "x.jpg"), who)
        s.check(f"hp bar {who} {got:.2f}", abs(got - share) < 0.04, True)
    s.check("monster health full", m.monster_health(full(frames / "mh-214511.jpg")) > 0.95, True)
    s.check("monster health low", m.monster_health(full(frames / "mh-214853.jpg")) < 0.15, True)
    stuck = Path.home() / ".cache/player-two/game2-work/stuck"
    if stuck.exists():
        s.check("take 2: cursor opens on row 1", m.skill_row(full(stuck / "s143.jpg")), 1)
        s.check("take 2: then row 2", m.skill_row(full(stuck / "s144.jpg")), 2)
        s.check("take 2: row 2 greyed", m.skill_greyed(full(stuck / "s144.jpg"), 2), True)
        s.check("take 2: row 1 not greyed", m.skill_greyed(full(stuck / "s144.jpg"), 1), False)
    print(f"\n{s.ok}/{s.n}")
    return 0 if s.ok == s.n else 1


async def muse(d: Path) -> int:
    from playertwo.config import Settings
    from playertwo.muse import Muse, image_part
    client = Muse(Settings.load())
    ui = d / "ui"
    t = time.monotonic()
    sheet = full(ui / "wheel3.jpg")
    reply = await client.json([{"role": "user", "content": [
        image_part(sheet, 2400), {"type": "text", "text": m.READ_WHEEL}]}],
        effort="minimal", max_tokens=1500)
    print("wheel", round(time.monotonic() - t, 2), json.dumps(reply))
    for name in ("rt", "rt2", "m3"):
        img = full(ui / f"{name}.jpg")
        w, h = img.size
        crop = img.crop((940 * w // 1280, 70 * h // 720, w, 365 * h // 720))
        t = time.monotonic()
        reply = await client.json([{"role": "user", "content": [
            image_part(crop, 1000), {"type": "text", "text": m.READ_STATUS.format(
                elements="Cluster Hammer: none, Frozen Speartuna: ice, Seahorse Harp: water")}]}],
            effort="minimal", max_tokens=800)
        print(name, round(time.monotonic() - t, 2), json.dumps(reply))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pixels", "muse"))
    ap.add_argument("--dir", default=str(Path.home() / ".cache/player-two/games-bakeoff/mhs3"))
    a = ap.parse_args()
    d = Path(a.dir)
    return pixels(d) if a.what == "pixels" else asyncio.run(muse(d))


if __name__ == "__main__":
    sys.exit(main())
