"""Walk out of a place to another by the surveyed rooms alone, no model: for testing
rooms.walk_out in a box with the game open (PLAYER_TWO_KEYS=hyprland for RetroArch).

    python scripts/rooms_walk.py GAME TO [--limit 90] [--exit NAME]  # TO: a location (hex)
    python scripts/rooms_walk.py GAME TO --target NAME [--talk]  # someone or something in TO
    python scripts/rooms_walk.py GAME --where                # where it thinks it is
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from playertwo.controls import keyboard
from playertwo.rooms import Rooms, Target, describe, walk_out
from playertwo.screen import HyprlandCapture


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("to", nargs="?", default="")
    ap.add_argument("--limit", type=float, default=90.0)
    ap.add_argument("--where", action="store_true")
    ap.add_argument("--exit", default="", help="which way out into TO, by its name")
    ap.add_argument("--target", default="", help="someone or something in TO, by its name "
                    "(as the survey has it, without the place in brackets)")
    ap.add_argument("--talk", action="store_true", help="press a facing them")
    args = ap.parse_args()
    rooms = Rooms.for_game(args.game)
    if rooms is None:
        print("no rooms")
        return 2
    cap = HyprlandCapture("RetroArch")
    t0 = time.monotonic()

    keep = os.environ.get("ROOMS_FRAMES")  # a folder to keep every screen looked at in
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)

    async def shoot():
        img = (await asyncio.to_thread(cap.grab)).image
        if keep:
            img.save(Path(keep) / f"{time.monotonic() - t0:07.2f}.png")
        return img

    if args.where:
        t = time.monotonic()
        here = rooms.locate(await shoot())
        print(f"{time.monotonic() - t:.2f} s", here,
              describe(rooms, here) if here else "not in a surveyed place")
        return 0
    keys = keyboard(focus=cap.focus)

    async def keep_going():
        return True

    def log(text: str) -> None:
        print(f"{time.monotonic() - t0:6.1f} {text}", flush=True)

    target = None
    if args.target:
        room = rooms.rooms.get(int(args.to, 16))
        got = [Target(room.loc, p.at, p.kind != "save point", p.name)
               for p in room.people if p.name == args.target] if room else []
        got += [Target(room.loc, v, False, k) for k, v in room.spots.items()
                if k == args.target] if room else []
        if not got:
            print(f"RESULT no {args.target!r} in {args.to}")
            return 2
        target = got[0]
    try:
        while True:
            left = args.limit - (time.monotonic() - t0)
            out = await walk_out(rooms, int(args.to, 16), shoot, keys.steer, keep_going, log,
                                 left, label=args.exit, spot=target, press=keys.press,
                                 talk=args.talk)
            if not out.startswith("interrupted") or left < 5:
                break
            # What the model would do: read it and press on, or fight (a forest's
            # monsters: the attack, over and over, wins those).
            log(f"{out}: pressing a")
            for _ in range(8):
                keys.press("a")
                await asyncio.sleep(0.35)
    finally:
        keys.close()
    print(f"{time.monotonic() - t0:6.1f} RESULT {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
