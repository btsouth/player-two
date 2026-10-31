"""Walk out of a place to another by the surveyed rooms alone, no model: for testing
rooms.walk_out in a box with the game open (PLAYER_TWO_KEYS=hyprland for RetroArch).

    python scripts/rooms_walk.py GAME TO [--limit 90] [--exit NAME]  # TO: a location (hex)
    python scripts/rooms_walk.py GAME --where                # where it thinks it is
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from playertwo.controls import keyboard
from playertwo.rooms import Rooms, describe, walk_out
from playertwo.screen import HyprlandCapture


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("to", nargs="?", default="")
    ap.add_argument("--limit", type=float, default=90.0)
    ap.add_argument("--where", action="store_true")
    ap.add_argument("--exit", default="", help="which way out into TO, by its name")
    args = ap.parse_args()
    rooms = Rooms.for_game(args.game)
    if rooms is None:
        print("no rooms")
        return 2
    cap = HyprlandCapture("RetroArch")
    t0 = time.monotonic()

    async def shoot():
        return (await asyncio.to_thread(cap.grab)).image

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

    try:
        while True:
            left = args.limit - (time.monotonic() - t0)
            out = await walk_out(rooms, int(args.to, 16), shoot, keys.steer, keep_going, log,
                                 left, label=args.exit)
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
