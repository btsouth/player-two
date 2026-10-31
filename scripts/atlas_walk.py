"""Walk to a place by the atlas alone, no model: for testing atlas.walk_to in a box with
the game open (PLAYER_TWO_KEYS=hyprland for RetroArch).

    python scripts/atlas_walk.py GAME PLACE [--limit 90]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from playertwo.atlas import Atlas, walk_to
from playertwo.controls import keyboard
from playertwo.screen import HyprlandCapture


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("place")
    ap.add_argument("--limit", type=float, default=90.0)
    args = ap.parse_args()
    atlas = Atlas.for_game(args.game)
    if atlas is None:
        print("no atlas")
        return 2
    found = atlas.find_place(args.place)
    if found is None:
        print("no such place")
        return 2
    m, name = found
    cap = HyprlandCapture("RetroArch")
    keys = keyboard(focus=cap.focus)
    t0 = time.monotonic()

    async def shoot():
        return (await asyncio.to_thread(cap.grab)).image

    async def keep_going():
        return True

    def log(text: str) -> None:
        print(f"{time.monotonic() - t0:6.1f} {text}", flush=True)

    try:
        out = await walk_to(atlas, m, name, shoot, keys.steer, keys.press, keep_going, log,
                            args.limit)
    finally:
        keys.close()
    print(f"{time.monotonic() - t0:6.1f} RESULT {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
