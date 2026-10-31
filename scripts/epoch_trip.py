"""Go somewhere by the Epoch with no model, for testing epoch.Pilot in a box with the game
open (PLAYER_TWO_KEYS=hyprland for RetroArch): a place (walks when it can, else flies),
or an era.

    python scripts/epoch_trip.py GAME PLACE [--limit 150]
    python scripts/epoch_trip.py GAME --era "600 A.D."
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

from playertwo.atlas import Atlas, walk_to
from playertwo.controls import keyboard
from playertwo.epoch import Pilot, era_named, needs_epoch
from playertwo.screen import HyprlandCapture


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("place", nargs="?", default="")
    ap.add_argument("--era")
    ap.add_argument("--map", help="the place's map (era), when names repeat")
    ap.add_argument("--fly", action="store_true", help="fly even when walking would do")
    ap.add_argument("--limit", type=float, default=150.0)
    args = ap.parse_args()
    atlas = Atlas.for_game(args.game)
    if atlas is None:
        print("no atlas")
        return 2
    cap = HyprlandCapture("RetroArch")
    keys = keyboard(focus=cap.focus)
    t0 = time.monotonic()

    async def shoot():
        return (await asyncio.to_thread(cap.grab)).image

    async def keep_going():
        return time.monotonic() - t0 < args.limit

    def log(text: str) -> None:
        print(f"{time.monotonic() - t0:6.1f} {text}", flush=True)

    notes = Path(tempfile.mkdtemp(prefix="p2-epoch-"))
    pilot = Pilot(atlas, args.game, shoot, keys.steer, keys.press, keep_going, log, notes)
    try:
        if args.era:
            era = era_named(args.era)
            if era is None:
                print("no such era")
                return 2
            out = await pilot.go_to_era(era)
        else:
            s = await pilot.where(near=False)
            found = None
            if args.map or s is not None:  # the map asked for, else this one first
                want = args.map or s.map
                found = next(((atlas.map(want), n) for n in atlas.map(want).places
                              if n.lower() == args.place.lower()), None)
            found = found or atlas.find_place(args.place)
            if found is None:
                print("no such place")
                return 2
            m, name = found
            log(f"going to {name} on the {m.name} map")
            out = ""
            if s is not None and s.map == m.name and not args.fly:
                out = await walk_to(atlas, m, name, shoot, keys.steer, keys.press,
                                    keep_going, log, args.limit)
                log(f"walk: {out}")
            if not out or needs_epoch(out) or (s is not None and s.map != m.name):
                out = await pilot.travel(m, name)
    finally:
        keys.steer(None)
        keys.close()
    print(f"{time.monotonic() - t0:6.1f} RESULT {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
