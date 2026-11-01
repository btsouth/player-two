"""How well plain words find the right person (or thing) in the fair and in Truce:
people.resolve on the room survey, no model and no game.

Each case is a place, where the player stands in it, what they say, and what should
happen: go to one (by the survey's name), ask between two or three (all of the named ones
among the options), or find no one. Phrasings are the player's own ways of putting it,
and the requests made live on Steam on 2026-11-01 (night-2026-11-01/live2, audit).

    .venv/bin/python scripts/people_bench.py [--rooms FOLDER] [-v] [--held | --replay]

The rooms folder defaults to PLAYER_TWO_ROOMS, else the installed survey
(~/.config/player-two/atlas/chrono-trigger/rooms).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOMS = Path.home() / ".config/player-two/atlas/chrono-trigger/rooms"

LEENE, NORTH, INN, MANOR, MANOR_UP, MARKET, CRONO = 0x5, 0x1B7, 0xC, 0xD, 0xE, 0x11, 0x1
GATE = (392.0, 700.0)  # Leene Square, just in from the world map
FOUNTAIN = (360.0, 560.0)  # Leene Square, west of the fountain

MELCHIOR = "Melchior the sword seller in the middle"
LOWER = "the old man at the lower stalls"
ITEMS = "the item seller at the purple tent"
ARMOR = "the armor seller at the blue tent on the left"
GIRL = "the girl in the green dress"
GUARD = "the guard at the big tents"
WALKER = "the man walking by the west stalls"
BELLMAN = "the man by the bell game"

# (place, where the player stands, what they say, expected): expected is a name, a set of
# names to ask between, or None for no one.
CASES: list[tuple[int, tuple[float, float], str, object]] = [
    # Melchior, every way
    (LEENE, GATE, "talk to Melchior", MELCHIOR),
    (LEENE, GATE, "Melchior", MELCHIOR),
    (LEENE, FOUNTAIN, "take me to Melchior", MELCHIOR),
    (NORTH, (290.0, 330.0), "talk to Melchior in Leene Square", MELCHIOR),
    (LEENE, GATE, "the guy selling swords", MELCHIOR),
    (LEENE, GATE, "talk to the guy who sells swords", MELCHIOR),
    (LEENE, GATE, "talk to the sword seller", MELCHIOR),
    (LEENE, GATE, "the weapon shop", MELCHIOR),
    (LEENE, GATE, "the swordsmith", MELCHIOR),
    (LEENE, FOUNTAIN, "the old man selling swords", MELCHIOR),
    (LEENE, GATE, "talk to the shopkeeper in the middle", MELCHIOR),  # live2 talk-to-1
    (LEENE, FOUNTAIN, "talk to the shopkeeper at the southwest stall up the north stairs",
     "the shopkeeper at the southwest stall"),  # live2 talk-to-2
    (LEENE, GATE, "talk to the merchant with the blades", MELCHIOR),
    # the old man at the lower stalls
    (LEENE, GATE, "talk to the old man at the lower stalls", LOWER),
    (LEENE, FOUNTAIN, "talk to the old man at the lower stalls in Leene Square", LOWER),
    (LEENE, GATE, "the old man at the bottom", LOWER),
    (LEENE, GATE, "talk to the old guy who talked about the anniversary", LOWER),
    # too close to call: ask
    (LEENE, GATE, "the old man by the stalls", {MELCHIOR, LOWER}),
    (LEENE, GATE, "talk to the old man", {MELCHIOR, LOWER}),
    (LEENE, GATE, "talk to the shopkeeper", {MELCHIOR, ITEMS, ARMOR}),
    (LEENE, GATE, "the shopkeeper on the left", {ITEMS, ARMOR}),  # live2 talk-to-5/6
    # sellers
    (LEENE, GATE, "talk to the item seller", ITEMS),
    (LEENE, GATE, "the item seller at the purple tent", ITEMS),
    (LEENE, GATE, "the guy who sells potions", ITEMS),
    (LEENE, FOUNTAIN, "the item shop", ITEMS),
    (LEENE, GATE, "the potion seller", ITEMS),
    (LEENE, GATE, "talk to the armor seller", ARMOR),
    (LEENE, GATE, "the armor shop", ARMOR),
    (LEENE, GATE, "the man at the left blue tent", ARMOR),
    # others in the square
    (LEENE, GATE, "the girl", GIRL),
    (LEENE, GATE, "talk to the girl in green", GIRL),
    (LEENE, GATE, "the lady", GIRL),
    (LEENE, GATE, "talk to the woman next to the swords", GIRL),  # audit talk-to-5
    (LEENE, GATE, "that soldier", GUARD),
    (LEENE, GATE, "talk to the soldier", GUARD),  # audit talk-to-1/2
    (LEENE, GATE, "the guard", GUARD),
    (LEENE, GATE, "talk to the knight", GUARD),
    (LEENE, GATE, "the man walking around", WALKER),
    (LEENE, GATE, "talk to the man in the green cap at the stall on the left", WALKER),
    (LEENE, GATE, "the guy who talked about the war", WALKER),
    (LEENE, GATE, "the man by the bell", BELLMAN),
    (LEENE, GATE, "the strength test", "the bell"),
    (LEENE, GATE, "the bell", "the bell"),
    (LEENE, GATE, "the cat", "the cat"),
    (LEENE, GATE, "talk to the kid at the stairs", "the kid at the stairs"),
    (LEENE, GATE, "the boy", "the kid at the stairs"),
    (LEENE, GATE, "the musician", "the old musician to the east"),
    (LEENE, GATE, "the man at the race tent", "the race tent keeper at the right blue tent"),
    (LEENE, GATE, "the man by the flower cart", "the man by the flower cart"),
    (LEENE, GATE, "the fountain", "the fountain"),
    # no one: say so (or the model's turn)
    (LEENE, GATE, "the purple monster", None),
    (LEENE, GATE, "talk to Lucca", None),
    (LEENE, GATE, "talk to the innkeeper", None),
    # the fair, up the north stairs
    (NORTH, (290.0, 330.0), "talk to the shopkeeper",
     {"the shopkeeper at the southwest stall", "the shopkeeper at the north stall"}),
    (NORTH, (290.0, 330.0), "talk to the shopkeeper at the north stall",
     "the shopkeeper at the north stall"),
    (NORTH, (290.0, 330.0), "the guy eating lunch", "the shopkeeper at the southwest stall"),
    (NORTH, (290.0, 330.0), "the man at the tent", "the man at the tent"),
    # Truce Inn
    (INN, (700.0, 470.0), "talk to the innkeeper", "the innkeeper"),
    (INN, (700.0, 470.0), "the man at the bar", "the man at the bar"),
    (INN, (700.0, 470.0), "the guy upstairs", "the man upstairs"),
    (INN, (700.0, 470.0), "talk to the girl", "the girl by the table"),
    # Mayor's Manor
    (MANOR, (200.0, 500.0), "talk to the man by the table", "the man by the table"),
    (MANOR, (200.0, 500.0), "the woman in the kitchen", "the woman in the kitchen"),
    (MANOR, (200.0, 500.0), "the kid on the left", "the boy on the left"),
    (MANOR, (200.0, 500.0), "the girl", "the girl on the right"),
    (MANOR, (200.0, 500.0), "the guy who asked about shelters", "the man by the table"),
    (MANOR_UP, (300.0, 200.0), "talk to the soldier", "the soldier"),
    (MANOR_UP, (300.0, 200.0), "the old man", "the old man"),
    # Truce Market, Crono's house
    (MARKET, (352.0, 719.0), "talk to the shopkeeper", "the shopkeeper"),
    (MARKET, (352.0, 719.0), "the chest", "the chest"),
    (CRONO, (130.0, 200.0), "talk to Mom", "Crono's mom"),
    (CRONO, (130.0, 200.0), "talk to my mother", "Crono's mom"),
    (CRONO, (130.0, 200.0), "talk to Crono's mom", "Crono's mom"),
]

# Written after the resolver was tuned on CASES, and not tuned on: how it does on
# phrasings it hasn't seen (--held).
HELD: list[tuple[int, tuple[float, float], str, object]] = [
    (LEENE, GATE, "go see the sword guy", MELCHIOR),
    (LEENE, GATE, "chat with the blacksmith", MELCHIOR),
    (LEENE, GATE, "talk to the man with the beard", MELCHIOR),
    (LEENE, GATE, "the lady in the green dress", GIRL),
    (LEENE, GATE, "the soldier by the tents", GUARD),
    (LEENE, GATE, "the guy at the purple tent", ITEMS),
    (LEENE, GATE, "talk to the potion guy", ITEMS),
    (LEENE, GATE, "the shop with the armor", ARMOR),
    (LEENE, GATE, "the man who sells armor", ARMOR),
    (LEENE, GATE, "the man with the green hat", WALKER),
    (LEENE, GATE, "the old fellow", {MELCHIOR, LOWER}),
    (LEENE, GATE, "the kid", {"the kid in green by the grass", "the kid at the stairs"}),
    (LEENE, GATE, "the old guy near the fountain", {MELCHIOR, LOWER}),
    (LEENE, GATE, "the music guy", "the old musician to the east"),
    (LEENE, GATE, "the cat by the fountain", "the cat"),
    (LEENE, GATE, "the woman", GIRL),
    (INN, (700.0, 470.0), "the guy sitting at the bar", "the man at the bar"),
    (MANOR, (200.0, 500.0), "the lady in the kitchen", "the woman in the kitchen"),
    (MANOR, (200.0, 500.0), "the little boy", "the boy on the left"),
    (MANOR, (200.0, 500.0), "the soldier upstairs", "the soldier"),
    (MANOR, (200.0, 500.0), "the guy who talked about shelters", "the man by the table"),
    (MARKET, (352.0, 719.0), "the shop owner", "the shopkeeper"),
    (CRONO, (130.0, 200.0), "my mom", "Crono's mom"),
]


def run(folder: Path, verbose: bool = False, cases=None) -> tuple[int, int, list[str]]:
    """Every case: (passed, total, lines about the misses)."""
    from playertwo import people
    from playertwo.rooms import Here, Rooms

    rooms = Rooms(folder)
    ok, misses = 0, []
    cases = CASES if cases is None else cases
    for loc, (x, y), said, want in cases:
        here = Here(loc, (0, 0), x, y, 1.0)
        choices, targets, _hints = rooms.options(here)
        targets = {k: t for k, t in targets.items() if k in choices}
        cands = people.candidates(rooms, here, {k: choices[k] for k in targets}, targets)
        pick = people.resolve(said, cands, (x, y), people.landmarks(rooms, here),
                              people.MARGIN, people.middle(rooms, here))
        got_name = next((c.name for c in cands if c.key == pick.key), None)
        asked = {o.name for o in pick.options}
        if isinstance(want, set):
            good = not pick.sure and want <= asked and len(asked) <= 3
            got = f"asks: {pick.ask}" if pick.options else f"goes to {got_name}"
        elif want is None:
            good = not pick.sure and not pick.options
            got = f"goes to {got_name}" if pick.sure else (
                f"asks: {pick.ask}" if pick.options else f"no one ({pick.why})")
        else:
            good = pick.sure and got_name == want
            got = f"goes to {got_name}" if pick.sure else (
                f"asks: {pick.ask}" if pick.options else f"no one ({pick.why})")
        ok += good
        line = f"{'PASS' if good else 'MISS'} {rooms.name(loc)[:22]:22} {said!r}: {got}"
        if not good:
            misses.append(line + f" (wanted {want})")
        if verbose:
            print(line)
    return ok, len(cases), misses


# Last night's live "talk to" requests (night-2026-11-01/live2/<run>.txt), with who each
# meant (from the run's own result and the line heard): replayed on the frame recorded
# just before the request (<run>-before.png), located on the survey (--replay).
LIVE2 = Path.home() / ".cache/player-two/night-2026-11-01/live2"
REPLAY = {
    "talk-to-1": MELCHIOR,
    "talk-to-2": "the shopkeeper at the southwest stall",
    "talk-to-3": "the shopkeeper at the north stall",
    "talk-to-4": LOWER,
    "talk-to-5": {ITEMS, ARMOR},
    "talk-to-6": {ITEMS, ARMOR},
    "talk-to-7": MELCHIOR,
    "talk-to-8": ITEMS,
    "talk-to-9": "Crono's mom",
    "talk-to-10": "the man by the table",
    "talk-melchior-1": MELCHIOR,
}


def replay(folder: Path, live: Path = LIVE2) -> tuple[int, int, list[str]]:
    """The live requests on their recorded frames: (right, located, lines)."""
    import re

    from PIL import Image

    from playertwo import people
    from playertwo.atlas import Atlas
    from playertwo.rooms import Rooms

    atlas = Path.home() / ".config/player-two/atlas/chrono-trigger"
    rooms = Rooms(folder, Atlas(atlas) if (atlas / "atlas.json").exists() else None)
    ok = n = 0
    lines = []
    for name, want in REPLAY.items():
        said = re.search(r"REQUEST: (.*)", (live / f"{name}.txt").read_text()).group(1)
        t = time.monotonic()
        here = rooms.locate(Image.open(live / f"{name}-before.png"))
        if here is None:
            lines.append(f"---- {name:16} {said!r}: a text box was up (not located)")
            continue
        choices, targets, _ = rooms.options(here)
        targets = {k: v for k, v in targets.items() if k in choices}
        cands = people.candidates(rooms, here, {k: choices[k] for k in targets}, targets)
        pick = people.resolve(people.wanted(said) or said, cands, (here.x, here.y),
                              people.landmarks(rooms, here), people.MARGIN,
                              people.middle(rooms, here))
        got = next((c.name for c in cands if c.key == pick.key), None)
        good = got == want if isinstance(want, str) else (
            not pick.sure and want <= {o.name for o in pick.options})
        ok += good
        n += 1
        what = f"goes to {got}" if pick.sure else (
            f"asks: {pick.ask}" if pick.options else f"no one ({pick.why})")
        lines.append(f"{'PASS' if good else 'MISS'} {name:16} in {rooms.name(here.loc)} "
                     f"({time.monotonic() - t:.2f} s) {said!r}: {what}")
    return ok, n, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rooms", type=Path,
                    default=Path(os.environ.get("PLAYER_TWO_ROOMS") or ROOMS).expanduser())
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--held", action="store_true", help="the held-out phrasings")
    ap.add_argument("--replay", action="store_true",
                    help="last night's live requests on their recorded Steam frames")
    a = ap.parse_args()
    t = time.monotonic()
    if a.replay:
        ok, n, lines = replay(a.rooms)
        print("\n".join(lines))
        print(f"{ok}/{n} located frames right ({a.rooms})")
        return 0 if ok == n else 1
    ok, n, misses = run(a.rooms, a.verbose, HELD if a.held else None)
    for m in misses:
        print(m)
    print(f"{ok}/{n} in {time.monotonic() - t:.1f} s ({a.rooms})")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
