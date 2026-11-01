"""How the room survey does on real Steam screens: each screen of a surveyed place found
in it (and where), and screens of places not surveyed, battles and menus found nowhere.

    python scripts/rooms_steam_check.py [--rooms DIR] [--labels FILE] [--out FILE]

Screens: Steam trip screenshots (one per place entered, named after the place),
recordings of walks about (rec_fair: Leene Square; rec_down: the Mountain of Woe), and
the saved frames of a live session. --labels is a JSON file {"screen path": loc or
null} saying what each screen shows (null: no surveyed place); without one, these
defaults (the 2026-10-31 and 11-01 sessions). Prints k/n per place and the time each
look took.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from playertwo.atlas import ATLASES
from playertwo.rooms import Rooms

LIVE = Path.home() / ".cache/player-two/live"
REC = Path.home() / ".cache/player-two/recordings"
TRIPS = LIVE / "steam-2026-10-31/trips"
# Trip screens by name: the place each shows.
TRIP = {"crono-s-house": 0x1, "guardia-forest": 0x13, "lucca-s-house": 0x4,
        "truce-inn": 0xC, "truce-market": 0x11, "truce-mayor-s-manor": 0xD}


def screens(labels: Path | None) -> list[tuple[str, object, int | None]]:
    """(name, picture or path, the place it shows or None)."""
    out: list[tuple[str, object, int | None]] = []
    if labels is not None:
        for p, loc in json.loads(labels.read_text()).items():
            out.append((p, Path(p), loc))
        return out
    for f in sorted(TRIPS.glob("*.png")):
        stem = f.stem.rsplit("-", 1)[0]
        out.append((f.name, f, TRIP.get(stem)))
    for name, loc in (("rec_fair", 0x5), ("rec_down", None)):
        if (REC / f"{name}.npz").exists():
            for i, fr in enumerate(np.load(REC / f"{name}.npz")["frames"]):
                out.append((f"{name}#{i}", fr, loc))
    frames = LIVE / "steam-2026-11-01/frames"
    # The Mountain of Woe, battles and menus (13:46 to 16:56): no place surveyed then.
    for f in sorted(frames.glob("*.jpg")):
        if f.name <= "165609-00.jpg":
            out.append((f.name, f, 0x189 if "135812" <= f.name[:6] <= "135938" else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=Path, default=ATLASES / "chrono-trigger" / "rooms")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--skip-moving", action="store_true",
                    help="leave the places whose ground drifts out (the Mountain of Woe)")
    args = ap.parse_args()
    rooms = Rooms(args.rooms)
    if args.skip_moving:
        rooms.rooms = {k: v for k, v in rooms.rooms.items() if not v.moving}
    print(f"{len(rooms.rooms)} places surveyed")
    by: dict[str, list[bool]] = defaultdict(list)
    wrong: list[str] = []
    times: list[float] = []
    rows = []
    near: dict[str, object] = {}  # recordings: tracked from the last screen
    for name, pic, loc in screens(args.labels):
        if loc is not None and loc not in rooms.rooms:
            continue  # a place this survey doesn't have
        rgb = np.asarray(Image.open(pic).convert("RGB")) if isinstance(pic, Path) else pic
        track = name.split("#")[0] if "#" in name else None
        t = time.monotonic()
        h = rooms.locate(rgb, near.get(track) if track else None)
        times.append(time.monotonic() - t)
        if track:
            near[track] = h
        got = None if h is None else h.loc
        ok = got == loc
        key = f"{loc:#x} {rooms.name(loc)}" if loc is not None else "not surveyed"
        by[key].append(ok)
        if not ok:
            wrong.append(f"{name}: wanted {loc if loc is None else hex(loc)}, got "
                         f"{got if got is None else hex(got)}")
        rows.append({"screen": name, "want": loc, "got": got,
                     "cam": None if h is None else list(h.cam),
                     "feet": None if h is None else [round(h.x), round(h.y)],
                     "sure": None if h is None else h.sure})
    for k, v in sorted(by.items()):
        print(f"{k:60} {sum(v)}/{len(v)}")
    print(f"look: median {np.median(times) * 1000:.0f} ms, 95% {np.percentile(times, 95) * 1000:.0f}"
          f" ms, max {max(times) * 1000:.0f} ms")
    for w in wrong[:40]:
        print("MISS", w)
    if args.out:
        args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
