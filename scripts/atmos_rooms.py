"""Draft each surveyed place's line of what it's like (the rooms' atmos.json), from its
survey picture.

    python scripts/atmos_rooms.py ROOMS_DIR OUT.json [--locs 5,19] [--game "Chrono Trigger"]

A survey picture is a whole place stitched from the game with its people left out, so a line
says what the place itself is like (the kind of place, materials, colours, light, mood),
never who is there. Muse drafts the lines; check each against its picture, and against a
frame of the place as the player sees it when there is one, before they're used: Player Two
says them as fact, the first time the player walks in. OUT.json is {"lines": {"5": "..."},
"names": {"5": "Leene Square"}}; the session reads "lines" from <rooms>/atmos.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playertwo.config import Settings
from playertwo.muse import Muse, image_part
from playertwo.rooms import Rooms

PROMPT = """\
This picture is a map of one whole place in a top-down 2D role-playing game ({game}), \
stitched from the game's screens and seen from above; black is outside it, and its people \
and animals aren't drawn. The place is called "{name}". For a blind player who has just \
walked in, write one sentence, 8 to 16 words, of what this place is like: the kind of \
place, its main materials and colours, its light and its mood. Only what the picture \
shows: no people or animals, no names, no story, and no time of day or weather unless the \
picture plainly shows it (a night sky, rain). Don't start with the place's name, "This \
is" or "You". Like: "A snug wooden cabin, lamplit, with a bed by the hearth and shelves \
of jars." or "A dark stone corridor, damp and narrow, lit by a few torches." \
JSON: {{"line": "..."}}"""


def picture(path: Path) -> Image.Image:
    """The survey picture, scaled up (pixel art, by whole steps) to about 800 pixels."""
    img = Image.open(path).convert("RGB")
    k = max(1, 800 // max(img.size))
    return img.resize((img.width * k, img.height * k), Image.Resampling.NEAREST) if k > 1 else img


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rooms")
    ap.add_argument("out")
    ap.add_argument("--locs", default="")
    ap.add_argument("--game", default="Chrono Trigger")
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    folder = Path(a.rooms).expanduser()
    rooms = Rooms(folder)  # the names as the session says them (no atlas: "Residence")
    want = {int(x, 0) for x in a.locs.split(",") if x} if a.locs else set(rooms.rooms)
    s = Settings.load()
    s.brain = "meta"
    muse = Muse(s)
    out_path = Path(a.out)
    done = json.loads(out_path.read_text()) if out_path.exists() else {"lines": {}, "names": {}}
    sem = asyncio.Semaphore(4)

    async def one(loc: int) -> None:
        name = rooms.name(loc)
        async with sem:
            msgs = [{"role": "user", "content": [
                image_part(picture(folder / f"loc-{loc:03x}.png"), 1280),
                {"type": "text", "text": PROMPT.format(game=a.game, name=name)}]}]
            try:
                r = await muse.json(msgs, effort=a.effort, max_tokens=2000)
            except Exception as exc:  # noqa: BLE001  the others still get theirs
                print(f"{loc:#x} {name}: {exc!r}"[:200], flush=True)
                return
        line = " ".join(str(r.get("line", "")).split())
        if line:
            done["lines"][str(loc)] = line
            done["names"][str(loc)] = name
        print(f"{loc:#05x} {name:<48} {line}", flush=True)

    await asyncio.gather(*(one(loc) for loc in sorted(want) if loc in rooms.rooms))
    done["lines"] = dict(sorted(done["lines"].items(), key=lambda kv: int(kv[0])))
    done["names"] = dict(sorted(done["names"].items(), key=lambda kv: int(kv[0])))
    out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False) + "\n")
    print(f"cost ${muse.usage.cost_usd():.4f} for {muse.usage.calls} calls")


if __name__ == "__main__":
    asyncio.run(main())
