"""Run the narrator on still screenshots, one fresh session per image, for grading.

    uv run python scripts/eval_screens.py DIR_OR_IMAGES... --out results.jsonl

Each line of the output has the first-look narration (as when a new screen settles) and the
full "describe" answer, with timings. Grade them against the images by hand or with a
reviewer model; nothing here decides what's right.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from PIL import Image

from playertwo.config import Settings
from playertwo.muse import Muse
from playertwo.narrator import Narrator
from playertwo.screen import Frame


async def run_one(muse: Muse, s: Settings, path: Path) -> dict:
    frame = Frame(Image.open(path).convert("RGB"), time.time(), path.name)
    out: dict = {"file": str(path)}
    for kind in ("narrate", "describe"):
        nar = Narrator(muse, s.verbosity, s.game, s.language, s.team_up)
        t = time.monotonic()
        first = None
        text = ""
        async for delta in getattr(nar, kind)(frame):
            if first is None:
                first = time.monotonic() - t
            text += delta
        out[kind] = text.strip() or "<silent>"
        out[f"{kind}_first_s"] = round(first, 2) if first is not None else None
        out[f"{kind}_total_s"] = round(time.monotonic() - t, 2)
    return out


def append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--brain", choices=["auto", "meta", "open"])
    args = p.parse_args()

    s = Settings.load()
    if args.brain:
        s.brain = args.brain
    muse = Muse(s)
    files: list[Path] = []
    for raw in args.paths:
        q = Path(raw)
        files += sorted(q.glob("*.jpg")) + sorted(q.glob("*.png")) if q.is_dir() else [q]
    done = set()
    out = Path(args.out)
    if out.exists():
        done = {json.loads(line)["file"] for line in out.read_text().splitlines() if line}
    todo = [f for f in files if str(f) not in done]
    print(f"{len(todo)} to run ({len(done)} already done) on {muse.model}")

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(path: Path) -> None:
        async with sem:
            try:
                r = await run_one(muse, s, path)
            except Exception as exc:  # noqa: BLE001  keep going; the line records it
                r = {"file": str(path), "error": repr(exc)}
            append(out, r)
            print(f"  {path.name}: {r.get('narrate_total_s', 'error')}s", flush=True)

    await asyncio.gather(*(guarded(f) for f in todo))
    u = muse.usage
    print(f"{u.calls} calls, ${u.cost_usd():.3f} at Muse Spark prices")


if __name__ == "__main__":
    asyncio.run(main())
