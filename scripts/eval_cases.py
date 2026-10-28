"""Score first-look narration against hand-written cases from a real game.

    uv run python scripts/eval_cases.py tests/eval/p4g.json [--runs 2] [--verbosity normal]

Each case lists facts the narration must contain, phrases it must never use, and a word
budget (dialogue read word for word doesn't count against it). Every run is a fresh
narrator; a case can give "recent" lines to play a screen in the middle of a session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from PIL import Image

from playertwo.config import Settings
from playertwo.muse import Muse
from playertwo.narrator import Narrator
from playertwo.screen import Frame

QUOTED = re.compile(r"[\"“][^\"”]*[\"”]")


def words_outside_quotes(text: str) -> int:
    return len(re.findall(r"[\w']+", QUOTED.sub(" ", text)))


def score(case: dict, text: str) -> list[str]:
    """Problems with this narration; empty means it passed."""
    problems = []
    low = text.lower()
    for group in case["must"]:
        if not any(re.search(p, low) for p in group):
            problems.append(f"missing {group[0]!r}")
    for p in case["never"]:
        if re.search(p, low):
            problems.append(f"said {p!r}")
    n = words_outside_quotes(text)
    if n > case["max_words"]:
        problems.append(f"{n} words (max {case['max_words']})")
    return problems


async def narrate_once(muse: Muse, s: Settings, path: Path,
                       recent: list[str]) -> tuple[str, float]:
    nar = Narrator(muse, s.verbosity, s.game, s.language, s.team_up)
    nar.recent.extend(recent)
    frame = Frame(Image.open(path).convert("RGB"), time.time(), path.name)
    t = time.monotonic()
    text = "".join([d async for d in nar.narrate(frame)]).strip()
    return text or "<silent>", time.monotonic() - t


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cases")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--verbosity", default="normal")
    p.add_argument("--brain", choices=["auto", "meta", "open"])
    p.add_argument("--only", help="regex on file names")
    p.add_argument("--out", help="write every result as JSON lines")
    args = p.parse_args()

    spec = json.loads(Path(args.cases).read_text())
    images = Path(spec["images"]).expanduser()
    s = Settings.load()
    s.game, s.verbosity = spec["game"], args.verbosity
    if args.brain:
        s.brain = args.brain
    muse = Muse(s)
    cases = [c for c in spec["cases"] if not args.only or re.search(args.only, c["file"])]
    sem = asyncio.Semaphore(4)

    async def run(case: dict, i: int) -> dict:
        async with sem:
            text, secs = await narrate_once(muse, s, images / case["file"],
                                            case.get("recent", []))
        return {"file": case["file"], "what": case["what"], "run": i, "text": text, "secs": round(secs, 2),
                "problems": score(case, text)}

    results = await asyncio.gather(*(run(c, i) for c in cases for i in range(args.runs)))
    passed = sum(not r["problems"] for r in results)
    for case in cases:
        rs = [r for r in results if r["what"] == case["what"]]
        ok = sum(not r["problems"] for r in rs)
        print(f"{ok}/{len(rs)} {case['file']:12} {case['what']}")
        for r in rs:
            if r["problems"]:
                print(f"     x {'; '.join(r['problems'])}\n       {r['text'][:220]}")
    secs = [r["secs"] for r in results]
    words = [words_outside_quotes(r["text"]) for r in results]
    print(f"\n{passed}/{len(results)} passed on {muse.model} ({args.verbosity}); "
          f"median {statistics.median(secs):.2f}s, median {statistics.median(words)} words; "
          f"${muse.usage.cost_usd():.3f}")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results))


if __name__ == "__main__":
    asyncio.run(main())
