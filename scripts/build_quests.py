"""Build a game's quest list from its walkthrough with Muse (see playertwo/quests.py).

    .venv/bin/python scripts/build_quests.py                      # Chrono Trigger, all
    .venv/bin/python scripts/build_quests.py --chapters 3,9-10    # redo some chapters
    .venv/bin/python scripts/build_quests.py --link-only          # atlas grew: relink

Reads ~/.config/player-two/guides/<game>.txt, writes <game>.quests.json next to it. The
places are linked to the atlas's world-map names when the game has an atlas.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playertwo.config import Settings
from playertwo.guide import chapters, guide_path
from playertwo.muse import Muse
from playertwo.quests import Quests, build, link_places, quests_path


def numbers(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def atlas_maps(game: str) -> dict[str, list[str]]:
    from playertwo.atlas import Atlas

    atlas = Atlas.for_game(game)
    return {m.name: list(m.places) for m in atlas.maps} if atlas else {}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="Chrono Trigger")
    ap.add_argument("--chapters", help="only these chapter numbers, e.g. 1-3,7")
    ap.add_argument("--link-only", action="store_true", help="relink places, no extraction")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    src = guide_path(a.game)
    out = quests_path(a.game)
    if src is None or out is None:
        sys.exit(f"no guide for {a.game}")
    muse = Muse(Settings.load())
    # Whole chapters take longer than a live call's limit.
    muse.brain = muse.brain.with_options(timeout=300, max_retries=2)
    maps = atlas_maps(a.game)
    print(f"guide {src.name}; atlas maps: "
          + (", ".join(f"{k} ({len(v)} places)" for k, v in maps.items()) or "none"))

    if a.link_only:
        q = Quests.load(out)
    else:
        text = src.read_text(encoding="utf-8", errors="replace")
        only = numbers(a.chapters) if a.chapters else None
        q = await build(muse, a.game, text, list(maps), only, a.concurrency, a.effort)
        if only is not None and out.exists():
            # Put the redone chapters in among the rest, in the guide's order.
            order = {t: i for i, (t, _) in enumerate(chapters(text))}
            redone = {s.chapter for s in q.steps}
            kept = [s for s in Quests.load(out).steps if s.chapter not in redone]
            q.steps = sorted(kept + q.steps, key=lambda s: (order.get(s.chapter, 999), s.n))
            for i, s in enumerate(q.steps, 1):
                s.n = i
    linked = await link_places(muse, q, maps, a.game, a.effort) if maps else 0
    q.save(out)
    u = muse.usage
    print(f"{len(q.steps)} steps, {linked} linked to world-map places -> {out}")
    print(f"{u.calls} calls, {u.prompt_tokens} in, {u.completion_tokens} out, "
          f"${u.cost_usd():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
