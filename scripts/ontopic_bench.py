#!/usr/bin/env python3
"""How often the on-topic check (src/playertwo/ontopic.py) would hold back a good line, and
how often it catches an unrelated one.

Good lines: every line the agent said in the given event logs (say events from the agent
and each request's outcome), checked against what it was given that request as far as the
log shows it (the player's words, the last four requests, tool arguments and results cut
to 200 characters, the instructions and the game pack): less than it had live, so a
conservative count. Unrelated lines: OFF below (sentences from elsewhere, like the football
rule live 3 said), each checked against the context of a real request.

    python scripts/ontopic_bench.py ~/.cache/player-two/live/*/events.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from playertwo import ontopic  # noqa: E402
from playertwo.agent import TOOLS, CT_TOOLS  # noqa: E402
from playertwo.prompts import AGENT, BATTLE_CT  # noqa: E402

OFF = [
    "If a tackle involves contact that endangers the safety of an opponent, it must be "
    "sanctioned as serious foul play.",
    "Preheat the oven to 350 degrees and grease a nine inch baking pan.",
    "The mitochondria is the powerhouse of the cell.",
    "In 1066, William the Conqueror defeated Harold Godwinson at the Battle of Hastings.",
    "The function returns a promise that resolves when the request completes.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Interest rates rose by a quarter point after the central bank's meeting.",
    "Please consult your doctor before starting any new exercise program.",
    "The offside rule applies when an attacker is nearer the goal line than the ball.",
    "Paris is the capital of France and its largest city.",
    "To reset your password, click the link we sent to your email address.",
    "The quarterback threw for three hundred yards in the second half.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Our return policy allows refunds within thirty days of purchase with a receipt.",
    "The defendant pleaded not guilty to all charges in court on Tuesday.",
    "Add two cups of flour, a pinch of salt and stir until smooth.",
    "The stock market closed higher as technology shares rallied.",
    "Shakespeare wrote Hamlet around the year sixteen hundred.",
    "Make sure the referee signals before the free kick is taken.",
    "The weather tomorrow will be cloudy with a chance of afternoon showers.",
    "Install the package with pip and import it at the top of your script.",
    "Elephants are the largest land animals on Earth.",
    "The meeting has been moved to three o'clock in conference room B.",
    "Vote early: polling stations open at seven in the morning.",
    "A yellow card is shown for unsporting behaviour or persistent infringement.",
    "Thank you for calling, your call is important to us.",
    "The novel follows a young woman who moves to London to become a writer.",
    "Lithium batteries should be stored at room temperature away from heat.",
    "Your package has shipped and should arrive within five business days.",
    "The Pythagorean theorem relates the sides of a right triangle.",
    "Tennis players switch ends after every odd game in a set.",
    "Brush your teeth twice a day and floss at least once.",
    "As an AI language model, I cannot browse the internet.",
    "The striker scored twice before halftime to give the visitors the lead.",
    "Sales tax is added at checkout based on your shipping address.",
    "The museum is closed on Mondays and public holidays.",
    "Hydrogen is the most abundant element in the universe.",
    "Remember to water the plants every other day during summer.",
    "The bill passed the Senate by a narrow margin late last night.",
    "Turn left at the next intersection and continue for two miles.",
]


def contexts(paths: list[str]):
    """(line, context texts) for every line the agent said, request by request."""
    base = [AGENT, BATTLE_CT, json.dumps(TOOLS), json.dumps(CT_TOOLS)]
    pack = Path("~/.config/player-two/games/chrono-trigger.pack.txt").expanduser()
    if pack.exists():
        base.append(pack.read_text())
    base.append("Chrono Trigger")
    for path in paths:
        history: list[str] = []
        said: list[str] = []
        for raw in Path(path).read_text().splitlines():
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "say" and e.get("source") == "agent" and e.get("text"):
                said.append(e["text"])
            if e.get("kind") == "heard":
                said = []
            if e.get("kind") != "agent":
                continue
            steps = e.get("steps") or []
            own = [e.get("request") or "", *history[-4:],
                   *(json.dumps(s.get("args")) + " " + str(s.get("result") or "")
                     for s in steps)]
            ctx = ontopic.Grounds(*own, *base)
            lines = list(dict.fromkeys([*said, e.get("outcome") or ""]))
            for line in lines:
                if line and line not in ("cancelled", "gave up", "done"):
                    yield line, ctx, e.get("request") or "", path
            history.append(f"{e.get('request')} -> {e.get('outcome')}")
            said = []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", nargs="+")
    p.add_argument("--show", type=int, default=30, help="good lines held back to list")
    p.add_argument("--judge", action="store_true",
                   help="also ask the model about every line held back (Muse calls)")
    a = p.parse_args()
    good = list(contexts(a.logs))
    held = [(ln, req, path) for ln, ctx, req, path in good if ctx.off_topic(ln)]
    # Each unrelated line against the context of five real requests.
    rng = random.Random(7)
    caught = total = 0
    missed = []
    for line in OFF:
        for _ln, ctx, _req, _path in rng.sample(good, min(5, len(good))):
            total += 1
            if ctx.off_topic(line):
                caught += 1
            else:
                missed.append(line)
    print(f"good lines: {len(good)} from {len(a.logs)} logs; held back {len(held)} "
          f"({100 * len(held) / max(1, len(good)):.1f}%)")
    print(f"unrelated lines: caught {caught}/{total}")
    for ln, req, _path in held[: a.show]:
        print(f"  held: {ln[:110]!r}  (asked {req[:40]!r})")
    for line in sorted(set(missed)):
        print(f"  missed: {line[:110]!r}")
    if a.judge:
        asyncio.run(judge(held, [ln for ln in OFF]))


async def judge(held: list[tuple[str, str, str]], off: list[str]) -> None:
    """The model's second look at the good lines held back (a no is a false reject) and
    at every unrelated line, each beside a real request."""
    from playertwo.config import Settings
    from playertwo.muse import Muse

    muse = Muse(Settings.load())
    rng = random.Random(11)
    asks = [req for _ln, req, _p in held] or ["what's on the screen?"]
    jobs = [(ln, req, True) for ln, req, _p in held] + [
        (ln, rng.choice(asks + ["what's the battle mode set to?", "buy three potions",
                                "where am I?", "heal Frog"]), False) for ln in off]
    sem = asyncio.Semaphore(6)
    times: list[float] = []

    async def one(ln: str, req: str, good: bool):
        async with sem:
            t = time.monotonic()
            ok = await ontopic.second_look(muse, "Chrono Trigger", req, ln)
            times.append(time.monotonic() - t)
            return ln, req, good, ok

    res = await asyncio.gather(*(one(*j) for j in jobs))
    bad_no = [(ln, req) for ln, req, good, ok in res if good and ok is not True]
    let_through = [(ln, req) for ln, req, good, ok in res if not good and ok is not False]
    n_good = sum(1 for j in jobs if j[2])
    print(f"second look: good lines said no to {len(bad_no)}/{n_good}; unrelated said yes "
          f"to {len(let_through)}/{len(off)}; median {sorted(times)[len(times) // 2]:.2f} s, "
          f"slowest {max(times):.2f} s; ${muse.usage.cost_usd():.3f}")
    for ln, req in bad_no:
        print(f"  no to good: {ln[:100]!r} (asked {req[:40]!r})")
    for ln, req in let_through:
        print(f"  yes to unrelated: {ln[:100]!r} (asked {req[:40]!r})")


if __name__ == "__main__":
    main()
