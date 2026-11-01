"""Offline battle bench: last night's battle frames, read and decided on without the game.

    python scripts/battle_replay.py reader --gt GT.json [--gt MORE.json] [--out RUN.jsonl]
    python scripts/battle_replay.py heal --gt GT.json [--gt MORE.json]
    python scripts/battle_replay.py foes --gt FOES.json [--out RUN.jsonl] [--ids a,b]

reader: what the pixel reader makes of each labelled battle frame (kind, whose turn, the
command or list entry under the hand, party HP, who's in danger), scored field by field.
heal: "heal <each member>" planned on each labelled frame with the party's HP; the plan is
scored against what the labels say should happen (full HP: nothing pressed; knocked out:
left to the model; hurt: that member's heal, by whoever is up).
foes: the enemy count the fight lines are built on (one model call a frame; Muse key).

The labels are kept out of the repo (they quote the game): narrator-bench/gt.json's battle
frames take the same fields as a reader label.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playertwo import ctsteam

KINDS = {"battle_command": "command", "battle_list": "list", "battle_target": "target",
         "battle_anim": "none", "battle_message": "none", "battle_paused": "paused"}


def labels(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        for g in json.loads(Path(p).read_text()):
            if g.get("kind", "").startswith("battle"):
                out.append(g)
    return out


def expect(g: dict) -> dict:
    """What a label says the reader should find, field by field (absent: not labelled)."""
    kind = KINDS[g["kind"]]
    want: dict = {"kind": kind}
    must = [m for m in g.get("must", []) if "|" not in m]
    if kind == "command" and len(must) >= 2:
        want["actor"], want["command"] = must[0], must[1]
    if kind == "list" and must:
        want["entry"] = must[0]
    hp = (g.get("q") or {}).get("hp")
    if hp:
        want["hp"] = {n: h for n, h in hp}
    if "danger" in g and kind != "paused":  # (a paused frame's label is the word only)
        want["danger"] = sorted(n for n, _h in g["danger"])
    if kind == "target":
        want["glove"] = True
    return want


def got(st: ctsteam.State) -> dict:
    kind = "paused" if st.paused and st.kind == "none" else st.kind
    return {"kind": kind, "actor": st.actor, "command": st.command, "entry": st.entry,
            "hp": {m.name: m.hp for m in st.party},
            "danger": sorted(m.name for m in st.party
                             if m.hp is not None and m.hp_max and m.hp < m.hp_max / 3),
            "glove": bool(st.gloves)}


def reader(args) -> None:
    rows, score = [], Counter()
    for g in labels(args.gt):
        img = cv2.imread(g.get("src") or g["path"])
        if img is None:
            continue
        t = time.monotonic()
        st = ctsteam.read(img)
        ms = (time.monotonic() - t) * 1000
        want, have = expect(g), got(st)
        res = {}
        for k, v in want.items():
            if k == "hp":
                ok = all(have["hp"].get(n) == h for n, h in v.items())
            elif k == "kind" and v == "paused":
                ok = st.paused
            elif k == "kind" and v == "none":
                ok = have["kind"] in ("none", "paused") and st.in_battle
            else:
                ok = have.get(k) == v
            res[k] = ok
            score[k, ok] += 1
        score["all", all(res.values())] += 1
        rows.append({"id": g["id"], "ok": all(res.values()), "bad": [k for k, v in res.items() if not v],
                     "want": want, "have": {k: have[k] for k in want}, "ms": round(ms)})
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        if not r["ok"]:
            print(f"  {r['id']}: {', '.join(r['bad'])}  want {r['want']}  have {r['have']}")
    for k in ("all", "kind", "actor", "command", "entry", "hp", "danger", "glove"):
        n = score[k, True] + score[k, False]
        if n:
            print(f"{k:8s} {score[k, True]}/{n}")
    print(f"ms median {sorted(r['ms'] for r in rows)[len(rows) // 2]}")


def heal(args) -> None:
    """Plan "heal <member>" on each frame whose party HP is labelled, and check the plan
    against the labels: the reader's HP drives the plan, the labels judge it."""
    from playertwo.battle import heal_plan, needless

    score, bad = Counter(), []
    for g in labels(args.gt):
        hp = dict((g.get("q") or {}).get("hp") or [])
        if not hp:
            continue
        img = cv2.imread(g.get("src") or g["path"])
        if img is None:
            continue
        st = ctsteam.read(img)
        tops = {m.name: m.hp_max for m in st.party}
        for who, h in hp.items():
            plan = heal_plan(st, who)
            top = tops.get(who)
            if h == 0:
                want = "model"  # a revive: the model's call
            elif top is not None and h == top:
                want = "say"
            elif top is None:
                want = "?"
            else:
                want = "heal"
            have = ("model" if plan is None else "say" if plan.say else
                    "heal" if plan.intents and all(i.target == who or not i.target
                                                   for i in plan.intents[:1]) else "other")
            if want == "?":
                continue
            # Whatever comes first in the plan must not be a heal the guard would refuse.
            if have == "heal" and needless(st, plan.intents[0]):
                have = "needless"
            ok = have == want
            score[ok] += 1
            score[want, ok] += 1
            if not ok:
                bad.append(f"  {g['id']} heal {who} ({h}): want {want}, plan {have}"
                           + (f" {plan.intents[0]}" if plan and plan.intents else ""))
    print("\n".join(bad))
    print(f"heal decisions right: {score[True]}/{score[True] + score[False]}  (" + ", ".join(
        f"{w} {score[w, True]}/{score[w, True] + score[w, False]}"
        for w in ("heal", "say", "model")) + ")")


def foes(args) -> None:
    from PIL import Image

    from playertwo.agent import Agent
    from playertwo.config import Settings
    from playertwo.muse import Muse

    gt = json.loads(Path(args.gt[0]).read_text())
    if args.ids:
        keep = set(args.ids.split(","))
        gt = [g for g in gt if g["id"] in keep]
    agent = Agent.__new__(Agent)
    agent.muse = Muse(Settings.load())
    agent._executor, agent._party = None, []
    agent.on_action = lambda kind, d: print(f"  {d.get('text', d)}", file=sys.stderr)
    rows = []

    async def go():
        for g in gt:
            t = time.monotonic()
            f = await agent.count_foes(Image.open(g["path"]).convert("RGB"), g.get("party", []))
            rows.append({"id": g["id"], "want": g["count"], "have": f.count if f else None,
                         "look": f.look if f else "", "s": round(time.monotonic() - t, 2)})
            print(json.dumps(rows[-1]), flush=True)

    asyncio.run(go())
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    right = sum(r["have"] == r["want"] for r in rows)
    off1 = sum(r["have"] is not None and abs(r["have"] - r["want"]) == 1 for r in rows)
    print(f"count right {right}/{len(rows)}, off by one {off1}, "
          f"median {sorted(r['s'] for r in rows)[len(rows) // 2]} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("reader", "heal", "foes"))
    ap.add_argument("--gt", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ids")
    args = ap.parse_args()
    {"reader": reader, "heal": heal, "foes": foes}[args.mode](args)


if __name__ == "__main__":
    main()
