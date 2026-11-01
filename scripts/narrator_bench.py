"""Narration benchmark on labelled real Steam frames.

    python scripts/narrator_bench.py run --gt GT.json --out RUN.jsonl [--mode model|fast] [--ids a,b]
    python scripts/narrator_bench.py score RUN.jsonl --gt GT.json
    python scripts/narrator_bench.py ask --gt GT.json --out ASK.jsonl [--mode model|fast]

Each frame is narrated as a first look (nothing said before). The ground truth (labelled by
eye, kept out of the repo since it quotes the game) says what must be said: text to read word
for word, the selected option, who's in danger, text still typing out, the names shown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playertwo.config import Settings  # noqa: E402
from playertwo.muse import Muse  # noqa: E402
from playertwo.narrator import Narrator  # noqa: E402
from playertwo.screen import Frame  # noqa: E402

# Proper nouns a narration could make up: places, people, eras, monsters.
CT_NAMES = [
    "Truce", "Leene Square", "Millennial Fair", "Guardia", "Guardia Forest", "Guardia Castle",
    "Medina", "Porre", "Zenan Bridge", "Mountain of Woe", "Mt. Woe", "Ioka", "Tyrano Lair",
    "Arris Dome", "Proto Dome", "Trann Dome", "Keeper's Dome", "Zeal", "Kajar", "Enhasa",
    "Ocean Palace", "Lab 16", "Lab 32", "Denadoro", "Cursed Woods", "Heckran", "Sun Keep",
    "Magic Cave", "Magus's Castle", "Mystic Mountains", "Forest Maze", "Hunting Range",
    "Laruba", "Dactyl Nest", "Giant's Claw", "North Cape", "Choras", "Black Omen",
    "Death Peak", "End of Time", "Lucca's House", "Crono's House", "Fiona", "Sunken Desert",
    "Geno Dome", "Bangor Dome", "Skyway", "Terra Cave", "Algetty", "Last Village",
    "Mayor's Manor", "Truce Inn", "Market", "Residence", "Leene", "Millennial",
    "Crono", "Marle", "Lucca", "Robo", "Frog", "Ayla", "Magus", "Schala", "Janus", "Gaspar",
    "Melchior", "Spekkio", "Belthasar", "Dalton", "Azala", "Kino", "Lavos", "Taban", "Toma",
    "Fritz", "Doan", "Cyrus", "Glenn", "Ozzie", "Flea", "Yakra", "Gato",
    "Gargoyle", "Imp", "Rubble", "Bellbird", "Golem", "Mud Imp", "Blue Beast", "Red Beast",
    "Present", "Middle Ages", "Prehistory", "Antiquity", "Future", "Dark Ages",
    "1000 A.D.", "600 A.D.", "2300 A.D.", "65,000,000 B.C.", "12,000 B.C.",
    "Guru of Woe", "Kilwala", "Hench", "Mystic", "Reptite",
]


def norm(text: str) -> str:
    t = text.lower().replace("’", "'").replace("h.p.", "hp").replace("m.p.", "mp")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"[^a-z0-9' /]+", " ", t).replace("/", " ")
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return norm(text).replace("'", "").split()


def lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


# The player's own character may be "you" (the narrator's rule for renamable heroes).
HERO = "Crono"


def has(text: str, phrase: str) -> bool:
    if phrase == HERO and (has(text, "you") or has(text, "your")):
        return True
    p = norm(phrase).replace("'", "")
    t = norm(text).replace("'", "")
    return bool(p) and re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t) is not None


def score(e: dict, said: str) -> dict:
    """How a narration (or answer) measures up to one frame's ground truth."""
    out: dict = {"words": len(words(said))}
    if e.get("read"):
        gw = words(e["read"])
        out["read"] = lcs(gw, words(said)) / len(gw)
    if e.get("must"):
        out["must"] = sum(has(said, m) for m in e["must"]) / len(e["must"])
    if e.get("danger"):
        ok = 0
        for name, hp in e["danger"]:
            if not has(said, name):
                continue
            if hp == 0:
                ok += any(has(said, w) for w in ("knocked out", "ko", "ko'd", "down", "0", "zero",
                                                    "fallen", "out"))
            else:
                ok += has(said, str(hp))
        out["danger"] = ok / len(e["danger"])
    if e.get("partial"):
        p = words(e["partial"])
        out["partial_read"] = len(p) >= 2 and " ".join(p) in " ".join(words(said)) and \
            not (e.get("read") and " ".join(words(e["read"])).startswith(" ".join(p)) and
                 out.get("read", 0) == 1)
    allowed = " ".join([*e.get("shown", []), *e.get("must", []), e.get("read", ""),
                        e.get("true_place", "")]).lower()
    made_up = [n for n in CT_NAMES if has(said, n) and norm(n) not in norm(allowed)
               and not any(norm(n) in norm(a) for a in e.get("shown", []))]
    # A longer name already counted covers its parts ("Leene" in "Leene Square").
    made_up = [n for n in made_up if not any(n != m and n in m for m in made_up)]
    out["made_up"] = made_up
    tp = e.get("true_place")
    out["unshown_true"] = bool(tp) and has(said, tp) and tp not in e.get("shown", [])
    return out


def settings() -> Settings:
    s = Settings.load()
    s.brain = "meta"
    return s


async def narrate_model(narr: Narrator, frame: Frame) -> tuple[str, float | None]:
    t0 = time.monotonic()
    first, text = None, ""
    async for d in narr.narrate(frame):
        if first is None and d.strip():
            first = time.monotonic() - t0
        text += d
    return text.strip(), first


async def run(args) -> None:
    gt = json.load(open(args.gt))
    if args.ids:
        keep = set(args.ids.split(","))
        gt = [e for e in gt if e["id"] in keep]
    if args.kinds:
        kinds = set(args.kinds.split(","))
        gt = [e for e in gt if e["kind"] in kinds]
    s = settings()
    muse = Muse(s)
    sem = asyncio.Semaphore(args.par)
    out = open(args.out, "a")

    async def one(e):
        async with sem:
            img = Image.open(e["path"]).convert("RGB")
            if args.full and img.size != (2560, 1440):
                img = img.resize((2560, 1440), Image.Resampling.LANCZOS)
            frame = Frame(img, time.monotonic(), e["id"])
            narr = Narrator(muse, s.verbosity, "Chrono Trigger")
            before = (muse.usage.prompt_tokens, muse.usage.cached_tokens,
                      muse.usage.completion_tokens)
            t0 = time.monotonic()
            try:
                if args.mode == "fast":
                    text, first = await narrate_model(narr, frame)  # replaced by the fast path
                else:
                    text, first = await narrate_model(narr, frame)
                err = ""
            except Exception as exc:  # noqa: BLE001
                text, first, err = "", None, repr(exc)[:200]
            total = time.monotonic() - t0
            u = (muse.usage.prompt_tokens - before[0], muse.usage.cached_tokens - before[1],
                 muse.usage.completion_tokens - before[2])
            rec = {"id": e["id"], "kind": e["kind"], "mode": args.mode, "said": text,
                   "first_s": None if first is None else round(first, 2),
                   "total_s": round(total, 2), "tokens": u, "error": err}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{e['id']:<22} {e['kind']:<15} {rec['first_s']!s:>5} {text[:110]!r}",
                  flush=True)

    await asyncio.gather(*(one(e) for e in gt))
    print(f"cost ${muse.usage.cost_usd():.3f} for {muse.usage.calls} calls")


def summarize(args) -> None:
    gt = {e["id"]: e for e in json.load(open(args.gt))}
    recs = {}
    for line in open(args.run):
        r = json.loads(line)
        recs[r["id"]] = r  # the last run of a frame counts
    rows = []
    for k, r in recs.items():
        if k not in gt:
            continue
        sc = score(gt[k], r["said"])
        rows.append((gt[k], r, sc))
    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")
    def show(label, rs):
        reads = [sc["read"] for _e, _r, sc in rs if "read" in sc]
        exact = [sc["read"] == 1 for _e, _r, sc in rs if "read" in sc]
        must = [sc["must"] for _e, _r, sc in rs if "must" in sc]
        dz = [sc["danger"] for _e, _r, sc in rs if "danger" in sc]
        part = [sc["partial_read"] for _e, _r, sc in rs if "partial_read" in sc]
        made = [bool(sc["made_up"]) for _e, _r, sc in rs]
        firsts = sorted(r["first_s"] for _e, r, _sc in rs if r.get("first_s") is not None)
        med = firsts[len(firsts) // 2] if firsts else float("nan")
        p90 = firsts[int(0.9 * (len(firsts) - 1))] if firsts else float("nan")
        silent = sum(not r["said"] for _e, r, _sc in rs)
        print(f"{label:<15} n={len(rs):>3} read={mean(reads):.2f} exact={sum(exact)}/{len(exact)}"
              f" must={mean(must):.2f} danger={mean(dz):.2f} partial_read={sum(part)}/{len(part)}"
              f" made_up={sum(made)}/{len(made)} words={mean([sc['words'] for *_x, sc in rs]):.1f}"
              f" silent={silent} first_med={med:.2f} p90={p90:.2f}")
    kinds = sorted({e["kind"] for e, _r, _s in rows})
    for k in kinds:
        show(k, [x for x in rows if x[0]["kind"] == k])
    show("ALL", rows)
    toks = [r.get("tokens") or (0, 0, 0) for _e, r, _s in rows]
    fresh = sum(t[0] - t[1] for t in toks)
    cached = sum(t[1] for t in toks)
    comp = sum(t[2] for t in toks)
    cost = (fresh * 1.25 + cached * 0.15 + comp * 4.25) / 1e6
    print(f"tokens: prompt {fresh + cached} (cached {cached}), completion {comp}; "
          f"cost ${cost:.3f}, ${cost / max(1, len(rows)) * 1000:.2f} per 1000 screens")
    if args.v:
        for e, r, sc in rows:
            bad = (sc.get("read", 1) < 1 or sc.get("must", 1) < 1 or sc.get("danger", 1) < 1
                   or sc.get("partial_read") or sc["made_up"])
            if bad:
                print(f"- {e['id']} [{e['kind']}] read={sc.get('read')} must={sc.get('must')} "
                      f"danger={sc.get('danger')} partial={sc.get('partial_read')} "
                      f"made_up={sc['made_up']}\n    GT read={e.get('read')!r} must={e.get('must')}"
                      f" danger={e.get('danger')}\n    said={r['said']!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mode", default="model")
    r.add_argument("--ids", default="")
    r.add_argument("--kinds", default="")
    r.add_argument("--par", type=int, default=4)
    r.add_argument("--full", action="store_true", help="scale small frames up to 2560x1440")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--gt", required=True)
    s.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    else:
        summarize(a)


if __name__ == "__main__":
    main()
