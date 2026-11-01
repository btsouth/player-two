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

from playertwo.config import Settings
from playertwo.muse import Muse
from playertwo.narrator import Narrator
from playertwo.screen import Frame

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
    if "|" in phrase:
        return any(has(text, p) for p in phrase.split("|"))
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
    # Names as names: capitalised ("a market" is no place name).
    made_up = [n for n in CT_NAMES if has(said, n) and n in said and norm(n) not in norm(allowed)
               and not any(norm(n) in norm(a) for a in e.get("shown", []))]
    # A longer name already counted covers its parts ("Leene" in "Leene Square").
    made_up = [n for n in made_up if not any(n != m and n in m for m in made_up)]
    out["made_up"] = made_up
    tp = e.get("true_place")
    out["unshown_true"] = bool(tp) and has(said, tp) and tp not in e.get("shown", [])
    if e["kind"] == "field" and tp:
        out["place"] = has(said, tp)
    return out


_ROOMS: list = []


def place_of(image) -> str:
    """The surveyed place on screen, as Session._place names it (no memory of the last)."""
    from playertwo.rooms import Rooms

    if not _ROOMS:
        _ROOMS.append(Rooms.for_game("Chrono Trigger"))
    rooms = _ROOMS[0]
    here = rooms.locate(image) if rooms else None
    if here is None or here.loc in rooms.worlds:
        return ""
    name = rooms.name(here.loc)
    if name.startswith("place 0x"):
        return ""
    return name.split(",")[0] if len(name.split()) > 4 else name


# The party in menu order, as the screens before these showed it (the reader remembers it).
PARTY = ["Crono", "Frog", "Marle", "Ayla", "Lucca", "Robo"]


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
    gt = json.loads(Path(args.gt).read_text())
    if args.ids:
        keep = set(args.ids.split(","))
        gt = [e for e in gt if e["id"] in keep]
    if args.kinds:
        kinds = set(args.kinds.split(","))
        gt = [e for e in gt if e["kind"] in kinds]
    s = settings()
    muse = Muse(s)
    sem = asyncio.Semaphore(args.par)
    out = Path(args.out).open("a")  # noqa: SIM115, ASYNC230  appended to as frames finish

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
                    from playertwo import ctscreen

                    lk = await asyncio.to_thread(ctscreen.look, frame.image, PARTY)
                    if not ctscreen.sharp(frame.image):
                        lk = ctscreen.Look()  # too small to read: the model's, as live
                    reader = ctscreen.Reader()
                    if lk.readable:
                        text = reader.update(lk, first=True)
                        first = time.monotonic() - t0
                    if lk.readable and reader.model_read:
                        # As the session does: a whole box that reads badly goes to the model.
                        more, f2 = await narrate_model(narr, frame)
                        first = first if text else (f2 or 0) + (time.monotonic() - t0 - (f2 or 0))
                        text = f"{text} {more}".strip()
                    elif not lk.readable:
                        # As the session does: a surveyed place's name first, then the model.
                        place = await asyncio.to_thread(place_of, frame.image)
                        text, first = await narrate_model(narr, frame)
                        text = f"{place}. {text}".strip() if place else text
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
    gt = {e["id"]: e for e in json.loads(Path(args.gt).read_text())}
    recs = {}
    for line in Path(args.run).read_text().splitlines():
        r = json.loads(line)
        recs[r["id"]] = r  # the last run of a frame counts
    rows = []
    for k, r in recs.items():
        if k not in gt:
            continue
        if args.res and (Image.open(gt[k]["path"]).width >= 1600) != (args.res == "full"):
            continue  # full: frames as the app sees them; small: 960 previews
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
        places = [sc["place"] for _e, _r, sc in rs if "place" in sc]
        print(f"{label:<15} n={len(rs):>3} read={mean(reads):.2f} exact={sum(exact)}/{len(exact)}"
              f" must={mean(must):.2f} danger={mean(dz):.2f} partial_read={sum(part)}/{len(part)}"
              f" made_up={sum(made)}/{len(made)} words={mean([sc['words'] for *_x, sc in rs]):.1f}"
              f" silent={silent} first_med={med:.2f} p90={p90:.2f}"
              + (f" place_named={sum(places)}/{len(places)}" if places else ""))
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


QUESTIONS = [
    ("134639-00", "what's my HP", ["780", "1", "430", "Frog"]),
    ("164538-03", "what's my HP", ["780", "Frog", "Marle", "1"]),
    ("135105-01", "how's everyone doing", ["50", "188", "Frog"]),
    ("134832-08", "who's in my party", ["Crono", "Frog", "Marle"]),
    ("140314-00", "what are we fighting", ["Gargoyle|Gargoyles"]),
    ("164519-00", "what are we fighting", ["Gargoyle|Gargoyles"]),
    ("134854-00", "what items do I have", ["Potion", "Mid-Potion", "Hi-Potion", "Mid-Ether",
                                           "Panacea", "Athenian Water"]),
    ("135428-00", "how's everyone doing", ["227", "241", "188"]),
    ("135629-31", "who's in my party", ["Crono", "Frog", "Marle"]),
    ("135445-06", "what items do I have", ["Potion", "Hi-Potion", "Panacea", "Shelter",
                                           "Elixir", "Athenian Water", "Lapis", "Ether"]),
    ("135445-06", "what does that say", ["Fully restores HP/MP at a save point."]),
    ("134639-00", "what does it say", ["Combo Attack: Reduces HP to 1"]),
    ("s3", "what does that say", ["Resume bookmarked game?", "No"]),
    ("q4", "what does that say", ["Exit?"]),
    ("135212-14", "what does that say", ["Crono's level increased!"]),
    ("164526-01", "what does that say", ["Attack all enemies with water."]),
    ("140421-06", "what does that say", ["Restore moderate HP to all allies."]),
    ("a6", "what does that say", ["Leene Square"]),
    ("140406-00", "what's my HP", ["Crono", "811", "347", "430"]),
    ("175358-00", "what's Frog's HP", ["Frog", "494"]),
    ("h-menu-summit", "how's everyone doing", ["811", "455", "199"]),
    ("h-004556-00", "what's that person saying",
     ["Frog", "So this is the prison that holds the Guru of Life"]),
]


async def ask(args) -> None:
    """Questions on frames: the agent (model, tools) as before, or the screen first."""
    from playertwo import agent as agent_mod
    from playertwo import ctscreen
    from playertwo.controls import DryRun

    gt = {e["id"]: e for e in json.loads(Path(args.gt).read_text())}
    s = settings()
    muse = Muse(s)

    async def no_pause(self, st):  # the frames don't move: count it as paused at once
        return st

    agent_mod.Agent._hold_fight = no_pause
    out = Path(args.out).open("a")  # noqa: SIM115, ASYNC230  appended to as answers come
    for fid, q, must in QUESTIONS:
        e = gt[fid]
        img = Image.open(e["path"]).convert("RGB")
        frame = Frame(img, time.monotonic(), fid)
        said: list[tuple[float, str]] = []
        t0 = time.monotonic()
        via = "agent"
        line = None
        if args.mode == "fast" and ctscreen.sharp(img):
            lk = await asyncio.to_thread(ctscreen.look, img, PARTY)
            line = ctscreen.answer(q, lk, PARTY)
            if line:
                said.append((time.monotonic() - t0, line))
                via = "screen"
        if not line:
            async def speak(text, said=said, t0=t0):
                said.append((time.monotonic() - t0, text))

            async def grab(frame=frame):
                return frame

            ag = agent_mod.Agent(muse, DryRun(), grab, speak, game="Chrono Trigger",
                                 max_turns=4)
            before = muse.usage.cost_usd()
            try:
                await ag.run(q, frame)
            except Exception as exc:  # noqa: BLE001
                said.append((time.monotonic() - t0, f"ERROR {exc!r}"[:200]))
            cost = muse.usage.cost_usd() - before
        else:
            cost = 0.0
        text = " ".join(t for _t, t in said)
        ok = sum(has(text, m) for m in must) / len(must)
        rec = {"id": fid, "q": q, "mode": args.mode, "via": via, "said": text,
               "first_s": round(said[0][0], 2) if said else None,
               "total_s": round(time.monotonic() - t0, 2), "must": ok, "cost": round(cost, 4)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{fid:<10} {q:<24} {via:<6} {rec['first_s']!s:>5}s must={ok:.2f} {text[:120]!r}",
              flush=True)
    print(f"cost ${muse.usage.cost_usd():.3f}")


def ask_summary(path: str) -> None:
    recs = [json.loads(x) for x in Path(path).read_text().splitlines()]
    for r in recs:  # re-scored, so a fixed label counts
        must = next(m for f, q, m in QUESTIONS if f == r["id"] and q == r["q"])
        r["must"] = sum(has(r["said"], m) for m in must) / len(must)
    firsts = sorted(r["first_s"] for r in recs if r["first_s"] is not None)
    print(f"n={len(recs)} must={sum(r['must'] for r in recs) / len(recs):.2f} "
          f"all_right={sum(r['must'] == 1 for r in recs)}/{len(recs)} "
          f"first_med={firsts[len(firsts) // 2]:.2f}s max={firsts[-1]:.2f}s "
          f"screen={sum(r['via'] == 'screen' for r in recs)} cost=${sum(r['cost'] for r in recs):.3f}"
          f" words={sum(len(words(r['said'])) for r in recs) / len(recs):.1f}")


def seq_frames(folder: str, start: str, end: str) -> list[tuple[float, Path]]:
    """Frames named HHMMSS-n.jpg, timed by their names (several in a second spread evenly)."""
    by_sec: dict[str, list[Path]] = {}
    for p in sorted(Path(folder).glob("*.jpg")):
        hms = p.name[:6]
        if start <= hms <= end:
            by_sec.setdefault(hms, []).append(p)
    out = []
    for hms, ps in sorted(by_sec.items()):
        base = int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])
        out += [(base + i / len(ps), p) for i, p in enumerate(ps)]
    t0 = out[0][0] if out else 0
    return [(t - t0, p) for t, p in out]


def repeats(lines: list[tuple[float, str]], within: float = 30.0) -> int:
    """Lines that say again what was said in the last `within` seconds."""
    n = 0
    for i, (t, line) in enumerate(lines):
        w = set(words(line))
        for t2, old in lines[:i]:
            o = set(words(old))
            if t - t2 <= within and w and o and len(w & o) / len(w | o) >= 0.75:
                n += 1
                break
    return n


async def seq(args) -> None:
    """A recorded stretch of play, narrated as the session would: the watcher and the
    model (old), or the reader with the model for what it can't read (new)."""
    from playertwo import ctscreen
    from playertwo.screen import ChangeWatcher

    frames = seq_frames(args.folder, args.start, args.end)
    s = settings()
    muse = Muse(s)
    narr = Narrator(muse, s.verbosity, "Chrono Trigger")
    watcher = ChangeWatcher()
    reader = ctscreen.Reader()
    reader.party = list(PARTY)
    said: list[tuple[float, str, str]] = []
    busy_until, quiet_until, streak = -1.0, -1.0, 0
    pending = None
    last_look, owned, owned_t = -9.0, False, -9.0
    for t, path in frames:
        img = Image.open(path).convert("RGB")
        frame = Frame(img, t, path.name)
        if args.mode == "fast" and t - last_look >= 0.3:
            last_look = t
            lk = await asyncio.to_thread(ctscreen.look, img, reader.party)
            owned, owned_t = lk.readable, t
            line = reader.update(lk, now=t)
            if line:
                said.append((t + lk.ms / 1000, line, "reader"))
                busy_until = max(busy_until, t) + len(words(line)) / 2.7
        if watcher.update(frame):
            pending = frame
        if pending is None or t < busy_until or t < quiet_until:
            continue
        if args.mode == "fast" and owned and owned_t >= pending.t - 0.3:
            pending = None
            continue
        f, pending = pending, None
        text, first = await narrate_model(narr, f)
        lat = narr.log[-1].latency_s if narr.log else 0.0
        if text:
            said.append((t + (first or lat), text, "model"))
            busy_until = t + lat + len(words(text)) / 2.7
            streak = 0
        else:
            streak += 1
            quiet_until = t + lat + min(16.0, 2.0 ** streak)
            busy_until = t + lat
    span = frames[-1][0] if frames else 0
    rep = repeats([(t, x) for t, x, _w in said])
    nw = sum(len(words(x)) for _t, x, _w in said)
    print(f"{args.mode}: {len(frames)} frames over {span:.0f} s; {len(said)} lines "
          f"({sum(w == 'reader' for *_x, w in said)} by the reader), {nw} words "
          f"({nw / max(span, 1) * 60:.0f} a minute), repeats {rep}, "
          f"cost ${muse.usage.cost_usd():.3f}")
    Path(args.out).write_text("".join(json.dumps({"t": round(t, 2), "who": who, "said": x}) + "\n"
                                      for t, x, who in said))


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
    k = sub.add_parser("ask")
    k.add_argument("--gt", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--mode", default="model")
    q = sub.add_parser("seq")
    q.add_argument("folder")
    q.add_argument("--start", default="000000")
    q.add_argument("--end", default="999999")
    q.add_argument("--out", required=True)
    q.add_argument("--mode", default="model")
    k2 = sub.add_parser("asked")
    k2.add_argument("run")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--gt", required=True)
    s.add_argument("-v", action="store_true")
    s.add_argument("--res", default="", choices=["", "full", "small"])
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    elif a.cmd == "ask":
        asyncio.run(ask(a))
    elif a.cmd == "seq":
        asyncio.run(seq(a))
    elif a.cmd == "asked":
        ask_summary(a.run)
    else:
        summarize(a)


if __name__ == "__main__":
    main()
