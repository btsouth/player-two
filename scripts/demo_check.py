#!/usr/bin/env python3
"""A one-command rehearsal of the Chrono Trigger takes, held to the bar for the video.

    # Live: a running Player Two (with --log) and the game on DP-2
    python3 scripts/demo_check.py run --take A --repeats 3
    python3 scripts/demo_check.py run --take B --repeats 3 --skip B3

    # A recorded log, with the take's video if there is one (frames for the HP and gold checks)
    python3 scripts/demo_check.py replay events.jsonl --video take-A-raw.mp4 --video take-B-raw.mp4

Each line of the run sheet goes through `player-two ctl do`, in order, with a reload between
runs. The event log then says, per line: pass or fail, first-word time, longest silence while
the request ran, total time and everything spoken. A line fails on:

- first words after 2 s;
- a silence over 3 s, unless the line names game animation as its reason (listed, and held to
  a cap);
- a total over the line's budget ("predicted": SPEED's predictions in speed-profile.md for this
  build; "sheet": the run sheet's measured range, for logs from before those changes). Total
  means the same as the run sheet's times: until its result starts being said;
- an internal or off-topic phrase, or a line said twice;
- a spoken HP or gold figure that doesn't match the reader's reading of the frame at that moment,
  where the reader can tell, or that the log itself contradicts while it's being said;
- not saying what the line is there to show, or something the run sheet has seen it get wrong.

Each run gets its own folder with frames and report.txt. The final table goes to check-report.md,
and a live run writes check-rates.txt for `make_video.py --rates`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Self

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from demo_captions import _heard_text, _words

NIGHT = Path.home() / ".cache/player-two/night-2026-11-01"
SAVES = Path("/data/SteamLibrary/steamapps/compatdata/613830/pfx/drive_c/users/steamuser/"
             "Documents/My Games/Chrono Trigger")
LAUNCHER = Path.home() / ".cache/player-two/live/tools/ct_launch.py"
FIRST_WORD_S = 2.0
SILENCE_S = 3.0
WATCH_AFTER_S = 12.0  # what the reader and narrator say next still counts
CHARS_PER_S = 14.0  # Player Two's voice, for logs without "spoken" events
# Lines that answer the request. Reader and narrator lines count as speech, never as the answer.
OWN = {"p2", "agent", "screen", "describe", "atmosphere", "answer", "ask"}
STREAMED = {"describe", "atmosphere", "narrator"}  # logged when written, not when begun


@dataclass(frozen=True)
class Allow:
    """A silence with a stated reason: up to cap seconds, on evidence if `needs` says."""
    reason: str
    cap: float
    needs: str = ""  # "walk": the log must show the walker moving during the silence


@dataclass(frozen=True)
class Line:
    id: str
    take: str
    request: str
    match: str
    budget: float  # SPEED's predicted worst case, with a little room
    sheet: float  # the run sheet's measured worst case
    must: str = ""
    must_not: str = ""
    allow: tuple[Allow, ...] = ()
    ready: str = ""  # wait for this one's command window ("any": anyone's)
    repeats_ok: bool = False  # a long fight says "Frog: Heal." more than once
    optional: bool = False  # in the run sheet, but off the default order


LINES = (
    Line("A1", "A", "Where am I?", r"^where am i\b", 2.5, 6.0, must=r"leene square"),
    Line("A2", "A", "What's on the screen?", r"^what'?s on the screen", 32.0, 32.0,
         must=r"castle|tent|fair|town|sea", must_not=r"\byou are (marle|crono)\b"),
    Line("A3", "A", "Take me to the fair.", r"^take me to the fair", 4.5, 4.5,
         must=r"leene square"),
    Line("A3b", "A", "What does it look like?", r"^what does it look like", 20.0, 20.0,
         optional=True),
    Line("A4", "A", "Buy three potions.", r"^buy three potions", 16.0, 27.0,
         must=r"potion", allow=(Allow("the walk to the seller", 6.0, "walk"),)),
    Line("A5", "A", "Take me back outside.", r"^take me back outside", 16.5, 16.5,
         must=r"outside|world map", allow=(Allow("the walk out", 14.5, "walk"),)),
    Line("B1", "B", "What are we fighting?", r"^what are we fighting", 1.5, 3.5,
         must=r"\bimps?\b|gargoyle", must_not=r"reptite"),
    Line("B2", "B", "Crono, use Cyclone.", r"^crono,? use cyclone", 10.0, 12.5, must=r"cyclone",
         must_not=r"can'?t use|isn'?t ready", allow=(Allow("Cyclone's animation", 4.5),),
         ready="Crono"),
    Line("B3", "B", "Attack the one on the left.", r"^attack the one on the left", 4.5, 9.5,
         must=r"\bleft\b|attack|\bhit\b", ready="any"),
    Line("B4", "B", "Win this fight for me.", r"^win this fight", 58.0, 76.0,
         must=r"\bwon\b|earn|experience|\bexp\b",
         allow=(Allow("fight animations and turns", 5.0),), ready="any", repeats_ok=True),
)
SLOT = {"A": 4, "B": 1}
SAVE_FILE = {4: "save_20.bin", 1: "save_00.bin"}
BATTLE_SETUP = "set the battle mode to wait"

# Internal words that must never reach the player's ears.
INTERNAL = re.compile(
    r"tool calls?\b|\bthe model\b|train of thought|\bfinish:|\bjson\b|\bapi\b|image file|"
    r"truncated|traceback|\bas an ai\b|\bthe prompt\b|\bscreenshot\b|can'?t reach|"
    r"back online|exclamation mark", re.IGNORECASE)
CT_WORDS = """Crono Marle Frog Lucca Robo Ayla Magus Leene Square Truce Guardia Melchior Cyclone
Slurp Cure Heal Aura Spincut Hi-Potion Tonic Panacea Athenian Water Shelter imp imps gargoyle
gargoyles golden winged purple tent tents stall stalls fountain fairground combo EXP"""
NAMES = "Crono|Marle|Lucca|Frog|Robo|Ayla|Magus"
PROPER = rf"\b(?:{NAMES}|Leene|Truce|Guardia|Melchior|Cyclone)\b"


# ---------------------------------------------------------------- the log


def read_log(path: Path) -> list[dict]:
    out = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        try:
            e = json.loads(raw)
        except ValueError:
            continue  # a line still being written
        if isinstance(e, dict) and "t" in e and "kind" in e:
            out.append(e)
    return out


def merged(spans: list[tuple[float, float]], join: float = 0.25) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= join:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def status_spans(events: list[dict]) -> list[tuple[float, float]]:
    """When the voice's status said it was speaking (logs before "spoken" events)."""
    spans, cur = [], None
    for e in events:
        if e["kind"] != "status":
            continue
        if e.get("state") == "speaking" and cur is None:
            cur = e["t"]
        elif e.get("state") != "speaking" and cur is not None:
            spans.append((cur, e["t"]))
            cur = None
    if cur is not None:
        spans.append((cur, events[-1]["t"]))
    return merged(spans)


@dataclass
class Said:
    t: float  # logged
    source: str
    text: str
    began: float | None = None
    ended: float | None = None


def said_lines(events: list[dict], t0: float, end: float,
               status: list[tuple[float, float]]) -> list[Said]:
    """Everything said in a window, with when it was heard: from the voice's own record, or
    worked out from its status in older logs."""
    lines = [Said(e["t"], str(e.get("source") or e["kind"]), str(e["text"]))
             for e in events
             if e["kind"] in ("say", "ask") and e.get("text") and t0 <= e["t"] < end]
    played = sorted((e for e in events if e["kind"] == "spoken" and e.get("text")
                     and t0 - 0.05 <= e["began"] < end), key=lambda e: e["began"])
    if not played and any(e["kind"] == "spoken" for e in events
                          if t0 - 60 <= e["t"] <= end + 60):
        return lines  # the voice's record exists and nothing played in this window
    if played:
        for e in played:
            w = _words(_heard_text(e["text"]))
            # Pieces of a say were queued no later than they began (a streamed one is logged
            # with its first sentence).
            owners = [s for s in lines if w <= _words(_heard_text(s.text))]
            owner = min(owners, key=lambda s: abs(s.t - e["began"]), default=None)
            if owner is None:
                owner = Said(e["began"], "?", _heard_text(e["text"]))
                lines.append(owner)
            owner.began = e["began"] if owner.began is None else min(owner.began, e["began"])
            owner.ended = e["ended"] if owner.ended is None else max(owner.ended, e["ended"])
        return sorted(lines, key=lambda s: s.began if s.began is not None else s.t)
    cur = t0
    for s in sorted(lines, key=lambda s: s.t):
        dur = max(0.6, len(s.text) / CHARS_PER_S)
        ready = s.t - dur if s.source in STREAMED else s.t
        ready = max(ready, t0)
        span = next((p for p in status if p[1] > ready), None)
        began = ready if span is None or span[0] <= ready else span[0]
        s.began = max(cur, began)
        s.ended = s.began + dur
        cur = s.ended
    return sorted(lines, key=lambda s: s.began or s.t)


# ---------------------------------------------------------------- what a line claims


@dataclass(frozen=True)
class Claim:
    kind: str  # all_full, full, hp, gold_left, gold_spent
    name: str = ""
    value: int = 0
    most: int | None = None
    at: float = 0.0  # how far through the line it's said, 0 to 1

    def __str__(self) -> str:
        return " ".join(str(x) for x in (self.kind, self.name, self.value or "", self.most or "")
                        if x != "").strip()


def claims(text: str) -> list[Claim]:
    """The HP and gold a line states."""
    t = _heard_text(text)
    n = max(1, len(t))
    out: list[Claim] = []
    seen: set[tuple[str, int]] = set()

    def hp(m: re.Match, name: str, value: str, most: str | None = None) -> None:
        if (name, int(value)) not in seen:
            seen.add((name, int(value)))
            out.append(Claim("hp", name, int(value), int(most) if most else None, m.start() / n))

    m = re.search(r"\b(?:everyone|everybody)(?:'s| is| are)?(?:\s+\w+){0,3}?\s+full\b", t, re.IGNORECASE)
    if m:
        out.append(Claim("all_full", at=m.start() / n))
    for m in re.finditer(rf"\b({NAMES})(?:'s| is)?(?: at|:|,)? (\d+) of (\d+)", t):
        hp(m, *m.groups())
    for m in re.finditer(rf"\b({NAMES})(?:'s)?(?: is| are)?(?: still)?(?: at| down to| dropped "
                         rf"to| to)? (\d+) ?HP\b", t):
        hp(m, *m.groups())
    for m in re.finditer(rf"\b((?:{NAMES})(?:(?:, | and |, and )(?:{NAMES}))+) (?:is |are )?"
                         rf"(?:still )?(?:at |dropped to |down to )(\d+) ?HP\b", t):
        for name in re.findall(NAMES, m.group(1)):
            hp(m, name, m.group(2))
    for m in re.finditer(rf"\b({NAMES}) took \d+, down to (\d+)", t):
        hp(m, *m.groups())
    for m in re.finditer(rf"\b({NAMES})(?:'s| is)(?: at| still)? full\b", t):
        out.append(Claim("full", m.group(1), at=m.start() / n))
    m = re.search(r"\byou have (\d[\d,]*)(?: gold| g\b)?|\b(\d[\d,]*) (?:gold|g) left\b", t, re.IGNORECASE)
    if m:
        out.append(Claim("gold_left", value=int((m.group(1) or m.group(2)).replace(",", "")),
                         at=m.start() / n))
    m = re.search(r"\bfor (\d[\d,]*) gold\b|, (\d[\d,]*) gold\.", t, re.IGNORECASE)
    if m:
        out.append(Claim("gold_spent", value=int((m.group(1) or m.group(2)).replace(",", "")),
                         at=m.start() / n))
    return out


class VideoFrames:
    """Frames of a recording at given moments, read by the game's own readers."""

    def __init__(self, video: Path, start: float, out: Path):
        self.video, self.start, self.out = Path(video), start, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, Path | None] = {}

    def at(self, t: float) -> Path | None:
        name = f"{max(0.0, t - self.start):08.2f}"
        if name not in self.cache:
            p = self.out / f"{name}.jpg"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", name, "-i", str(self.video),
                            "-frames:v", "1", "-q:v", "3", str(p)], capture_output=True, check=False)
            self.cache[name] = p if p.exists() else None
        return self.cache[name]


class ShotFrames:
    """Frames taken during a run with `ctl shot`: the nearest one to a moment."""

    def __init__(self, shots: list[tuple[float, Path]]):
        self.shots = shots

    def at(self, t: float) -> Path | None:
        near = min(self.shots, key=lambda s: abs(s[0] - t), default=None)
        return near[1] if near and abs(near[0] - t) < 1.0 else None


def reading(path: Path | None) -> dict:
    """{"hp": {name: (hp, most)}, "funds": n}: what the reader can tell from a frame."""
    if path is None:
        return {}
    import cv2

    from playertwo import ctsteam, shop

    img = cv2.imread(str(path))
    if img is None or not ctsteam.fits(img):
        return {}
    out: dict = {}
    st = ctsteam.read(img, full=False)
    hp = {m.name: (m.hp, m.hp_max) for m in st.party if m.hp is not None}
    if hp:
        out["hp"] = hp
    else:  # no fight on screen: a shop may be
        hands, lines = ctsteam.menu_read(img)
        funds = shop.parse([h.tip for h in hands], lines).funds
        if funds is not None:
            out["funds"] = funds
    return out


def check_figures(s: Said, later: list[Said], frames) -> tuple[list[str], list[str]]:
    """Fails and notes for the figures in one line. A "full" claim must hold for as long as it's
    being said. A number must match around the moment its own words are said (a long line's
    second sentence comes seconds after the first, and HP can change in between)."""
    fails: list[str] = []
    notes: list[str] = []
    said = claims(s.text)
    if not said or s.began is None:
        return fails, notes
    began, ended = s.began, s.ended or s.began + len(s.text) / CHARS_PER_S
    fulls = [c for c in said if c.kind in ("all_full", "full")]
    for o in later if fulls else []:
        hurt = re.search(rf"\b({NAMES}) took \d+, down to (\d+)", o.text)
        when = o.began if o.began is not None else o.t
        if hurt and began - 0.2 <= when <= ended + 0.5 and any(
                c.kind == "all_full" or c.name == hurt.group(1) for c in fulls):
            fails.append(f"stale: said \"{s.text[:60]}\" while \"{o.text[:40]}\" was being said "
                         f"(+{when - began:.1f} s)")
    if frames is None:
        notes.append("no frames, figures unchecked: " + "; ".join(map(str, said)))
        return fails, notes
    for c in said:
        moment = began + (ended - began) * c.at
        if c.kind in ("all_full", "full"):
            reads = [(t, reading(frames.at(t))) for t in (moment, (moment + ended) / 2, ended)]
            for t, r in reads:
                low = [f"{n} {a}/{b}" for n, (a, b) in r.get("hp", {}).items()
                       if a is not None and b and a < b
                       and (c.kind == "all_full" or n == c.name)]
                if low:
                    fails.append(f"stale: said \"{s.text[:60]}\" but the frame at +{t - began:.1f} "
                                 f"s into it shows {', '.join(low)}")
                    break
            else:
                if not any(r.get("hp") for _t, r in reads):
                    notes.append(f"reader can't tell: {c}")
        elif c.kind == "hp":
            reads = [reading(frames.at(t)) for t in (moment - 0.4, moment, moment + 0.4)]
            seen = [r["hp"][c.name] for r in reads if c.name in r.get("hp", {})]
            if not seen:
                notes.append(f"reader can't tell: {c}")
            elif not any(v[0] == c.value and (c.most is None or v[1] == c.most) for v in seen):
                fails.append(f"figure: said {c.name} {c.value} at +{moment - began:.1f} s into "
                             f"\"{s.text[:50]}\" but the frame shows "
                             f"{c.name} {seen[0][0]}/{seen[0][1]}")
        else:
            fails_, notes_ = _check_gold(c, began, frames)
            fails += fails_
            notes += notes_
    return fails, notes


def _check_gold(c: Claim, began: float, frames) -> tuple[list[str], list[str]]:
    """The gold on the shop's last screen before the line, looking back up to 12 s."""
    funds: list[int] = []
    t = began + 0.3
    while t > began - 12 and len(funds) < 2:
        n = reading(frames.at(t)).get("funds")
        if n is not None and n not in funds:
            funds.append(n)
        t -= 0.75
    if not funds:
        return [], [f"reader can't tell: {c}"]
    if c.kind == "gold_left":
        if funds[0] != c.value:
            return [f"figure: said {c.value} gold left but the shop's frame shows {funds[0]} G"], []
    elif len(funds) > 1 and funds[1] - funds[0] != c.value:
        return [(f"figure: said {c.value} gold spent but the shop's frames show "
                 f"{funds[1]} G to {funds[0]} G")], []
    return [], []


# ---------------------------------------------------------------- judging one line


@dataclass
class Result:
    line: Line
    run: str
    heard: dict
    first: float | None = None
    silence: float = 0.0
    total: float | None = None
    heard_by: float | None = None
    budget: float = 0.0
    fails: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    said: list[Said] = field(default_factory=list)
    frames: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fails


def off_topic(text: str, request: str, results: str) -> bool:
    from playertwo import ontopic

    words = ontopic.content(text)
    if len(words) < 6 or re.search(PROPER, text):
        return False  # too short to judge, or it names something in this game
    # The same floor the app falls back on: the football line's share was 0.2.
    share, _loose = ontopic.Grounds(request, results, CT_WORDS).grounding(text)
    return share < ontopic.FALLBACK_SHARE


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _heard_text(text).lower()).strip()


def repeats(lines: list[Said], allow_gap: float | None) -> list[str]:
    """A line said twice, or nearly: the same words again. allow_gap: a long fight's own lines
    may come back after this many seconds."""
    out = []
    for i, s in enumerate(lines):
        a = _norm(s.text)
        for o in lines[:i]:
            b = _norm(o.text)
            gap = (s.began or s.t) - (o.began or o.t)
            same = a == b or (min(len(a), len(b)) >= 12
                              and SequenceMatcher(None, a, b).ratio() >= 0.9)
            if same and (allow_gap is None or gap < allow_gap):
                out.append(f"said twice ({gap:.1f} s apart): \"{s.text[:70]}\"")
                break
    return out


def judge(line: Line, heard: dict, events: list[dict], end: float, run: str = "",
          budgets: str = "predicted", frames=None) -> Result:
    """One request: heard at heard["t"], its window ending at `end` (the next request, or Player
    Two stopping). `events` may be the whole log."""
    t0 = heard["t"]
    near = [e for e in events if t0 - 60 <= e["t"] < end + 60]
    seg = [e for e in events if t0 <= e["t"] < end]
    status = status_spans([e for e in events if t0 - 120 <= e["t"] < end + 5] or near)
    r = Result(line, run, heard, budget=line.budget if budgets == "predicted" else line.sheet)
    all_said = said_lines(near, t0, end, status)
    own = [s for s in all_said if s.source in OWN and s.began is not None]
    agent = next((e for e in seg if e["kind"] == "agent"), None)
    if agent is not None and agent.get("outcome") == "cancelled":
        r.fails.append("the request was cancelled")
    if not own:
        r.fails.append("no answer")
    else:
        r.first = own[0].began - t0
        if r.first > FIRST_WORD_S:
            r.fails.append(f"first word at {r.first:.2f} s (limit {FIRST_WORD_S:.0f} s)")
    if own:
        if agent is not None:
            result = [s for s in own if s.t <= agent["t"] + 0.3] or own
            done = max(agent["t"], result[-1].began)
        else:  # a local answer (a description): done when it has been said
            result = own
            done = max(s.ended or s.began for s in own)
            if not any(e["kind"] == "spoken" for e in near):
                # An old log: its length is a guess, the voice's status isn't.
                last = own[-1].began
                done = next((b for a, b in status if a <= last + 0.05 < b), done)
        r.total = done - t0
        r.heard_by = max(max(s.ended or s.began for s in result) - t0, r.total)
        if agent is None:
            r.heard_by = r.total
        if r.total > r.budget:
            r.fails.append(f"total {r.total:.1f} s (budget {r.budget:.1f} s)")
    else:
        done = agent["t"] if agent is not None else t0

    # Silence while it ran: any voice counts, the reader's too.
    played = [(e["began"], e["ended"]) for e in near if e["kind"] == "spoken"]
    spans = merged(played) if played else status
    cur = t0
    gaps: list[tuple[float, float]] = []
    for a, b in spans:
        if b <= t0:
            continue
        if a >= done:
            break
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if done > cur:
        gaps.append((cur, done))
    r.silence = max((b - a for a, b in gaps), default=0.0)
    for a, b in gaps:
        if b - a <= SILENCE_S:
            continue
        where = f"{b - a:.1f} s of silence at +{a - t0:.1f} s"
        walking = any(e["kind"] in ("walk", "walk_note") and a <= e["t"] <= b for e in seg)
        why = next((w for w in line.allow if not w.needs or walking), None)
        if why is not None and b - a <= why.cap:
            r.allowed.append(f"{where}: {why.reason} (allowed up to {why.cap:.1f} s)")
        elif why is not None:
            r.fails.append(f"{where}: over the {why.cap:.1f} s allowed for {why.reason}")
        else:
            r.fails.append(where)

    # What was said, up to a moment after it was done.
    watch = done + WATCH_AFTER_S
    r.said = [s for s in all_said if (s.began if s.began is not None else s.t) <= watch]
    results = " ".join(str(st.get("result", "")) for st in (agent or {}).get("steps", []))
    for s in r.said:
        if INTERNAL.search(s.text):
            r.fails.append(f"internal phrase: \"{s.text[:80]}\"")
        elif s.source in ("agent", "screen", "ask") and off_topic(s.text, heard["text"], results):
            r.fails.append(f"off topic: \"{s.text[:80]}\"")
    r.fails += repeats(r.said, 4.0 if line.repeats_ok else None)
    told = " ".join(s.text for s in r.said if s.source != "narrator")
    if line.must and not re.search(line.must, told, re.IGNORECASE):
        r.fails.append(f"didn't say what this line is for (/{line.must}/)")
    if line.must_not:
        m = re.search(line.must_not, told, re.IGNORECASE)
        if m:
            r.fails.append(f"said \"{m.group(0)}\": the run sheet retakes this")

    for s in r.said:
        if s.began is None or claims(s.text) == []:
            continue
        fails, notes = check_figures(s, r.said, frames)
        r.fails += fails
        r.notes += notes
    if frames is not None:  # a picture at each line, for the report
        r.frames = [p for s in r.said if s.began is not None and (p := frames.at(s.began))]
    return r


def find_line(text: str, lines: tuple[Line, ...] = LINES) -> Line | None:
    t = text.lower().strip(" .!")
    return next((ln for ln in lines if re.search(ln.match, t)), None)


def window_end(events: list[dict], t0: float, limit: float = 300.0) -> float:
    return next((e["t"] for e in events
                 if t0 < e["t"] < t0 + limit and e["kind"] in ("heard", "bye", "ready")),
                t0 + limit)


def replay(events: list[dict], takes: str = "AB", budgets: str = "sheet",
           frames_for=None, run: str = "replay") -> list[Result]:
    """Every run sheet line found in a log. frames_for(t): the frames covering a moment."""
    out = []
    seen: dict[str, int] = {}
    for e in events:
        if e["kind"] != "heard" or not e.get("text"):
            continue
        line = find_line(str(e["text"]))
        if line is None or line.take not in takes:
            continue
        seen[line.id] = seen.get(line.id, 0) + 1
        end = window_end(events, e["t"])
        out.append(judge(line, e, events, end, f"{run} #{seen[line.id]}", budgets,
                         frames_for(e["t"]) if frames_for else None))
    return out


# ---------------------------------------------------------------- reports


def _stamp(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def line_report(r: Result) -> str:
    verdict = "PASS" if r.ok else "FAIL"
    head = (f"{r.line.id} {verdict}  \"{r.heard['text']}\" at {_stamp(r.heard['t'])}  first "
            f"{_f(r.first)} s, longest silence {_f(r.silence)} s, total {_f(r.total)} s "
            f"(budget {r.budget:.1f}), all heard by {_f(r.heard_by)} s")
    rows = [head]
    rows += [f"    FAIL: {x}" for x in r.fails]
    rows += [f"    allowed: {x}" for x in r.allowed]
    rows += [f"    note: {x}" for x in r.notes]
    for s in r.said:
        at = f"+{s.began - r.heard['t']:5.1f}" if s.began is not None else "  not played"
        rows.append(f"    {at} {s.source:<10} {s.text}")
    if r.frames:
        rows.append(f"    frames: {r.frames[0].parent}")
    return "\n".join(rows)


def table(results: list[Result]) -> str:
    rows = ["| run | line | result | first s | silence s | total s | budget s | why |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        why = "; ".join(x.replace("|", "/") for x in (r.fails or r.allowed))
        rows.append(f"| {r.run} {_stamp(r.heard['t'])} | {r.line.id} {r.line.request} | "
                    f"{'pass' if r.ok else 'FAIL'} | {_f(r.first)} | {_f(r.silence)} | "
                    f"{_f(r.total)} | {r.budget:.1f} | {why} |")
    return "\n".join(rows)


def summary(results: list[Result]) -> str:
    rows = ["| line | passed | first s | longest silence s | total s |", "|---|---|---|---|---|"]
    for ln in LINES:
        rs = [r for r in results if r.line.id == ln.id]
        if not rs:
            continue

        def span(vals: list[float | None]) -> str:
            got = [v for v in vals if v is not None]
            return f"{min(got):.1f} to {max(got):.1f}" if got else "-"

        rows.append(f"| {ln.id} {ln.request} | {sum(r.ok for r in rs)}/{len(rs)} | "
                    f"{span([r.first for r in rs])} | {span([r.silence for r in rs])} | "
                    f"{span([r.total for r in rs])} |")
    return "\n".join(rows)


def write_report(results: list[Result], path: Path, title: str, extra: list[str] = ()) -> None:
    passed = sum(r.ok for r in results)
    limits = ("Limits: first word 2 s, silence 3 s (or the line's stated animation, listed), the "
              "line's budget, no internal or repeated lines, figures match the frames.")
    body = [f"# {title}", "", f"{_stamp(time.time())}: {passed}/{len(results)} lines passed. "
            + limits, "", summary(results), "", table(results), "", *extra]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n")


def write_rates(results: list[Result], path: Path, heading: str) -> None:
    """For `make_video.py --rates`: its heading, then a rate on each line. Only lines the
    video's card names."""
    rows = [heading]
    for ln in LINES:
        rs = [r for r in results if r.line.id == ln.id]
        if rs and ln.id in ("A3", "A4", "B1", "B4"):
            rows.append(f"“{ln.request.rstrip('.')}” {sum(r.ok for r in rs)}/{len(rs)}")
    path.write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------- live


class Live:
    """A running Player Two and the game: only `player-two ctl`, the log, and the launcher."""

    def __init__(self, a: argparse.Namespace):
        self.p2 = a.p2
        self.log = Path(a.log).expanduser()
        a.out = str(Path(a.out).expanduser())
        self.a = a
        self.tmp = Path(a.out) / "shot.jpg"
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        # The log is read several times a second during a measured request: only what's new,
        # and the last few MB of what came before (a session's log only grows).
        self.seen: list[dict] = []
        self.pos = max(0, self.log.stat().st_size - (4 << 20)) if self.log.exists() else 0
        self.partial = self.pos > 0

    def ctl(self, *args: str) -> str:
        return subprocess.run([self.p2, "ctl", *args], capture_output=True, text=True,
                              check=False, timeout=60).stdout.strip()

    def events(self, since: float = 0.0) -> list[dict]:
        if self.log.exists():
            with self.log.open("rb") as f:
                f.seek(self.pos)
                chunk = f.read()
            done = chunk.rfind(b"\n") + 1  # a line still being written waits for the next look
            self.pos += done
            lines = chunk[:done].decode(errors="replace").splitlines()
            if self.partial:  # began in the middle of a line
                lines, self.partial = lines[1:], False
            for raw in lines:
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(e, dict) and "t" in e and "kind" in e:
                    self.seen.append(e)
        return [e for e in self.seen if e["t"] >= since]

    def quiet(self, secs: float, most: float) -> None:
        """Until Player Two has been quiet for secs, or `most` has gone by."""
        end = time.time() + most
        while time.time() < end:
            last = next((e for e in reversed(self.events(time.time() - 120))
                         if e["kind"] == "status"), None)
            if last is None or (last.get("state") != "speaking"
                                and time.time() - last["t"] >= secs):
                return
            time.sleep(0.25)

    def state(self):
        import cv2

        from playertwo import ctsteam

        self.ctl("shot", str(self.tmp.resolve()))
        img = cv2.imread(str(self.tmp))
        return ctsteam.read(img) if img is not None and ctsteam.fits(img) else None

    def wait_ready(self, who: str, most: float = 20.0) -> bool:
        end = time.time() + most
        while time.time() < end:
            st = self.state()
            if st is not None and (who in st.ready() or (who == "any" and st.ready())):
                return True
            time.sleep(0.4)
        return False

    def send(self, text: str, most: float) -> dict | None:
        """The request, then wait until it's done and what follows it has been said."""
        t = time.time() - 0.2
        self.ctl("do", text)
        end = time.time() + most
        heard = None
        while time.time() < end:
            ev = self.events(t)
            heard = heard or next((e for e in ev if e["kind"] == "heard"
                                   and _norm(str(e.get("text"))) == _norm(text)), None)
            if heard is not None:
                after = [e for e in ev if e["t"] >= heard["t"]]
                if any(e["kind"] == "agent" for e in after):
                    break
                said = [e for e in after if e["kind"] == "say"]
                stat = [e for e in after if e["kind"] == "status"]
                if said and stat and stat[-1].get("state") == "watching" \
                        and time.time() - stat[-1]["t"] > 1.0:
                    break  # a local answer: said, and the voice has stopped
            time.sleep(0.25)
        self.quiet(2.0, WATCH_AFTER_S)
        return heard

    def game_up(self) -> bool:
        cmd = ["hyprctl", "-j", "clients"]
        if shutil.which("omabox"):
            cmd = ["omabox", "host", "--", *cmd]
        return "steam_app_613830" in subprocess.run(cmd, capture_output=True, text=True, check=False).stdout

    def _goto(self, want: str, order: tuple[str, ...]) -> bool:
        import cv2

        from playertwo import ctscreen

        for _ in range(12):
            self.ctl("shot", str(self.tmp.resolve()))
            focus = ctscreen.look(cv2.imread(str(self.tmp))).focus
            if focus == want:
                return True
            up = focus in order and order.index(focus) > order.index(want)
            self.ctl("press", "up" if up else "down")
            time.sleep(0.4)
        return False

    def quit_game(self) -> None:
        """From the field: menu, Settings, Quit, Yes, reading the cursor each step. Never saves.
        (From the title, after a lost fight, the launcher carries on by itself.)"""
        import cv2

        from playertwo import ctscreen

        if not self.game_up():
            return
        self.ctl("shot", str(self.tmp.resolve()))
        if ctscreen.look(cv2.imread(str(self.tmp))).focus not in ctscreen.MAIN:
            self.ctl("press", "b wait0.5 b wait0.5 y")
            time.sleep(1.5)
        if not self._goto("Settings", ctscreen.MAIN):
            print("reload: no Settings in the menu, leaving it to the launcher", file=sys.stderr)
            return
        self.ctl("press", "a")
        time.sleep(1.5)
        if not self._goto("Quit", (*ctscreen.SETTINGS, "Quit")):
            print("reload: no Quit in Settings, leaving it to the launcher", file=sys.stderr)
            return
        self.ctl("press", "a")
        time.sleep(1.5)
        self._goto("Yes", ("Yes", "No"))
        self.ctl("press", "a")
        for _ in range(12):
            time.sleep(1)
            if not self.game_up():
                break

    def reload(self, slot: int) -> None:
        self.quit_game()
        python = Path(self.p2).parent / "python"
        subprocess.run([str(python if python.exists() else sys.executable),
                        str(self.a.launcher), str(slot)], check=True, timeout=300)
        self.quiet(2.0, 20.0)

    def walk_to_fight(self, most: float = 90.0) -> bool:
        """Slot 1: walk down from where the save starts until something attacks."""
        end = time.time() + most
        while time.time() < end:
            st = self.state()
            if st is not None and st.in_battle:
                return True
            self.ctl("press", "down:1.5")
        return False


class Recorder:
    """Frames for the run with no load on Player Two: the recorder the takes use. If it won't
    start, `ctl shot` once a second as JPEGs instead (`shot` runs on Player Two's own loop,
    so a PNG, or a faster pace, would change what's being measured)."""

    def __init__(self, live: Live, folder: Path, mode: str):
        self.live, self.folder, self.mode = live, folder, mode
        self.video = folder / "run.mp4"
        self.shots: list[tuple[float, Path]] = []
        self.child = None
        self.start_t = 0.0

    def _host(self, *cmd: str) -> list[str]:
        return ["omabox", "host", "--", *cmd] if shutil.which("omabox") else list(cmd)

    def __enter__(self) -> Self:
        if self.mode == "none":
            return self
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.mode == "record":
            script = HERE / "demo_record.sh"
            cmd = (f"DEMO_MIC=0 setsid -f {script} {self.live.a.monitor} {self.video} "
                   f"> {self.folder / 'record.log'} 2>&1")
            subprocess.run(self._host("sh", "-c", cmd), capture_output=True, check=False)
            time.sleep(3.0)
            up = subprocess.run(self._host("pgrep", "-f", "gpu-screen-recorder -w"),
                                capture_output=True, check=False)
            if up.returncode == 0:
                return self
            print("recorder didn't start: taking a frame a second instead", file=sys.stderr)
            self.mode = "shot"
        self.child = subprocess.Popen(
            [sys.executable, __file__, "_shots", self.live.p2, str(self.folder)])
        return self

    def __exit__(self, *_exc) -> None:
        if self.mode == "record":
            pat = f"gpu-screen-recorder -w {self.live.a.monitor}"
            subprocess.run(self._host("pkill", "-INT", "-f", pat), capture_output=True, check=False)
            time.sleep(3.0)
        if self.child is not None:
            self.child.terminate()
            self.child.wait(timeout=10)

    def frames(self):
        if self.mode == "record" and self.video.exists():
            start = float(Path(f"{self.video}.start").read_text().split()[0])
            return VideoFrames(self.video, start, self.folder / "frames")
        if self.mode == "shot":
            return ShotFrames(sorted((float(p.stem), p) for p in self.folder.glob("*.jpg")))
        return None


def _shots(p2: str, folder: str) -> None:
    while True:
        t = time.time()
        subprocess.run([p2, "ctl", "shot", f"{folder}/{t:.2f}.jpg"], capture_output=True, check=False)
        time.sleep(max(0.0, 1.0 - (time.time() - t)))


def save_hashes(slot: int) -> dict[str, str]:
    p = SAVES / SAVE_FILE[slot]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]} if p.exists() else {}


def run_take(live: Live, take: str, run: int, lines: list[Line], reload_it: bool) -> list[Result]:
    a = live.a
    slot = SLOT[take]
    folder = Path(a.out) / f"take-{take}-{time.strftime('%H%M%S')}-run{run}"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"take {take} run {run}: {folder}", flush=True)
    if reload_it:
        live.reload(slot)
    if take == "B":
        live.send(BATTLE_SETUP, 30.0)
    heards: list[tuple[Line, dict]] = []
    problems: list[str] = []
    with Recorder(live, folder, a.frames) as rec:
        if take == "B" and not live.walk_to_fight():
            problems.append("no fight came up walking down: the fight lines ran anyway")
        for ln in lines:
            live.quiet(1.0, 15.0)
            if ln.ready and not live.wait_ready(ln.ready):
                problems.append(f"{ln.id}: no command window for {ln.ready} in 20 s, sent anyway")
            heard = live.send(ln.request, max(60.0, ln.sheet * 2.5))
            if heard is None:
                problems.append(f"{ln.id}: the request never reached the log")
                continue
            heards.append((ln, heard))
            print(f"  {ln.id} sent", flush=True)
    events = live.events(time.time() - 3600)
    frames = rec.frames()
    label = f"{take}{run}"
    results = []
    for i, (ln, heard) in enumerate(heards):
        end = heards[i + 1][1]["t"] if i + 1 < len(heards) else window_end(events, heard["t"])
        results.append(judge(ln, heard, events, end, label, a.budgets, frames))
    text = [f"Take {take}, run {run} (slot {slot}, log {live.log})", *problems, ""]
    text += [line_report(r) + "\n" for r in results]
    (folder / "report.txt").write_text("\n".join(text))
    with (folder / "events.jsonl").open("w") as f:
        for e in events:
            if heards and e["t"] >= heards[0][1]["t"] - 5:
                f.write(json.dumps(e) + "\n")
    for r in results:
        print(line_report(r), flush=True)
    return results


def to_json(r: Result) -> dict:
    return {"id": r.line.id, "run": r.run, "t": r.heard["t"], "text": r.heard["text"],
            "first": r.first, "silence": r.silence, "total": r.total, "heard_by": r.heard_by,
            "budget": r.budget, "fails": r.fails, "allowed": r.allowed}


def from_json(d: dict) -> Result:
    line = next(ln for ln in LINES if ln.id == d["id"])
    return Result(line, d["run"], {"t": d["t"], "text": d["text"]}, d["first"], d["silence"],
                  d["total"], d["heard_by"], d["budget"], d["fails"], d["allowed"])


def latest_takes(store: Path, take: str, results: list[Result]) -> list[Result]:
    """This take's results, with the latest of the other take from the same report."""
    store.parent.mkdir(parents=True, exist_ok=True)
    (store.parent / f"{store.stem}-{take}.json").write_text(
        json.dumps([to_json(r) for r in results]))
    out = []
    for t in "AB":
        f = store.parent / f"{store.stem}-{t}.json"
        if f.exists():
            out += [from_json(d) for d in json.loads(f.read_text())]
    return out


def cmd_run(a: argparse.Namespace) -> int:
    live = Live(a)
    if not live.log.exists():
        sys.exit(f"no log at {live.log}: start Player Two with --log there, or pass --log")
    if not live.ctl("status"):
        sys.exit("Player Two isn't answering `ctl status`: start it first (run sheet, setup 6)")
    skip = set(filter(None, a.skip.split(",")))
    report = Path(a.report).expanduser()
    store = Path(a.out).expanduser() / report.stem
    all_results: list[Result] = []
    extra: list[str] = []
    ok = True
    for take in a.take:
        slot = SLOT[take]
        lines = [ln for ln in LINES if ln.take == take and ln.id not in skip
                 and (not ln.optional or ln.id in a.add.split(","))]
        before = save_hashes(slot)
        results: list[Result] = []
        for run in range(1, a.repeats + 1):
            results += run_take(live, take, run, lines, a.reload == "always"
                                or (a.reload == "between" and run > 1))
        after = save_hashes(slot)
        if before != after:
            extra.append(f"**The save for slot {slot} changed during take {take}: {before} to "
                         f"{after}. Restore it from the backup (run sheet, After recording).**")
        ok = ok and all(r.ok for r in results)
        all_results = latest_takes(store, take, results)
    write_report(all_results, report, "Rehearsal of the Chrono Trigger takes", extra)
    write_rates(all_results, report.with_name("check-rates.txt"),
                f"Measured on live Steam, {time.strftime('%b %-d')}:")
    print(f"\n{summary(all_results)}\n\n{report}")
    for line in extra:
        print(line)
    return 0 if ok and not extra else 1


def cmd_replay(a: argparse.Namespace) -> int:
    events = read_log(Path(a.events).expanduser())
    if a.since:
        events = [e for e in events if e["t"] >= a.since]
    if a.until:
        events = [e for e in events if e["t"] <= a.until]
    out = Path(a.out).expanduser() / f"replay-{Path(a.events).parent.name}"
    videos = []
    for v in a.video:
        video = Path(v).expanduser()
        start = float(Path(f"{video}.start").read_text().split()[0])
        raw = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(video)], capture_output=True, text=True,
                             check=True)
        videos.append((start, start + float(raw.stdout.strip() or 0),
                       VideoFrames(video, start, out / f"frames-{video.stem}")))

    def frames_for(t: float):
        return next((f for a0, b0, f in videos if a0 <= t <= b0), None)

    results = replay(events, a.take, a.budgets, frames_for, Path(a.events).parent.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text("\n\n".join(line_report(r) for r in results) + "\n")
    for r in results:
        print(line_report(r), "\n", flush=True)
    print(summary(results))
    if a.report:
        write_report(results, Path(a.report).expanduser(),
                     f"Replay of {a.events} ({a.budgets} budgets)")
    return 0 if all(r.ok for r in results) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="send the take's lines to a running Player Two and judge them")
    r.add_argument("--take", required=True, choices=["A", "B", "AB"],
                   help="A: slot 4, looking and shopping. B: slot 1, a fight. AB: both")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--log", default="~/Videos/player-two/recording.events.jsonl",
                   help="the --log Player Two was started with")
    r.add_argument("--p2", default=_default_p2())
    r.add_argument("--skip", default="", help="line ids to leave out, e.g. B3,A5")
    r.add_argument("--add", default="", help="optional lines to put in, e.g. A3b")
    r.add_argument("--reload", choices=["always", "between", "never"], default="always",
                   help="always: a fresh load before each run, as the run sheet says. between: "
                   "the first run uses what's loaded")
    r.add_argument("--frames", choices=["record", "shot", "none"], default="record")
    r.add_argument("--monitor", default="DP-2")
    r.add_argument("--launcher", default=str(LAUNCHER), help="loads a slot: LAUNCHER SLOT")
    r.add_argument("--budgets", choices=["predicted", "sheet"], default="predicted")
    r.add_argument("--out", default=str(NIGHT / "check"))
    r.add_argument("--report", default=str(NIGHT / "check-report.md"))
    r.set_defaults(fn=cmd_run)
    y = sub.add_parser("replay", help="judge the run sheet lines in a recorded log")
    y.add_argument("events")
    y.add_argument("--take", default="AB", choices=["A", "B", "AB"])
    y.add_argument("--video", action="append", default=[],
                   help="a take's recording with its .start: frames check the figures in the "
                   "lines it covers (repeatable)")
    y.add_argument("--since", type=float)
    y.add_argument("--until", type=float)
    y.add_argument("--budgets", choices=["predicted", "sheet"], default="sheet")
    y.add_argument("--out", default=str(NIGHT / "check"))
    y.add_argument("--report", default="", help="also write the table here")
    y.set_defaults(fn=cmd_replay)
    if len(sys.argv) > 1 and sys.argv[1] == "_shots":
        _shots(sys.argv[2], sys.argv[3])
        return 0
    a = p.parse_args()
    return a.fn(a)


def _default_p2() -> str:
    dev = Path("/home/bts/Projects/player-two-dev/.venv/bin/player-two")
    if dev.exists():
        return str(dev)
    return shutil.which("player-two") or os.environ.get("PLAYER_TWO_BIN", "player-two")


if __name__ == "__main__":
    sys.exit(main())
