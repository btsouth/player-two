#!/usr/bin/env python3
"""The Player Two demo video, cut from a folder of takes and their event logs.

    python scripts/make_video.py TAKES --out video.mp4 [--no-code game2,game3] [--rates rates.txt]
    python scripts/make_video.py TAKES --out video.mp4 --cut video.mp4.cut.json   # hand edits
    python scripts/make_video.py TAKES --out video.mp4 --final                    # no placeholders

TAKES holds take-A.mp4, take-B.mp4, game2.mp4 and game3.mp4 (symlinks are fine), each with
<name>.mp4.start (demo_record.sh writes it) and <name>.events.jsonl (the session log; only the
video's time range is used), and optionally vo-why.* and vo-how.* (voice-over audio). A missing
game clip becomes a labelled placeholder. No footage belongs in the repo.

Every request is found in its take's log by the run sheet's words. Its window runs from the
request to its last line. Speech stays at normal speed; only the gaps between lines are sped up,
always with a label. Captions come from demo_captions (the voice's "spoken" record when the log
has one) and are burned once, per piece, from the raw take. The resolved cut goes to
<out>.cut.json; edit it and pass it back with --cut. Pieces whose settings haven't changed are
not encoded again.

The order is the one in games-plan.md: hook, why, Chrono Trigger, "not one game", how and what's
next. The budget is Chrono Trigger about 1:55, game 2 20 to 25 s, game 3 8 to 10 s, and 3:00 in
all. Over it, B3 goes first, then A5, then the description gets shorter.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_captions

W, H, FPS = 1920, 1080, 60
PROXY = False
GAME_X, GAME_Y, GAME_W, GAME_H = 368, 64, 1488, 837
BG = (9, 15, 25, 255)
ACCENT = (120, 235, 214, 255)
CODE = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()  # a new layout redoes the pieces
FONT_DIRS = ("/usr/share/fonts/liberation", "/usr/share/fonts/truetype/liberation")
BLUR = 45
TOTAL_LIMIT = 180.0
CT_ABOUT = 105.0
CT_LIMIT = 110.0  # "about 1:55"
GAME_BUDGET = {"game2": (20.0, 25.0), "game3": (10.0, 15.0)}
PLACEHOLDER_S = {"game2": 22.0, "game3": 12.0}
GAMES = ("game2", "game3")
NO_CODE = "No code written for this game, only a short notes file."

LINES = {
    "A1": ("A", r"^where am i\b"),
    "A2": ("A", r"^what'?s on the screen"),
    "A3": ("A", r"^take me to the fair"),
    "A4": ("A", r"^buy three potions"),
    "A5": ("A", r"^take me back outside"),
    "B1": ("B", r"^what are we fighting"),
    "B2": ("B", r"^crono,? use cyclone"),
    "B3": ("B", r"^attack the one on the left"),
    "B4": ("B", r"^win this fight"),
}
SPEED = {"A": 2.0, "B": 3.0}
CHAIN = {"A": 2.0, "B": 0.5}  # a fight's reader goes on talking: stop at its first pause
END_S = 5.0
DROP_ORDER = ("B3", "A5")
MAX_TALK_STEPS = (15.0, 11.0, 8.0)  # A2's description, seconds kept
CARRY_S = 3.0  # a line this far along as a request starts is kept whole

# True on every build: on this one a plain "buy three potions" has no model turn (fast.py),
# so the plan's "Muse Spark works out what you want" would be false here.
LOWER_A4 = "Code on the PC walks there, presses the buttons and checks the gold."
LOWER_B = ("In a fight, it reads the menus, HP and turns from the screen's pixels in about a "
           "tenth of a second, and checks every press.")
META_SPARK = ("Muse Spark understands free-form requests, decides what to do about them and "
              "describes scenes.")
META_READER = "A pixel reader on the PC handles menus, HP and gold. It's there for speed."
NEXT_LINE = "Creator-tested prototype. No blind-player testing yet. Lavos is still ahead."
DRAFT_RATES = ["DRAFT RATES / awaiting fresh live checks", "Measured results go here."]
HOOK_1 = ("In 2019, the RetroArch emulator added accessibility features for blind players. "
          "Its release post said:")
QUOTE = ("“A blind person still cannot get Super mario into the castle in Super Mario 64, "
         "nor defeat Lavos in Chrono Trigger...”")
VO_LABEL = "[voice-over goes here]"
DRAFT_MARK = "DRAFT RATES"


@dataclass
class Overlay:
    text: str
    size: int
    where: str
    f0: float  # seconds into the piece's output (or, before placing, the line's)
    f1: float


@dataclass
class Piece:
    id: str
    section: str  # hook, why, ct, game2, game3, outro
    line: str = ""  # the group it's listed under
    kind: str = "clip"  # clip, card or slate (a placeholder)
    take: str = ""
    a: float = 0.0
    b: float = 0.0
    speed: float = 1.0
    sharp_at: float | None = None  # take time the blur lifts; past b, it never does
    captions: bool = True
    vol: float = 1.0
    fade_in: bool = False
    fade_out: bool = False
    vo: str = ""
    blocks: list = field(default_factory=list)  # a card's (text, size, bold)
    secs: float = 0.0
    overlays: list = field(default_factory=list)

    @property
    def out_s(self) -> float:
        return self.secs if self.kind != "clip" else (self.b - self.a) / self.speed


def _piece_from(d: dict) -> Piece:
    d = dict(d)
    d["overlays"] = [Overlay(**o) for o in d.get("overlays", [])]
    return Piece(**d)


@dataclass
class Take:
    name: str
    video: Path
    start: float | None
    duration: float
    events: list
    caps: list  # (from, to, who, text) in seconds of the video
    top: bool = False

    def heard(self) -> list[tuple[float, dict]]:
        if self.start is None:
            return []
        return [(e["t"] - self.start, e) for e in self.events
                if e.get("kind") == "heard" and e.get("text")
                and 0 <= e["t"] - self.start <= self.duration]

    def game_name(self) -> str:
        return next((str(e["name"]) for e in self.events
                     if e.get("kind") == "game" and e.get("name")), "")


# ---------- the pure part: windows, speed, budget, wording ----------

def merge_spans(spans: list[tuple[float, float]], pad: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((a - pad, b + pad) for a, b in spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def split_speed(a: float, b: float, spans: list[tuple[float, float]], speed: float,
                min_gap: float = 2.0, pad: float = 0.3) -> list[tuple[float, float, float]]:
    """[a, b] as pieces (from, to, speed): speech at normal speed, and only a gap of min_gap
    or more between lines at `speed`."""
    cuts: list[tuple[float, float, float]] = []
    cur = a

    def add(x: float, y: float, sp: float) -> None:
        if y <= x:
            return
        if cuts and cuts[-1][2] == sp:
            cuts[-1] = (cuts[-1][0], y, sp)
        else:
            cuts.append((x, y, sp))

    for s, e in merge_spans([(max(x, a), min(y, b)) for x, y in spans if y > a and x < b], pad):
        s, e = max(s, a), min(e, b)
        if s > cur:
            add(cur, s, speed if speed > 1 and s - cur >= min_gap else 1.0)
        add(max(s, cur), e, 1.0)
        cur = max(cur, e)
    if b > cur:
        add(cur, b, speed if speed > 1 and b - cur >= min_gap else 1.0)
    return cuts


def request_lines(caps: list, heard_t: float, next_t: float, done_t: float | None,
                  chain: float = 2.0) -> list:
    """Player Two's captions that belong to a request: all up to its end (the log's "agent"
    event), then any that follow on within `chain` s (the narrator's line about a place just
    entered, "Crono's level increased!"). With no end logged (a description), the first line
    and what follows on from it."""
    mine = [c for c in caps if c[2] == "Player Two" and heard_t - 0.2 <= c[0] < next_t]
    if not mine:
        return []
    own = mine[:1]
    if done_t is not None:
        own = [c for c in mine if c[0] <= done_t + 0.5] or own
    for c in mine[len(own):]:
        if c[0] - own[-1][1] > chain:
            break
        own.append(c)
    return own


def talk_cut(own: list, max_talk: float) -> tuple[list, float]:
    """A long description cut after whole captions within max_talk s of its first word: the
    captions kept, and the seconds of talk left out."""
    if not own:
        return own, 0.0
    t0 = own[0][0]
    kept = [c for c in own if c[1] <= t0 + max_talk] or own[:1]
    return kept, max(0.0, own[-1][1] - kept[-1][1])


def fit_budget(line_s: dict[str, float], ct_extra: float, other: float,
               limit: float = TOTAL_LIMIT, ct_limit: float = CT_LIMIT) -> list[str]:
    """Which lines to drop, in the plan's order, to get under the budget."""
    drop: list[str] = []
    for name in DROP_ORDER:
        ct = ct_extra + sum(s for k, s in line_s.items() if k not in drop)
        if ct <= ct_limit and ct + other <= limit:
            break
        if name in line_s:
            drop.append(name)
    return drop


def requests_wording(shown: list[tuple[str, bool]], sped: bool) -> str:
    """The end card's line, from the log's own "typed" marks: (who, typed) per request shown."""
    typed = sorted({who for who, t in shown if t}, key=[w for w, _ in shown].index)
    spoken = sorted({who for who, t in shown if not t}, key=[w for w, _ in shown].index)
    if typed and spoken:
        text = (f"Requests spoken live in {_and(spoken)}, typed in {_and(typed)}."
                if not set(typed) & set(spoken) else
                "Some requests spoken live, some typed.")
    elif typed:
        text = "Requests typed, not spoken, in this cut."
    elif spoken:
        text = "Requests spoken live."
    else:
        text = ""
    return f"{text} Sped-up parts are marked.".strip() if sped else text


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def place_overlays(pieces: list[Piece], overlays: list[Overlay]) -> None:
    """Overlays timed in a line's output, handed to the pieces they fall in."""
    t = 0.0
    for p in pieces:
        d = p.out_s
        for o in overlays:
            f0, f1 = max(o.f0 - t, 0.0), min(o.f1 - t, d)
            if f1 > f0 + 0.01:
                p.overlays.append(Overlay(o.text, o.size, o.where, round(f0, 3), round(f1, 3)))
        t += d


def speech_spans(events: list, start: float) -> list[tuple[float, float]]:
    """When the voice was really talking, in video seconds: its "spoken" record, else the
    status changes of an older log."""
    played = [(e["began"] - start, e["ended"] - start) for e in events
              if e.get("kind") == "spoken" and "began" in e]
    if played:
        return merge_spans(played)
    spans, cur = [], None
    for e in events:
        if e.get("kind") != "status":
            continue
        if e.get("state") == "speaking" and cur is None:
            cur = e["t"] - start
        elif e.get("state") != "speaking" and cur is not None:
            spans.append((cur, e["t"] - start))
            cur = None
    return merge_spans(spans)


FACT = re.compile(r"full health|health is full|everyone's (?:full|fine)|no one got hurt",
                  re.IGNORECASE)
HURT = re.compile(r"\btook \d+|down to \d+|\bat 1 HP\b|knocked out", re.IGNORECASE)


def caption_problems(caps: list, speech: list[tuple[float, float]], pieces: list[Piece],
                     slack: float = 1.0) -> list[tuple[float, str]]:
    """What can be seen wrong without looking: (take time, problem), for captions shown. Same
    speaker over itself, a caption outliving its speech, speech sped up, a caption carried in
    from before a cut, a "full health" still up (or just said) as someone is hurt."""
    shown = [p for p in pieces if p.kind == "clip" and p.captions]
    windows = merge_spans([(p.a, p.b) for p in shown])
    out: list[tuple[float, str]] = []
    vis = [c for c in caps if any(c[0] < b and c[1] > a for a, b in windows)]
    for i, (a, b, who, text) in enumerate(vis):
        short = text[:60]
        nxt = next((c for c in vis[i + 1:] if c[2] == who), None)
        if nxt is not None and nxt[0] < b - 0.01:
            out.append((a, f"overlap: {short!r} still up {b - nxt[0]:.2f} s into {nxt[3][:40]!r}"))
        for wa, _wb in windows:
            if a < wa - 0.05 < b - 0.05:
                out.append((a, f"carried over a cut: {short!r} started {wa - a:.1f} s earlier"))
        if who != "Player Two":
            continue
        under = [(s, e) for s, e in speech if s < b and e > a]
        if not under:
            out.append((a, f"no speech under it: {short!r}"))
        elif b - under[-1][1] > slack:
            out.append((a, f"outlives its speech by {b - under[-1][1]:.1f} s: {short!r}"))
        for p in shown:
            if p.speed > 1 and min(b, p.b) - max(a, p.a) > 0.3:
                out.append((a, f"speech sped up {p.speed:g}x: {short!r}"))
        if FACT.search(text):
            soon = [c for c in vis[i + 1:] if c[0] < b + 1.5 and HURT.search(c[3])]
            if soon:
                out.append((a, f"stale: {short!r} next to {soon[0][3][:50]!r}"))
    return sorted(out)


def out_time(pieces: list[Piece], take: str, t: float) -> float | None:
    """Where a moment of a take lands in the video."""
    cur = 0.0
    for p in pieces:
        if p.kind == "clip" and p.take == take and p.a <= t < p.b:
            return cur + (t - p.a) / p.speed
        cur += p.out_s
    return None


# ---------- the plan ----------

def _attempts(take: Take, rx: str) -> list[float]:
    return [t for t, e in take.heard() if re.search(rx, e["text"].lower().strip())]


def _lead(take: Take, heard_t: float, event: dict) -> float:
    """How long before the "heard" the request began: a spoken one was said, and transcribed,
    before it's logged. It starts where the status went to listening."""
    if event.get("typed") or take.start is None:
        return 0.4
    listen = [e["t"] - take.start for e in take.events
              if e.get("kind") == "status" and e.get("state") == "listening"
              and heard_t - 15 < e["t"] - take.start <= heard_t]
    return min(heard_t - listen[-1] + 0.3, 15.0) if listen else 2.5


@dataclass
class LineCut:
    id: str
    take: str
    a: float
    b: float
    heard: float
    typed: bool
    first_word: float | None
    left_out: float = 0.0


def cut_line(line: str, take: Take, pick: int | None, max_talk: float | None) -> LineCut | None:
    rx = LINES[line][1]
    times = _attempts(take, rx)
    if not times:
        return None
    heard_t = times[pick - 1] if pick and 0 < pick <= len(times) else times[-1]
    event = next(e for t, e in take.heard() if t == heard_t)
    later = [t for t, _e in take.heard() if t > heard_t]
    next_t = later[0] if later else take.duration
    done = next((e["t"] - take.start for e in take.events
                 if e.get("kind") == "agent" and heard_t < e["t"] - take.start < next_t), None)
    own = request_lines(take.caps, heard_t, next_t, done, CHAIN[LINES[line][0]])
    left = 0.0
    if max_talk is not None:
        own, left = talk_cut(own, max_talk)
    end = (own[-1][1] if own else heard_t + 3.0) + 0.5
    start = heard_t - _lead(take, heard_t, event)
    # A line still being said as the request starts stays whole, not cut into mid-word.
    said = [c[0] for c in take.caps if c[2] == "Player Two" and c[0] < start < c[1]
            and start - c[0] <= CARRY_S]
    if said:
        start = min(said)
    return LineCut(line, take.name, max(0.0, start), min(end, next_t - 0.2, take.duration),
                   heard_t, bool(event.get("typed")), own[0][0] if own else None, left)


def line_pieces(cut: LineCut, take: Take, speed: float) -> list[Piece]:
    spans = [(c[0], c[1]) for c in take.caps if c[2] == "Player Two"]
    spans.append((cut.a, cut.heard + 1.5))  # the player's own words, and time to read them
    pieces = [Piece(f"{cut.id}_{i}", "ct", cut.id, "clip", cut.take, round(x, 3), round(y, 3), sp)
              for i, (x, y, sp) in enumerate(split_speed(cut.a, cut.b, spans, speed))]
    label_where = "trl" if take.top else "tr"
    for p in pieces:
        if p.speed > 1:
            p.overlays.append(Overlay(f"{p.speed:g}x", 48, label_where, 0.0, round(p.out_s, 3)))
    return pieces


def _first_gap_out(pieces: list[Piece]) -> float:
    t = 0.0
    for p in pieces:
        if p.speed > 1:
            return t
        t += p.out_s
    return 1.0


def ct_pieces(takes: dict[str, Take], picks: dict[str, int], drop: set[str], max_talk: float,
              warnings: list[str]) -> dict[str, list[Piece]]:
    out: dict[str, list[Piece]] = {}
    for line, (name, _rx) in LINES.items():
        if line in drop:
            continue
        take = takes.get(f"take-{name}")
        if take is None:
            warnings.append(f"{line}: no take-{name}")
            continue
        cut = cut_line(line, take, picks.get(line), max_talk if line == "A2" else None)
        if cut is None:
            warnings.append(f"{line}: not found in take-{name}'s log")
            continue
        pieces = line_pieces(cut, take, SPEED[name])
        overlays: list[Overlay] = []
        if line == "A1":
            overlays.append(Overlay("1. Where am I?", 44, "tl", 0.0, 2.5))
            # The voice-over already revealed the game; never blur it a second time.
        elif line == "A2":
            # Keep the revealed picture visible for the viewer to check the description.
            pieces[-1].fade_out = True
            if cut.left_out > 2:
                more = max(5, 5 * round(cut.left_out / 5))
                total = sum(p.out_s for p in pieces)
                overlays.append(Overlay(f"(it describes for about {more} seconds more)", 36, "bc",
                                        max(0.0, total - 3.5), total))
        elif line == "A3":
            overlays.append(Overlay("2. Getting around", 44, "tl", 0.0, 2.5))
        elif line == "A4":
            t0 = _first_gap_out(pieces)
            overlays.append(Overlay(LOWER_A4, 36, "bl", t0, t0 + 5.0))
        elif line == "B2":
            overlays.append(Overlay(LOWER_B, 36, "ml", 2.0, 7.0))
        elif line == "B4":
            pieces[-1].fade_out = True
        place_overlays(pieces, overlays)
        out[line] = pieces
    return out


def game_pieces(name: str, take: Take | None, no_code: bool, clip: tuple[float, float] | None,
                warnings: list[str]) -> list[Piece]:
    lo, hi = GAME_BUDGET[name]
    label = {"game2": "Monster Hunter Stories 3", "game3": "Red Dead Redemption"}[name]
    if take is None:
        secs = PLACEHOLDER_S[name]
        p = Piece(name, name, name, "slate", secs=secs, blocks=[
            (f"PLACEHOLDER: {label}", 64, True),
            (f"Accepted take pending / {lo:.0f} to {hi:.0f} seconds", 40, False)])
        if no_code:
            p.overlays.append(Overlay(NO_CODE, 40, "tl2", 0.0, min(5.0, secs)))
        return [p]
    title = take.game_name() or label
    heard = take.heard()
    if clip:
        a, b = clip
    elif heard:
        a = max(0.0, heard[0][0] - _lead(take, heard[0][0], heard[0][1]))
        ends = [c[1] for c in take.caps if c[2] == "Player Two" and c[0] >= a]
        b = min((ends[-1] if ends else take.duration) + 0.5, take.duration)
    else:
        a, b = 0.0, take.duration
    if b - a > hi:  # end after a whole caption, if one fits
        fit = [c[1] for c in take.caps if a < c[1] <= a + hi - 0.3]
        b = (fit[-1] + 0.3) if fit else a + hi
    if b - a < lo or b - a > hi:
        warnings.append(f"{name}: {b - a:.1f} s, the budget is {lo:.0f} to {hi:.0f} s")
    p = Piece(name, name, name, "clip", name, round(a, 3), round(b, 3), fade_out=True)
    p.overlays.append(Overlay(title, 40, "tl", 0.0, min(4.0, b - a)))
    if no_code:
        p.overlays.append(Overlay(NO_CODE, 40, "tl2", 0.0, min(5.0, b - a)))
    return [p]


def _shown_requests(takes: dict[str, Take], parts: list[Piece]) -> list[tuple[str, bool]]:
    shown: list[tuple[str, bool]] = []
    for p in parts:
        if p.kind != "clip" or not p.captions or p.take not in takes:
            continue
        take = takes[p.take]
        who = "Chrono Trigger" if p.take.startswith("take-") else (take.game_name() or p.take)
        shown += [(who, bool(e.get("typed"))) for t, e in take.heard() if p.a <= t < p.b]
    return shown


def rates_blocks(rates: list[str]) -> list[tuple[str, int, bool]]:
    return [(rates[0], 44, True), (" · ".join(rates[1:]), 38, False), (NEXT_LINE, 40, True)]


def make_plan(takes: dict[str, Take], picks: dict[str, int], no_code: set[str],
              rates: list[str], vo: dict[str, str], vo_s: dict[str, float],
              clips: dict[str, tuple[float, float]], force_drop: set[str] = frozenset(),
              ) -> tuple[list[Piece], list[str], list[str]]:
    """Every piece in order, what was dropped, and warnings."""
    warnings: list[str] = []
    a = takes.get("take-A")
    why_s = max(8.0, vo_s.get("why", 0.0) + 0.5)
    hook = [Piece("c1", "hook", "hook", "card", secs=1.5, blocks=[
        ("What if you could play by asking?", 88, True),
        ("Player Two / an AI co-op partner", 38, False)])]
    if a is not None:
        why = Piece("why", "why", "why", "clip", "take-A", 0.0, min(why_s, a.duration),
                    sharp_at=2.0, captions=False, vol=0.15 if "why" in vo else 0.0,
                    fade_in=True, fade_out=True, vo=vo.get("why", ""))
        first = next(iter(t for t, _e in a.heard()), None)
        if first is not None and first < why.b:
            warnings.append(f"why: the footage runs {why.b - first:.1f} s into the first request "
                            f"(blurred and quiet, but start the take earlier if you can)")
    else:
        why = Piece("why", "why", "why", "slate", secs=why_s, vo=vo.get("why", ""),
                    blocks=[("PLACEHOLDER: blurred world map", 64, True)])
    if "why" not in vo:
        why.overlays.append(Overlay(VO_LABEL, 26, "tl", 0.0, why.out_s))
        why.overlays.append(Overlay("Games hide their choices in pixels.\nI built a co-op partner that reads them,\nso you can choose what happens next.", 42, "bc", 0.0, why.out_s))
    fight_card = Piece("c3", "ct", "fight card", "card", secs=3.0, blocks=[
        ("3. A fight", 72, True),
        ("My own save, later in the game. Battle Mode set to Wait.", 40, False)])
    games = [Piece("c-games", "games", "games card", "card", secs=2.0,
                   blocks=[("It's not one game.", 72, True)])]
    for g in GAMES:
        games += game_pieces(g, takes.get(g), g in no_code, clips.get(g), warnings)
    how_s = max(6.5, vo_s.get("how", 0.0) + 0.5)
    outro = [Piece("c-how", "outro", "how", "card", secs=how_s, vo=vo.get("how", ""),
                   blocks=[("Muse understands scenes and open-ended requests.", 60, True),
                           ("Local readers and routines handle common actions.", 42, False)]),
             Piece("c-rates", "outro", "rates", "card", secs=8.0, blocks=rates_blocks(rates))]
    if "how" not in vo:
        outro[0].overlays.append(Overlay(VO_LABEL, 26, "tl", 0, how_s))
    missing_fight = None if "take-B" in takes else Piece(
        "missing-fight", "ct", "fight", "slate", secs=50.0, blocks=[
            ("PLACEHOLDER: Chrono Trigger fight", 64, True),
            ("Fresh take required / older health narration is incorrect", 40, False)])
    ct_extra = fight_card.out_s + (missing_fight.out_s if missing_fight else 0)
    fixed = sum(p.out_s for p in [*hook, why, *games, *outro]) + END_S
    # The budget, in the plan's order.
    drop = set(force_drop)
    lines: dict[str, list[Piece]] = {}
    for max_talk in MAX_TALK_STEPS:
        warned: list[str] = []
        lines = ct_pieces(takes, picks, drop, max_talk, warned)
        secs = {k: sum(p.out_s for p in v) for k, v in lines.items()}
        more = fit_budget(secs, ct_extra, fixed)
        if more:
            drop |= set(more)
            lines = ct_pieces(takes, picks, drop, max_talk, warned)
            secs = {k: sum(p.out_s for p in v) for k, v in lines.items()}
        ct = ct_extra + sum(secs.values())
        if ct <= CT_LIMIT and ct + fixed <= TOTAL_LIMIT:
            break
    warnings += warned
    ct_parts: list[Piece] = []
    for line in LINES:
        if line == "B1":
            ct_parts.append(fight_card)
            if missing_fight:
                ct_parts.append(missing_fight)
        ct_parts += lines.get(line, [])
    parts = [*hook, why, *ct_parts, *games, *outro]
    parts.append(end_card(takes, parts))
    return parts, sorted(drop), warnings


def end_card(takes: dict[str, Take], parts: list[Piece]) -> Piece:
    sped = any(p.speed > 1 for p in parts)
    wording = requests_wording(_shown_requests(takes, parts), sped)
    return Piece("c-end", "outro", "end", "card", secs=END_S, blocks=[
        ("Player Two", 80, True),
        ("github.com/btsouth/player-two · open source (MIT)", 40, False),
        ("Built with Meta Muse Spark and Muse Voice Transcribe", 40, False),
        (wording, 32, False)])


def final_refusals(parts: list[Piece], warnings: list[str]) -> list[str]:
    """What --final won't ship."""
    out = [f"placeholder: {p.id}" for p in parts if p.kind == "slate"]
    for p in parts:
        if p.id in ("why", "c-how") and not p.vo:
            out.append(f"missing voice-over audio: {p.id}")
        if any(o.text == VO_LABEL for o in p.overlays):
            out.append(f"no voice-over for {p.id} (vo-why.* in the folder)")
        if any(DRAFT_MARK in str(b[0]) for b in p.blocks):
            out.append("draft rates: pass --rates")
        if p.kind == "clip" and p.speed != 1 and not any(
                o.text == f"{p.speed:g}x" and o.f0 <= 0.01 and o.f1 >= p.out_s - 0.05
                for o in p.overlays):
            out.append(f"no {p.speed:g}x label all through {p.id}")
    total = sum(p.out_s for p in parts)
    if total > TOTAL_LIMIT:
        out.append(f"total {total:.1f} s is over {TOTAL_LIMIT:.0f} s")
    return out + [w for w in warnings if "not found" in w or "no take-" in w or "missing timing" in w]


# ---------- pictures ----------

def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    for d in FONT_DIRS:
        if (Path(d) / name).exists():
            return ImageFont.truetype(str(Path(d) / name), size)
    return ImageFont.truetype(name, size)


def _wrap(draw, text: str, font, width: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = (line + " " + word).strip()
            if draw.textlength(test, font=font) <= width:
                line = test
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def card_png(path: Path, blocks: list, bg=BG) -> Path:
    img = Image.new("RGBA", (W, H), tuple(bg))
    d = ImageDraw.Draw(img)
    d.line((96, 120, 180, 120), fill=ACCENT, width=6)
    d.text((96, 66), "PLAYER TWO  /  CO-OP, BY VOICE", font=_font(True, 25), fill=ACCENT)
    laid = [( _wrap(d, text, _font(bold, size), W - 240), _font(bold, size), size)
            for text, size, bold in blocks]
    total = sum(len(ls) * int(size * 1.3) for ls, _, size in laid) + 36 * (len(laid)-1)
    y = max(200, (H-total)//2)
    for ls, f, size in laid:
        for ln in ls:
            d.text((96, y), ln, font=f, fill="white")
            y += int(size * 1.3)
        y += 36
    d.text((96, H-76), "A second player. A different way in.", font=_font(False, 26), fill=(150,166,181))
    img.save(path)
    return path


def label_png(path: Path, o: Overlay) -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if o.where == "draft":
        d.text((W-64, 20), o.text, anchor="ra", font=_font(True, 20), fill=(200,166,190))
    elif o.where == "trace":
        title = "REQUEST / RESULT"
        d.text((56, 72), title, font=_font(True, 23), fill=(150,166,181))
        d.line((56, 224, 304, 224), fill=(45,65,80), width=2)
        y = 260
        for i, para in enumerate(o.text.split("\n")):
            f = _font(i == 0, 30 if i == 0 else 27)
            for ln in _wrap(d, para, f, 272):
                d.text((56, y), ln, font=f, fill=(255,206,124) if para == "Local action" else ACCENT if i == 0 or para.startswith("Muse") or para == "SAM 3.1" else "white")
                y += 40
            y += 20
    elif o.where in ("tl", "tl2"):
        for i, ln in enumerate(_wrap(d, o.text, _font(True, 30), 272)):
            d.text((56, 114 + i*40), ln, font=_font(True, 30), fill=ACCENT)
    else:
        f = _font(True, min(o.size, 42))
        lines = _wrap(d, o.text, f, 1260)
        y = 700 if o.where == "bc" else 76
        if o.text.endswith("x"):
            d.rounded_rectangle((56, 800, 260, 866), radius=12, fill=(24,45,58,255))
            d.text((80, 812), o.text + " SPEED", font=_font(True, 27), fill=ACCENT)
        else:
            # Cards / voice-over stand-ins only. Gameplay keeps its own pixels clear.
            d.rectangle((GAME_X, y-20, W-64, y+len(lines)*56+20), fill=(9,15,25,235))
            for i, ln in enumerate(lines):
                d.text((GAME_X+28, y+i*56), ln, font=f, fill="white")
    img.save(path)
    return path


def evidence_overlays(parts: list[Piece], takes: dict[str, Take]) -> list[dict]:
    """Completion badges: explicit logged model work only, never guessed from request words.

    Durations are labelled by scope. A local action's seconds include walking/animation;
    model_s is inference time, not end-to-end response time. Badges start at completion.
    """
    report = []
    for p in parts:
        if p.kind != "clip" or not p.captions or p.take not in takes:
            continue
        take = takes[p.take]
        if take.start is None:
            continue
        es = take.events
        heard = [(t, e) for t, e in take.heard() if t < p.b]
        if not heard:
            continue
        t, request = heard[-1]
        next_t = next((x for x, _ in take.heard() if x > t), take.duration)
        done = [e for e in es if t <= e.get("t", 0)-take.start < min(next_t, p.b)]
        brain = next((e.get("brain", "") for e in reversed(es)
                      if e.get("kind") == "ready" and e["t"] <= take.start+t), "")
        facts = []
        for e in done:
            kind = e.get("kind")
            if kind == "agent" and isinstance(e.get("seconds"), (int,float)):
                turns = e.get("turns")
                ms = [x["model_s"] for x in turns or [] if isinstance(x.get("model_s"), (int,float))]
                if ms and "muse" in brain.lower():
                    facts.append((e, "Muse Spark", f"Model time {sum(ms):.2f} s"))
                elif turns == [] and e.get("steps") and all(
                        x.get("tool") in {"go_to_place", "go_to_exit", "battle", "fight", "shop"}
                        for x in e["steps"]):
                    facts.append((e, "Local action", f"Action time {e['seconds']:.2f} s"))
                else:
                    # Some legacy screen answers call a model internally but log no turns.
                    facts.append((e, "Player Two", f"Total time {e['seconds']:.2f} s"))
            elif kind == "say" and e.get("source") in ("describe", "atmosphere", "narrator") \
                    and isinstance(e.get("latency"), (int,float)) and "muse" in brain.lower():
                facts.append((e, "Muse Spark / scene", f"Generation {e['latency']:.2f} s"))
            elif kind == "where" and e.get("via") == "sam-3.1" and isinstance(e.get("seconds"), (int,float)):
                facts.append((e, "SAM 3.1", f"Locate time {e['seconds']:.2f} s"))
            # Future instrumentation may supply explicit stage records; no inferred SAM/STT.
            elif kind == "model_stage" and e.get("model") in ("Muse Voice Transcribe", "SAM 3.1") \
                    and isinstance(e.get("seconds"), (int,float)):
                facts.append((e, e["model"], f"Stage time {e['seconds']:.2f} s"))
        label = "You: typed" if request.get("typed") else "You: heard"
        base = f"{label}\n{request['text']}"
        heard_at = max(0, (t-p.a)/p.speed)
        if heard_at > 0:
            p.overlays.append(Overlay("You:\nRequest begins", 30, "trace", 0, heard_at))
        state = {}
        cur = heard_at
        for e, layer, timing in facts:
            at = max(cur, (e['t']-take.start-p.a)/p.speed)
            text = base + "".join(f"\n{k}\n{v}" for k,v in state.items())
            if at > cur:
                p.overlays.append(Overlay(text, 30, "trace", cur, at))
            state[layer] = timing
            cur = at
            report.append({"piece": p.id, "request": request['text'], "layer": layer,
                           "timing": timing, "event_t": e['t'], "badge_from": at})
        text = base + "".join(f"\n{k}\n{v}" for k,v in state.items())
        if cur < p.out_s:
            p.overlays.append(Overlay(text, 30, "trace", cur, p.out_s))
    return report


# ---------- ffmpeg ----------

def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def probe_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def _enc(vcodec: str) -> list[str]:
    v = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21"] if "nvenc" in vcodec else
         ["-c:v", vcodec, "-preset", "veryfast", "-crf", "20", "-threads", "6"])
    return [*v, "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar",
            "48000", "-ac", "2"]


def _tempo(speed: float) -> str:
    """atempo takes up to 2x at a time."""
    out = ""
    while speed > 2.0:
        out += "atempo=2.0,"
        speed /= 2.0
    return out if speed == 1.0 else f"{out}atempo={speed:g},"


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def _overlays_chain(p: Piece, work: Path, first_input: int) -> tuple[list[str], str, str]:
    inputs: list[str] = []
    chain, last = "", "v0"
    for i, o in enumerate(p.overlays):
        png = work / ("ov-" + hashlib.sha1(f"{CODE}|{o.text}|{o.size}|{o.where}".encode())
                      .hexdigest()[:12] + ".png")
        if not png.exists():
            label_png(png, o)
        inputs += ["-i", str(png)]
        chain += (f";[{last}][{first_input + i}:v]overlay=0:0:"
                  f"enable='between(t,{o.f0},{o.f1})'[v{i + 1}]")
        last = f"v{i + 1}"
    if PROXY:
        chain += f";[{last}]scale=960:540[proxy]"
        last = "proxy"
    return inputs, chain, last


def encode_card(p: Piece, work: Path, out: Path, vcodec: str) -> None:
    bg = BG if p.kind == "card" else (32, 25, 43, 255)
    png = card_png(work / f"{p.id}.png", p.blocks, bg)
    audio = ["-i", p.vo] if p.vo else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    inputs, chain, last = _overlays_chain(p, work, 2)
    fc = f"[0:v]scale={W}:{H},format=yuv420p,fade=t=in:d=0.15,fade=t=out:st={max(0,p.secs-0.15)}:d=0.15[v0]{chain}"
    _ff("-loop", "1", "-framerate", str(FPS), "-i", str(png), *audio, *inputs, "-filter_complex",
        fc, "-map", f"[{last}]", "-map", "1:a", "-t", f"{p.secs}", "-af", "apad", *_enc(vcodec),
        str(out))


def encode_clip(p: Piece, takes: dict[str, Take], asses: dict[str, Path], work: Path, out: Path,
                vcodec: str) -> None:
    take = takes[p.take]
    inputs = ["-ss", f"{p.a}", "-to", f"{p.b}", "-i", str(take.video)]
    n = 1
    if p.vo:
        inputs += ["-i", p.vo]
        n = 2
    game_audio = take.video.with_suffix(".game.wav")
    voice_audio = take.video.with_suffix(".voice.wav")
    stems = game_audio.exists() and voice_audio.exists()
    stem_n = n
    if stems:
        for stem in (game_audio, voice_audio):
            inputs += ["-ss", f"{p.a}", "-to", f"{p.b}", "-i", str(stem)]
        n += 2
    ov_inputs, chain, last = _overlays_chain(p, work, n)
    v = f"[0:v]scale={GAME_W}:{GAME_H},fps={FPS},setpts=PTS-STARTPTS"
    if p.sharp_at is not None:
        if p.sharp_at >= p.b:
            v += f",gblur=sigma={BLUR}"
        else:
            t0 = p.sharp_at - p.a
            k = f"clip((T-{t0:.2f})/1.5,0,1)"
            v += (f",split[s][b];[b]gblur=sigma={BLUR}[bl];"
                  f"[s][bl]blend=all_expr='A*{k}+B*(1-{k})'")
    v += f",pad={W}:{H}:{GAME_X}:{GAME_Y}:color=0x090f19,setsar=1"
    if p.captions and p.take in asses:
        v += f",setpts=PTS+{p.a}/TB,subtitles='{_esc(asses[p.take])}',setpts=PTS-STARTPTS"
    v += f",setpts=PTS/{p.speed:g}[v0]"
    tempo = _tempo(p.speed)
    dur = p.out_s
    # Accelerated gaps are silent: no sped-up voices or frantic game music.
    gain = 0 if p.speed > 1 else p.vol
    if stems:
        # Voice stem contains Player Two + the microphone; the game stem contains only game.
        a = (f"[{stem_n+1}:a]asetpts=PTS-STARTPTS,{tempo}asplit[voice][side];"
             f"[{stem_n}:a]asetpts=PTS-STARTPTS,{tempo}anull"
             "[game];[game][side]sidechaincompress=threshold=0.025:ratio=8:"
             "attack=15:release=220[duck];[duck][voice]amix=inputs=2:normalize=0:"
             f"duration=first,volume={gain}")
    else:
        a = f"[0:a]asetpts=PTS-STARTPTS,{tempo}volume={gain}"
    a += f",afade=t=in:d=0.012,afade=t=out:st={max(0,dur-0.018):.3f}:d=0.018"
    if p.fade_in:
        a += ",afade=t=in:d=0.5"
    if p.fade_out:
        a += f",afade=t=out:st={max(0.0, dur - 0.5):.2f}:d=0.5"
    if p.vo:
        a += "[ta];[ta][1:a]amix=inputs=2:duration=first:normalize=0[a]"
    else:
        a += "[a]"
    _ff(*inputs, *ov_inputs, "-filter_complex", f"{v}{chain};{a}", "-map", f"[{last}]", "-map",
        "[a]", "-t", f"{dur:.3f}", *_enc(vcodec), str(out))


def ass_text(take: Take) -> str:
    """Speech captions in the dedicated band below gameplay, with a short opacity fade."""
    # Requests live in the left rail; speech has a dedicated band below gameplay.
    head = demo_captions.ASS_HEAD.replace("PlayResX: 512", "PlayResX: 1920").replace(
        "PlayResY: 288", "PlayResY: 1080")
    styles = head.index("Style: P2,")
    events = head.index("[Events]")
    head = head[:styles] + ("Style: P2,Liberation Sans,40,&H00FFFFFF,&H00FFFFFF,"
        "&H00190F09,&H00190F09,0,0,0,0,100,100,0,0,1,1,0,2,368,64,30,1\n\n") + head[events:]
    rows = []
    for a, b, who, txt in take.caps:
        if who != "Player Two":
            continue
        # Escape ASS controls from transcript text, and keep a visible speaker identity.
        txt = txt.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
        if not txt.startswith("Player Two:"):
            txt = "Player Two: " + txt
        rows.append(f"Dialogue: 0,{demo_captions._ass_stamp(a)},"
                    f"{demo_captions._ass_stamp(b)},P2,,0,0,0,,"
                    + r"{\fad(90,70)}" + txt.replace(chr(10), " "))
    return head + "\n".join(rows) + "\n"



def build(parts: list[Piece], takes: dict[str, Take], out: Path, vcodec: str, jobs: int) -> None:
    work = out.parent / f"{out.stem}-work"
    work.mkdir(parents=True, exist_ok=True)
    asses: dict[str, Path] = {}
    for name, take in takes.items():
        if take.caps:
            text = ass_text(take)
            path = work / f"{name}.ass"
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
            asses[name] = path

    def one(p: Piece) -> Path:
        dest = work / f"{p.id}.mp4"
        key = {**dataclasses.asdict(p), "code": CODE, "proxy": PROXY, "codec": vcodec}
        sources = [takes[p.take].video] if p.kind == "clip" else []
        if p.kind == "clip":
            sources += [f for suffix in (".game.wav", ".voice.wav")
                        if (f := takes[p.take].video.with_suffix(suffix)).exists()]
        if p.vo:
            sources.append(Path(p.vo))
        key["sources"] = [(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in sources]
        if p.kind == "clip" and p.take in asses:
            key["ass"] = hashlib.sha1(asses[p.take].read_bytes()).hexdigest()
        stamp = work / f"{p.id}.key"
        text = json.dumps(key, sort_keys=True)
        if dest.exists() and stamp.exists() and stamp.read_text() == text:
            return dest
        if p.kind == "clip":
            encode_clip(p, takes, asses, work, dest, vcodec)
        else:
            encode_card(p, work, dest, vcodec)
        stamp.write_text(text)
        return dest

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        files = list(pool.map(one, parts))
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    joined = work / "joined.mp4"
    _ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined))
    normalize_mix(joined, out, work)


def normalize_mix(joined: Path, out: Path, work: Path) -> None:
    """Two-pass EBU R128. -1.5 dBTP leaves AAC reconstruction headroom."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(joined), "-vn", "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.DOTALL)
    if not match:
        raise RuntimeError("No loudnorm measurement")
    stats = json.loads(match.group())
    (work / "loudnorm.json").write_text(json.dumps(stats, indent=2))
    if not all(math.isfinite(float(stats[k])) for k in ("input_i", "input_tp", "input_thresh")):
        # A completely silent layout proxy has no meaningful LUFS; preserve silence.
        _ff("-i", str(joined), "-c", "copy", "-movflags", "+faststart", str(out))
        return
    af = ("loudnorm=I=-14:TP=-1.5:LRA=9:linear=true:"
          f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
          f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
          f"offset={stats['target_offset']}")
    _ff("-i", str(joined), "-c:v", "copy", "-af", af, "-ar", "48000", "-c:a", "aac",
        "-b:a", "192k", "-movflags", "+faststart", str(out))


def _grab(video: Path, t: float, width: int) -> Image.Image:
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v",
                          "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "png", "-"],
                         capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(png)).convert("RGB")


def tile_frames(video: Path, shots: list[tuple[float, str]], path: Path, cols: int, width: int,
                jobs: int = 4) -> None:
    """Frames at the given moments, each with its label under it."""
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        frames = list(pool.map(lambda s: _grab(video, s[0], width), shots))
    fh, band = frames[0].height, 26
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * width, rows * (fh + band)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    f = _font(False, 17)
    for i, (frame, (_t, label)) in enumerate(zip(frames, shots, strict=True)):
        x, y = (i % cols) * width, (i // cols) * (fh + band)
        sheet.paste(frame, (x, y))
        d.text((x + 4, y + fh + 3), label[:int(width / 8.5)], font=f, fill=(255, 230, 90))
    sheet.save(path)


def contact_sheet(video: Path, sheet: Path, every: float = 3.0, jobs: int = 4) -> None:
    total = probe_s(video)
    shots = [(t, f"{int(t // 60)}:{t % 60:04.1f}") for t in
             (i * every + 0.5 for i in range(int(total / every))) if t < total]
    tile_frames(video, shots, sheet, 6, 426, jobs)


def caption_shots(parts: list[Piece], takes: dict[str, Take]) -> list[tuple[float, str]]:
    """A moment for every caption shown: just before it goes, when a stale one is plainest."""
    shots = []
    for name, take in takes.items():
        for a, b, _who, text in take.caps:
            for t in (b - 0.1, (a + b) / 2):
                at = out_time(parts, name, t)
                if at is not None:
                    shots.append((at, f"{int(at // 60)}:{at % 60:04.1f} {text}"))
                    break
    return sorted(shots)


def caption_sheets(video: Path, parts: list[Piece], takes: dict[str, Take], jobs: int = 4
                   ) -> list[Path]:
    shots = caption_shots(parts, takes)
    out = []
    for i in range(0, len(shots), 20):
        path = video.with_name(f"{video.stem}.captions-{i // 20 + 1}.png")
        tile_frames(video, shots[i:i + 20], path, 4, 640, jobs)
        out.append(path)
    return out


# ---------- the folder ----------

def _old_narrations(events: list[dict]) -> list[dict]:
    """A log with no "spoken" record: a narration is logged as soon as the model has written it,
    about when it starts being said. demo_captions takes that stamp as about when it ended, which
    puts the caption seconds early (a description's before its own request), so move the stamp
    to the end of what was said."""
    if any(e.get("kind") == "spoken" for e in events):
        return events
    out = []
    for e in events:
        if e.get("kind") == "say" and e.get("source") in ("describe", "atmosphere", "narrator"):
            dur = max(0.6, len(str(e.get("text") or "").strip()) / demo_captions.CHARS_PER_S)
            e = {**e, "t": e["t"] + dur}
        out.append(e)
    return out


def load_take(folder: Path, name: str, top: bool) -> Take | None:
    manifest_path = folder / "takes.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config = manifest.get(name, {})
    video = folder / config.get("video", f"{name}.mp4")
    if config.get("reject") or not video.exists():
        return None
    duration = probe_s(video)
    stems = [video.with_suffix(x) for x in (".game.wav", ".voice.wav")]
    if any(x.exists() for x in stems):
        if not all(x.exists() for x in stems):
            raise ValueError(f"Both game and voice stems are required for {video}")
        if any(probe_s(x) < duration - 0.2 for x in stems):
            raise ValueError(f"Audio stem shorter than take: {video}")
    stamp = folder / config.get("start", f"{name}.mp4.start")
    start = float(stamp.read_text().split()[0]) if stamp.exists() else None
    log = folder / config.get("events", f"{name}.events.jsonl")
    events: list[dict] = []
    if start is not None and log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("kind") == "ready" or start - 30 <= e.get("t", 0) <= start + duration + 30:
                events.append(e)
    caps = demo_captions.captions(_old_narrations(events), start) if events else []
    caps = [c for c in caps if c[0] < duration]
    return Take(name, video, start, duration, events, caps, top)


def _vo(folder: Path, which: str) -> str:
    return next((str(p) for p in sorted(folder.glob(f"vo-{which}.*"))
                 if p.suffix in (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")), "")


def table(parts: list[Piece], actual: dict[str, float] | None = None) -> str:
    rows: dict[str, tuple[str, float, str]] = {}
    for p in parts:
        s = (actual or {}).get(p.id, p.out_s)
        sec, t, sp = rows.get(p.line, (p.section, 0.0, ""))
        if p.speed > 1 and f"{p.speed:g}x" not in sp:
            sp = f"{p.speed:g}x gaps"
        rows[p.line] = (sec, t + s, sp)
    sections: dict[str, float] = {}
    out, t0 = ["part          section   from    secs  speed"], 0.0
    for line, (sec, t, sp) in rows.items():
        out.append(f"{line:13} {sec:8} {int(t0 // 60)}:{t0 % 60:05.2f} {t:6.1f}  {sp}")
        t0 += t
        sections[sec] = sections.get(sec, 0.0) + t
    budgets = {"ct": f"about {CT_ABOUT:.0f}", "game2": "20-25", "game3": "10-15"}
    out.append("")
    for sec, t in sections.items():
        out.append(f"{sec:8} {t:6.1f} s" + (f"  (budget {budgets[sec]})" if sec in budgets else ""))
    flag = "" if t0 <= TOTAL_LIMIT else "  OVER"
    out.append(f"total    {t0:6.1f} s = {int(t0 // 60)}:{t0 % 60:04.1f}  (limit 3:00){flag}")
    return "\n".join(out)


def _picks(items: list[str]) -> dict[str, int]:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        if k not in LINES or not v.isdigit():
            sys.exit(f"--pick wants LINE=N, like A4=2: {item}")
        out[k] = int(v)
    return out


def _clips(items: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        a, _, b = v.partition("-")
        out[k] = (float(a), float(b))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path, help="the folder of takes")
    p.add_argument("--out", type=Path, help="the video (default: video.mp4 beside the folder)")
    p.add_argument("--cut", type=Path, help="a <out>.cut.json to build from, hand-edited or not")
    p.add_argument("--pick", action="append", default=[], help="which attempt: A4=2 (default last)")
    p.add_argument("--drop", default="", help="lines to leave out whatever the budget: B3,A5")
    p.add_argument("--clip", action="append", default=[],
                   help="the part of a game clip to use, seconds: game2=12.5-35")
    p.add_argument("--no-code", default="",
                   help=f"games that get the line {NO_CODE!r}. Only if it's true: game2,game3")
    p.add_argument("--rates", type=Path,
                   help="the rates card: its heading on the first line, a rate on each line after")
    p.add_argument("--top", default="take-B", help="takes with captions at the top")
    p.add_argument("--final", action="store_true",
                   help="refuse placeholders, voice-over labels and draft rates")
    p.add_argument("--accept-captions", action="store_true",
                   help="deprecated; never overrides stale captions or sped-up speech")
    p.add_argument("--vcodec", default="libx264", help="libx264, or h264_nvenc on a free NVIDIA")
    p.add_argument("--jobs", type=int, default=4, help="pieces encoded at once")
    p.add_argument("--sheet", action="store_true",
                   help="also write contact sheets: every 3 s, and every caption as it ends")
    p.add_argument("--dry-run", action="store_true", help="the plan and checks, no encoding")
    p.add_argument("--proxy", action="store_true", help="fast 960x540 / 30fps preview")
    a = p.parse_args()
    global PROXY, FPS
    PROXY = a.proxy
    if PROXY:
        if a.final:
            sys.exit("--final requires full resolution; remove --proxy")
        FPS = 30

    out = a.out or a.folder.parent / "video.mp4"
    tops = set(filter(None, a.top.split(",")))
    takes = {}
    for name in ("take-A", "take-B", *GAMES):
        take = load_take(a.folder, name, name in tops)
        if take is not None:
            takes[name] = take
    rates = ([ln.strip() for ln in a.rates.read_text().splitlines() if ln.strip()]
             if a.rates else DRAFT_RATES)
    no_code = set(filter(None, a.no_code.split(",")))
    if no_code - set(GAMES):
        sys.exit(f"--no-code takes {','.join(GAMES)}")
    warnings: list[str] = []
    dropped: list[str] = []
    if a.cut:
        parts = [_piece_from(d) for d in json.loads(a.cut.read_text())]
    else:
        vo = {w: v for w in ("why", "how") if (v := _vo(a.folder, w))}
        vo_s = {w: probe_s(Path(v)) for w, v in vo.items()}
        parts, dropped, warnings = make_plan(takes, _picks(a.pick), no_code, rates, vo, vo_s,
                                             _clips(a.clip), set(filter(None, a.drop.split(","))))
    # Legacy verbose lower thirds would cover the game; use the evidence rail instead.
    for piece in parts:
        if piece.kind == "clip":
            piece.overlays = [o for o in piece.overlays if o.text not in (LOWER_A4, LOWER_B) and o.where != "trace" and o.where != "draft"]
    evidence = evidence_overlays(parts, takes)
    if not a.final:
        for piece in parts:
            piece.overlays.append(Overlay("DRAFT / NOT SUBMISSION FOOTAGE", 20, "draft", 0, piece.out_s))
    for name in {p_.take for p_ in parts if p_.kind == "clip" and p_.captions}:
        take = takes.get(name)
        if take is None or take.start is None or not take.events or not take.caps:
            warnings.append(f"missing timing or captions: {name}")
    print(table(parts))
    if dropped:
        print("dropped for the budget:", ", ".join(dropped))
    print("end card:", next((b[0] for p_ in parts if p_.id == "c-end" for b in p_.blocks[-1:]), ""))
    for w in warnings:
        print("warning:", w)
    problems = []
    for name, take in takes.items():
        if take.start is None or not take.caps:
            continue
        shown = [p_ for p_ in parts if p_.take == name]
        for t, what in caption_problems(take.caps, speech_spans(take.events, take.start), shown):
            at = out_time(parts, name, t)
            when = f"{int(at // 60)}:{at % 60:04.1f}" if at is not None else "--"
            problems.append(f"{when}  {name} {t:6.1f}  {what}")
    print(f"caption check: {len(problems)} problem(s)")
    print("\n".join(problems))
    refused = final_refusals(parts, warnings) if a.final else []
    if a.final:  # a stale fact on screen is what this is all for
        refused += [x for x in problems if "  stale:" in x or "  speech sped up" in x]
    if refused:
        sys.exit("--final refuses:\n  " + "\n  ".join(refused))
    out.parent.mkdir(parents=True, exist_ok=True)
    cut_file = out.with_name(out.name + ".cut.json")
    out.with_suffix(".evidence.json").write_text(json.dumps(evidence, indent=2))
    cut_file.write_text(json.dumps([dataclasses.asdict(p_) for p_ in parts], indent=1))
    out.with_name(out.name + ".captions.txt").write_text("\n".join(problems) + "\n")
    print(f"cut -> {cut_file}")
    if a.dry_run:
        return
    build(parts, takes, out, a.vcodec, a.jobs)
    actual_s = probe_s(out)
    if a.final and actual_s > TOTAL_LIMIT:
        rejected = out.with_suffix(".rejected.mp4")
        out.rename(rejected)
        sys.exit(f"--final refuses encoded duration {actual_s:.3f}s; moved to {rejected}")
    print(f"{out}: {actual_s:.1f} s")
    if a.sheet:
        sheet = out.with_suffix(".sheet.png")
        contact_sheet(out, sheet, jobs=a.jobs)
        pages = caption_sheets(out, parts, takes, a.jobs)
        print(f"sheets -> {sheet}, {len(pages)} of captions: {pages[0] if pages else ''}")


if __name__ == "__main__":
    main()
