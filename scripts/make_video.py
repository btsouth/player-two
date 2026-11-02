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
import itertools
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
ALLOW_REJECTED = False
GAME_X, GAME_Y, GAME_W, GAME_H = 0, 0, 1920, 1080
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "video"
BG = (9, 15, 25, 255)
ACCENT = (120, 235, 214, 255)
CODE = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()  # a new layout redoes the pieces
FONT_DIRS = (ASSETS,)
BLUR = 45
TOTAL_LIMIT = 180.0
CT_ABOUT = 105.0
CT_LIMIT = 110.0  # "about 1:55"
GAME_BUDGET = {"game2": (20.0, 25.0), "game3": (10.0, 15.0)}
PLACEHOLDER_S = {"game2": 22.0, "game3": 12.0}
GAMES = ("game2", "game3")
NO_CODE = "No code written for this game. One line of notes."
LIMIT_LINE = "Creator-tested. No blind player has tried it yet. That's next."
SE_NOTICE = ("The copyright of the materials used on this page or video is owned by SQUARE ENIX CO., LTD. "
             "or its licensors. The unauthorized reproduction and distribution of the images is prohibited.\n"
             "© 1995, 2018 SQUARE ENIX\nIllustration: © 1995 BIRD STUDIO / SHUEISHA\n"
             "Story and Screenplay: © 1995, 2008 ARMOR PROJECT / SQUARE ENIX")
LEGAL_LINE = ("Games and footage belong to their publishers. Shown to demonstrate accessibility. "
              "No affiliation or endorsement.")

LINES = {
    "A1": ("A", r"^where am i\b"),
    "A2": ("A", r"^what'?s on the screen"),
    "A3": ("A", r"^take me to the fair"),
    "A3b": ("A", r"^what does it look like"),
    "A4": ("A", r"^buy three potions"),
    "A5": ("A", r"^take me back outside"),
    "B0": ("B", r"^how are we doing"),
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
    reveal_s: float = 1.0
    black_for: float = 0.0  # seconds of black, then a 1s reveal; audio remains continuous
    layout: str = ""  # map, fight, shop, black
    proof: str = ""  # path to saved atlas proof sidecar
    sharp_at: float | None = None  # take time the blur lifts; past b, it never does
    captions: bool = True
    vol: float = 1.0
    duck: bool = True
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
    edit: dict = field(default_factory=dict)

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
    own = request_lines(take.caps, heard_t, next_t, done,
                        3.0 if line=="A3b" else CHAIN[LINES[line][0]])
    left = 0.0
    if max_talk is not None:
        own, left = talk_cut(own, max_talk)
    end = (own[-1][1] if own else heard_t + 3.0) + 0.5
    if done is not None and left == 0:
        end = max(end, done + 1.2)  # let the completed action's real timing remain readable
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
        cut = cut_line(line, take, picks.get(line), 12.0 if line == "A3b" else max_talk if line == "A2" else None)
        if cut is None:
            warnings.append(f"{line}: not found in take-{name}'s log")
            continue
        pieces = line_pieces(cut, take, float(take.edit.get("speeds",{}).get(line,SPEED[name])))
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
        elif line == "A3b":
            if cut.left_out > 1:
                total=sum(p.out_s for p in pieces)
                overlays.append(Overlay("Description shortened",34,"trim",max(0,total-2.5),total))
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
    if name == "game3" and clip is None:
        # The point of Red Dead is the game speaking after narration stops.
        b = min(a+hi,take.duration)
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




def mh_story_pieces(take: Take) -> list[Piece]:
    """One continuous source run; selected true speech at 1x, other narration suppressed."""
    config=take.edit["story"]
    normal=[tuple(map(float,w)) for w in config["normal"]]
    speed=float(config.get("gap_speed",28))
    if speed<=1 or any(b<=a for a,b in normal) or any(a[1]>b[0] for a,b in itertools.pairwise(normal)):
        raise ValueError("MH story windows must be ordered, nonoverlapping and accelerated gaps labelled")
    for event in take.events:
        if event.get("kind")=="spoken" and "coming at you with technical" in event.get("text","").lower():
            a,b=event["began"]-take.start,event["ended"]-take.start
            if any(min(b,y)>max(a,x) for x,y in normal):
                raise ValueError("False MH line overlaps an audible story window")
    parts=[]
    for i,(a,b) in enumerate(normal):
        if i:
            gap=Piece(f"mh-gap-{i}","game2",line="MHS3",take=take.name,a=normal[i-1][1],b=a,
                      speed=speed,captions=False,layout="mh")
            gap.overlays=[Overlay(f"{speed:g}x",40,"speed",0,gap.out_s),
                          Overlay("Narration muted",34,"gap",0,gap.out_s)]
            parts.append(gap)
        p=Piece(f"mh-beat-{i}","game2",line="MHS3",take=take.name,a=a,b=b,layout="mh",
                duck=not any(min(b,y)>max(a,x) for x,y in config.get("unducked_windows",[])))
        if i==0:
            model=next((e for e in take.events if e.get("kind")=="mh" and e.get("name")=="mh_muse"
                        and e.get("read")=="wheel" and e["t"]-take.start<=a),None)
            if not model or not isinstance(model.get("seconds"),(float,int)):
                raise ValueError("MH story needs its real completed Muse wheel read")
            p.overlays=[Overlay(f"Muse Spark / wheel read {model['seconds']:.2f} s",34,"tl",0,min(4,b-a)),
                        Overlay("Monster Hunter Stories 3",34,"trim",0,min(2,b-a))]
            if b-a>4:
                p.overlays.append(Overlay("Custom combat code acts on the read",34,"tl",4,min(8,b-a)))
        elif i==1:
            p.overlays=[Overlay("Custom combat code acts on the read",34,"tl",0,min(3,b-a))]
        parts.append(p)
    return parts


def rdr_pieces(look: Take, dialogue: Take) -> list[Piece]:
    a,b=map(float,look.edit.get("clip",[18.4,28.1]))
    x,y=map(float,dialogue.edit.get("clip",[24,35]))
    vista=Piece("rdr-look","game3",line="RDR view",take=look.name,a=a,b=b,layout="rdr")
    vista.overlays += [Overlay("Red Dead Redemption",40,"tl",0,min(3,b-a)),
                       Overlay("Description shortened",34,"trim",max(0,b-a-2),b-a)]
    if look.edit.get("notes_only_verified"):
        vista.overlays.append(Overlay(NO_CODE,40,"tl2",3,min(8,b-a)))
    if dialogue.start is not None and any(e.get("kind")=="spoken" and
            min(y,e["ended"]-dialogue.start)>max(x,e["began"]-dialogue.start)
            for e in dialogue.events):
        raise ValueError("Red Dead quiet window contains logged Player Two speech")
    quiet=Piece("rdr-dialogue","game3",line="RDR voices",take=dialogue.name,a=x,b=y,
                layout="map",captions=False,duck=False)
    quiet.overlays.append(Overlay("Player Two stays quiet while the game's voices talk.",42,"bc",0,y-x))
    return [vista,quiet]


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


def atlas_beat(folder: Path) -> list[Piece]:
    path = folder / "atlas-proof.json"
    if not path.exists():
        return [
            Piece(
                "atlas-missing",
                "ct",
                kind="slate",
                secs=5,
                blocks=[("PLACEHOLDER: checked map-label response", 58, True)],
            )
        ]
    data = json.loads(path.read_text())
    raw = data["raw_place"]
    checked = data["checked_place"]
    if (
        raw["name"] != checked["name"]
        or not data.get("raw_sha256")
        or not data.get("checked_sha256")
    ):
        raise ValueError(
            "Atlas proof needs matching saved and checked destinations with source hashes"
        )
    words = "Muse labels the map. I check it. Local code navigates."
    return [
        Piece(
            "atlas-model",
            "ct",
            line="atlas proof",
            kind="card",
            secs=1.7,
            proof=str(path),
            blocks=[
                ("Muse Spark / saved response", 38, False),
                (
                    json.dumps(
                        {"name": raw["name"], "icon": raw["icon"]}, ensure_ascii=False
                    ),
                    48,
                    True,
                ),
                (words, 38, False),
            ],
        ),
        Piece(
            "atlas-check",
            "ct",
            line="atlas proof",
            kind="card",
            secs=1.6,
            proof=str(path),
            blocks=[
                ("Human checked", 38, False),
                (f"{checked['name']}  [{checked['x']}, {checked['y']}]", 64, True),
                (words, 38, False),
            ],
        ),
        Piece(
            "atlas-destination",
            "ct",
            line="atlas proof",
            kind="card",
            secs=1.7,
            proof=str(path),
            blocks=[
                ("A destination the local navigator can use", 38, False),
                (checked["name"], 88, True),
                (words, 38, False),
            ],
        ),
    ]


def cold_open(take: Take | None) -> tuple[list[Piece], list[str]]:
    """Fight narration, player choice, Cyclone reveal, then delegation: one real run."""
    if not take:
        return [Piece("cold-missing","hook",line="cold",kind="slate",secs=45,
                      blocks=[("PLACEHOLDER: continuous fight take",58,True)])], []
    doing=cut_line("B0",take,None,None)
    cyclone=cut_line("B2",take,None,None)
    win=cut_line("B4",take,None,None)
    if not doing or not cyclone or not win or not doing.heard<cyclone.heard<win.heard:
        return [Piece("cold-missing","hook",line="cold",kind="slate",secs=45,
                      blocks=[("PLACEHOLDER: health, Cyclone, then delegation",58,True)])], [
                          "cold open incomplete: health, Cyclone and Win must occur in that order"]
    anchor=str(take.edit.get("opening_line","Fight.")).strip().lower()
    fights=[c for c in take.caps if c[2]=="Player Two" and c[0]<doing.heard
            and c[3].removeprefix("Player Two: ").strip().lower()==anchor]
    warnings=[]
    if fights:
        a=max(0,fights[-1][0]-0.3)
    else:
        prior=[c[0] for c in take.caps if c[2]=="Player Two" and c[0]<doing.heard]
        a=max(0,min(prior,default=doing.a)-0.3)
        warnings.append(f"cold open incomplete: original opening speech record missing: {anchor}")
    reveal=take.edit.get("reveal_at")
    if not isinstance(reveal,(int,float)):
        reveal=cyclone.heard+2
        warnings.append("cold open incomplete: choose reveal_at from actual Cyclone frames")
    reveal_s=float(take.edit.get("reveal_duration",0.3))
    if not cyclone.heard<reveal<win.heard or not 0<reveal_s<=1:
        warnings.append("cold open incomplete: measured reveal must follow Cyclone and precede Win")
    first_end=min(float(take.edit.get("normal_until",reveal+2)),win.b)
    if first_end<reveal+reveal_s:
        raise ValueError("normal_until must include the entire Cyclone reveal")
    # All original speech remains normal speed. The manifest may tighten only silent gaps.
    spans=speech_spans(take.events,take.start)
    spans += [(t-_lead(take,t,e),t+1.8) for t,e in take.heard()]
    spans += [(reveal-0.1,first_end)]
    if "gap_speed" not in take.edit:
        spans.append((a,first_end))
    gap_speed=float(take.edit.get("gap_speed",3))
    gap_cap=take.edit.get("gap_max_output")
    cuts=split_speed(a,win.b,spans,gap_speed,
                     min_gap=float(take.edit.get("min_gap",2)),
                     pad=float(take.edit.get("speech_pad",0.3)))
    parts=[]
    for x,y,speed in cuts:
        # Keep both edges of an accelerated gap; removed middle silence is explicitly labelled.
        windows=[(x,y)]
        trimmed=gap_cap is not None and speed>1 and (y-x)/speed>float(gap_cap)
        if trimmed:
            edge=float(gap_cap)*speed/2
            windows=[(x,x+edge),(y-edge,y)]
        for lo,hi in windows:
            bounds=[lo]+[t for t in (reveal,first_end) if lo<t<hi]+[hi]
            for xx,yy in itertools.pairwise(bounds):
                before=xx<first_end
                p=Piece(f"cold-{len(parts)}","hook",line="cold",take=take.name,a=xx,b=yy,
                        speed=speed,black_for=(999 if yy<=reveal else max(0,reveal-xx)),
                        reveal_s=reveal_s,layout="black" if before else "fight")
                if speed>1:
                    p.overlays.append(Overlay(f"{speed:g}x",40,"speed",0,p.out_s))
                if trimmed:
                    p.overlays.append(Overlay("Silent gap shortened",34,"gap",0,p.out_s))
                parts.append(p)
    elapsed=0.0
    for p in parts:
        if elapsed<4.5:
            p.overlays.append(Overlay("This is the game with the picture off.",34,"intro",0,
                                      min(p.out_s,4.5-elapsed)))
        elapsed+=p.out_s
    return parts,warnings


def reserve_win_line(opening: list[Piece], take: Take | None) -> tuple[list[Piece], list[Piece]]:
    """The same run's earned result closes on black; never repeat or invent it."""
    if not take or take.start is None or not opening or any(p.kind != "clip" for p in opening):
        return opening, []
    win=cut_line("B4",take,None,None)
    if not win:
        return opening, []
    results=[e for e in take.events if e.get("kind")=="spoken"
             and e.get("text","").lower().startswith("won the fight")
             and win.heard <= e["began"]-take.start < win.b]
    if not results:
        return opening, []
    e=results[-1]
    a=max(0,e["began"]-take.start-0.05)
    b=min(take.duration,e["ended"]-take.start+0.12)
    kept=[]
    for p in opening:
        if p.kind!="clip" or p.a>=a:
            continue
        p.b=min(p.b,a)
        p.overlays=[dataclasses.replace(o,f1=min(o.f1,p.out_s)) for o in p.overlays
                    if o.f0<p.out_s]
        kept.append(p)
    ending=Piece("last-line","outro",line="last line",take=take.name,a=a,b=b,
                 black_for=999,layout="black")
    ending.overlays.append(Overlay("The same fight. The result.",34,"intro",0,2.5))
    return kept,[ending]



def goodbye_piece(take: Take) -> Piece:
    """Use a bounded, genuine last exchange; never fill it with a new synthetic voice."""
    request=next(((t,e) for t,e in take.heard() if re.fullmatch(r"goodbye[.!?]?",e["text"].strip(),re.IGNORECASE)),None)
    if request is None or take.start is None:
        raise ValueError("Goodbye take needs its real logged request")
    heard,event=request
    said=[e for e in take.events if e.get("kind")=="spoken" and e["began"]-take.start>=heard]
    joined=" ".join(e["text"] for e in said).lower()
    if not re.search(r"bye for now[.!]?\s+have fun[.!]?",joined):
        raise ValueError("Goodbye take needs real Bye for now and Have fun spoken records")
    end=next(e for e in said if "have fun" in e["text"].lower())
    return Piece("goodbye","outro",line="goodbye",take=take.name,a=max(0,heard-_lead(take,heard,event)),
                 b=min(take.duration,end["ended"]-take.start+.3),
                 black_for=0 if take.edit.get("field") else 999,
                 layout="fight" if take.edit.get("field") else "black")


def shop_scene_layouts(pieces: list[Piece], take: Take | None) -> list[Piece]:
    """Change only chip placement; keep source video and audio completely continuous."""
    menu=take.edit.get("shop_menu") if take else None
    if not menu:
        return pieces
    lo,hi=map(float,menu)
    for p in pieces:
        if p.kind!="clip" or p.line!="A4":
            continue
        a,b=(lo-p.a)/p.speed,(hi-p.a)/p.speed
        placed=[]
        for o in p.overlays:
            if o.where!="trace":
                placed.append(o)
                continue
            bounds=[o.f0]+[t for t in (a,b) if o.f0<t<o.f1]+[o.f1]
            for x,y in itertools.pairwise(bounds):
                where="trace-shop-menu" if a<=(x+y)/2<b else "trace-shop"
                placed.append(dataclasses.replace(o,f0=x,f1=y,where=where))
        p.overlays=placed
    return pieces


def make_plan(
    takes: dict[str, Take],
    picks: dict[str, int],
    no_code: set[str],
    rates: list[str],
    vo: dict[str, str],
    vo_s: dict[str, float],
    clips: dict[str, tuple[float, float]],
    force_drop: set[str] = frozenset(),
) -> tuple[list[Piece], list[str], list[str]]:
    del rates  # the film ends with a human line, not a rates wall
    folder = Path(vo.get("_folder", "."))
    opening, warnings = cold_open(takes.get("take-B"))
    # Each request appears once; the same fight's real result is reserved for the ending.
    drop = set(force_drop) | {"B0", "B1", "B2", "B3", "B4"}
    lines = ct_pieces(takes, picks, drop, 8, warnings)
    maps = []
    for line in ("A1", "A2", "A3", "A3b", "A4", "A5"):
        if line == "A3":
            maps += atlas_beat(folder)
        maps += lines.get(line, [])
    for p in maps:
        p.sharp_at = None
        if p.line == "A4":
            p.layout = "shop"
    sam = takes.get("sam")
    if sam:
        # Caller selects a continuous excerpt; never synthesize a model result or stereo ping.
        aa, bb = clips.get("sam", (0, min(sam.duration, 9)))
        sam_parts = [
            Piece("sam-demo", "ct", line="sam", take="sam", a=aa, b=bb, layout="map")
        ]
        if not any(e.get("via") == "sam-3.1" and sam.start is not None
                   and aa <= e.get("t", 0) - sam.start < bb
                   and any(valid_box(box) for box in e.get("boxes", [])) for e in sam.events):
            warnings.append("missing SAM geometry: no via=sam-3.1 boxes in this take")
    else:
        sam_parts = []
    games = [
        Piece(
            "c-games",
            "games",
            kind="card",
            secs=2,
            blocks=[("It’s not one game.", 72, True)],
        )
    ]
    mh=takes.get("game2")
    games += mh_story_pieces(mh) if mh and mh.edit.get("story") else game_pieces("game2",mh,"game2" in no_code,clips.get("game2"),warnings)
    if takes.get("rdr-look") and takes.get("rdr-dialogue"):
        games += rdr_pieces(takes["rdr-look"], takes["rdr-dialogue"])
    else:
        games += game_pieces("game3", takes.get("game3"), "game3" in no_code, clips.get("game3"), warnings)
    opening,reserved_ending=reserve_win_line(opening,takes.get("take-B"))
    end_parts=reserved_ending or [Piece("win-missing","outro",kind="slate",secs=3.5,
                    blocks=[("PLACEHOLDER: real win line",48,True)])]
    goodbye=takes.get("goodbye")
    if goodbye:
        farewell=goodbye_piece(goodbye)
        fight=takes.get("take-B")
        if (reserved_ending and goodbye.edit.get("field") and fight and
                fight.video.resolve()==goodbye.video.resolve() and fight.start==goodbye.start):
            end_parts=[dataclasses.replace(reserved_ending[0],id="closing-exchange",b=farewell.b,
                                          black_for=0,layout="fight")]
        else:
            end_parts.append(farewell)
    else:
        end_parts.append(Piece("goodbye-missing","outro",line="goodbye",kind="slate",secs=3.5,
                blocks=[("ENDING DESIGN / clean take pending",34,False),
                        ("You: Goodbye.",64,True),
                        ("Player Two: Bye for now. Have fun.",64,True)]))
    how = Piece(
        "c-how",
        "outro",
        kind="card",
        secs=max(5.5, vo_s.get("how", 0) + 0.3),
        vo=vo.get("how", ""),
        blocks=[
            ("Muse understands scenes and open-ended requests.", 56, True),
            ("Local readers and routines handle common actions.", 42, False),
        ],
    )
    parts = [
        *opening,
        *round2_cards(folder),
        *maps,
        *sam_parts,
        *games,
        how,
        Piece("limit","outro",kind="card",secs=3.2,blocks=[(LIMIT_LINE,58,True)]),
        *end_parts,
    ]
    # Cut whole optional requests, never the payoff of one already shown.
    dropped = []
    for line in ("A2", "A5", "A1"):
        if sum(p.out_s for p in parts) + 8 <= TOTAL_LIMIT - 1.0:
            break
        parts = [p for p in parts if p.line != line]
        dropped.append(line)
    parts.append(
        Piece(
            "c-end",
            "outro",
            kind="card",
            secs=8,
            blocks=[
                ("Player Two", 96, True),
                ("You choose. We play.", 56, True),
                ("github.com/btsouth/player-two", 40, False),
                (
                    requests_wording(
                        _shown_requests(takes, parts), any(p.speed > 1 for p in parts)
                    ),
                    30,
                    False,
                ),
                (LEGAL_LINE,24,False),
                (SE_NOTICE,22,False),
            ],
        )
    )
    return parts, dropped, warnings


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
        if p.id == "c-how" and not p.vo:
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
    return out + [w for w in warnings if "not found" in w or "no take-" in w or "missing timing" in w or "cold open incomplete" in w or "missing SAM geometry" in w]


def round2_cards(folder: Path) -> list[Piece]:
    ahead_vo = _vo(folder, "ahead")
    ahead_s = max(2.5, probe_s(Path(ahead_vo)) + 0.25) if ahead_vo else 2.5
    cards = [
        Piece(
            "identity",
            "hook",
            kind="card",
            secs=2,
            blocks=[
                ("Player Two", 96, True),
                ("An AI co-op partner for blind and motor-impaired gamers.", 48, False),
            ],
        ),
        Piece(
            "lavos",
            "hook",
            kind="card",
            secs=4,
            blocks=[
                (
                    "“A blind person still cannot ... nor defeat Lavos in Chrono Trigger...”",
                    58,
                    True,
                ),
                ("RetroArch 1.8.2 release post, December 2019", 36, False),
                ("Lavos is still ahead. Here’s how far we’ve got.",42,True),
            ],
        ),
        Piece(
            "ahead",
            "hook",
            kind="card",
            secs=ahead_s,
            vo=ahead_vo,
            blocks=[("Lavos is still ahead. Here’s how far we’ve got.", 64, True)],
        ),
    ]
    return cards if ahead_vo else cards[:2]


def sample_plan(takes: dict[str, Take], folder: Path) -> tuple[list[Piece], list[str]]:
    take = takes.get("take-B")
    if not take:
        raise ValueError("The style sample needs the old fight source")
    cut = cut_line("B4", take, None, None)
    if cut is None:
        raise ValueError("No real win-this-fight request")
    # Old questions are 65s apart. Use ONE continuous segment, never splice a fake exchange.
    cold = Piece(
        "cold-sample",
        "hook",
        line="cold",
        take="take-B",
        a=cut.a,
        b=cut.a + 17.5,
        black_for=13,
        layout="black",
    )
    cold.overlays.append(
        Overlay("This is the game with the picture off.", 34, "intro", 0, 4.5)
    )
    a = takes["take-A"]
    map_cut = cut_line("A3", a, None, None)
    shop_cut = cut_line("A4", a, None, None)
    if map_cut is None or shop_cut is None:
        raise ValueError("Need actual fair and shop requests")
    map_p = Piece(
        "map-sample",
        "ct",
        line="map",
        take="take-A",
        a=map_cut.a,
        b=map_cut.b,
        layout="map",
    )
    cold.b = cold.a + (
        45 - sum(p.out_s for p in round2_cards(folder)) - map_p.out_s - 11
    )
    if cold.out_s < 14:
        raise ValueError(
            "Map request too long for the 45s sample; choose another real request"
        )
    # A continuous 11s shop excerpt, including actual menus and the spoken receipt.
    shop_p = Piece(
        "shop-sample",
        "ct",
        line="shop",
        take="take-A",
        a=shop_cut.b - 11,
        b=shop_cut.b,
        layout="shop",
    )
    return [cold, *round2_cards(folder), map_p, shop_p], [
        "cold open incomplete: old What-are-we-fighting and Win requests are 65s apart; sample uses Win only"
    ]


# ---------- pictures ----------

def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    name = "Inter-SemiBold.ttf" if bold else "Inter-Regular.ttf"
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


def card_png(path: Path, blocks: list, bg=BG, end: bool = False) -> Path:
    img = Image.new("RGBA", (W, H), tuple(bg))
    d = ImageDraw.Draw(img)
    if end:
        positions=[160,290,410,560,720,820]
        for (text,size,bold),y in zip(blocks,positions,strict=True):
            f=_font(bold,size)
            for paragraph in text.splitlines():
                for ln in _wrap(d,paragraph,f,W-192):
                    d.text((96,y),ln,font=f,fill="white" if size>30 else (176,190,202))
                    y+=int(size*1.35)
        img.save(path)
        return path
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
        d.text(
            (W - 64, 20),
            o.text,
            anchor="ra",
            font=_font(True, 20),
            fill=(200, 166, 190),
        )
    elif o.where.startswith("trace"):
        lines = o.text.split("\n")
        # Request chip and measured result are independent, floating surfaces.
        req = lines[1] if len(lines) > 1 else lines[0]
        typed = "typed" in lines[0]
        top = o.where in ("trace-fight", "trace-shop", "trace-black")
        x, y = 64, (200 if o.where == "trace-black" else 60 if top else 700)
        if o.where=="trace-shop-menu":
            x,y=1000,550
        if o.where=="trace-rdr":
            y=520
        black=o.where=="trace-black"
        f = _font(True, 64 if black else 42)
        text = ("You (typed): " if typed else "You: ") + req
        wrapped = _wrap(d, text, f, 1000 if o.where=="trace-shop-menu" else 1450)
        if black:
            x=(W-int(max(d.textlength(t,font=f) for t in wrapped))-108)//2
            y=250
        width = int(max(d.textlength(t, font=f) for t in wrapped)) + 108
        line_h=84 if black else 56
        height = len(wrapped) * line_h + 32
        d.rounded_rectangle(
            (x, y, x + width, y + height), radius=20, fill=(9, 15, 25, 220)
        )
        # Draw the microphone rather than depending on a font's emoji coverage.
        mx, my = x + 32, y + 27
        d.rounded_rectangle(
            (mx, my, mx + 13, my + 22), radius=6, outline=ACCENT, width=3
        )
        d.arc((mx - 5, my + 6, mx + 18, my + 30), 0, 180, fill=ACCENT, width=3)
        d.line((mx + 6, my + 30, mx + 6, my + 37), fill=ACCENT, width=3)
        for i, ln in enumerate(wrapped):
            d.text((x + 66, y + 16 + i * line_h), ln, font=f, fill=ACCENT if black else "white")
        by = y + height + 14
        # Latest results, retaining scope labels so action and inference aren't conflated.
        bx = x
        pairs=list(zip(lines[2::2], lines[3::2], strict=False))[-2:]
        if o.where=="trace-black":
            pairs=[]  # Black opening is the game heard through dialogue, not a dashboard.
        for layer, timing in pairs:
            label = layer.replace("Local action", "Local").replace(
                " / scene", " / scene"
            )
            txt = f"{label}  {timing}"
            bf = _font(True, 34)
            bw = int(d.textlength(txt, font=bf)) + 44
            d.rounded_rectangle(
                (bx, by, bx + bw, by + 58), radius=16, fill=(9, 15, 25, 225)
            )
            d.text(
                (bx + 22, by + 9),
                txt,
                font=bf,
                fill=(255, 206, 124) if layer == "Local action" else ACCENT,
            )
            bx += bw + 16
        img.save(path)
        return path
    elif o.where == "box":
        # Coordinates are normalized source-frame xyxy, supplied by the SAM event itself.
        x0, y0, x1, y1 = json.loads(o.text)
        d.rectangle((x0 * W, y0 * H, x1 * W, y1 * H), outline=ACCENT, width=6)
    else:
        f = _font(True, max(34, o.size))
        lines = _wrap(d, o.text, f, 1600)
        is_speed=bool(re.fullmatch(r"\d+(?:\.\d+)?x",o.text))
        if is_speed:
            lines = [o.text + " SPEED"]
        y = 52 if o.where in ("tl", "tl2", "tr", "trl", "intro") else 700
        if o.where == "intro":
            y = 66
        if o.where == "gap":
            y = 150
        if is_speed or o.where in ("speed","trim"):
            y = 64
        if o.where=="speed-shop":
            y = 700
        tw = int(max(d.textlength(ln, font=f) for ln in lines))
        x = ((W-tw)//2 if o.where in ("bc","intro") else W-tw-64 if is_speed or o.where in ("speed","trim") else 64)
        bh = len(lines) * int(f.size * 1.35) + 28
        d.rounded_rectangle(
            (x - 20, y - 12, x + tw + 20, y + bh), radius=18, fill=(9, 15, 25, 218)
        )
        for i, ln in enumerate(lines):
            d.text((x, y + i * int(f.size * 1.35)), ln, font=f, fill="white")
    img.save(path)
    return path


def _evidence_for_window(parts: list[Piece], takes: dict[str, Take]) -> list[dict]:
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
                        x.get("tool") in {"go_to_place", "go_to_exit", "battle", "fight", "shop", "where_am_i"}
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


def evidence_overlays(parts: list[Piece], takes: dict[str, Take]) -> list[dict]:
    """Handle every request in a continuous clip, including the paired black opening."""
    report = []
    for p in parts:
        take = takes.get(p.take)
        if not take or p.kind != "clip" or not p.captions or p.id=="last-line" or (p.take=="game2" and take.edit.get("story")):
            continue
        heard = take.heard()
        before = [t for t, _ in heard if t <= p.a]
        inside = [t for t, _ in heard if p.a < t < p.b]
        starts = (
            [p.a] if before and (not inside or p.a - before[-1] < 15) else []
        ) + inside
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else p.b
            virtual = dataclasses.replace(p, a=start, b=end, overlays=[])
            found = _evidence_for_window([virtual], takes)
            offset = (start - p.a) / p.speed
            p.overlays.extend(
                dataclasses.replace(o, f0=o.f0 + offset, f1=o.f1 + offset)
                for o in virtual.overlays
            )
            for row in found:
                row["badge_from"] += offset
            if p.layout != "black":
                report += found
    return report


def valid_box(box) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in box)
            and box[0] < box[2] and box[1] < box[3])


def sam_boxes(parts: list[Piece], takes: dict[str, Take]) -> None:
    for p in parts:
        take = takes.get(p.take)
        if p.kind != "clip" or not take or take.start is None:
            continue
        for e in take.events:
            if e.get("kind") != "where" or e.get("via") != "sam-3.1":
                continue
            for box in e.get("boxes", []):
                if not valid_box(box):
                    continue
                t = (e["t"] - take.start - p.a) / p.speed
                # Never pin a box across a changing scene or before the result arrived.
                if 0 <= t < p.out_s:
                    p.overlays.append(
                        Overlay(json.dumps(box), 34, "box", t, min(t + 1.5, p.out_s))
                    )


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
        if p.layout in ("shop","shop-menu") and re.fullmatch(r"\d+(?:\.\d+)?x",o.text):
            o=dataclasses.replace(o,where="speed-shop")
        if o.where == "trace":
            o = dataclasses.replace(o, where="trace-" + (p.layout or ("fight" if p.take=="take-B" else "map")))
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
    bg = (0,0,0,255) if p.id=="goodbye-missing" else BG if p.kind == "card" else (32,25,43,255)
    png = card_png(work / f"{p.id}.png", p.blocks, bg, end=p.id=="c-end")
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
    v = f"[0:v]scale={GAME_W}:{GAME_H}:flags=lanczos,fps={FPS},setpts=PTS-STARTPTS"
    if p.black_for or (p.line=="cold" and p.layout=="black"):
        v += f",fade=t=in:st={p.black_for}:d={p.reveal_s}"
    if p.sharp_at is not None:
        if p.sharp_at >= p.b:
            v += f",gblur=sigma={BLUR}"
        else:
            t0 = p.sharp_at - p.a
            k = f"clip((T-{t0:.2f})/1.5,0,1)"
            v += (f",split[s][b];[b]gblur=sigma={BLUR}[bl];"
                  f"[s][bl]blend=all_expr='A*{k}+B*(1-{k})'")
    v += f",pad={W}:{H}:{GAME_X}:{GAME_Y}:color=0x090f19,setsar=1"
    ass_key = p.take + ":" + (p.layout or ("fight" if take.top else "map"))
    if p.captions and ass_key in asses:
        v += (f",setpts=PTS+{p.a}/TB,subtitles='{_esc(asses[ass_key])}':"
              f"fontsdir='{_esc(ASSETS)}',setpts=PTS-STARTPTS")
    v += f",setpts=PTS/{p.speed:g}[v0]"
    tempo = _tempo(p.speed)
    dur = p.out_s
    # Accelerated gaps are silent: no sped-up voices or frantic game music.
    gain = 0 if p.speed > 1 else p.vol
    if p.speed>1:
        # Very short accelerated gaps can exhaust atempo before AAC receives a frame.
        # They are silent by design, so generate exact-length silence without time-stretching.
        a=f"anullsrc=r=48000:cl=stereo,atrim=duration={dur},asetpts=PTS-STARTPTS"
    elif stems and p.duck:
        # Voice stem contains Player Two + the microphone; the game stem contains only game.
        a = (f"[{stem_n+1}:a]asetpts=PTS-STARTPTS,{tempo}asplit[voice][side];"
             f"[{stem_n}:a]asetpts=PTS-STARTPTS,{tempo}anull"
             "[game];[game][side]sidechaincompress=threshold=0.025:ratio=8:"
             "attack=15:release=220[duck];[duck][voice]amix=inputs=2:normalize=0:"
             f"duration=first,volume={gain}")
    elif stems:
        # Preserve both stems untouched by the sidechain, including any unexpected voice.
        a = (f"[{stem_n}:a]asetpts=PTS-STARTPTS,{tempo}anull[game];"
             f"[{stem_n+1}:a]asetpts=PTS-STARTPTS,{tempo}anull[voice];"
             f"[game][voice]amix=inputs=2:normalize=0:duration=first,volume={gain}")
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


def ass_text(take: Take, layout: str = "") -> str:
    """Full-bleed captions, positioned by scene; black opening gets generous central type."""
    layout = layout or ("fight" if take.top else "map")
    size, align, ml, mr, mv = {
        "map": (48, 2, 100, 100, 60),
        "rdr": (48, 2, 100, 100, 60),
        "mh": (48, 8, 100, 100, 740),
        "fight": (48, 8, 100, 100, 240),
        "shop": (48, 9, 1000, 64, 420),
        "black": (80, 5, 200, 200, 30),
    }[layout]
    head = demo_captions.ASS_HEAD.replace("PlayResX: 512", "PlayResX: 1920").replace(
        "PlayResY: 288", "PlayResY: 1080"
    )
    styles = head.index("Style: P2,")
    events = head.index("[Events]")
    head = (
        head[:styles]
        + (
            f"Style: P2,Inter,{size},&H00FFFFFF,&H00FFFFFF,"
            f"&H00190F09,&H99190F09,0,0,0,0,100,100,0,0,3,2,0,{align},{ml},{mr},{mv},1\n\n"
        )
        + head[events:]
    )
    rows = []
    for a, b, who, txt in take.caps:
        if who != "Player Two":
            continue
        txt = txt.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
        if not txt.startswith("Player Two:"):
            txt = "Player Two: " + txt
        rows.append(
            f"Dialogue: 0,{demo_captions._ass_stamp(a)},"
            f"{demo_captions._ass_stamp(b)},P2,,0,0,0,,"
            + (r"{\fad(90,70)\1c&HFFFFFF&}" if layout != "black" else
               r"{\fad(90,70)\1c&HD7EFFF&}")
            + txt.replace(chr(10), " ")
        )
    return head + "\n".join(rows) + "\n"


def build(parts: list[Piece], takes: dict[str, Take], out: Path, vcodec: str, jobs: int) -> None:
    work = out.parent / f"{out.stem}-work"
    work.mkdir(parents=True, exist_ok=True)
    asses: dict[str, Path] = {}
    for name, take in takes.items():
        if take.caps:
            for layout in ("map","fight","shop","black","rdr","mh"):
                text=ass_text(take,layout)
                path=work/f"{name}-{layout}.ass"
                path.write_text(text,encoding="utf-8")
                asses[name+":"+layout]=path

    def one(p: Piece) -> Path:
        dest = work / f"{p.id}.mp4"
        key = {**dataclasses.asdict(p), "code": CODE, "proxy": PROXY, "codec": vcodec,
               "font": hashlib.sha1((ASSETS/"Inter-Regular.ttf").read_bytes()).hexdigest()}
        sources = [takes[p.take].video] if p.kind == "clip" else []
        if p.kind == "clip":
            sources += [f for suffix in (".game.wav", ".voice.wav")
                        if (f := takes[p.take].video.with_suffix(suffix)).exists()]
        if p.vo:
            sources.append(Path(p.vo))
        if p.proof:
            sources.append(Path(p.proof))
        key["sources"] = [(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in sources]
        if p.kind == "clip":
            ak = p.take+":"+(p.layout or ("fight" if takes[p.take].top else "map"))
            if ak in asses:
                key["ass"] = hashlib.sha1(asses[ak].read_bytes()).hexdigest()
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
    """Two-pass EBU R128. -4 dBTP leaves headroom for AAC reconstruction transients."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(joined), "-vn", "-af",
        "loudnorm=I=-14:TP=-4:LRA=9:print_format=json", "-f", "null", "-"],
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
    af = ("loudnorm=I=-14:TP=-4:LRA=9:linear=true:"
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
                if not any(p.kind=="clip" and p.captions and p.take==name and p.a<=t<p.b for p in parts):
                    continue
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
    if (config.get("reject") and not ALLOW_REJECTED) or not video.exists():
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
    if start is not None:
        # GSR begins after the shell writes its timestamp. Measure against the voice stem.
        start += float(config.get("edit", {}).get("recording_delay", 0.0))
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
    return Take(name, video, start, duration, events, caps, top, config.get("edit",{}))


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
    p.add_argument("--sample", action="store_true", help="45-second Round 2 style sample")
    p.add_argument("--proxy", action="store_true", help="fast 960x540 / 30fps preview")
    a = p.parse_args()
    global PROXY, FPS, ALLOW_REJECTED
    PROXY = a.proxy
    ALLOW_REJECTED = a.sample
    if a.sample and a.final:
        sys.exit("--sample cannot be --final")
    if PROXY:
        if a.final:
            sys.exit("--final requires full resolution; remove --proxy")
        FPS = 30

    out = a.out or a.folder.parent / "video.mp4"
    tops = set(filter(None, a.top.split(",")))
    takes = {}
    for name in ("take-A", "take-B", "sam", "ending", "goodbye", "rdr-look", "rdr-dialogue", *GAMES):
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
    manifest_path=a.folder/"takes.json"
    edit=json.loads(manifest_path.read_text()).get("edit",{}) if manifest_path.exists() else {}
    if a.cut:
        parts = [_piece_from(d) for d in json.loads(a.cut.read_text())]
    else:
        vo = {w: v for w in ("why", "how") if (v := _vo(a.folder, w))}
        vo_s = {w: probe_s(Path(v)) for w, v in vo.items()}
        vo["_folder"]=str(a.folder)
        parts, dropped, warnings = make_plan(takes, _picks(a.pick), no_code, rates, vo, vo_s,
                                             _clips(a.clip), set(filter(None, a.drop.split(","))) | set(edit.get("drop",[])))
    # Remove legacy overlays before rebuilding the floating chips from source evidence.
    for piece in parts:
        if piece.kind == "clip":
            piece.overlays = [o for o in piece.overlays if o.text not in (LOWER_A4, LOWER_B) and o.where != "trace" and o.where != "draft"]
    for piece in parts:
        if piece.kind == "clip" and not piece.layout:
            piece.layout = "fight" if takes[piece.take].top else "map"
        if piece.line=="A4" and piece.layout not in ("shop","shop-menu"):
            piece.layout="shop"
    evidence = evidence_overlays(parts, takes)
    sam_boxes(parts,takes)
    parts=shop_scene_layouts(parts,takes.get("take-A"))
    if not a.final:
        for piece in parts:
            piece.overlays.append(Overlay("DRAFT / NOT SUBMISSION FOOTAGE", 20, "draft", 0, piece.out_s))
    for name in {p_.take for p_ in parts if p_.kind == "clip" and p_.captions}:
        take = takes.get(name)
        if take is None or take.start is None or not take.events or not take.caps:
            warnings.append(f"missing timing or captions: {name}")
    if a.sample:
        parts,warnings = sample_plan(takes,a.folder)
        evidence=evidence_overlays(parts,takes)
        for piece in parts:
            piece.overlays.append(Overlay("STYLE SAMPLE / OLD TYPED TAKE / FINAL TAKE PENDING",20,"draft",0,piece.out_s))
    for piece in parts:
        if not piece.line:
            piece.line=piece.id
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
