#!/usr/bin/env python3
"""The Player Two demo video, cut from a folder of takes and their event logs.

    python scripts/make_video.py TAKES --out video.mp4 [--no-code game2,game3] [--rates rates.txt]
    python scripts/make_video.py TAKES --out video.mp4 --cut video.mp4.cut.json   # hand edits
    python scripts/make_video.py TAKES --out video.mp4 --final                    # no placeholders

TAKES holds take-A.mp4, take-B.mp4, game2.mp4 and rdr-sam.mp4 (symlinks are fine), each with
<name>.mp4.start (demo_record.sh writes it) and <name>.events.jsonl (the session log; only the
video's time range is used), plus nav/goodbye takes and optional vo-intro/how/limit audio. A missing
game clip becomes a labelled placeholder. No footage belongs in the repo.

Every request is found in its take's log by the run sheet's words. Its window runs from the
request to its last line. Speech stays at normal speed; only the gaps between lines are sped up,
always with a label. Captions come from demo_captions (the voice's "spoken" record when the log
has one) and are burned once, per piece, from the raw take. The resolved cut goes to
<out>.cut.json; edit it and pass it back with --cut. Pieces whose settings haven't changed are
not encoded again.

The opening fight pays off before the audience title and historical quote. Navigation, scene
description and shopping lead into the general/specific explanation and other games. The limit,
real goodbye and end card close the film. The manifest keeps source-specific timings editable;
final mode refuses unsupported evidence or a film over 3:00 rather than cutting a spoken payoff.
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
INTRO_VO = "Player Two. For players who can't see the screen or can't hold a controller. You ask. It reads, answers and presses the buttons."
LIMIT_VO = "I've tested this myself. No blind player has tried it yet. That's next."
GENERAL_LINE = "Red Dead has no code at all, just Muse looking at the picture. Chrono Trigger and Monster Hunter add code written for them."
HOW_SECONDS = 9.0
MAP_LINE = "Muse labelled this map. I checked it. Local code walks it."
RETRO_QUOTE = "A blind person still cannot ... nor defeat Lavos in Chrono Trigger ..."
RETRO_SOURCE = "Devin Prater, blind accessibility instructor, for libretro, 2019"
FRIEND_QUOTE = "In the end, I had to give up as I couldn’t keep calling my friend."
FRIEND_SOURCE = "Gamer with sight loss, RNIB Accessible Gaming Research, 2022"
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
    audio_mode: str = "mix"
    voice_boosts: list = field(default_factory=list)
    game_ducks: list = field(default_factory=list)
    stand_in: bool = False
    box_until: float = 0.0
    result_at: float | None = None
    hold_after: float = 0.0
    picture_until: float | None = None
    dip: bool = False
    hold_at: float | None = None  # explicit held purchase result; original speech clock is unchanged
    freeze: bool = False
    timing_group: str = ""
    vo_prompt: str = ""
    vo_offset: float = 0.0  # continuous founder narration across the bridge shots
    fade_in: bool = False
    fade_out: bool = False
    vo: str = ""
    blocks: list = field(default_factory=list)  # a card's (text, size, bold)
    secs: float = 0.0
    overlays: list = field(default_factory=list)

    @property
    def out_s(self) -> float:
        return self.secs if self.kind != "clip" else (self.b - self.a) / self.speed + self.hold_after


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
    before it's logged. Prefer measured hands-free audio start: the listening
    status is set only after the wake name has already been spoken."""
    if event.get("typed") or take.start is None:
        return 0.4
    started = event.get("utterance_audio_started_at", event.get("utterance_started_at"))
    if isinstance(started, (int, float)) and heard_t - 20 <= started - take.start <= heard_t:
        return min(heard_t - (started - take.start) + 0.3, 20.0)
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
    end = (own[-1][1] if own else heard_t + 3.0) + (0 if left else 0.5)
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
        cut = cut_line(line, take, picks.get(line), float(take.edit.get('description_seconds',12)) if line == "A3b" else max_talk if line == "A2" else None)
        if cut is None:
            warnings.append(f"{line}: not found in take-{name}'s log")
            continue
        pieces = line_pieces(cut, take, float(take.edit.get("speeds",{}).get(line,SPEED[name])))
        overlays: list[Overlay] = []
        if line == "A1":
            pass
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
            pass
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
    if config.get('request_intro'):
        a,b,speed_intro=map(float,config['request_intro'])
        if b!=normal[0][0] or speed_intro<=1:
            raise ValueError('MH request intro must meet the first read and mark accelerated muted narration')
        p=Piece('mh-request','game2',line='MHS3',take=take.name,a=a,b=b,speed=speed_intro,captions=False,layout='mh')
        req=next(((t,e) for t,e in take.heard() if a<=t<b),None)
        if not req:raise ValueError('MH intro needs its original request')
        t,e=req
        p.overlays=[Overlay(('You: typed' if e.get('typed') else 'You: heard')+'\n'+e['text'],42,'trace-mh',(t-a)/speed_intro,p.out_s)]
        parts.append(p)
    models=[e for e in take.events if e.get("kind")=="mh" and e.get("name")=="mh_muse"
            and e.get("read")=="wheel" and isinstance(e.get("seconds"),(float,int))
            and normal[0][0] <= e["t"]-take.start <= normal[-1][1]]
    # Older reviewed cuts begin just after completion; retain that original record.
    if not models:
        models=[e for e in take.events if e.get("kind")=="mh" and e.get("name")=="mh_muse"
                and e.get("read")=="wheel" and isinstance(e.get("seconds"),(float,int))
                and e["t"]-take.start<=normal[0][0]][-1:]
    if not models:
        raise ValueError("MH story needs its real completed Muse weak-point read")
    for i,(a,b) in enumerate(normal):
        if i and not config.get('cuts_only'):
            parts.append(Piece(f"mh-gap-{i}","game2",line="MHS3",take=take.name,a=normal[i-1][1],b=a,
                               speed=speed,captions=False,layout="mh"))
        p=Piece(f"mh-beat-{i}","game2",line="MHS3",take=take.name,a=a,b=b,layout="mh",
                duck=not any(min(b,y)>max(a,x) for x,y in config.get("unducked_windows",[])))
        # The opening chip is the actual logged request, not a fabricated command.
        for t,e in take.heard():
            if a<=t<b:
                p.overlays.append(Overlay(("You: typed" if e.get("typed") else "You: heard")+"\n"+e["text"],42,"trace-mh",max(0,t-a),b-a))
        if config.get('carry_requests') and not any(o.where=='trace-mh' for o in p.overlays):
            prior=[(t,e) for t,e in take.heard() if t<a]
            if prior:
                _,e=prior[-1]
                if not e.get('typed'):raise ValueError('Keep the real microphone request in a spoken take')
                p.overlays.append(Overlay('You: typed\n'+e['text'],42,'trace-mh',0,b-a))
        model=next((e for e in reversed(take.events) if e.get('kind')=='mh' and e.get('name')=='mh_muse'
                    and e.get('read') in ('status','wheel') and isinstance(e.get('seconds'),(int,float))
                    and a-1<=e['t']-take.start<b),None)
        if model:
            at=max(0,model['t']-take.start-a)
            request=next((t for t,_ in reversed(take.heard()) if t<=model['t']-take.start),None)
            response=event_answer(take,request,model) if request is not None else None
            words='Muse read the weak points' if model['read']=='wheel' else 'Muse Spark'
            if response:
                at=max(at,response[0]-a)
                words+=f' · Answer {response[1]:.1f} s'
            p.overlays.append(Overlay(words,34,'mh-model',at,min(at+4,b-a)))
            if model['read']=='wheel':p.overlays.append(Overlay('Harp-shaped icon = piercing damage',34,'mh-gloss',at,min(at+4,b-a)))
        if b-a>6:
            start=min(4,max(0,b-a-4))
            p.overlays.append(Overlay("Muse reads weak points. Code plays the turns.",34,"mh-code",start,start+4))
        for c,d,who,text in take.caps:
            if who=="Player Two" and "Speed beats that" in text and min(b,d)>max(a,c):
                p.overlays.append(Overlay("Speed beats Power in this game's attack triangle.",34,"mh-gloss",max(0,c-a),min(b-a,d-a+.8)))
        parts.append(p)
    if not any(o.where=='mh-code' for p in parts for o in p.overlays):
        at=float(config.get('code_chip_at',max(0,sum(p.out_s for p in parts)-8)))
        place_overlays(parts,[Overlay('Muse reads weak points. Code plays the turns.',34,'mh-code',at,at+4)])
    steady_timing(parts,"mh-cuts" if config.get('cuts_only') else "mh")
    return parts




def rdr_bench(folder: Path) -> tuple[str,str]:
    """Recompute the saved same-clip bench. Its footage is never used in the film."""
    root=folder/"rdr-bench"
    if not all((root/name).is_file() for name in ('words.json','before.t0.json','after.t0.json','before.events.jsonl','after.events.jsonl')):
        return "",""
    words=json.loads((root/"words.json").read_text())
    spans=[]
    for a,b,*_ in words:
        if a>=170:continue
        if spans and a-spans[-1][1]<.35:spans[-1][1]=max(b,spans[-1][1])
        else:spans.append([a,b])
    scores=[];positions=[];hashes={}
    for name in ("before","after"):
        epoch=json.loads((root/f"{name}.t0.json").read_text());positions.append(epoch['pos'])
        es=[json.loads(x) for x in (root/f"{name}.events.jsonl").read_text().splitlines()]
        total=0
        for e in es:
            if e.get('kind')!='spoken':continue
            a,b=e['began']-epoch['t0'],e['ended']-epoch['t0']
            if b<0 or a>170:continue
            total+=sum(max(0,min(b,y)-max(a,x)) for x,y in spans)
        scores.append(round(total,1))
    if positions[0]!=positions[1] or scores!=[12.7,1.1]:
        print(f"RDR benchmark does not support the caption; shot omitted: {scores}",file=sys.stderr)
        return "",""
    for f in root.glob('*.json*'):
        if f.name!='verified.json':hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    proof=root/'verified.json'
    content=json.dumps({'seconds':scores,'duration':170,'same_clip_position':positions[0],
                       'merged_word_gap':.35,'source_hashes':hashes},indent=2)+'\n'
    if not proof.exists() or proof.read_text()!=content:proof.write_text(content)
    return "Talking over actors, same clip: 12.7 s before this fix, 1.1 s after.",str(proof)


def rdr_sam_pieces(take: Take) -> list[Piece]:
    e=take.edit
    parts=[]
    keys=[("rdr-look","view"),("rdr-horse","horse"),("rdr-wagon","wagon"),("rdr-dialogue","dialogue")]
    for ident,key in keys:
        a,b=map(float,e[key])
        quiet=key=="dialogue"
        p=Piece(ident,"game3",line="RDR "+key,take=take.name,a=a,b=b,layout="rdr",captions=not quiet,duck=not quiet)
        if quiet:
            if any(x.get("kind")=="spoken" and min(b,x["ended"]-take.start)>max(a,x["began"]-take.start) for x in take.events):
                raise ValueError("RDR actors window contains Player Two speech")
        elif key!="view":
            events=[x for x in take.events if x.get("kind")=="where" and a<=x["t"]-take.start<b]
            if not events or any(x.get("via")!="sam-3.1" or not isinstance(x.get("seconds"),(float,int)) or not x.get("boxes") or not all(valid_box(z) for z in x['boxes']) for x in events):
                raise ValueError("RDR SAM beat needs actual sam-3.1 geometry")
            p.voice_boosts=e.get("ping_boosts",{}).get(key,[])
            p.box_until=max((x[1] for x in p.voice_boosts),default=b)
            p.overlays.append(Overlay("No 3D autopilot. You walk, it tells you where things are.",36,"tl",0,b-a))
        if key=='view':p.overlays.append(Overlay('Description shortened',34,'trim',0,b-a))
        if key=="horse" and e.get("horse_wait_cut"):
            x,y=map(float,e["horse_wait_cut"])
            if not a<x<y<b or any(max(x,c)<min(y,d) for c,d in speech_spans(take.events,take.start)):
                raise ValueError("Horse wait cut must remove silence only")
            cut=[dataclasses.replace(p,id=ident+"-ask",b=x,voice_boosts=[]),dataclasses.replace(p,id=ident+"-result",a=y)]
            for q in cut:
                q.overlays=[dataclasses.replace(o,f1=q.out_s) for o in p.overlays]
            steady_timing(cut,"horse")
            parts+=cut
        else:parts.append(p)
    return parts


def menu_records(take: Take, parts: list[Piece] | None = None) -> tuple[list[float],list[dict]]:
    """The accepted keyboard take must prove both reads and its actual journal press."""
    if take.edit.get('backend')!='keyboard':raise ValueError('RDR menu proof needs the accepted keyboard backend')
    requests=take.edit.get('menu_requests',['Read this menu.','Open the journal.','Read it.']);times=[];agents=[]
    clean=lambda text:re.sub(r'[^a-z0-9]+','',text.lower())
    for request in requests:
        t=next((t for t,e in take.heard() if 0<=t<take.duration and clean(e['text'])==clean(request)),None)
        e=next((e for e in take.events if e.get('kind')=='agent' and clean(e.get('request',''))==clean(request) and t is not None and t<=e['t']-take.start),None)
        if t is None or not e or not e.get('turns'):raise ValueError('RDR menu needs its real three Muse requests')
        times.append(t);agents.append(e)
    if times!=sorted(times):raise ValueError('RDR menu requests out of order')
    presses=[x for x in agents[1].get('steps',[]) if x.get('tool')=='press']
    if not any(x.get('args',{}).get('buttons')==['a'] for x in presses):
        raise ValueError('RDR menu needs the real journal navigation presses')
    at=times[1]+next(x['t'] for x in presses if x['args'].get('buttons')==['a'])
    if parts is None and not any(a<=at<b for a,b in take.edit['story']):raise ValueError('RDR menu must show its own journal press')
    if parts is not None and not any(p.take==take.name and p.a<=at<p.b and not p.freeze and p.speed==1 for p in parts):
        raise ValueError('RDR edit must show its own journal press at1x')
    return times,agents


def rdr_menu_pieces(take: Take) -> list[Piece]:
    menu_records(take);parts=[]
    for i,(a,b) in enumerate(take.edit['story']):
        if not 0<=a<b<=take.duration:raise ValueError('RDR menu window outside its take')
        p=Piece(f'rdr-menu-{i}','game3',line='RDR menu',take=take.name,a=a,b=b,layout='menu',proof=str(take.video.parent))
        p.overlays=[Overlay('No code for this game. Muse reads the menu, presses, and reads the page.',34,'tl',0,p.out_s)]
        parts.append(p)
    steady_timing(parts,'rdr-menu')
    return parts


def moving_atlas(folder: Path, take: Take | None) -> list[Piece]:
    base=atlas_beat(folder)
    if not take or any(p.kind=="slate" for p in base):
        return base
    raw=json.loads(base[0].blocks[1][0])
    texts=[f"Muse Spark / saved result\n{raw['name']} · icon {raw['icon']}",
           "Human checked\n"+base[1].blocks[1][0],"Local destination\n"+base[2].blocks[1][0]]
    t=12.0
    out=[]
    for old,text in zip(base,texts,strict=True):
        p=Piece(old.id,"ct",line="atlas proof",take=take.name,a=t,b=t+old.secs,captions=False,
                audio_mode="game",vol=.6,proof=old.proof,layout="map")
        p.overlays=[Overlay(text,38,"atlas-stage",0,p.out_s),
                    Overlay("Muse labels the map. I check it. Local code navigates.",40,"how",0,p.out_s)]
        out.append(p);t+=old.secs
    return out


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
        request_chip=any(o.where.startswith('trace') for o in p.overlays)
        if p.kind != "clip" or not (p.captions or request_chip) or p.take not in takes:
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
    spans += [tuple(w) for w in take.edit.get('action_windows',[])]
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
    excluded=[(e['began']-take.start-.04,e['ended']-take.start+.04) for e in take.events
              if e.get('kind')=='spoken' and e['text'] in take.edit.get('omit_speech',[])]
    kept=[]
    for p in parts:
        windows=[(p.a,p.b)]
        for lo,hi in excluded:
            windows=[(x,y) for a,b in windows for x,y in [(a,min(b,lo)),(max(a,hi),b)] if y>x]
        for x,y in windows:
            kept.append(dataclasses.replace(p,id=f'cold-{len(kept)}',a=x,b=y,overlays=[dataclasses.replace(o,f0=0,f1=(y-x)/p.speed) for o in p.overlays]))
    parts=kept
    reveal_out=sum(p.out_s for p in parts if p.b<=reveal)
    elapsed=0
    for p in parts:
        if p.black_for==999:
            p.overlays.append(Overlay('This is the game with the picture off.',34,'intro',0,p.out_s))
        for x,y,text in take.edit.get('action_labels',[]):
            lo,hi=max(p.a,x),min(p.b,y)
            if hi>lo:p.overlays.append(Overlay(text,34,'action-proof',(lo-p.a)/p.speed,(hi-p.a)/p.speed))
        elapsed+=p.out_s
    place_overlays(parts,[Overlay('Picture on.',42,'picture-on',reveal_out,reveal_out+1.2)])
    return parts,warnings


def timing_text(parts: list[Piece], group: str) -> str:
    source=max(p.b for p in parts)-min(p.a for p in parts);shown=sum(p.out_s for p in parts)
    if group=='mh':return f'{source:.0f} s shown in {shown:.0f} s · Fast narration muted'
    if group in ('horse','fair'):return f'{source-shown:.1f} s wait cut'
    suffix='Cuts + speed-ups' if any(p.speed>1 for p in parts) else 'Cuts'
    return f'{source:.0f} s shown in {shown:.0f} s · {suffix}'


def timing_starts(parts: list[Piece], group: str) -> list[float]:
    delay=0
    if group=='cold' and any(p.black_for==999 for p in parts):
        delay=sum(p.out_s for p in itertools.takewhile(lambda p:p.black_for==999,parts))+1.2
    at=0;starts=[]
    for p in parts:
        starts.append(max(0,min(p.out_s,delay-at)));at+=p.out_s
    return starts


def steady_timing(parts: list[Piece], group: str) -> None:
    if not parts or any(p.kind!='clip' for p in parts):return
    for p,start in zip(parts,timing_starts(parts,group),strict=True):
        p.timing_group=group
        p.overlays=[o for o in p.overlays if o.where not in ('speed','gap')]
        if start<p.out_s:p.overlays.append(Overlay(timing_text(parts,group),34,'time-'+group,start,p.out_s))


def valid_timing_group(parts: list[Piece], group: str) -> bool:
    selected=[p for p in parts if p.timing_group==group]
    return bool(selected) and len({p.take for p in selected})==1 and all(a.b<=b.a+.0001 for a,b in itertools.pairwise(selected)) and all(
        (not any(o.where=='time-'+group for o in p.overlays) if start==p.out_s else
         any(o.where=='time-'+group and o.text==timing_text(selected,group) and abs(o.f0-start)<.001 and o.f1>=p.out_s-.001 for o in p.overlays))
        for p,start in zip(selected,timing_starts(selected,group),strict=True))


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
                where="trace-shop-menu" if p.freeze or a<=(x+y)/2<b else "trace-shop"
                placed.append(dataclasses.replace(o,f0=x,f1=y,where=where))
        p.overlays=placed
    return pieces


def map_walk(take: Take | None, folder: Path) -> list[Piece]:
    if not take:
        return [Piece("nav-pending","ct",line="world-map walk",kind="slate",secs=12,
                     blocks=[("WORLD-MAP WALK / TAKE PENDING",48,True),
                             ("12-second slot for the reviewed real walk at normal speed",36,False)])]
    if take.edit.get('story'):
        agents=[e for e in take.events if e.get('kind')=='agent']
        castle=next((e for e in agents if e.get('request')=='Take me to Guardia Castle.'),None)
        resume=next((e for e in agents if e.get('request')=='Keep going to the castle.'),None)
        if not castle or not resume or not any(x.get('tool')=='fight' and x.get('args',{}).get('mode')=='win' for x in castle.get('steps',[])):
            raise ValueError('Navigation story needs the real interrupted route and fight decision')
        route={x.get('tool') for x in resume.get('steps',[])}
        if not {'exits','walk_to','keep_walking','go_to_place','remember'}<=route or not resume.get('turns'):
            raise ValueError('Navigation story needs its real multi-turn forest-exit plan')
        parts=[]
        spans=speech_spans(take.events,take.start)
        for index,w in enumerate(take.edit['story']):
            a,b=float(w['a']),float(w['b']);speed=float(w.get('speed',1));role=w['role']
            windows=split_speed(a,b,spans,speed,min_gap=.2,pad=.08) if speed>1 else [(a,b,1)]
            for j,(x,y,rate) in enumerate(windows):
                p=Piece(f'nav-{index}-{j}','ct',line='castle and fair',take=take.name,a=x,b=y,speed=rate,
                        captions=rate==1,layout='fight' if role in ('interrupt','fight','fight-win') else 'map',audio_mode='live')
                if role=='fight' and rate>1:
                    p.overlays.append(Overlay('Muse picks the tool. Code finds the path and plays the turns.',34,'nav-route',0,p.out_s))
                if role=='walk' and p.out_s>=2:p.overlays.append(Overlay(MAP_LINE,40,'nav-map',0,p.out_s))
                if role=='exit':p.overlays.append(Overlay('Muse Spark planned the forest exit. Local code moves and takes turns.',34,'nav-route',0,p.out_s))
                if role=='summary':
                    steps=resume.get('steps',[])
                    if not any(q.get('tool')=='walk_to' and 'not_found' in q.get('result','') for q in steps) or not any(q.get('tool')=='remember' and q.get('result')=='updated that note' for q in steps):
                        raise ValueError('Recovery needs the logged wrong guess and saved note')
                    count=len(resume['turns'])
                    p.overlays.append(Overlay(f'Not scripted. Muse took {count} turns, recovered from a wrong guess and saved a note for next time.',34,'nav-recovery',0,p.out_s))
                parts.append(p)
        steady_timing(parts,'nav')
        return parts
    a,b=map(float,take.edit['clip'])
    if any(p.kind=='slate' for p in atlas_beat(folder)):
        raise ValueError('Map walking needs its real saved Muse label/check proof')
    local=[e for e in take.events if e.get('kind')=='agent' and a<=e.get('t',0)-take.start<=b+.5
           and e.get('turns')==[] and any(x.get('tool')=='go_to_place' for x in e.get('steps',[]))]
    if not local:
        raise ValueError('Map walking needs the logged local go_to_place result')
    p=Piece("nav-walk","ct",line="world-map walk",take=take.name,a=a,b=b,layout="map")
    p.proof=str(folder/'atlas-proof.json')
    p.overlays=[Overlay(MAP_LINE,40,"how",0,b-a)]
    return [p]


def paced_plan(folder: Path, takes: dict[str,Take], vo: dict, vo_s: dict) -> list[Piece]:
    config=json.loads((folder/'pacing.json').read_text());parts=[]
    for item in config['shots']:
        p=_piece_from(item)
        if p.take=='rdr-menu':p.proof=str(folder)
        if p.kind=='clip':
            take=takes[p.take]
            if not 0<=p.a<p.b<=take.duration:raise ValueError(f'Paced source out of range: {p.id}')
        if p.vo_prompt:
            key={'identity':'intro','c-how':'how','limit':'limit'}[p.id]
            if vo_s.get(key,0)>p.out_s:raise ValueError(f'{key} VO exceeds paced slot {p.out_s}s')
            p.vo=vo.get(key,'')
        parts.append(p)
    for p in parts:
        if p.id=='lavos':p.overlays=[Overlay(RETRO_QUOTE+'\n'+RETRO_SOURCE,48,'lavos-credit',0,p.out_s)]
    if config.get('friend_quote'):
        at=next(i for i,p in enumerate(parts) if p.id=='identity')
        parts.insert(at,Piece('friend-quote','hook',kind='card',secs=5,blocks=[(FRIEND_QUOTE,52,True),(FRIEND_SOURCE,32,False),('About a phone touch game.',28,False)]))
    # Recheck the specific source proofs, not an invented model or game-specific macro.
    menu_records(takes['rdr-menu'],parts)
    verify_recovery(parts,takes.get('nav'))
    return parts


def verify_recovery(parts: list[Piece], take: Take | None) -> None:
    claims=[o.text for p in parts for o in p.overlays if 'Not scripted. Muse took' in o.text]
    if not claims:return
    if take is None:raise ValueError('Recovery claim needs its navigation log')
    events=[e for e in take.events if e.get('kind')=='agent' and e.get('request','').rstrip('.?!').lower()=='keep going to the castle']
    if not events:raise ValueError('Recovery claim lacks its agent event')
    event=events[-1];n=len(event.get('turns',[]))
    if not all(f'Muse took {n} turns' in text for text in claims) or not any(x.get('tool')=='walk_to' and 'not_found' in x.get('result','') for x in event.get('steps',[])) or not any(x.get('tool')=='remember' for x in event.get('steps',[])):
        raise ValueError('Paced recovery proof changed')


def word_count(text: str) -> int:
    return sum(any(c.isalnum() for c in word) for word in text.split())


def pacing_report(parts: list[Piece], takes: dict[str,Take], exceptions: dict) -> tuple[str,list[str]]:
    issues=[];shots=[];t=0
    def issue(code,detail):issues.append((code,detail))
    for p in parts:
        shots.append((t,p));t+=p.out_s
        if p.kind=='card':
            words=sum(word_count(b[0]) for b in p.blocks)
            if p.out_s<words*.4+1:issue('card:'+p.id,f'{words} words in{p.out_s:.2f}s; needs{words*.4+1:.2f}s')
        if t>=180:issue('duration:film',f'{t:.3f}s must stay below180s')
        if p.out_s<2.5:issue('shot:'+p.id,f'{p.out_s:.2f}s shot below2.5s')
        if p.result_at is not None and p.b-p.result_at+p.hold_after<1:issue('result:'+p.id,'Less than1s result dwell')
        if p.speed!=1:issue('speed:'+p.id,'Paced cut must keep all selected action at1x')
        take=takes.get(p.take)
        if take and not p.freeze and p.audio_mode!='game':
            for i,e in enumerate(take.events):
                if e.get('kind')!='spoken':continue
                a,b=e['began']-take.start,e['ended']-take.start
                if max(a,p.a)>=min(b,p.b):continue
                if a<p.a-.025 or b>p.b+.025:issue('partial:'+p.id+':'+str(i),e['text'])
                if p.speed!=1:issue('speech-speed:'+p.id,e['text'])
            for heard_t,event in take.heard():
                if event.get('typed') or not p.a<=heard_t<p.b:continue
                starts=[e['t']-take.start for e in take.events if e.get('kind')=='status' and e.get('state')=='listening' and heard_t-15<=e['t']-take.start<=heard_t]
                if starts and starts[-1]<p.a-.025:issue('partial-request:'+p.id,'Clip starts after the logged listening interval began')
                previous=[e['ended']-take.start for e in take.events if e.get('kind')=='spoken' and p.a<=e['began']-take.start<(starts[-1] if starts else heard_t)]
                if starts and previous and starts[-1]-max(previous)<.7-.001:issue('request-gap:'+p.id,f'Spoken reply to captured request: {starts[-1]-max(previous):.3f}s')
            selected=dataclasses.replace(take,caps=[c for c in take.caps if p.a-.025<=c[0]<p.b-.025])
            caps=[(a,b,txt) for a,b,txt in caption_rows(selected) if p.captions and p.a<=a<p.b]
            for i,(a,b,txt) in enumerate(caps):
                needed=.4*word_count(txt)+1;shown=min(p.b+p.hold_after,b)-a
                if shown+.01<needed:issue(f'caption:{p.id}:{i}',f'{txt!r}: shown{shown:.2f}s, reading rule{needed:.2f}s')
            utter=[e for e in take.events if e.get('kind')=='spoken' and p.a<=e['began']-take.start<p.b]
            for i,(a,b) in enumerate(itertools.pairwise(utter)):
                gap=b['began']-a['ended']
                if gap<.7:issue(f'cadence:{p.id}:{i}',f'Native recorded gap{gap:.3f}s: {a["text"]!r} -> {b["text"]!r}')
        display=render_overlays(p)
        for i,o in enumerate(display):
            if o.where in ('draft','box'):continue
            if o.where.startswith('trace'):
                request=o.text.splitlines()[-1]
                meta=(' Muse Voice Transcribe' if 'Muse Voice Transcribe' in o.text else '')+(' No button pressed' if 'No button pressed' in o.text else '')
                needed=.4*word_count(request+meta)+1
                if o.f1-o.f0+.01<needed:issue(f'request:{p.id}:{i}',f'{request!r}: {o.f1-o.f0:.2f}s / needed{needed:.2f}s')
                continue
            needed=.4*word_count(o.text)+1
            if o.f1-o.f0<needed:issue(f'text:{p.id}:{i}',f'{o.text!r}: {o.f1-o.f0:.2f}s / needed{needed:.2f}s')
        contexts=[o for o in display if o.where.startswith('context')]
        for at in {o.f0+.001 for o in contexts}:
            if sum(o.f0<=at<o.f1 for o in contexts)>1:issue('chips:'+p.id,'More than one context chip')
        # Count visible onset changes, including authentic requests and captions.
        onsets=[(o.f0,o.where,o.text) for o in display if o.where not in ('draft','box')]
        if take and p.captions:
            onsets += [(a-p.a,'caption',txt) for a,b,txt in caption_rows(dataclasses.replace(take,caps=[c for c in take.caps if p.a-.025<=c[0]<p.b-.025])) if p.a<=a<p.b]
        onsets.sort()
        for j,(left,right) in enumerate(itertools.pairwise(onsets)):
            if 0<=right[0]-left[0]<1 and left[2]!=right[2]:issue(f'onset:{p.id}:{j}',f'{left[1]} -> {right[1]} after{right[0]-left[0]:.2f}s')
        # Independent hard cuts; every edited gap in this plan has a visible dip.
    for (_at,left),(_next_at,right) in itertools.pairwise(shots):
        take=takes.get(left.take);following=takes.get(right.take)
        if not take or not following or left.audio_mode=='game' or left.freeze:continue
        ended=[e['ended']-take.start for e in take.events if e.get('kind')=='spoken' and left.a<=e['began']-take.start<left.b]
        incoming=[]
        for x,e in following.heard():
            if not right.a<=x<right.b:continue
            starts=[z['t']-following.start for z in following.events if z.get('kind')=='status' and z.get('state')=='listening' and x-15<=z['t']-following.start<=x] if not e.get('typed') else []
            incoming.append((starts[-1] if starts else x)-right.a)
        incoming += [e['began']-following.start-right.a for e in following.events if e.get('kind')=='spoken' and right.a<=e['began']-following.start<right.b]
        if ended and incoming:
            pause=left.out_s-(max(ended)-left.a)+min(incoming)
            if pause<.7:issue('join:'+left.id,f'Reply-to-next-ask gap{pause:.2f}s')
    cuts=[at for at,p in shots[1:] if not p.dip and p.kind=='clip']
    for i,x in enumerate(cuts):
        if i and x-cuts[i-1]<6:issue(f'cut:{i}',f'Two hard cuts within{x-cuts[i-1]:.2f}s')
    for group,needle in [('hook','Chrono Trigger'),('game2','Monster Hunter'),('game3','Red Dead')]:
        if not any(p.section==group and any(needle in o.text and o.f1-o.f0>=2 for o in p.overlays) for p in parts):issue('name:'+group,'Missing game name for2s')
    blocked=[code for code,_ in issues if code not in exceptions or code.startswith(('partial:','partial-request:','speech-speed:','duration:','speed:','shot:','result:','chips:','name:'))]
    rows=['## Pace report',f'{len(parts)} shots / {t:.3f}s. Short shots: {sum(p.out_s<2.5 for p in parts)}. Speed-ups: {sum(p.speed!=1 for p in parts)}. All edited source gaps use visible dips.',f'Unlisted violations: {len(blocked)}. Listed exceptions: {len(issues)-len(blocked)}.']
    for code,detail in issues:rows.append(f'- `{code}`: {detail}. '+('EXCEPTION: '+exceptions[code] if code in exceptions else 'BLOCKED'))
    return '\n'.join(rows)+'\n',blocked


def how_bridge(takes: dict[str, Take], vo: dict[str,str], vo_s: dict[str,float]) -> list[Piece]:
    """One continuous voice-over over three illustrative, real game shots."""
    if vo_s.get('how',0)>HOW_SECONDS:
        raise ValueError(f"How VO exceeds its {HOW_SECONDS:.2f}s footage slot")
    # Source windows are selected together with the takes, never borrowed from another run.
    config=takes.get('rdr-sam').edit.get('bridge',[]) if takes.get('rdr-sam') else []
    if not config:
        return [Piece('c-how','games',line='how VO / general and specific',kind='card',secs=HOW_SECONDS,
                      vo=vo.get('how',''),vo_prompt=GENERAL_LINE,blocks=[(GENERAL_LINE,52,True)])]
    if len(config)!=3 or [x['take'] for x in config]!=['take-A','game2','rdr-sam']:
        raise ValueError('How bridge needs CT, Monster Hunter and Red Dead in that order')
    out=[];elapsed=0
    for i,w in enumerate(config):
        name=w['take'];take=takes.get(name);a,b=float(w['a']),float(w['b'])
        if not take or not 0<=a<b<=take.duration:
            raise ValueError('How bridge source window outside its real take')
        if name=='game2' and any(max(a,x)<min(b,y) for x,y in speech_spans(take.events,take.start)):
            raise ValueError('Monster Hunter bridge must not omit a spoken line')
        p=Piece('c-how' if i==0 else f'c-how-{i}','games',line='how VO / general and specific',
                take=name,a=a,b=b,captions=False,audio_mode='mix' if name=='game2' else 'game',vol=.35,
                vo=vo.get('how',''),vo_offset=elapsed,vo_prompt=GENERAL_LINE,layout='map',
                overlays=[Overlay(GENERAL_LINE,44,'how',0,b-a)])
        out.append(p);elapsed+=p.out_s
    if abs(elapsed-HOW_SECONDS)>.001:raise ValueError('How bridge must cover its complete VO slot')
    return out


def make_plan(
    takes: dict[str, Take], picks: dict[str, int], no_code: set[str], rates: list[str],
    vo: dict[str, str], vo_s: dict[str, float], clips: dict[str, tuple[float, float]],
    force_drop: set[str] = frozenset(),
) -> tuple[list[Piece], list[str], list[str]]:
    del rates
    folder=Path(vo.get("_folder","."))
    opening,warnings=cold_open(takes.get("take-B"))
    fight=takes.get("take-B")
    if fight and not any(e.get('kind')=='spoken' and e.get('text','').lower().startswith('won the fight')
                         and any(p.a<=e['began']-fight.start<p.b for p in opening) for e in fight.events):
        warnings.append("cold open incomplete: actual win speech missing")
    if fight:
        for p in opening:
            for a,b in fight.edit.get('pause_windows',[]):
                lo,hi=max(p.a,a),min(p.b,b)
                if hi>lo:p.overlays.append(Overlay("Paused until you ask to continue",34,"pause",(lo-p.a)/p.speed,(hi-p.a)/p.speed))
        steady_timing(opening,"cold")
    drop=set(force_drop)|{"A2","A3","A5","B0","B1","B2","B3","B4"}
    lines=ct_pieces(takes,picks,drop,8,warnings)
    take=takes.get('take-A')
    if take and lines.get('A1'):
        first=next(((a,b) for a,b,who,txt in take.caps if who=='Player Two' and 'On the world map' in txt),None)
        if first:
            heard=next((t for t,e in take.heard() if LINES['A1'][1] and re.search(LINES['A1'][1],e['text'],re.IGNORECASE)),first[0])
            lines['A1']=[Piece('A1-local','ct',line='A1',take=take.name,a=max(0,heard-.4),b=first[1],layout='map')]
    if take and take.edit.get('fair_wait_cut') and lines.get('A3b'):
        lo,hi=map(float,take.edit['fair_wait_cut']);new=[]
        if any(max(lo,a)<min(hi,b) for a,b in speech_spans(take.events,take.start)):
            raise ValueError('Fair wait cut overlaps real speech')
        for p in lines['A3b']:
            for x,y in [(p.a,min(p.b,lo)),(max(p.a,hi),p.b)]:
                if y>x:new.append(dataclasses.replace(p,id=p.id+f'-{len(new)}',a=x,b=y,overlays=[]))
        steady_timing(new,'fair');lines['A3b']=new
        new[0].overlays.append(Overlay('Two runs of the same save',34,'trim',0,new[0].out_s))
    if take and take.edit.get('shop_end') and lines.get('A4'):
        for p in lines['A4']:p.b=min(p.b,float(take.edit['shop_end']))
        lines['A4']=[p for p in lines['A4'] if p.b>p.a]
    if take and lines.get('A3b'):
        for p in lines['A3b']:
            p.overlays=[o for o in p.overlays if o.where!='trim']
            p.overlays.append(Overlay('Another run of the same save. Description shortened.',34,'trim',0,p.out_s))
    if take and take.edit.get('shop_cut') and lines.get('A4'):
        lo,hi=map(float,take.edit['shop_cut']);new=[]
        if any(max(lo,a)<min(hi,b) for a,b in speech_spans(take.events,take.start)):
            raise ValueError('Shop cut overlaps real speech')
        for p in lines['A4']:
            for x,y in [(p.a,min(p.b,lo)),(max(p.a,hi),p.b)]:
                if y>x:new.append(dataclasses.replace(p,id=p.id+f'-cut-{len(new)}',a=x,b=y,overlays=[]))
        steady_timing(new,'shop');lines['A4']=new
    if take and take.edit.get('shop_hold') and lines.get('A4'):
        start,frame=map(float,take.edit['shop_hold']);new=[]
        for p in lines['A4']:
            if p.b<=start:new.append(p);continue
            if p.a<start:new.append(dataclasses.replace(p,id=p.id+'-live',b=start,overlays=[]))
            q=dataclasses.replace(p,id=p.id+'-held',a=max(p.a,start),freeze=True,hold_at=frame,overlays=[])
            q.overlays=[Overlay('Purchase result held',34,'trim',0,q.out_s)];new.append(q)
        steady_timing(new,'shop');lines['A4']=new
    maps=[*lines.get('A1',[]),*map_walk(takes.get('nav'),folder),*lines.get('A3b',[]),*lines.get('A4',[])]
    for p in maps:
        p.sharp_at=None
        if p.line=='A4':p.layout='shop'
    intro=round2_cards(folder)
    for i,p in enumerate(intro):
        if p.id=='identity' and fight:
            at=float(fight.edit.get('identity_at',max(x.b for x in opening)))
            intro[i]=Piece('identity','hook',line='founder intro',take=take.name if take else fight.name,a=0 if take else at,b=p.secs if take else at+p.secs,
                freeze=True,captions=False,audio_mode='game',vol=.3,vo=vo.get('intro',''),vo_prompt=INTRO_VO,
                layout='fight',overlays=[Overlay(INTRO_VO,58,'identity',0,p.secs)])
        elif p.id=='lavos' and take:
            intro[i]=Piece('lavos','hook',line='lavos',take=take.name,a=0,b=p.secs,captions=False,
                          audio_mode='game',vol=.5,layout='map',overlays=[Overlay('\n'.join(x[0] for x in p.blocks),44,'lavos',0,p.secs)])
    for key,limit in [('intro',8.5),('limit',6.0)]:
        if vo_s.get(key,0)>limit:raise ValueError(f"{key} VO exceeds its {limit:.2f}s slot")
    games=[]
    rdr=takes.get('rdr-sam')
    games.extend(how_bridge(takes,vo,vo_s))
    mh=takes.get('game2')
    games+=mh_story_pieces(mh) if mh and mh.edit.get('story') else game_pieces('game2',mh,False,clips.get('game2'),warnings)
    if rdr:
        scenes=rdr_sam_pieces(rdr);menu=takes.get('rdr-menu')
        if menu:
            scenes[0].overlays.append(Overlay('Red Dead · two takes',34,'trim',0,scenes[0].out_s))
            games += [scenes[0],*rdr_menu_pieces(menu),*[p for p in scenes[1:] if p.id!='rdr-dialogue']]
        else:games+=scenes
    else:games+=game_pieces('game3',None,False,None,warnings)
    goodbye=takes.get('goodbye')
    ending=[goodbye_piece(goodbye)] if goodbye else [Piece('goodbye-missing','outro',kind='slate',secs=3.5,
                        blocks=[('ENDING DESIGN / real goodbye pending',48,True)])]
    limit=Piece('limit','outro',kind='card',secs=6,vo=vo.get('limit',''),vo_prompt=LIMIT_VO,blocks=[(LIMIT_LINE,58,True)])
    parts=[*opening,*intro,*maps,*games,limit,*ending]
    parts.append(Piece('c-end','outro',kind='card',secs=5,blocks=[
        ('Player Two',96,True),('You choose. It reads and presses.',56,True),('github.com/btsouth/player-two',40,False),
        (requests_wording(_shown_requests(takes,parts),False)+' Waits are cut and fast parts are marked.',30,False),
        (LEGAL_LINE,24,False),(SE_NOTICE,22,False)]))
    return parts,[],warnings


def end_card(takes: dict[str, Take], parts: list[Piece]) -> Piece:
    sped = any(p.speed > 1 for p in parts)
    wording = requests_wording(_shown_requests(takes, parts), sped)
    return Piece("c-end", "outro", "end", "card", secs=END_S, blocks=[
        ("Player Two", 80, True),
        ("github.com/btsouth/player-two · open source (MIT)", 40, False),
        ("Built with Meta Muse Spark and Muse Voice Transcribe", 40, False),
        (wording, 32, False)])


def stand_in_voice(path: str) -> bool:
    if not path:return False
    file=Path(path)
    if 'TEMP-NOT-FOR-SUBMISSION' in str(file.resolve()):return True
    if file.is_file():
        with file.open('rb') as f:
            f.seek(max(0,file.stat().st_size-512))
            return b'PLAYER_TWO_SYNTHETIC_STAND_IN' in f.read()
    return False


def final_refusals(parts: list[Piece], warnings: list[str]) -> list[str]:
    """What --final won't ship."""
    out = [f"placeholder: {p.id}" for p in parts if p.kind == "slate"]
    for group in {p.timing_group for p in parts if p.timing_group}:
        if not valid_timing_group(parts,group):out.append(f"unsupported time summary: {group}")
    out += [f"stand-in media: {p.id}" for p in parts if p.stand_in or stand_in_voice(p.vo)]
    checked_menu=False
    for p in parts:
        if p.take=='rdr-menu' and not checked_menu:
            try:
                take=load_take(Path(p.proof),'rdr-menu',False)
                if not take:raise ValueError('menu take absent')
                menu_records(take,parts)
            except (OSError,ValueError,KeyError,StopIteration) as exc:out.append(f'unsupported menu proof: {exc}')
            checked_menu=True
        if (p.id.startswith("c-how") or p.id in ("identity","limit")) and not p.vo:
            out.append(f"missing voice-over audio: {p.id}")
        if p.hold_at is not None and not (p.freeze and p.line=='A4' and 0<=p.a-p.hold_at<=3 and any(o.text=='Purchase result held' for o in p.overlays)):
            out.append(f'unsupported held result: {p.id}')
        if any(o.text == VO_LABEL for o in p.overlays):
            out.append(f"no voice-over for {p.id} (vo-why.* in the folder)")
        if any(DRAFT_MARK in str(b[0]) for b in p.blocks):
            out.append("draft rates: pass --rates")
        for o in p.overlays:
            if o.where=='bench':
                text,_=rdr_bench(Path(p.proof).parent.parent)
                if not text or o.text!='Recorded-clip benchmark\n'+text:
                    out.append(f"unsupported benchmark caption: {p.id}")
        if p.kind == "clip" and p.speed != 1 and not any(
                o.text == f"{p.speed:g}x" and o.f0 <= 0.01 and o.f1 >= p.out_s - 0.05
                for o in p.overlays) and not (p.timing_group and valid_timing_group(parts,p.timing_group)):
            out.append(f"no {p.speed:g}x label all through {p.id}")
    total = sum(p.out_s for p in parts)
    if total > TOTAL_LIMIT:
        out.append(f"total {total:.1f} s is over {TOTAL_LIMIT:.0f} s")
    return out + [w for w in warnings if "not found" in w or "no take-" in w or "missing timing" in w or "cold open incomplete" in w or "missing SAM geometry" in w]


def round2_cards(folder: Path) -> list[Piece]:
    return [Piece('identity','hook',kind='card',secs=8.5,blocks=[
                ('Player Two.',80,True),("For players who can't see the screen or can't hold a controller.",58,True),
                ('You ask. It reads, answers and presses the buttons.',44,False)]),
            Piece('lavos','hook',kind='card',secs=6,blocks=[
                (RETRO_QUOTE,48,True),(RETRO_SOURCE,34,False),
                ("Lavos is the final boss. Not beaten yet. Here's how far we've got.",44,True)])]


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


def card_png(path: Path, blocks: list, bg=BG, end: bool = False, plain: bool = False) -> Path:
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
    if not plain:
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
    if not plain:d.text((96, H-76), "A second player. A different way in.", font=_font(False, 26), fill=(150,166,181))
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
    elif o.where.startswith('context'):
        f=_font(True,34);lines=[]
        for row in o.text.splitlines():lines+=_wrap(d,row,f,350 if o.where=='context-menu' else 600)
        x,y=(64,110) if o.where=='context-menu' else (W-664,330) if o.where=='context-shop-menu' else (W-664,600) if o.where=='context-dialogue' else (W-664,52)
        h=len(lines)*46+26
        d.rounded_rectangle((x-16,y-10,x+(366 if o.where=='context-menu' else 616),y+h),radius=16,fill=(9,15,25,225))
        for i,ln in enumerate(lines):d.text((x,y+i*46),ln,font=f,fill='white')
    elif o.where.startswith('time-'):
        f=_font(True,34)
        lines=_wrap(d,o.text,f,600)
        x,y=W-64-600,52
        h=len(lines)*46+26
        d.rounded_rectangle((x-16,y-10,W-48,y+h),radius=16,fill=(9,15,25,225))
        for i,ln in enumerate(lines):d.text((x,y+i*46),ln,font=f,fill='white')
    elif o.where.startswith("trace"):
        lines = o.text.split("\n")
        # Request chip and measured result are independent, floating surfaces.
        req = lines[1] if len(lines) > 1 else lines[0]
        typed = "typed" in lines[0]
        top = o.where in ("trace-fight", "trace-shop", "trace-black", "trace-mh")
        x, y = 64, (200 if o.where == "trace-black" else 60 if top else 700)
        if o.where=="trace-shop-menu":
            x,y=1000,550
        if o.where in ("trace-rdr","trace-menu"):
            y=520
        black=o.where=="trace-black"
        f = _font(True, 64 if black else 42)
        text = ("You (typed): " if typed else "You: ") + req
        meta=''
        if 'Muse Voice Transcribe' in lines[0]:
            meta='Muse Voice Transcribe'
            if 'No button pressed' in lines[0]:meta+=' · No button pressed'
        wrapped = _wrap(d, text, f, 1000 if o.where=="trace-shop-menu" else 1450)
        mf=_font(True,34)
        meta_lines=_wrap(d,meta,mf,1000 if o.where=='trace-shop-menu' else 1450) if meta else []
        content_width=max([d.textlength(t,font=f) for t in wrapped]+[d.textlength(t,font=mf) for t in meta_lines])
        if black:
            x=(W-int(content_width)-108)//2
            y=250
        width = int(content_width) + 108
        line_h=84 if black else 56
        height = len(wrapped) * line_h + len(meta_lines)*46 + 32
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
        for i,ln in enumerate(meta_lines):
            d.text((x+66,y+16+len(wrapped)*line_h+i*46),ln,font=mf,fill=ACCENT if black else 'white')
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
    elif o.where=='lavos-credit':
        d.rounded_rectangle((48,620,1872,990),radius=20,fill=(9,15,25,232))
        y=650
        for text,size in [(RETRO_QUOTE,48),(RETRO_SOURCE,34)]:
            f=_font(True,size)
            for line in _wrap(d,text,f,1728):
                d.text((80,y),line,font=f,fill='white');y+=int(size*1.3)
            y+=24
    elif o.where == "atlas-stage":
        lines=o.text.split("\n")
        d.rounded_rectangle((48,170,1130,355),radius=20,fill=(9,15,25,220))
        d.text((72,190),lines[0],font=_font(True,34),fill=ACCENT)
        d.text((72,250),lines[1],font=_font(True,44),fill="white")
    elif o.where in ("identity","lavos"):
        d.rounded_rectangle((48,140,1872,680),radius=24,fill=(9,15,25,232))
        if o.where=="identity":
            texts=[("Player Two.",80),("For players who can't see the screen or can't hold a controller.",58),
                   ("You ask. It reads, answers and presses the buttons.",44)]
        else:
            texts=[(RETRO_QUOTE,48),(RETRO_SOURCE,34),
                   ("Lavos is the final boss. Not beaten yet. Here's how far we've got.",44)]
        y=175
        for text,size in texts:
            f=_font(True,size)
            for line in _wrap(d,text,f,1728):
                d.text((80,y),line,font=f,fill="white");y+=int(size*1.3)
            y+=20
    elif o.where == "box":
        # Coordinates are normalized source-frame xyxy, supplied by the SAM event itself.
        x0, y0, x1, y1 = json.loads(o.text)
        d.rectangle((x0 * W, y0 * H, x1 * W, y1 * H), outline=ACCENT, width=6)
    else:
        f = _font(True, max(34, o.size))
        lines = _wrap(d, o.text, f, 1176 if o.where=="how" else 1080 if o.where=="nav-map" else 1728 if o.where=="lavos" else 1600)
        is_speed=bool(re.fullmatch(r"\d+(?:\.\d+)?x",o.text))
        if is_speed:
            lines = [o.text + " SPEED"]
        y = 52 if o.where in ("tl", "tl2", "tr", "trl", "intro") else 700
        if o.where == "intro":
            y = 66
        if o.where in ("gap","pause"):
            y = 150
        if o.where=="lavos":
            y=240
        if is_speed or o.where in ("speed","trim"):
            y = 64
        if o.where=="speed-shop":
            y = 700
        tw = int(max(d.textlength(ln, font=f) for ln in lines))
        x = ((W-tw)//2 if o.where in ("bc","intro") else W-tw-64 if is_speed or o.where in ("speed","trim") else 64)
        if o.where=="how":
            x,y=640,700
        if o.where=="pause":
            x,y=W-tw-64,680
        if o.where=='nav-map':x,y=64,52
        if o.where=='nav-route':x,y=64,500
        if o.where=='bench':
            x,y=64,65
        if o.where=='mh-model':x,y=64,170
        if o.where in ('mh-code','mh-gloss'):
            x,y=64,300
        bh = len(lines) * int(f.size * 1.35) + 28
        d.rounded_rectangle(
            (x - 20, y - 12, x + tw + 20, y + bh), radius=18, fill=(9, 15, 25, 218)
        )
        for i, ln in enumerate(lines):
            d.text((x, y + i * int(f.size * 1.35)), ln, font=f, fill="white")
    img.save(path)
    return path


def answer_after(take: Take, request: float, text: str) -> tuple[float,float] | None:
    """Recorded heard -> actual content speech, excluding acknowledgement/generation time."""
    clean=lambda x:re.sub(r'[^a-z0-9]+','',x.lower())
    target=clean(text)
    if not target:return None
    next_request=next((t for t,_ in take.heard() if t>request),take.duration)
    candidates=[]
    for e in take.events:
        if e.get('kind')!='spoken':continue
        at=e['began']-take.start;words=clean(e['text'])
        if request<=at<next_request and words and (target.startswith(words) or words.startswith(target)):
            candidates.append(at)
    return (min(candidates),min(candidates)-request) if candidates else None


def event_answer(take: Take, request: float, event: dict) -> tuple[float,float] | None:
    text=event.get('outcome') or event.get('text') or event.get('reply',{}).get('say','')
    prefix=take.edit.get('scene_prefix','')
    if event.get('source')=='atmosphere' and prefix and text.startswith(prefix):text=text[len(prefix):].lstrip()
    if event.get('kind')=='where':text='Found it' if len(event.get('boxes',[]))==1 else 'I see'
    return answer_after(take,request,text)


def _evidence_for_window(parts: list[Piece], takes: dict[str, Take]) -> list[dict]:
    """Completion badges: explicit logged model work only, never guessed from request words.

    Raw computation durations remain in the evidence report. Visible durations are
    heard-to-content-speech from spoken.began; without a matched answer, show only the layer.
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
                facts.append((e, e["model"], f"Stream {e['seconds']:.2f} s"
                              if e.get("timing") == "stream" else f"Stage time {e['seconds']:.2f} s"))
        label = "You: typed" if request.get("typed") else "You: heard"
        if not request.get('typed'):
            label += ' · Muse Voice Transcribe'
            if request.get('hands_free') is True or request.get('input_mode')=='hands_free':
                label += ' · No button pressed'
        base = f"{label}\n{request.get('raw_transcript') or request['text']}"
        heard_at = max(0, (t-p.a)/p.speed)
        if heard_at > 0:
            p.overlays.append(Overlay("You:\nRequest begins", 30, "trace", 0, heard_at))
        state = {}
        cur = heard_at
        for e, layer, timing in facts:
            response=event_answer(take,t,e)
            at = max(cur, (e['t']-take.start-p.a)/p.speed, (response[0]-p.a)/p.speed if response else 0)
            if response and response[0]>=p.b:continue
            text = base + "".join(f"\n{k}\n{v}" for k,v in state.items())
            if at > cur:
                p.overlays.append(Overlay(text, 30, "trace", cur, at))
            display = timing if layer == 'Muse Voice Transcribe' else (
                f'Answer {response[1]:.1f} s' if response else '')
            state[layer] = display
            cur = at
            report.append({"piece": p.id, "request": request['text'], "layer": layer,
                           "timing": timing, "answer_seconds": response[1] if response else None, "display_timing":display, "event_t": e['t'], "badge_from": at})
        text = base + "".join(f"\n{k}\n{v}" for k,v in state.items())
        if cur < p.out_s:
            p.overlays.append(Overlay(text, 30, "trace", cur, p.out_s))
    return report


def evidence_overlays(parts: list[Piece], takes: dict[str, Take]) -> list[dict]:
    """Handle every request in a continuous clip, including the paired black opening."""
    report = []
    for p in parts:
        take = takes.get(p.take)
        if not take or p.kind != "clip" or not p.captions or p.id=="last-line" or (p.take=="game2" and take.edit.get("story") and not take.edit.get("paced")):
            continue
        heard = take.heard()
        if take.edit.get('paced') and not any(p.a<=x<p.b for x,_ in heard):continue
        before = [t for t, _ in heard if t <= p.a]
        inside = [t for t, _ in heard if p.a < t < p.b]
        near_new_request=bool(inside and inside[0]-p.a<=.5)
        starts = (
            [p.a] if before and not near_new_request and (not inside or p.a - before[-1] < 15) else []
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
        if p.timing_group in ('cold','nav'):
            for o in p.overlays:
                if o.where.startswith('trace'):o.text='\n'.join(o.text.splitlines()[:2])
            for row in report:
                if row['piece']==p.id:row['display']='Request only; measurement retained in evidence'
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
                        Overlay(json.dumps(box), 34, "box", t, min((p.box_until-p.a)/p.speed if p.box_until else t+1.5, p.out_s))
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


def render_overlays(p: Piece) -> list[Overlay]:
    """One request, one context panel and the real spoken caption, with no stacked chips."""
    contextual=('time-','nav-','mh-')
    kinds={'tl','tl2','tr','trl','trim','pause','metric','action-proof','picture-on'}
    raw=[];keep=[]
    for o in p.overlays:
        if o.where.startswith('trace'):
            rows=o.text.splitlines();keep.append(dataclasses.replace(o,text='\n'.join(rows[:2])))
            if len(rows)>2:
                layer,value=list(zip(rows[2::2],rows[3::2],strict=False))[-1]
                raw.append(Overlay((layer.replace('Local action','Local')+' '+value).strip(),34,'metric',o.f0,o.f1))
        elif o.where.startswith(contextual) or o.where in kinds:raw.append(o)
        else:keep.append(o)
    points=sorted({v for o in raw for v in (o.f0,o.f1)})
    for a,b in itertools.pairwise(points):
        active=[o for o in raw if o.f0<=(a+b)/2<o.f1]
        if not active:continue
        # Timing is a compact first line. Other facts share its surface, rather than piling up.
        active.sort(key=lambda o:not o.where.startswith('time-'))
        texts=list(dict.fromkeys(o.text for o in active))
        where='context'
        if p.id=='shop-buy' or (p.line=='A4' and (p.freeze or any(o.where=='trace-shop-menu' and o.f0<=(a+b)/2<o.f1 for o in keep))):where='context-shop-menu'
        elif p.take=='nav' and 102<=p.a<112:where='context-dialogue'
        elif p.layout=='menu':where='context-menu'
        if where=='context-shop-menu' and 'Purchase result held' in texts and any('shown in' in t for t in texts):
            texts=[next(t.split(' · ')[0] for t in texts if 'shown in' in t)+' · Result held']
        keep.append(Overlay('\n'.join(texts),34,where,a,b))
    merged=[]
    for o in keep:
        previous=next((x for x in merged if x.where==o.where and x.text==o.text and abs(x.f1-o.f0)<.001),None)
        if previous:previous.f1=o.f1
        else:merged.append(o)
    return merged


def _overlays_chain(p: Piece, work: Path, first_input: int) -> tuple[list[str], str, str]:
    inputs: list[str] = []
    chain, last = "", "v0"
    for i, o in enumerate(render_overlays(p)):
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
    png = card_png(work / f"{p.id}.png", p.blocks, bg, end=p.id=="c-end",plain=p.id=="friend-quote")
    audio = ["-i", p.vo] if p.vo else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    inputs, chain, last = _overlays_chain(p, work, 2)
    fc = f"[0:v]scale={W}:{H},format=yuv420p,fade=t=in:d=0.15,fade=t=out:st={max(0,p.secs-0.15)}:d=0.15[v0]{chain}"
    _ff("-loop", "1", "-framerate", str(FPS), "-i", str(png), *audio, *inputs, "-filter_complex",
        fc, "-map", f"[{last}]", "-map", "1:a", "-t", f"{p.secs}", "-af", "apad", *_enc(vcodec),
        str(out))


def game_duck_filter(p: Piece) -> str:
    terms=[]
    for a,b,gain in p.game_ducks:
        if not p.a<=a<b<=p.b or not 0<=gain<=1:
            raise ValueError('Game duck must be a measured interval inside this shot')
        a=(a-p.a)/p.speed;b=(b-p.a)/p.speed
        terms.append(f"+({gain}-1)*min(1,min(max(0,(t-{a})/0.025),max(0,({b}-t)/0.025)))")
    expr='1'+''.join(terms)
    return f",aeval=exprs='val(0)*({expr})|val(1)*({expr})':c=stereo" if terms else ''


def encode_clip(p: Piece, takes: dict[str, Take], asses: dict[str, Path], work: Path, out: Path,
                vcodec: str) -> None:
    take = takes[p.take]
    inputs = ["-ss", f"{p.hold_at if p.hold_at is not None else p.a}", "-t", f"{p.b-p.a}", "-i", str(take.video)]
    n = 1
    if p.vo:
        inputs += ["-ss", f"{p.vo_offset}", "-i", p.vo]
        n = 2
    game_audio = take.video.with_suffix(".game.wav")
    voice_audio = take.video.with_suffix(".voice.wav")
    stems = game_audio.exists() and voice_audio.exists()
    if p.hold_at is not None and not stems:raise ValueError('Held purchase result needs its original separate stems')
    stem_n = n
    if stems:
        for stem in (game_audio, voice_audio):
            inputs += ["-ss", f"{p.a}", "-to", f"{p.b}", "-i", str(stem)]
        n += 2
    ov_inputs, chain, last = _overlays_chain(p, work, n)
    v = f"[0:v]scale={GAME_W}:{GAME_H}:flags=lanczos,fps={FPS},setpts=PTS-STARTPTS"
    if p.picture_until is not None:
        if not p.a<p.picture_until<p.b or p.result_at is None or p.picture_until<p.result_at or not any(o.text=='Purchase result held' for o in p.overlays):
            raise ValueError('Purchase picture hold needs its real result and disclosure')
        shown=p.picture_until-p.a
        v+=f",trim=duration={shown},tpad=stop_mode=clone:stop_duration={p.b-p.picture_until},setpts=PTS-STARTPTS"
    if p.freeze:
        v+=f",trim=end_frame=1,tpad=stop_mode=clone:stop_duration={p.out_s},setpts=PTS-STARTPTS"
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
    if take.edit.get('paced') and p.captions:
        selected=dataclasses.replace(take,caps=[c for c in take.caps if p.a-.025<=c[0]<p.b-.025])
        path=work/f'{p.id}-paced.ass';path.write_text(ass_text(selected,p.layout),encoding='utf-8')
        asses=dict(asses);asses[ass_key]=path
    if p.captions and ass_key in asses:
        v += (f",setpts=PTS+{p.a}/TB,subtitles='{_esc(asses[ass_key])}':"
              f"fontsdir='{_esc(ASSETS)}',setpts=PTS-STARTPTS")
    v += f",setpts=PTS/{p.speed:g}"
    if p.hold_after:v+=f",tpad=stop_mode=clone:stop_duration={p.hold_after}"
    if p.dip:v+=f",fade=t=in:d=0.25,fade=t=out:st={max(0,p.out_s-.25)}:d=0.25"
    v+='[v0]'
    tempo = _tempo(p.speed)
    dur = p.out_s
    # Accelerated gaps are silent: no sped-up voices or frantic game music.
    gain = 0 if p.speed > 1 and p.audio_mode!='live' else p.vol
    if p.speed>1 and p.audio_mode=='live' and any(max(p.a,a)<min(p.b,b) for a,b in speech_spans(take.events,take.start)):
        raise ValueError('Live accelerated audio must contain no spoken line')
    if p.speed>1 and p.audio_mode!='live':
        # Very short accelerated gaps can exhaust atempo before AAC receives a frame.
        # They are silent by design, so generate exact-length silence without time-stretching.
        a=f"anullsrc=r=48000:cl=stereo,atrim=duration={dur},asetpts=PTS-STARTPTS"
    elif p.audio_mode=="game":
        stream=f"{stem_n}:a" if stems else "0:a"
        a=f"[{stream}]asetpts=PTS-STARTPTS,{tempo}apad,atrim=duration={dur},volume={gain}"
    elif stems and p.duck:
        # Voice stem contains Player Two + the microphone; the game stem contains only game.
        boost=""
        if p.voice_boosts:
            expr="1"+"".join(f"+({g}-1)*between(t,{(x-p.a)/p.speed},{(y-p.a)/p.speed})" for x,y,g in p.voice_boosts)
            boost=f"aeval=exprs='val(0)*({expr})|val(1)*({expr})':c=stereo,"
        # FFmpeg framesync can stop the compressor early when the two stems arrive
        # in different block sizes. Flush it with padding, then trim to the real cut.
        a = (f"[{stem_n+1}:a]asetpts=PTS-STARTPTS,{tempo}apad=pad_dur=2,{boost}asplit[voice][side];"
             f"[{stem_n}:a]asetpts=PTS-STARTPTS,{tempo}apad=pad_dur=2"
             + game_duck_filter(p) + "[game];[game][side]sidechaincompress=threshold=0.025:ratio=8:"
             "attack=15:release=220[duck];[duck][voice]amix=inputs=2:normalize=0:"
             f"duration=first,atrim=duration={dur},volume={gain}")
    elif stems:
        # Preserve both stems untouched by the sidechain, including any unexpected voice.
        a = (f"[{stem_n}:a]asetpts=PTS-STARTPTS,{tempo}anull[game];"
             f"[{stem_n+1}:a]asetpts=PTS-STARTPTS,{tempo}anull[voice];"
             f"[game][voice]amix=inputs=2:normalize=0:duration=first,volume={gain}")
    else:
        a = f"[0:a]asetpts=PTS-STARTPTS,{tempo}volume={gain}"
    a += f",apad,atrim=duration={dur},afade=t=in:d=0.012,afade=t=out:st={max(0,dur-0.018):.3f}:d=0.018"
    if p.fade_in:
        a += ",afade=t=in:d=0.5"
    if p.fade_out:
        a += f",afade=t=out:st={max(0.0, dur - 0.5):.2f}:d=0.5"
    if p.vo and p.audio_mode=='game':
        # Founder narration is a speech sidechain too; pad through the whole shot.
        a += (f"[ta];[1:a]apad=pad_dur={dur+2},asplit[vovoice][voside];"
              "[ta][voside]sidechaincompress=threshold=0.01:ratio=8:attack=15:release=220[vogame];"
              f"[vogame][vovoice]amix=inputs=2:duration=first:normalize=0,atrim=duration={dur}[a]")
    elif p.vo:
        a += "[ta];[ta][1:a]amix=inputs=2:duration=first:normalize=0[a]"
    else:
        a += "[a]"
    _ff(*inputs, *ov_inputs, "-filter_complex", f"{v}{chain};{a}", "-map", f"[{last}]", "-map",
        "[a]", "-t", f"{dur:.3f}", *_enc(vcodec), str(out))


def caption_rows(take: Take) -> list[tuple]:
    source=[(a,b,txt.removeprefix('Player Two: ')) for a,b,who,txt in take.caps if who=='Player Two']
    if not take.edit.get('paced'):return source
    summaries={
      "Frog's at 1 HP, Crono and Marle are full.":None,
      'Crono took 779, down to 1.':'Frog and Crono: 1 HP.',
      'Frog uses Heal.':'Queued: Heal.', 'Marle uses Cure on Crono.':'Queued: Cure on Crono.',
      'Going to Guardia Castle, through Guardia Forest.':'Going to Guardia Castle.',
      "We're at Guardia Castle.":'Guardia Castle. Stop right there!', 'Soldier, Stop right there!':None,
      'Marle, Where are you going?':'Check out the fair instead.', "C'mon, let's check out the fair!":None,
      'We made it to Guardia Castle,':'The guard stopped you… Marle said: fair instead.',
      'but the guard stopped you, and Marle said to check out the fair instead,':None,
      "so we're back on the world map.":None,
      'Three parts you can break.':'Checking three parts…', 'Let me see what each one takes.':None,
      'Hit the head with the harp, the stomach with the hammer, and the tail with the sword.':'Head: pierce. Stomach: blunt. Tail: slash.',
      "It's coming at you, and these usually hit with speed.":None,
      'Technical beats speed.':'Technical beats speed.', 'Sleep Chaser on the stomach.':None,
      'Another part broken!':None,"That's the fight.":None,'We won.':'We won.',
      'Paused menu is open.':'Paused menu open.',
      'Your Journal is open on Brief, for Can a Swindler Change His Spots.':'Journal open: Brief.',
      'The Brief for Can a Swindler Change His Spots says: A medical breakthrough from the East!':'A medical breakthrough from the East!',
      'Found it: on the right.':'Found it: right.',
      'Going to the item seller at the purple tent.':'Walking to the seller.',
      'Three Potions, 30 gold.':'Three potions.',
      'Bought.':None,'You have 370 gold left.':'370 gold left.'}
    out=[]
    for a,b,text in source:
        if paced_caption(take,text) is None:continue
        text=summaries.get(text,text)
        if text is not None:out.append([a,max(b+.7,a+.4*word_count(text)+1),text])
    for i in range(len(out)-1):out[i][1]=min(out[i][1],out[i+1][0]-.1)
    return out


def paced_caption(take: Take, text: str) -> str | None:
    text=text.removeprefix('Player Two: ')
    if not take.edit.get('paced'):return text
    if text in {'Let me look.','On it.','One second.','Just a sec.',"Okay, I'll take this fight.","Okay, I'll open the journal.","Okay, I'll buy three potions.",'Okay, Crono will use Cyclone.'}:return None
    if text=='Marle uses Cure on Crono.':return 'Queued: '+text
    return text


def ass_text(take: Take, layout: str = "") -> str:
    """Full-bleed captions, positioned by scene; black opening gets generous central type."""
    layout = layout or ("fight" if take.top else "map")
    size, align, ml, mr, mv = {
        "map": (52, 2, 100, 100, 60),
        "rdr": (52, 2, 100, 100, 60),
        "menu": (52, 8, 500, 100, 700),
        "mh": (52, 8, 100, 100, 740),
        "fight": (52, 8, 100, 100, 240),
        "shop": (52, 9, 1000, 64, 420),
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
    for a,b,txt in caption_rows(take):
        txt = txt.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
        rows.append(
            f"Dialogue: 0,{demo_captions._ass_stamp(a)},"
            f"{demo_captions._ass_stamp(b)},P2,,0,0,0,,"
            + r"{\fad(90,70)\1c&HFFFFFF&}"
            + txt.replace(chr(10), " ")
        )
    return head + "\n".join(rows) + "\n"


def build(parts: list[Piece], takes: dict[str, Take], out: Path, vcodec: str, jobs: int) -> None:
    work = out.parent / f"{out.stem}-work"
    work.mkdir(parents=True, exist_ok=True)
    asses: dict[str, Path] = {}
    for name, take in takes.items():
        if take.caps:
            for layout in ("map","fight","shop","black","rdr","mh","menu"):
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
    joined = work / "joined.mp4"
    join_pieces(files,joined,work)
    normalize_mix(joined, out, work)


def join_pieces(files: list[Path], joined: Path, work: Path) -> None:
    """Trim decoded AAC per piece; copying packets carries priming across every cut."""
    lengths=[]
    for f in files:
        data=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=nb_frames,r_frame_rate","-of","json",str(f)]))["streams"][0]
        num,den=map(int,data["r_frame_rate"].split("/"))
        lengths.append(int(data["nb_frames"])*den/num)
    lst=work/"list.txt"
    lst.write_text("".join(f"file '{f.resolve()}'\nduration {s:.9f}\n" for f,s in zip(files,lengths,strict=True)))
    inputs=list(itertools.chain.from_iterable(("-i",str(f)) for f in files))
    filters=";".join(f"[{i}:a]apad,atrim=duration={s:.9f},asetpts=N/SR/TB[a{i}]" for i,s in enumerate(lengths))
    filters+=";"+"".join(f"[a{i}]" for i in range(len(files)))+f"concat=n={len(files)}:v=0:a=1[a]"
    wav=work/"joined-audio.wav"
    _ff(*inputs,"-filter_complex",filters,"-map","[a]","-c:a","pcm_s24le",str(wav))
    _ff("-f","concat","-safe","0","-i",str(lst),"-i",str(wav),"-map","0:v:0","-map","1:a:0",
        "-c:v","copy","-c:a","aac","-b:a","192k",str(joined))


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
          f"offset={stats['target_offset']},aresample=192000,"
          "alimiter=limit=0.707945784:level=false:attack=5:release=50:latency=true,aresample=48000")
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
                    shots.append((at, f"{int(at // 60)}:{at % 60:04.1f} {text.removeprefix('Player Two: ')}"))
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
    if (a.folder/'STAND-IN.json').exists() and a.final:sys.exit('--final refuses stand-in rehearsal media')
    if (a.folder/'STAND-IN.json').exists() and 'TEMP-NOT-FOR-SUBMISSION' not in str(out):sys.exit('Stand-in output name must contain TEMP-NOT-FOR-SUBMISSION')
    tops = set(filter(None, a.top.split(",")))
    takes = {}
    names={"take-A", "take-B", "sam", "ending", "goodbye", "rdr-look", "rdr-dialogue", "rdr-sam", "rdr-menu", "nav", *GAMES}
    manifest=json.loads((a.folder/'takes.json').read_text()) if (a.folder/'takes.json').exists() else {}
    names.update(k for k,v in manifest.items() if isinstance(v,dict) and 'video' in v)
    for name in sorted(names):
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
        vo = {w: v for w in ("why", "how", "intro", "limit") if (v := _vo(a.folder, w))}
        vo_s = {w: probe_s(Path(v)) for w, v in vo.items()}
        vo["_folder"]=str(a.folder)
        if (a.folder/'pacing.json').exists() and not a.sample:
            parts=paced_plan(a.folder,takes,vo,vo_s)
        else:
            parts, dropped, warnings = make_plan(takes, _picks(a.pick), no_code, rates, vo, vo_s,
                                                 _clips(a.clip), set(filter(None, a.drop.split(","))) | set(edit.get("drop",[])))
    # Remove legacy overlays before rebuilding the floating chips from source evidence.
    for piece in parts:
        if piece.kind == "clip":
            piece.overlays = [o for o in piece.overlays if o.text not in (LOWER_A4, LOWER_B)
                              and (not o.where.startswith('trace') or o.where=='trace-mh') and o.where != "draft"]
    for piece in parts:
        if piece.kind == "clip" and not piece.layout:
            piece.layout = "fight" if takes[piece.take].top else "map"
        if piece.line=="A4" and piece.layout not in ("shop","shop-menu"):
            piece.layout="shop"
    paced=(a.folder/'pacing.json').exists() and not a.sample
    if paced:
        for take in takes.values():take.edit['paced']=True
    for piece in parts:
        if manifest.get('_stand_in') or any(e.get('synthetic') for e in takes.get(piece.take,Take('',Path(''),None,0,[],[],False)).events):piece.stand_in=True
    evidence = evidence_overlays(parts, takes)
    if paced:
        for p in parts:
            for o in p.overlays:
                if o.where.startswith('trace'):
                    o.text='\n'.join(o.text.splitlines()[:2])
                    if p.id in ('mh-weak','rdr-menu-read'):o.f0=max(o.f0,1.1)
                    if p.id=='shop-buy':o.where='trace-shop-menu'
        for p in parts:p.overlays=[o for o in p.overlays if o.text!='You:\nRequest begins' and o.f1>o.f0 and not (p.id in ('ct-castle','ct-forest-walk') and o.where.startswith('trace'))]
    if any(p.stand_in or stand_in_voice(p.vo) for p in parts) and 'TEMP-NOT-FOR-SUBMISSION' not in str(out):
        sys.exit('Stand-in media cannot enter a real render path')
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
    if (a.folder/'pacing.json').exists() and not a.sample:
        config=json.loads((a.folder/'pacing.json').read_text())
        report,blocked=pacing_report(parts,takes,config.get('exceptions',{}))
        (a.folder.parent/'pace-report.md').write_text(report)
        qc=a.folder.parent/'QC.md';previous=qc.read_text() if qc.exists() else '# Video QC\n'
        previous=previous.split('<!-- PACE REPORT -->')[0]
        qc.write_text(previous+'\n<!-- PACE REPORT -->\n'+report)
        print(report)
        if blocked:sys.exit('Pacing blocks render: '+', '.join(blocked))
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
