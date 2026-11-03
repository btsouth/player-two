"""Monster Hunter Stories 3 fights: Muse Spark studies the monster, local code plays the turns.

What each side does:
- Muse Spark, once a fight: reads the target wheel (which part takes which damage type, what
  breaking it gives) and the Status screen (level, state, element weaknesses), and answers open
  questions ("what are we fighting?", "where's it weak?") in plain words. Every call is logged
  as mh_muse with its kind and model time, and each turn logs what Muse told it (mh_turn).
- Local code, every turn: the cursor and the timing. Pixels give the arc, the attack types on
  the ribbons, health bars, which menu row is lit. Nothing is said about a skill until the
  screen shows it selected and usable.

The game's rules, checked against saved frames and the published guides:
- Power beats Technical beats Speed beats Power (the triangle on the Skills screen).
- The arc says who's targeted and whether it can be countered, not the monster's type. Yellow:
  a typed attack at the rider. Blue: a typed attack at an ally. Red: an untyped hit on one. No
  arc: usually a hit on the whole party, says the guide, but not always (turn 1 of take 2 had no
  arc and only Chirpy was hit), so nothing's claimed. The colour doesn't change with Zerk's type.
- The monster's type shows on its ribbon as a move plays, and changes with its phase. Allies
  usually pick the type that wins, which is a hint, not a fact.
- Each part takes one damage type (crossed-out diamonds on the wheel). Weapon swaps are free.
- A plain attack of the winning type, matching the monstie's, can make a Double Attack. A skill
  can't.
- The Skills list keeps its cursor between turns, so rows are read, never counted from the top.
  A greyed row (not enough gauge) ignores A: that's what stalled the party in take 2.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .muse import image_part

GAME = "monster hunter stories 3"
GEAR_FILE = Path.home() / ".cache/player-two/mhs3-gear.json"
WHEEL_PX = int(os.environ.get("PLAYER_TWO_MH_WHEEL_PX", "1800"))
STATUS_PX = int(os.environ.get("PLAYER_TWO_MH_STATUS_PX", "800"))
TYPES = ("Power", "Speed", "Technical")  # the order the left stick steps through them
ANSWER = {"Speed": "Technical", "Technical": "Power", "Power": "Speed"}  # what beats each
ALLIES = ("Canyne", "Zerk", "Ogden", "Chirpy")

WHOLE_FIGHT = re.compile(
    r"\b(win (this|the) (fight|battle)|win this|fight for me|finish (this|the) (fight|battle)"
    r"|you (fight|play) (this|it)|take (this|the) (fight|battle)|play (this|the) (fight|battle)"
    r"|take over)\b")
ONE_COMMAND = re.compile(r"\b(attack|hit|strike|take your turn|your best move|best move|break)\b")
HEAL = re.compile(r"\b(heal|potion|herb)\b")
WHAT = re.compile(r"\b(what|which)\b.*\b(fighting|monster|enemy|that thing|this thing)\b"
                  r"|\bwhat('?s| is) (that|this)\b")
WEAK = re.compile(r"\bweak|\bwhere (should|do) (i|we) (hit|aim)\b|\bwhat should (i|we) (hit|aim)")
HEALTH = re.compile(r"\b(how'?s|how is|check) (my|our|the party'?s?) (health|hp)\b"
                    r"|\bhow (am i|are we) doing\b|\bam i hurt\b")


# The type each monster mostly attacks with, normal and angry: gamerguides.com's table, and its
# Tetranadon and Somnacanth rows match what's on screen in saved fights. Only a first guess: what
# the monster's actually been seen using in this fight goes first.
KNOWN_TYPES = {"tetranadon": ("Power", "Technical"), "somnacanth": ("Technical", "Speed"),
               "izuchi": ("Speed", "Speed"), "green bnahabra": ("Technical", "Technical"),
               "bnahabra": ("Technical", "Technical")}


def is_game(name: str) -> bool:
    return (name or "").lower().startswith(GAME)


def asks(request: str) -> str:
    """The kind of request this routine plays, or "" for one it leaves to the agent."""
    said = " ".join(request.lower().split())
    for kind, pattern in (("fight", WHOLE_FIGHT), ("weak", WEAK), ("what", WHAT),
                          ("health", HEALTH), ("heal", HEAL), ("turn", ONE_COMMAND)):
        if pattern.search(said):
            return kind
    return ""


# Reading pixels. Coordinates are in a 1280x720 picture, whatever the real size.

def _arr(img: Image.Image, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    w, h = img.size
    box = (int(x0 * w / 1280), int(y0 * h / 720), int(x1 * w / 1280), int(y1 * h / 720))
    return np.asarray(img.crop(box).convert("RGB"), dtype=float)


def _patch(img: Image.Image, fx: float, fy: float, r: float = 0.005) -> np.ndarray:
    w, h = img.size
    x, y, d = int(fx * w), int(fy * h), max(2, int(r * w))
    return np.asarray(img.crop((x - d, y - d, x + d, y + d)).convert("RGB"), dtype=float).mean(
        axis=(0, 1))


def _near(c: np.ndarray, want: tuple[int, int, int], tol: float = 24) -> bool:
    return bool(np.abs(c - np.array(want)).max() <= tol)


def menu_up(img: Image.Image) -> bool:
    """The command menu is up: its Y and A glyphs and the Skills diamond (fixed, whatever the
    camera does behind them)."""
    return (_near(_patch(img, 167 / 1280, 345 / 720), (180, 180, 180))
            and _near(_patch(img, 167 / 1280, 388 / 720), (165, 165, 165))
            and _near(_patch(img, 110 / 1280, 365 / 720), (133, 127, 115)))


def menu_type(img: Image.Image) -> str:
    """The attack type the command menu is on, from the colour of its main diamond."""
    c = (_patch(img, 141 / 1280, 420 / 720) + _patch(img, 193 / 1280, 420 / 720)) / 2
    r, g, b = c
    if r > g + 35 and r > b + 35:
        return "Power"
    if b > r + 35:
        return "Speed"
    if g > r + 20 and g > b + 15:
        return "Technical"
    return ""


def skills_tab(img: Image.Image) -> int | None:
    """Which weapon tab the Skills menu is on, from the bright teal of the chosen tab; None when
    that menu isn't open."""
    best, which, dark = 0.0, None, []
    for i, (a, b) in enumerate(((70, 190), (197, 320), (327, 450))):
        c = _arr(img, a, 102, b, 130)
        teal = float(((c[..., 1] > 150) & (c[..., 2] > 150) & (c[..., 0] < 130)).mean())
        dark.append(float((c.max(axis=2) < 120).mean()))
        if teal > best:
            best, which = teal, i
    if best <= 0.2 or which is None:
        return None
    # The other two tabs are dark. Teal sky or water behind the same spot has no dark tabs.
    return which if min(d for i, d in enumerate(dark) if i != which) > 0.35 else None


LIST_X = (150, 390)
SKILL_ROWS = (318, 359, 400)  # the skill rows; the row above them is the plain attack
ITEM_ROWS = (169, 210, 251, 292, 333, 374, 415, 456)


def _row_teal(img: Image.Image, y: float) -> float:
    c = _arr(img, LIST_X[0], y - 10, LIST_X[1], y + 10)
    return float(((c[..., 1] - c[..., 0] > 30) & (c[..., 2] - c[..., 0] > 30)
                  & (c[..., 1] > 75)).mean())


def lit_row(img: Image.Image, rows: tuple[int, ...]) -> int | None:
    """Which of these list rows is lit, or None."""
    shares = [_row_teal(img, y) for y in rows]
    best = max(range(len(rows)), key=lambda i: shares[i])
    return best if shares[best] > 0.35 else None


def skill_row(img: Image.Image) -> int:
    """The Skills list's cursor: 0 on the plain attack, then 1, 2, 3 for the skills."""
    row = lit_row(img, SKILL_ROWS)
    return 0 if row is None else row + 1


def skill_greyed(img: Image.Image, row: int) -> bool:
    """This skill row (1 to 3) is greyed out: no bright letters in it."""
    y = SKILL_ROWS[row - 1]
    c = _arr(img, 165, y - 9, 350, y + 9).min(axis=2)
    return float((c > 175).mean()) < 0.015


def items_open(img: Image.Image) -> bool:
    """The item list: the top tab says ALL and a row of the list is lit."""
    return skills_tab(img) is not None and lit_row(img, ITEM_ROWS) is not None \
        and _row_teal(img, 268) < 0.2


def wheel_title(img: Image.Image) -> np.ndarray:
    """The white letters of the part's name at the top of the target wheel."""
    return _arr(img, 575, 86, 755, 112).min(axis=2) > 205


def wheel_open(img: Image.Image) -> bool:
    """The target wheel or a small monster's ring is open: its cyan ring, whatever's behind."""
    c = _arr(img, 520, 115, 810, 405)
    cyan = (c[..., 1] > 215) & (c[..., 2] > 205) & (c[..., 0] < 140)
    return float(cyan.mean()) > 0.008 and not menu_up(img)


def wheel_has_parts(img: Image.Image) -> bool:
    return wheel_open(img) and int(wheel_title(img).sum()) >= 40


DIAMONDS = {"slash": 635.5, "pierce": 670.0, "blunt": 704.0}


def ring_is_small(img: Image.Image) -> bool:
    """The ring around a far-off monster, about 170 wide against the big wheel's 290."""
    c = _arr(img, 380, 60, 900, 470)
    cyan = (c[..., 1] > 215) & (c[..., 2] > 205) & (c[..., 0] < 140)
    cols = np.flatnonzero(cyan.sum(axis=0) >= 2)
    return bool(cols.size) and (cols.max() - cols.min()) * 520 / c.shape[1] < 230


def _ring_dx(img: Image.Image) -> int:
    """How far the wheel sits right of its usual place (it moves a few pixels as a fight goes
    on), from the left edge of its cyan ring."""
    c = _arr(img, 496, 200, 530, 320)
    cols = np.flatnonzero(((c[..., 1] > 215) & (c[..., 2] > 205) & (c[..., 0] < 140)).mean(axis=0) > 0.05)
    if not cols.size:
        return 0
    dx = round(496 + cols.min() * 34 / c.shape[1]) - 512
    return dx if abs(dx) <= 14 else 0


def wheel_works(img: Image.Image) -> list[str]:
    """The damage types this part takes: the diamonds with no red line through them."""
    out = []
    dx = _ring_dx(img)
    for name, x in DIAMONDS.items():
        c = _arr(img, x + dx - 15, 246, x + dx + 15, 276)
        red = float(((c[..., 0] > 150) & (c[..., 1] < 90) & (c[..., 2] < 95)).mean())
        if red < 0.012:
            out.append(name)
    return out


def part_broken(img: Image.Image) -> bool:
    """The wheel's chosen part says Broken: gold letters where its blue gauge would be."""
    dx = _ring_dx(img)
    c = _arr(img, 630 + dx, 284, 705 + dx, 298)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    gold = ((r > 170) & (g > 140) & (b < 140) & (r - b > 45)).mean()
    cyan = ((b > 140) & (g > 120) & (r < 120)).mean()
    # Any cyan here is the ring flickering behind the word, unless it's solid at the left of the
    # gauge's own band: an almost empty gauge, not the word Broken.
    h, w = c.shape[:2]
    fill = ((b > 140) & (g > 120) & (r < 120))[h // 3:2 * h // 3, :w // 6].mean()
    return bool(gold > 0.05 and cyan < 0.12 and fill < 0.35)


def wheel_gauge(img: Image.Image) -> float:
    """How much of this part's break gauge is left, 0 to 1."""
    dx = _ring_dx(img)
    c = _arr(img, 632 + dx, 285, 697 + dx, 295)
    fill = (c[..., 2] > 150) & (c[..., 1] > 120) & (c[..., 0] < 130)
    cols = np.flatnonzero(fill.mean(axis=0) > 0.4)
    return float((cols.max() + 1) / c.shape[1]) if cols.size else 0.0


RIBBON_X = {"Canyne": 192, "Zerk": 512, "Ogden": 819, "Chirpy": 1152}
BAR_X = {"Canyne": (97, 228), "Zerk": (437, 567), "Ogden": (738, 869), "Chirpy": (1078, 1208)}


def _ribbon_type(c: np.ndarray) -> str:
    """"Power", "Speed", "Technical", "none" (a gold ribbon: no type) or "" (no ribbon here)."""
    r, g, b = np.median(c.reshape(-1, 3), axis=0)
    if r > g * 1.6 and r > b * 1.6 and g < 90:
        return "Power"
    if b > r * 1.4 and b > g * 1.1:
        return "Speed"
    if g > r * 1.3 and g > b * 1.3:
        return "Technical"
    if r > b * 1.5 and g > b * 1.15 and r > 95 and r < 185:
        return "none"
    return ""


def ribbon(img: Image.Image, who: str) -> str:
    """The attack type on the ribbon above a party member's name ("" for none or no type)."""
    x = RIBBON_X[who]
    kind = _ribbon_type(_arr(img, x - 90, 638, x + 90, 654))
    return "" if kind == "none" else kind


def hud_up(img: Image.Image) -> bool:
    """The monster's name bar is in its place: not a close-up with the HUD hidden."""
    c = _arr(img, 575, 34, 715, 46)
    edge = _arr(img, 575, 44, 715, 46).mean(axis=(0, 1))
    return bool(float((c.max(axis=2) < 90).mean()) > 0.02 and edge.max() < 240)


def monster_ribbon(img: Image.Image) -> str:
    """The monster's move as it plays, from the ribbon under its name: a band darker than what's
    just above and below it. A cut-in or sky behind the same spot has no such edge."""
    c = _arr(img, 590, 90, 720, 101)
    band = np.median(c.reshape(-1, 3), axis=0)
    if band.max() > 135 or band.max() - band.min() < 45 or not hud_up(img):
        return ""
    for box in ((590, 70, 720, 76), (590, 104, 720, 110)):
        if np.abs(np.median(_arr(img, *box).reshape(-1, 3), axis=0) - band).sum() < 60:
            return ""
    return _ribbon_type(c)


def clash_type(img: Image.Image) -> str:
    """The type of the monster's attack, from the Head-to-Head screen: white title at the top, two
    name bars at the bottom, and the right hand ribbon in the colour of its attack."""
    title = _arr(img, 440, 125, 840, 185).min(axis=2) > 225
    left, right = _arr(img, 10, 656, 150, 676), _arr(img, 1130, 656, 1270, 676)
    if not 0.08 < float(title.mean()) < 0.5:
        return ""
    if np.median(left.max(axis=2)) > 95 or np.median(right.max(axis=2)) > 95:
        return ""
    c = _arr(img, 960, 606, 1270, 626).reshape(-1, 3)
    c = c[c.min(axis=1) < 150]  # not the white letters
    if not len(c):
        return ""
    r, g, b = np.median(c, axis=0)
    if r > g * 1.5 and r > b * 1.5:
        return "Power"
    if b > r * 1.3 and b > g * 1.05:
        return "Speed"
    if g > r * 1.3 and g > b * 1.2:
        return "Technical"
    return ""


def bar_share(img: Image.Image, who: str, y: float = 684) -> float:
    """How full a party member's bar is, 0 to 1. y 684 is health (teal), y 698 the skill gauge
    (yellow). The empty part shows the world behind it, darkened."""
    a, b = BAR_X[who]
    c = _arr(img, a, y - 2, b, y + 2)
    if y < 690:
        lit = (c[..., 1] > 185) & (c[..., 1] - c[..., 0] > 12)
    else:
        lit = (c[..., 0] > 190) & (c[..., 1] > 185)
    cols = np.flatnonzero(lit.mean(axis=0) > 0.5)
    return float((cols.max() + 1) / c.shape[1]) if cols.size else 0.0


def monster_health(img: Image.Image) -> float:
    """How full the monster's bar is, 0 to 1. One monster on screen. The bar's colour changes
    with poison and burn; the empty part is dark."""
    c = _arr(img, 579, 38, 708, 42)
    cols = np.flatnonzero((c.max(axis=2) > 150).mean(axis=0) > 0.5)
    return float((cols.max() + 1) / c.shape[1]) if cols.size else 0.0


def toast_up(img: Image.Image) -> bool:
    """The game's message band near the bottom ("The Tetranadon has returned to normal!")."""
    c = _arr(img, 420, 574, 860, 596)
    return float((c.min(axis=2) > 205).mean()) > 0.45


def ring_who(img: Image.Image) -> str:
    """Who the ally ring is on, by where it stands."""
    c = _arr(img, 0, 330, 1280, 620)
    cyan = (c[..., 1] > 190) & (c[..., 2] > 180) & (c[..., 0] < 150)
    cols = np.flatnonzero(cyan.sum(axis=0) >= 2)
    if cols.size < 10:
        return ""
    x = float(cols.mean()) * 1280 / c.shape[1]
    return "Canyne" if x < 250 else "Zerk" if x < 450 else "Ogden" if x < 700 else "Chirpy"


def status_open(img: Image.Image) -> bool:
    """The Status screen: the dark Effects header, and the tan row of element icons."""
    head = _arr(img, 860, 380, 1040, 400).mean(axis=(0, 1))
    c = _arr(img, 1012, 293, 1215, 317)
    tan = ((c[..., 0] > 100) & (c[..., 1] > 80) & (c[..., 2] < 120)
           & (c[..., 0] > c[..., 1]) & (c[..., 1] > c[..., 2])).mean()
    return bool(head.max() < 65 and tan > 0.08)


def _components(mask: np.ndarray, smallest: int = 30) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist(), strict=True):
        if seen[y0, x0]:
            continue
        stack, pts = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for ny in range(max(0, y - 2), min(h, y + 3)):
                for nx in range(max(0, x - 2), min(w, x + 3)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pts) >= smallest:
            found.append(np.array(pts))
    return found


WHEEL_CENTER = (670, 246)
WHEEL_RADIUS = 142


def arc_seen(img: Image.Image) -> tuple[str, str]:
    """The arc in the wheel's view: (colour, who), from the streak leaving the wheel. ("none", "")
    with no streak. who is "" when the streak's direction doesn't settle it."""
    small = img.convert("RGB").resize((1280, 720), Image.BILINEAR)
    a = np.asarray(small, dtype=int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    room = np.ones(r.shape, dtype=bool)
    room[:90] = False
    room[610:] = False
    room[240:330, 860:1200] = False  # the retreat bar that slides in on the right
    yy, xx = np.ogrid[:720, :1280]
    room &= ((xx - WHEEL_CENTER[0]) ** 2 + (yy - WHEEL_CENTER[1]) ** 2) > WHEEL_RADIUS ** 2
    masks = {"blue": (b > 215) & (b - r > 30) & (g > 170) & (b >= g - 8),
             "red": (r > 235) & (g < 200) & (b < 190) & (r - b > 70) & (r - g > 45),
             "yellow": (r > 238) & (g > 205) & (b < 165) & (r - b > 85) & (r - g < 45)}
    best: dict[str, np.ndarray] = {}
    for colour, mask in masks.items():
        for pts in _components(mask & room):
            span = np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))
            near = np.hypot(pts[:, 1] - WHEEL_CENTER[0], pts[:, 0] - WHEEL_CENTER[1]).min()
            if span >= 28 and near < 215 and (colour not in best
                                              or len(pts) > len(best[colour])):
                best[colour] = pts
    # A red streak's bright core reads yellow, and a yellow arc can flare red where it leaves the
    # wheel: only a yellow far longer than the red is the arc.
    def span(name: str) -> float:
        return float(np.hypot(*(best[name].max(axis=0) - best[name].min(axis=0)))) if name in best else 0.0

    if "red" in best and len(best["red"]) >= 150 and (
            span("yellow") < 2.2 * span("red") or span("yellow") < 120):
        colour = "red"
    elif "blue" in best and len(best["blue"]) >= 300:
        colour = "blue"
    elif "yellow" in best and len(best["yellow"]) >= 150:
        colour = "yellow"
    else:
        return "none", ""
    if colour == "yellow":
        return colour, "Zerk"
    pts = best[colour]
    far = pts[np.argmax(np.hypot(pts[:, 1] - WHEEL_CENTER[0], pts[:, 0] - WHEEL_CENTER[1]))]
    dy = float(far[0] - WHEEL_CENTER[1])
    if dy <= 20:
        return colour, ""
    x = WHEEL_CENTER[0] + (float(far[1]) - WHEEL_CENTER[0]) * (480 - WHEEL_CENTER[1]) / dy
    who = "Canyne" if x < 250 else "Zerk" if x < 430 else "Ogden" if x < 660 else "Chirpy"
    if colour == "blue" and who == "Zerk":  # blue is never the rider
        who = ""
    return colour, who


# What Muse is asked

READ_WHEEL = """\
This picture shows a Monster Hunter Stories 3 target wheel, once per step round it (left to right; \
the same part may come round again). For each wheel: the part's name is the word at the very top. \
Inside are three diamonds, always in this order: left Slash (a sword), middle Pierce (an arrowhead), \
right Blunt (a hammer block). A diamond crossed out with a red line doesn't work on this part. \
"Break Bonus" and the coloured words under it say what breaking the part does. At the bottom is a \
blue gauge (how much is left before it breaks) or the word Broken.
The player carries: {weapons}.
Then tell them where to hit it, the way a friend on the couch would who can see the screen and they \
can't: one short plain sentence, naming each unbroken part and the one weapon that fits it, as in \
"Hit the head with the harp, the legs with the hammer, and the tail with the sword." Go by the \
diamond that isn't crossed out. No jargon, nothing the picture doesn't show.
JSON: {{"parts": [{{"name": "...", "works": ["slash"|"pierce"|"blunt"], "broken": true|false, \
"bonus": "words under Break Bonus, or empty"}}], "say": "..."}}"""

WHEEL_HOLD = 1.0  # seconds the wheel stays up after its round, for the camera and the count
TOAST_AGAIN = 6.0  # seconds in which the same message is still the same message
STATUS_WAIT = 4.0  # seconds a fight waits on a Status reading already under way

READ_STATUS = """\
The Status screen of a Monster Hunter Stories 3 fight. The panel at the top right has the monster's \
name, level, its State, and under Elem Res a row of six element icons, in this order: neutral, fire, \
water, thunder, ice, dragon. Under each is a mark: red down arrows mean it's weak to that element \
(two arrows, very weak), blue up arrows mean it resists (two, strongly), a dash means neither. The \
weapons Zerk carries have these elements: {elements}.
Then tell them what they're up against, the way a friend on the couch would who can see the screen \
and they can't: one short plain sentence with its name and level and the most useful thing about \
its elements for those weapons. Two blue arrows is "strongly resists", said in those words. No \
figures of speech, no dashes.
JSON: {{"name": "...", "level": 0, "state": "...", "weak": ["..."], "very_weak": ["..."], \
"resists": ["..."], "say": "..."}}"""

READ_GEAR = """\
These are the three weapon pages of Monster Hunter Stories 3's Skills menu, left to right. For each \
page: the weapon's name is at the top of the right panel, with its damage type icon: a sword is \
slash, an arrowhead is pierce, a hammer block is blunt. Its element is the small icon beside Attack \
(a drop is water, a crystal is ice, a grey circle is none). The list on the left has its skills: \
name and cost.
JSON: {"weapons": [{"name": "...", "damage": "slash"|"pierce"|"blunt", "element": "...", \
"skills": [{"name": "...", "cost": 0}]}]}"""

READ_SKILL = """\
This is one skill's details from Monster Hunter Stories 3's Skills menu. JSON: {"name": "...", \
"power": 0, "wyvernfell": 0, "targets": "one"|"all"|"none", "part_damage": true if the words say it \
does heavy damage to a monster part, "deals_damage": false if it only charges or coats}"""

CHECK = """\
Is this Monster Hunter Stories 3 picture still a fight? JSON: {"screen": "fight", "results" or \
"field"}. "results": a big rank letter and a score bar, or rewards after a win. "field": walking \
around, a round minimap at the bottom right and no party HP bars along the bottom. Anything else, \
close-ups and attacks included, is "fight"."""

READ_TOAST = 'What does this game message say? JSON: {"text": "..."}'


@dataclass
class Part:
    name: str
    works: list[str]
    broken: bool = False
    gauge: float = 1.0
    bonus: str = ""
    mask: np.ndarray | None = None


@dataclass
class Skill:
    name: str
    cost: int = 0
    power: int = 0
    wyvernfell: int = 0
    targets: str = "one"
    part_damage: bool = False
    deals_damage: bool = True


@dataclass
class Weapon:
    name: str
    damage: str
    element: str = ""
    skills: list[Skill] = field(default_factory=list)


@dataclass
class View:
    """What's known at the start of a turn."""
    parts: list[Part]
    weapons: list[Weapon]
    in_hand: int
    menu_type: str
    arc: str = "none"
    toward: str = ""
    monster_type: str = ""  # its type, from its ribbon this phase
    hint_type: str = ""  # the type an ally has picked, if any
    hint_from: str = ""
    canyne_type: str = ""
    hp: dict[str, float] = field(default_factory=dict)
    gauge: int = 0  # Zerk's skill gauge
    target: str = ""  # the part being worked on
    resists: list[str] = field(default_factory=list)  # elements it shrugs off (Muse, Status)
    usual_type: str = ""  # what it usually hits with, by the guides, for the state Muse read
    top_up: bool = False  # one early heal this fight is still to come


@dataclass
class Plan:
    do: str  # attack, skill or item
    type: str = ""
    weapon: int = 0
    part: Part | None = None
    skill: Skill | None = None
    item: str = ""
    on: str = ""
    why_type: str = ""  # seen, ally, none
    double: bool = False


HURT = {"Zerk": 0.6, "Canyne": 0.5, "Ogden": 0.5, "Chirpy": 0.5}


def shrugged_off(view: View, part: Part) -> bool:
    """The only weapon that fits this part has an element the monster resists."""
    fit = [w for w in view.weapons if w.damage in part.works]
    return bool(fit) and all(w.element.lower() in view.resists for w in fit)


def fits_hand(view: View, part: Part) -> bool:
    return bool(view.weapons) and view.weapons[view.in_hand].damage in part.works


def pick_part(view: View) -> Part | None:
    """Stay on the part being worked on until it breaks. Otherwise the unbroken part nearest to
    breaking, with a weapon that fits; with all broken, whatever the weapon in hand works on."""
    if not view.parts:
        return None
    fits = {w.damage for w in view.weapons} if view.weapons else None
    open_parts = [p for p in view.parts if not p.broken
                  and (fits is None or not p.works or set(p.works) & fits)]
    for p in open_parts:
        if p.name == view.target:
            return p
    open_parts = [p for p in open_parts if not shrugged_off(view, p)] or (
        [] if view.resists else open_parts)
    if open_parts:
        return min(open_parts, key=lambda p: (round(p.gauge, 1), not fits_hand(view, p)))
    hand = view.weapons[view.in_hand].damage if view.weapons else ""
    return next((p for p in view.parts if hand in p.works), view.parts[0])


def pick_weapon(view: View, part: Part | None) -> int:
    if part is None or not view.weapons or not part.works:
        return view.in_hand
    if view.weapons[view.in_hand].damage in part.works:
        return view.in_hand
    return next((i for i, w in enumerate(view.weapons) if w.damage in part.works), view.in_hand)


def choose(view: View) -> Plan:
    """The move, from the rules above. Healing first; then the right part with the right weapon;
    the type that wins; a plain attack when it can make a Double Attack, otherwise the hardest
    skill on the part that the gauge can pay for."""
    low = [(view.hp[who] / HURT[who], who) for who in HURT if 0 < view.hp.get(who, 1.0) < HURT[who]]
    if not low and view.top_up:  # once a fight, before it's urgent; 0 is a bar hidden by riding
        low = [(view.hp[who], who) for who in HURT if 0 < view.hp.get(who, 1.0) < 0.8]
    if low:
        return Plan("item", item="Potion", on=min(low)[1], type=view.menu_type,
                    weapon=view.in_hand)
    part = pick_part(view)
    weapon = pick_weapon(view, part)
    kind, why = view.menu_type, "none"
    if view.arc in ("yellow", "blue"):
        if view.monster_type in ANSWER:
            kind, why = ANSWER[view.monster_type], "seen"
        elif view.usual_type in ANSWER:
            kind, why = ANSWER[view.usual_type], "usual"
        elif view.hint_type:
            kind, why = view.hint_type, "ally"
    plan = Plan("attack", type=kind, weapon=weapon, part=part, why_type=why)
    if why != "none" and view.canyne_type == kind:
        plan.double = True
        return plan
    if view.weapons and part is not None and not part.broken:
        usable = [s for s in view.weapons[weapon].skills
                  if s.deals_damage and s.targets == "one" and s.power > 50
                  and 0 < s.cost <= view.gauge]
        if usable:
            plan.do, plan.skill = "skill", max(usable, key=lambda s: (s.part_damage, s.power))
    return plan


def short(weapon: str) -> str:
    """How a weapon is called out loud."""
    low = weapon.lower()
    for word in ("hammer", "harp", "sword", "bow", "lance", "blade"):
        if word in low:
            return f"the {word}"
    if "speartuna" in low:
        return "the sword"
    return "your " + weapon.split()[-1].lower() if weapon else "it"


def amount(share: float) -> str:
    """A bar's fill, said the way a friend would."""
    if share >= 0.97:
        return "full"
    for most, words in ((0.2, "almost nothing"), (0.32, "about a quarter"), (0.42, "about a third"),
                        (0.6, "about half"), (0.72, "about two thirds"), (0.88, "about three quarters"),
                        (1.0, "most of it")):
        if share < most:
            return words
    return "most of it"


def is_are(part: str) -> str:
    return "are" if part.lower().endswith("s") else "is"


def weapon_words(weapon: Weapon) -> set[str]:
    """What a weapon may be called in a sentence."""
    words = {w for w in weapon.name.lower().split() if len(w) > 3}
    words.add(short(weapon.name).split()[-1])
    if "speartuna" in words:
        words.add("spear")
    return words


def muse_line_fits(line: str, parts: list[Part], weapons: list[Weapon]) -> bool:
    """Muse's sentence is said only if, part by part, the weapon it names is one the wheel's
    diamonds allow: pixels check the words, whatever Muse's own list of diamonds said."""
    if not weapons:
        return all(part.name.lower() in line.lower() for part in parts)
    clauses = re.split(r",|;|\.|\band\b", line.lower())
    for part in parts:
        clause = next((c for c in clauses if part.name.lower() in c), "")
        named_here = [w for w in weapons if any(x in clause for x in weapon_words(w))]
        if len(named_here) != 1 or named_here[0].damage not in part.works:
            return False
    return True


def named(part: Part) -> bool:
    """Muse gave this part its name: "part 2" is only a placeholder until then, never said."""
    return not part.name.startswith("part ")


class Lines:
    """What's said in a fight: short, varied, and never the same words twice in a fight."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.low_step = 0
        self.opened_on = ""  # the part the opening line just named

    def pick(self, *options: str) -> str:
        for line in options:
            if line not in self.used:
                self.used.add(line)
                return line
        return ""

    def reason(self, view: View, plan: Plan) -> str:
        """Why this turn goes the way it does, from what the screen shows now."""
        who = "you" if view.toward == "Zerk" else view.toward
        if plan.do == "item":
            return ""
        if view.arc == "yellow":
            if plan.why_type == "seen":
                # A guess from the type it showed last: it can change, so it's said as a guess.
                return self.pick(
                    f"It's coming at you. I'm guessing {view.monster_type.lower()} again, "
                    f"so {plan.type.lower()}.",
                    f"You again. Probably {view.monster_type.lower()}, so going "
                    f"{plan.type.lower()}.",
                    f"Coming at you. Still guessing {view.monster_type.lower()}.")
            if plan.why_type == "usual":
                return self.pick(
                    f"It's coming at you, and these usually hit with "
                    f"{view.usual_type.lower()}. {plan.type} beats {view.usual_type.lower()}.")
            if plan.why_type == "ally":
                return self.pick(
                    f"It's coming at you. Going {plan.type.lower()}, same as {view.hint_from}.",
                    f"It's lining up on you. {view.hint_from}'s going {plan.type.lower()}, "
                    "so I will too.")
            return self.pick("It's coming at you.", "You're next.")
        if view.arc == "blue":
            name = who or "an ally"
            if plan.why_type == "seen":
                return self.pick(
                    f"It's going for {name}. I'm guessing {view.monster_type.lower()} again, "
                    f"so {plan.type.lower()}.",
                    f"{name} is up next. Probably {view.monster_type.lower()}, "
                    f"so {plan.type.lower()}.")
            if plan.why_type == "usual":
                return self.pick(f"It's going for {name}. These usually hit with "
                                 f"{view.usual_type.lower()}, so {plan.type.lower()}.")
            if plan.why_type == "ally":
                return self.pick(f"It's going for {name}. {view.hint_from}'s picked "
                                 f"{plan.type.lower()}, so I'll match.",
                                 f"{name} is in its sights. Going {plan.type.lower()} too.")
            return self.pick(f"It's going for {name}.", f"{name} is in its sights.")
        # A red streak or no streak says nothing sure about what's coming, so nothing's said of it.
        return self.pick("I'll just go for damage.")

    def opening(self, view: View, plan: Plan) -> str:
        """The first turn has no streak to talk about: where it's starting, and with what, while
        the menus go by."""
        if plan.do == "item" or not plan.part or not named(plan.part) or not view.weapons:
            return ""
        self.opened_on = plan.part.name.lower()
        return self.pick(f"The {self.opened_on} first, with {short(view.weapons[plan.weapon].name)}.")

    def choice(self, view: View, plan: Plan, swapped: bool) -> str:
        """What's being done, said only once the screen shows it selected."""
        if plan.do == "item":
            return self.pick(f"{plan.item} on {'you' if plan.on == 'Zerk' else plan.on}.",
                             f"Another {plan.item.lower()} for "
                             f"{'you' if plan.on == 'Zerk' else plan.on}.")
        part = plan.part.name.lower() if plan.part and named(plan.part) else ""
        weapon = short(view.weapons[plan.weapon].name) if view.weapons else ""
        if plan.do == "skill" and plan.skill:
            where = f" on the {part}" if part and part != self.opened_on else ""
            self.opened_on = ""
            return self.pick(f"{plan.skill.name}{where}.", f"{plan.skill.name} again{where}.",
                             f"One more {plan.skill.name}{where}.")
        if not part:
            return self.pick(f"Hitting it with {weapon}." if weapon else "Hitting it.",
                             "Hitting it again.", "And again.")
        if swapped:
            damage = view.weapons[plan.weapon].damage
            return self.pick(f"{part.capitalize()} {'only take' if is_are(part) == 'are' else 'only takes'} "
                             f"{damage}, so {weapon} comes out.",
                             f"Swapping to {weapon} for the {part}.")
        if plan.double:
            # Only an intention: Canyne may use a skill instead, and the game then shows no double.
            return self.pick(f"Plain attack on the {part}, same type as Canyne.",
                             f"Same type as Canyne again, on the {part}.",
                             f"Matching Canyne on the {part}.")
        return self.pick(f"{weapon.capitalize()} on the {part}." if weapon else f"On the {part}.",
                         f"Staying on the {part}.", f"The {part} again.", f"More on the {part}.")

    def hurt(self, who: str, share: float) -> str:
        whose = "You're" if who == "Zerk" else f"{who}'s"
        return self.pick(f"{whose} down to {amount(share)}.", f"{whose} hurting.",
                         f"{whose} low again.")

    def health(self, hp: dict[str, float]) -> str:
        """Everyone's health, from the bars."""
        zerk = hp.get("Zerk", 1.0)
        bits = [f"You've got {amount(zerk)}." if zerk < 0.97 else "You're at full health."]
        hurt = [who for who in ("Canyne", "Ogden", "Chirpy") if hp.get(who, 1.0) < 0.85]
        for who in hurt[:2]:
            bits.append(f"{who}'s at {amount(hp[who])}.")
        if not hurt:
            bits.append("Everyone else is fine.")
        return " ".join(bits)

    def broke(self, part: str, heard: bool = False) -> str:
        """heard: the game's message has already been passed on; now the wheel says which part."""
        if heard:
            return self.pick(f"That was the {part.lower()}.", f"That one was the {part.lower()}.")
        return self.pick(f"That broke the {part.lower()}.", f"{part.capitalize()} {is_are(part)} broken.")

    def monster_low(self, share: float) -> str:
        """Each step down gets said once a fight."""
        step = 3 if share < 0.15 else 2 if share < 0.35 else 1 if 0.45 <= share < 0.58 else 0
        if step <= self.low_step:
            return ""
        self.low_step = step
        return ("", "It's down to about half.", "It's getting weak.", "It's almost done.")[step]


class Fight:
    """One session's fights. `agent` gives grab, controls, the model, speech and cancelling."""

    def __init__(self, agent):
        self.a = agent
        self.weapons: list[Weapon] = []
        self.weapon = 0  # the weapon page Zerk is on
        self.weapon_synced = False  # seen this fight
        self.parts: list[Part] = []
        self.studied = False
        self.status: dict = {}
        self.weak_line = ""
        self.what_line = ""
        self.target = ""
        self.seen_type = ""
        self.ended = ""
        self.lines = Lines()
        self.talking: asyncio.Task | None = None
        self.last_said = 0.0
        self.asides: set[asyncio.Future] = set()  # Muse looks that don't hold up the turn
        self.toasts: list[tuple[str, float]] = []
        self.pending: list[asyncio.Future] = []  # Muse readings of what's left the screen
        self.muse_seconds: dict[str, float] = {}  # model time per kind of reading, for the logs
        self.turns = 0
        self.won = False
        self.monster_share = 1.0
        self.break_heard = False
        self.healed = False
        self.muse_agrees = True
        self.note: Callable[[str, dict], None] = (
            lambda name, args: agent.on_action and agent.on_action(name, args))

    # Speech goes out without holding up the pad; lines stay in order.

    def say(self, line: str) -> None:
        if not line:
            return
        self.last_said = time.monotonic()
        before = self.talking

        async def out() -> None:
            if before is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(before)
            await self.a._speak(line)

        self.talking = asyncio.ensure_future(out())

    def aside(self, job) -> None:
        task = asyncio.ensure_future(job)
        self.asides.add(task)
        task.add_done_callback(self.asides.discard)

    async def said(self) -> None:
        if self.talking is not None:
            with contextlib.suppress(Exception):
                await self.talking

    # Pad, with the screen checked between presses

    async def _press(self, button: str, then: float = 0.0) -> None:
        await self.a.go.wait()
        await asyncio.to_thread(self.a.controls.press, button)
        if then:
            await asyncio.sleep(then)

    async def _flick(self, way: str, then: float = 0.3) -> None:
        await self.a.go.wait()
        await asyncio.to_thread(self.a.controls.stick, "left", way, 160)
        await asyncio.sleep(then)

    async def _shot(self) -> Image.Image:
        return (await self.a.grab()).image

    async def until(self, test: Callable[[Image.Image], bool], most: float = 1.2
                    ) -> Image.Image | None:
        """The first picture that passes, or None by then."""
        end = time.monotonic() + most
        while not self.a.cancelled:
            img = await self._shot()
            if test(img):
                return img
            if time.monotonic() >= end:
                return None
            await asyncio.sleep(0.04)
        return None

    async def ask(self, kind: str, images: list, text: str, tokens: int = 1500, most: float = 18.0
                  ) -> dict:
        """One look by Muse, logged with its kind and time; a reply cut short or not JSON gets
        one more try."""
        last: Exception | None = None
        msgs = [{"role": "user", "content": [*images, {"type": "text", "text": text}]}]
        t = time.monotonic()
        for _ in range(2):
            try:
                reply = await asyncio.wait_for(self.a.muse.json(
                    msgs, effort="minimal", max_tokens=tokens), most)
                if not reply and kind in ("status", "wheel"):
                    raise ValueError("empty reply")
                seconds = round(time.monotonic() - t, 2)
                self.muse_seconds[kind] = seconds
                self.note("mh_muse", {"read": kind, "seconds": seconds, "reply": reply})
                return reply
            except (ValueError, TimeoutError) as exc:
                last = exc
        self.note("mh_muse", {"read": kind, "seconds": round(time.monotonic() - t, 2),
                              "failed": str(last)})
        raise ValueError(f"no reading: {last}")

    # Once a session: the gear

    async def read_gear(self) -> None:
        """The weapons and their skills, from the Skills pages. Done before anything's on camera
        when it can be: it's a lot of menu."""
        await self._press("x")
        if await self.until(lambda i: skills_tab(i) is not None) is None:
            return
        img = await self._shot()
        start = skills_tab(img) or 0
        pages, details = [], []
        for i in range(3):
            img = await self._shot()
            w, h = img.size
            pages.append(img.crop((0, 90 * h // 720, 830 * w // 1280, 590 * h // 720)))
            rows = []
            for _ in range(4):  # up to the plain attack, then down the skills
                if skill_row(img) == 0:
                    break
                await self._flick("up", 0.2)
                img = await self._shot()
            for row in (1, 2, 3):
                await self._press("down", 0.2)
                img = await self._shot()
                if skill_row(img) != row:
                    break
                rows.append(img.crop((0, 470 * h // 720, 830 * w // 1280, 590 * h // 720)))
            details.append(rows)
            if i < 2:
                await self._press("rb", 0.35)
        await self._press("b", 0.4)
        sheet = Image.new("RGB", (sum(p.width for p in pages), pages[0].height))
        x = 0
        for p in pages:
            sheet.paste(p, (x, 0))
            x += p.width
        d = await self.ask("gear", [image_part(sheet, 2400)], READ_GEAR, 2500)
        got: list[Weapon] = []
        for i, entry in enumerate(d.get("weapons") or []):
            skills = [Skill(str(s.get("name") or ""), int(s.get("cost") or 0))
                      for s in entry.get("skills") or []]
            got.append(Weapon(str(entry.get("name") or f"weapon {i}"),
                              str(entry.get("damage") or "").lower(),
                              str(entry.get("element") or ""), skills))

        async def detail(skill: Skill, crop: Image.Image) -> None:
            with contextlib.suppress(Exception):
                r = await self.ask("skill", [image_part(crop, 1000)], READ_SKILL, 500)
                skill.power = int(r.get("power") or 0)
                skill.wyvernfell = int(r.get("wyvernfell") or 0)
                skill.targets = str(r.get("targets") or "one")
                skill.part_damage = r.get("part_damage") is True
                skill.deals_damage = r.get("deals_damage") is not False

        jobs = [detail(s, c) for w, rows in zip(got, details, strict=False)
                for s, c in zip(w.skills, rows, strict=False)]
        await asyncio.gather(*jobs)
        if len(got) == 3:  # pages were read from where it opened: put them in tab order
            self.weapons = [got[(i - start) % 3] for i in range(3)]
            self.weapon = start
        self.note("mh_gear", {"on": self.weapon, "weapons": [
            (w.name, w.damage, w.element, [vars(s) for s in w.skills]) for w in self.weapons]})

    async def sync_weapon(self) -> None:
        """Which weapon Zerk has out, from the page the Skills menu opens on."""
        await self._press("x")
        img = await self.until(lambda i: skills_tab(i) is not None)
        if img is not None:
            self.weapon = skills_tab(img) or 0
            self.weapon_synced = True
        await self._press("b", 0.3)
        await self.until(menu_up, 0.8)

    # Once a fight: Muse studies the monster

    def weapon_words(self) -> str:
        if not self.weapons:
            return "a hammer (blunt), a sword (slash), a harp (pierce)"
        return ", ".join(f"{short(w.name)[4:]} ({w.damage})" for w in self.weapons)

    async def settled_wheel(self, most: float = 1.4) -> Image.Image:
        """The wheel, once its title hasn't changed for a quarter of a second: it zooms in and
        turns, and two grabs close together can be the same frame."""
        end = time.monotonic() + most
        img = await self._shot()
        mask, since = wheel_title(img), time.monotonic()
        while time.monotonic() < end:
            await asyncio.sleep(0.05)
            img = await self._shot()
            now = wheel_title(img)
            if not _same(now, mask) or int(now.sum()) < 40:
                mask, since = now, time.monotonic()
            elif time.monotonic() - since >= 0.25:
                return img
        return img

    async def read_status(self, hold: bool = False) -> None:
        """Open Status, let Muse read it, close it. With hold the panel stays up while Muse
        reads (it's on camera, and it's what's being read); otherwise it closes right away and
        the reading finishes behind the next thing, through self.pending."""
        await self._press("rt")
        img = await self.until(status_open, 1.0)
        if img is None:
            self.note("mh_note", {"text": "status didn't open"})
            await self.back_out()
            return
        await asyncio.sleep(0.12)
        img = await self._shot()
        w, h = img.size
        crop = img.crop((940 * w // 1280, 70 * h // 720, w, 365 * h // 720))
        elements = ", ".join(f"{x.name}: {x.element or 'none'}" for x in self.weapons) or "unknown"

        async def reading() -> None:
            self.status = await self.ask("status", [image_part(crop, STATUS_PX)],
                                         READ_STATUS.format(elements=elements), 1500)
            self.what_line = str(self.status.get("say") or "")
            # Spoken, so a one-letter misreading of a name it knows is put right.
            name = str(self.status.get("name") or "")
            right = difflib.get_close_matches(name.lower(), KNOWN_TYPES, 1, 0.8)
            if right and right[0] != name.lower():
                self.what_line = re.sub(re.escape(name), right[0].title(), self.what_line, flags=re.IGNORECASE)
                self.status["name"] = right[0].title()

        job = asyncio.ensure_future(reading())
        if hold:
            with contextlib.suppress(Exception):
                await job
        else:
            self.pending.append(job)
        await self._press("rt", 0.15)
        if not menu_up(await self._shot()):
            await self._press("b", 0.2)

    async def read_arc(self, wheel: Image.Image) -> tuple[str, str]:
        """The streak sweeps and flares, so one picture can mislead: three, a moment apart. Any
        red makes it red: "it's coming at you" is only said when nothing else was seen."""
        seen = [arc_seen(wheel)]
        for _ in range(2):
            await asyncio.sleep(0.12)
            seen.append(arc_seen(await self._shot()))
        for colour in ("red", "blue", "yellow"):
            hits = [x for x in seen if x[0] == colour]
            if hits:
                return hits[-1]
        return "none", ""

    async def read_wheel(self, say_count: bool = False) -> asyncio.Future | None:
        """Round the wheel once with the camera watching, then one look by Muse at all of it,
        which finishes behind the next thing: the job is returned and in self.pending. Pixels
        check what Muse says the diamonds show. A small monster has no wheel. With say_count the
        count is said from the pixels and the wheel stays up a moment while it is."""
        await self._press("a")
        img = await self.until(wheel_open, 1.2)
        if img is None or not wheel_has_parts(img):
            await self._press("b", 0.3)
            self.parts, self.weak_line = [], ""
            self.note("mh_parts", {"parts": []})
            return None
        seen: list[tuple[np.ndarray, list[str], bool, float, Image.Image]] = []
        for _ in range(6):
            img = await self.settled_wheel()
            mask = wheel_title(img)
            if seen and _same(mask, seen[0][0]):
                break
            w, h = img.size
            seen.append((mask, wheel_works(img), part_broken(img), wheel_gauge(img),
                         img.crop((980 * w // 2560, 170 * h // 1440,
                                   1680 * w // 2560, 770 * h // 1440))))
            await self._press("rb", 0.1)
        # What the pixels say is good to go on right away; Muse adds the names and the words.
        self.parts = [Part(f"part {i + 1}", works, broken, gauge, "", mask)
                      for i, (mask, works, broken, gauge, _) in enumerate(seen)]
        sheet = Image.new("RGB", (len(seen) * 600, 514))
        for i, entry in enumerate(seen):
            sheet.paste(entry[4].resize((600, 514)), (i * 600, 0))
        if say_count:
            line = self.count_line()
            if line:
                self.say(line)

        async def reading() -> None:
            d = await self.ask("wheel", [image_part(sheet, WHEEL_PX)],
                               READ_WHEEL.format(weapons=self.weapon_words()), 1500)
            said = d.get("parts") or []
            names = [str(x.get("name") or "").lower() for x in said]
            for i, name in enumerate(names):  # round the wheel twice: the name comes up again
                if name and name in names[:i]:
                    said = said[:i]
                    break
            if said and len(self.parts) > len(said):
                self.parts = self.parts[:len(said)]
            for i, part in enumerate(self.parts):
                read = said[i] if i < len(said) else {}
                works = [str(x).lower() for x in read.get("works") or []]
                if works != part.works:  # the diamonds are plain pixels: they win
                    self.muse_agrees = False
                    self.note("mh_note", {"text": f"wheel: Muse {works} pixels {part.works}"})
                part.name = str(read.get("name") or part.name)
                part.works = part.works or works
                part.bonus = str(read.get("bonus") or "")
            self.weak_line = str(d.get("say") or "")
            self.note("mh_parts", {"parts": [(p.name, p.works, p.broken, round(p.gauge, 2), p.bonus)
                                             for p in self.parts], "said": self.weak_line})

        job = asyncio.ensure_future(reading())
        self.pending.append(job)
        if say_count:
            await asyncio.sleep(WHEEL_HOLD)
        await self._press("b", 0.25)
        return job

    def count_line(self) -> str:
        """What the pixels already know, while Muse reads the names."""
        count = {2: "two", 3: "three", 4: "four", 5: "five"}.get(len(self.parts))
        if not count:
            return ""
        fits = [next((w.name for w in self.weapons if w.damage in p.works), "") for p in self.parts]
        if all(fits) and len(set(fits)) == len(fits):
            return f"You can break {count} parts, each with a different weapon."
        return f"You can break {count} parts."

    def where_weak(self) -> str:
        """Muse's words, if Muse read every diamond the way the pixels do and its line names each
        unbroken part; otherwise plain ones built from the wheel."""
        left = [p for p in self.parts if not p.broken]
        low = self.weak_line.lower()
        if self.weak_line and len(self.weak_line) < 170 and left and muse_line_fits(
                low, left, self.weapons):
            return self.weak_line
        if not left:
            return "Everything on it is broken already. Just hit it." if self.parts else \
                "It's small: any weapon works on it."
        bits = []
        for p in left:
            fit = next((w for w in self.weapons if w.damage in p.works), None)
            what = short(fit.name)[4:] if fit and short(fit.name).startswith("the ") else (
                " or ".join(p.works))
            bits.append(f"{what} on the {p.name.lower()}")
        said = ", ".join(bits)
        return said[0].upper() + said[1:] + "."

    async def settle_reads(self, most: float | None = None) -> None:
        """Wait for Muse's readings, for at most `most` seconds. One that's later than that goes
        on behind the fight, and what it says counts from the turn after it lands."""
        if not self.pending:
            return
        _, left = await asyncio.wait(self.pending, timeout=most)
        for job in self.pending:
            if job.done() and not job.cancelled():
                job.exception()  # a failed read is logged by ask, and retried by no one
        self.pending = list(left)

    async def study(self, say_weak: bool = False) -> None:
        """What Muse learns once a fight: the wheel, then Status, read at the same time once
        both have been on screen. Each is logged with its model time; the camera sees what's
        being read; something true is said before the wait."""
        t = time.monotonic()
        wheel = not self.parts or not all(named(p) for p in self.parts)
        if wheel:
            self.say(self.lines.pick("Let me get a look at it.", "Checking it out."))
        if not self.weapons:
            await self.load_gear()
        if wheel:
            await self.read_wheel()
            await self.until(menu_up, 1.0)
        if not self.status and not self.pending:
            await self.read_status()
            await self.until(menu_up, 1.0)
        await self.settle_reads(None if wheel else STATUS_WAIT)
        self.studied = True
        self.note("mh_study", {"seconds": round(time.monotonic() - t, 2),
                               "monster": self.status.get("name"),
                               "parts": [p.name for p in self.parts]})
        if say_weak:
            self.say(self.where_weak())

    # Gear stays between takes: a walk through three pages of skills is not for the camera, and
    # it only changes when the player changes it. Muse read it, once; `read_gear` refreshes it.

    async def load_gear(self) -> None:
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            data = json.loads(GEAR_FILE.read_text())
            self.weapons = [Weapon(w["name"], w["damage"], w.get("element", ""),
                                   [Skill(**x) for x in w["skills"]]) for w in data]
            self.note("mh_gear", {"from": str(GEAR_FILE), "weapons": data})
            return
        await self.read_gear()
        with contextlib.suppress(OSError):
            GEAR_FILE.parent.mkdir(parents=True, exist_ok=True)
            GEAR_FILE.write_text(json.dumps(
                [{"name": w.name, "damage": w.damage, "element": w.element,
                  "skills": [vars(x) for x in w.skills]} for w in self.weapons], indent=1))

    # One turn

    def reset_fight(self) -> None:
        self.studied = False
        self.weapon_synced = False
        self.parts = []
        self.status = {}
        self.weak_line = self.what_line = ""
        self.target = self.seen_type = self.ended = ""
        self.lines = Lines()
        self.turns = 0
        self.monster_share = 1.0
        self.break_heard = False
        self.healed = False
        self.won = False
        self.muse_agrees = True

    def view(self, img: Image.Image, wheel: Image.Image, kind_now: str) -> View:
        hint, hint_from = "", ""
        for who in ("Ogden", "Chirpy"):
            hint = ribbon(wheel, who)
            if hint:
                hint_from = who
                break
        colour, toward = arc_seen(wheel)
        return View(self.parts, self.weapons, self.weapon, kind_now, colour, toward,
                    self.seen_type, hint, hint_from, ribbon(wheel, "Canyne"),
                    {who: bar_share(wheel, who) for who in ALLIES},
                    round(bar_share(wheel, "Zerk", 698) * 100), self.target,
                    [str(x).lower() for x in self.status.get("resists") or []],
                    self.usual())

    def usual(self) -> str:
        known = KNOWN_TYPES.get(str(self.status.get("name") or "").lower())
        if not known:
            return ""
        state = str(self.status.get("state") or "normal").lower()
        return known[0] if "normal" in state else known[1]

    def current_part(self, img: Image.Image) -> Part | None:
        mask = wheel_title(img)
        scored = [(_title_gap(mask, p.mask), i) for i, p in enumerate(self.parts)
                  if p.mask is not None]
        if not scored or min(scored)[0] > 0.7 * max(int(mask.sum()), 1):
            return None  # a part that only shows once another has broken, like the stomach
        return self.parts[min(scored)[1]]

    async def turn_wheel(self, part: Part) -> bool:
        """Round the open wheel to this part, checking its title each step. A monster up in the
        air has only a small ring with one target: nothing to turn."""
        for _ in range(len(self.parts) + 2):
            img = await self.settled_wheel()
            if ring_is_small(img):
                return False
            if self.current_part(img) is part:
                part.broken = part.broken or part_broken(img)
                part.gauge = wheel_gauge(img)
                return True
            await self._press("rb", 0.1)
        return False

    async def set_type(self, kind: str, img: Image.Image) -> bool:
        """Put the command menu on that attack type, by its colour."""
        for _ in range(3):
            now = menu_type(img)
            if now == kind:
                return True
            step = (TYPES.index(kind) - TYPES.index(now)) % 3 if now in TYPES else 1
            await self._flick("right" if step == 1 else "left")
            img = await self._shot()
        return menu_type(img) == kind

    async def to_skills(self, plan: Plan) -> bool:
        """In the Skills menu: the right weapon's page, and the cursor on the skill (or on the
        plain attack). Reads the cursor; a skill that's greyed falls back to the plain attack."""
        await self._press("x")
        img = await self.until(lambda i: skills_tab(i) is not None, 1.2)
        if img is None:
            return False
        for _ in range(3):
            tab = skills_tab(img)
            if tab == plan.weapon:
                break
            await self._press("rb" if (plan.weapon - (tab or 0)) % 3 == 1 else "lb", 0.25)
            img = await self._shot()
        if skills_tab(img) != plan.weapon:
            return False
        want = 0
        if plan.do == "skill" and plan.skill:
            await asyncio.sleep(0.2)  # the rows fade in; a half-faded one reads as greyed
            img = await self._shot()
            skills = self.weapons[plan.weapon].skills
            wanted = [plan.skill, *sorted(
                (x for x in skills if x.deals_damage and x.targets == "one" and x.power > 50
                 and x.cost > 0 and x is not plan.skill), key=lambda x: -x.power)]
            for skill in wanted:
                row = skills.index(skill) + 1 if skill in skills else 0
                if row and not skill_greyed(img, row):
                    want, plan.skill = row, skill
                    break
                self.note("mh_note", {"text": f"{skill.name} is greyed"})
        if not want:
            plan.do, plan.skill = "attack", None
        for _ in range(5):
            now = skill_row(img)
            if now == want:
                return True
            await self._flick("down" if want > now else "up", 0.2)
            img = await self._shot()
        return skill_row(img) == want

    async def select(self, plan: Plan, kind_now: str) -> str:
        """From the open wheel to everything selected, wheel on the part. "ready" to confirm,
        "played" when a skill wanted no target and already went, "" when it went wrong and the
        menu's been left as it was."""
        change_weapon = plan.weapon != self.weapon
        if change_weapon or plan.do == "skill" or (plan.type in TYPES and plan.type != kind_now):
            await self._press("b")
            img = await self.until(menu_up, 1.2)
            if img is None:
                return ""
            if plan.type in TYPES and not await self.set_type(plan.type, img):
                self.note("mh_note", {"text": f"menu stayed off {plan.type}"})
            if (change_weapon or plan.do == "skill") and not await self.to_skills(plan):
                await self.back_out()
                return ""
            await self._press("a")
            img = await self.until(lambda i: wheel_open(i) or not (
                skills_tab(i) is not None or menu_up(i)), 1.0)
            if img is None:
                self.keep_frame(await self._shot(), "no-wheel")
                await self.back_out()
                return ""
            # The list can be gone a moment before the wheel is up.
            if self.parts and not wheel_open(img) and await self.until(wheel_open, 0.7) is None:
                return "played"
        self.weapon = plan.weapon
        if plan.part is not None and not await self.turn_wheel(plan.part):
            self.note("mh_note", {"text": f"wheel never showed {plan.part.name}"})
            plan.part = None  # not checked, so not named: a far-off monster's wheel is small
        return "ready"

    def keep_frame(self, img: Image.Image, tag: str) -> None:
        if os.environ.get("PLAYER_TWO_FRAMES"):
            with contextlib.suppress(OSError):
                img.convert("RGB").save(Path(os.environ["PLAYER_TWO_FRAMES"]).expanduser()
                                        / f"mh-{time.strftime('%H%M%S')}-{tag}.jpg", quality=85)

    async def back_out(self) -> None:
        for _ in range(3):
            if menu_up(await self._shot()):
                return
            await self._press("b", 0.3)

    async def confirm(self) -> None:
        """A on the open wheel, checked: it has to close. A small monster's ring gets one press."""
        for _ in range(2 if self.parts else 1):
            await self._press("a")
            if await self.until(lambda i: not wheel_open(i), 0.6) is not None:
                return

    async def use_item(self, plan: Plan) -> bool:
        """From the wheel: back to the menu, the item's row (read, the list keeps its cursor),
        the ring on the ally, confirm. False, with the menu back, if any step didn't show."""
        await self._press("b")
        if await self.until(menu_up, 1.2) is None:
            return False
        await self._press("y")
        img = await self.until(items_open, 1.2)
        if img is None:
            await self.back_out()
            return False
        want = ("Herb", "Potion").index(plan.item) if plan.item in ("Herb", "Potion") else 1
        for _ in range(9):
            now = lit_row(img, ITEM_ROWS)
            if now == want:
                break
            await self._flick("down" if (now or 0) < want else "up", 0.2)
            img = await self._shot()
        if lit_row(img, ITEM_ROWS) != want:
            await self.back_out()
            return False
        await self._press("a", 0.3)
        for _ in range(4):
            img = await self._shot()
            if ring_who(img) == plan.on:
                break
            await self._flick("right", 0.25)
        else:
            await self.back_out()
            return False
        self.say(self.lines.choice(View([], [], 0, ""), plan, False))
        await self._press("a", 0.1)
        return True

    async def play_out(self) -> Image.Image | None:
        """The turn plays: watch for the monster's type (Head-to-Head banner, its ribbon), take
        a game message into account, and return at the next menu. None when the fight's over. A menu left open gets backed out of quickly: the party waits forever
        on a menu."""
        t0 = time.monotonic()
        hits, toast_band, looking = 0, None, None
        nearly_won = 0 < self.monster_share < 0.25
        last_screen, over, nudge = t0 + (3.0 if nearly_won else 9.0), 0, t0 + 14.0
        looks, worst, before = 0, 0.0, time.monotonic()

        def seen_all() -> None:
            self.note("mh_watch", {"looks": looks, "worst_gap": round(worst, 2),
                                   "seconds": round(time.monotonic() - t0, 1)})

        while time.monotonic() - t0 < 70 and not self.a.cancelled:
            img = await self._shot()
            stamp = time.monotonic()
            looks, worst, before = looks + 1, max(worst, stamp - before), stamp
            if menu_up(img):
                hits += 1
                if hits >= 2:
                    seen_all()
                    return img
                continue
            hits = 0
            now = time.monotonic()
            if now - t0 > 1.2 and (wheel_open(img) or skills_tab(img) is not None or items_open(img)):
                await self._press("b", 0.3)
                continue
            if now >= nudge:  # this long with no menu: one that no check knows may be open
                nudge = now + 10.0
                await self._press("b", 0.2)
            kind = clash_type(img) or monster_ribbon(img)
            if kind and kind in ANSWER and kind != self.seen_type:
                self.seen_type = kind
                self.note("mh_seen", {"type": kind})
            if toast_up(img):
                band = _arr(img, 420, 574, 860, 596).mean(axis=2)
                # On first sight: at double speed a message is up for under a second.
                if toast_band is None or float(np.abs(band - toast_band).mean()) > 12:
                    toast_band = band
                    self.note("mh_note", {"text": "message band up"})
                    self.aside(self.read_message(img))
            else:
                toast_band = None
            if self.won:
                self.ended = "won"
                return None
            if looking is None and now >= last_screen:
                looking = asyncio.ensure_future(self.screen(img))
            elif looking is not None and looking.done():  # Muse's look doesn't hold up the watching
                seen = looking.result()
                looking = None
                last_screen = now + (1.0 if nearly_won else 2.5)
                over = over + 1 if seen in ("results", "field") else 0
                self.ended = seen if over else ""
                if over >= 2 or (seen == "results" and not hud_up(img)):
                    return None
            await asyncio.sleep(0.1)
        return None

    async def read_message(self, img: Image.Image) -> None:
        """The game's own message band. A change of phase means its type may change too."""
        w, h = img.size
        fresh = await self._shot()  # it may have faded in further by now
        if toast_up(fresh):
            img = fresh
        crop = img.crop((330 * w // 1280, 566 * h // 720, 950 * w // 1280, 604 * h // 720))
        with contextlib.suppress(Exception):
            d = await self.ask("message", [image_part(crop, 800)], READ_TOAST, 200, 5)
            text = str(d.get("text") or "").lower()
            # A message read twice as it fades in is one message.
            now = time.monotonic()
            if text and any(now - when < TOAST_AGAIN and difflib.SequenceMatcher(
                    None, text, old).ratio() > 0.85 for old, when in self.toasts):
                return
            self.toasts = [*self.toasts[-3:], (text, now)]
            if self.parts and any(w in text for w in ("slay", "slain", "its den", "retreating")):
                self.won = True
                return
            if "broke" in text and "part" in text:  # which one, the next wheel shows
                self.break_heard = True
                self.say(self.lines.pick("That's a part broken!", "Another part broken!"))
            if "got us" in text:  # a rider or partner down, and a heart with them
                self.say(self.lines.pick("We lost a heart.", "That's another heart gone."))
            if "toppled" in text:
                self.say(self.lines.pick("It's down!", "Knocked it over again."))
                self.aside(self.synchro_rush())
            if "double attack" in text:
                self.say(self.lines.pick(
                    "There's the double attack." if "zerk" in text
                    else "Ogden and Chirpy doubled up." if "ogden" in text else "Double attack."))
            if "normal" in text or "angry" in text or "strange" in text:
                self.seen_type = ""
                if self.status:  # what it usually hits with goes by this
                    self.status["state"] = "Normal" if "normal" in text else "Angry"
                self.note("mh_seen", {"type": "", "phase": text})
                self.say(self.lines.pick("It's calmed down." if "normal" in text
                                         else "Careful, it's changed gears."))

    async def synchro_rush(self) -> None:
        """The game's Y prompt while the monster's toppled. Never into a menu: Y there is Items."""
        for _ in range(4):
            if menu_up(await self._shot()):
                return
            await self._press("y", 0.05)
            await asyncio.sleep(0.25)
        self.note("mh_note", {"text": "pressed Y for the Synchro Rush"})

    async def screen(self, img: Image.Image) -> str:
        try:
            d = await self.ask("still_fight", [image_part(img, 768)], CHECK, 400, 15)
        except Exception:  # noqa: BLE001  not known: the fight goes on
            return "fight"
        return str(d.get("screen") or "fight")

    async def take_turn(self, img: Image.Image, request: str, heal: bool = False) -> Plan | None:
        """Look, say why, select, say what, confirm. No model call in here: Muse's reading of the
        monster was done once, in study."""
        t = time.monotonic()
        self.turns += 1
        said = request.lower()
        for p in self.parts:
            if p.name.lower() in said:
                self.target = p.name
        kind_now = menu_type(img)
        await self._press("a")
        # A big monster's wheel is checked; a small one's ring is small and moves about.
        wheel = await self.until(wheel_open if self.parts else (lambda i: not menu_up(i)), 1.2)
        if wheel is None:
            await self.back_out()
            self.note("mh_note", {"text": "wheel didn't open"})
            return None
        wheel = await self.settled_wheel(0.9) if self.parts else await self._shot()
        arc, toward = await self.read_arc(wheel)
        if os.environ.get("PLAYER_TWO_FRAMES"):  # what each turn was picked from (debugging)
            with contextlib.suppress(OSError):
                wheel.convert("RGB").save(Path(os.environ["PLAYER_TWO_FRAMES"]).expanduser()
                                          / f"mh-{time.strftime('%H%M%S')}.jpg", quality=85)
        # A part this turn's wheel opened on may have broken since it was last looked at.
        on = self.current_part(wheel)
        just_broke = ""
        if on is not None:
            broken = part_broken(wheel)
            if broken and not on.broken and on.name == self.target:
                just_broke = on.name
            # A part stays broken: the word flickers as the wheel zooms in, and a reading of
            # "not broken" put it back on the list (live: a broken stomach hit again and again).
            on.broken, on.gauge = on.broken or broken, wheel_gauge(wheel)
        view = await asyncio.to_thread(self.view, img, wheel, kind_now)
        view.arc, view.toward = arc, toward
        low = monster_health(wheel)
        self.monster_share = low
        view.top_up = not self.healed and low > 0.2
        plan = choose(view)
        if heal and plan.do != "item":
            worst = min(ALLIES, key=lambda who: view.hp.get(who, 1.0) or 1.0)
            plan = Plan("item", item="Potion", on=worst if view.hp.get(worst, 1.0) < 0.9 else "Zerk",
                        type=view.menu_type, weapon=view.in_hand)
        t_read = time.monotonic() - t
        # Said now, while the presses go on: all true of this moment.
        reaction = self.lines.broke(just_broke, self.break_heard) if just_broke else (
            self.lines.monster_low(low) if self.parts else "")
        self.break_heard = False
        if plan.do == "item":
            first = self.lines.hurt(plan.on, view.hp.get(plan.on, 0.0))
        else:
            first = self.lines.reason(view, plan)
            if self.turns == 1 and view.arc not in ("yellow", "blue"):
                first = self.lines.opening(view, plan) or first
        self.say(" ".join(x for x in (reaction, first) if x))
        swapped = plan.weapon != self.weapon
        if plan.do == "item":
            ok = await self.use_item(plan)
            state = "ready" if ok else ""
            self.healed = self.healed or ok
            if not ok:
                self.note("mh_note", {"text": "item didn't go through"})
        else:
            state = await self.select(plan, kind_now)
            if state in ("ready", "played"):
                self.say(self.lines.choice(view, plan, swapped))
                if plan.part is not None:
                    self.target = plan.part.name
            if state == "ready":
                await self.confirm()
        self.note("mh_turn", {
            "read_s": round(t_read, 2), "seconds": round(time.monotonic() - t, 2),
            "state": state, "arc": view.arc, "toward": view.toward, "do": plan.do,
            "type": plan.type, "why_type": plan.why_type, "double": plan.double,
            "part": plan.part.name if plan.part else "",
            "weapon": view.weapons[plan.weapon].name if view.weapons else "",
            "skill": plan.skill.name if plan.skill else "", "item": plan.item, "on": plan.on,
            "hp": {k: round(v, 2) for k, v in view.hp.items()}, "gauge": view.gauge,
            "monster_health": round(low, 2), "seen_type": view.monster_type, "usual": view.usual_type,
            "hint": view.hint_type, "canyne": view.canyne_type,
            # What Muse read that this turn's choice rests on, and how long each reading took.
            "muse_informed": {k: self.muse_seconds.get(k) for k in (
                ["wheel"] if plan.part else []) + (["skill"] if plan.skill else [])
                + (["status"] if self.status else [])}})
        return plan

    async def run(self, request: str, mode: str, deadline: float = 240.0) -> str:
        """Plays what was asked; returns anything left to tell the player."""
        for _ in range(3):  # a menu left open by hand: back to the commands
            if menu_up(await self._shot()):
                break
            await self._press("b", 0.3)
        img = await self.until(menu_up, 6.0)
        if img is None:
            return "" if mode in ("turn", "what", "weak", "health", "heal") \
                else "There's no fight to play right now."
        if self.ended:
            self.reset_fight()
        if mode == "health":
            return self.lines.health({who: bar_share(img, who) for who in ALLIES})
        if mode == "what":
            await self.settle_reads(STATUS_WAIT)
            if not self.status:
                if not self.weapons:
                    await self.load_gear()
                await self.read_status(hold=True)
            name = self.status.get("name")
            return self.what_line or (f"It's a {name}." if name else "I couldn't read that one.")
        if mode == "weak":
            if not self.parts or not all(named(p) for p in self.parts):
                if not self.weapons:
                    await self.load_gear()
                job = await self.read_wheel(say_count=True)
                if job is not None:
                    # Status too, behind the answer, so a fight asked for next starts at once.
                    await self.until(menu_up, 1.0)
                    if not self.status:
                        await self.read_status()
                    with contextlib.suppress(Exception):
                        await job
            return self.where_weak()
        await self.settle_reads(STATUS_WAIT)
        if not self.parts or not self.status or not all(named(p) for p in self.parts):
            await self.study(say_weak=mode == "fight" and not self.parts)
            img = await self.until(menu_up, 3.0) or img
        if not self.weapon_synced:
            await self.sync_weapon()
            img = await self.until(menu_up, 1.0) or img
        self.studied = True
        t0 = time.monotonic()
        while not self.a.cancelled and time.monotonic() - t0 < deadline:
            await self.take_turn(img, request, heal=mode == "heal")
            if mode in ("turn", "heal"):
                await self.said()
                return ""
            next_img = await self.play_out()
            if next_img is None:
                break
            img = next_img
        await self.said()
        if self.a.cancelled:
            return ""
        if not self.ended:
            return "I lost track of the fight there. Say describe to hear where things stand."
        won = self.won
        self.reset_fight()
        if not won:
            await self._press("a", 1.0)  # past the rank screen
        return "That's the fight. We won."


def _title_gap(a: np.ndarray, b: np.ndarray) -> int:
    """How far apart two title masks are. The wheel can sit a few pixels off, so the best of the
    small shifts counts."""
    k = max(1, a.shape[1] // 180)  # the grab can be bigger than 1280 wide
    return min(int(np.logical_xor(a, np.roll(np.roll(b, dx, axis=1), dy, axis=0)).sum())
               for dx in range(-12 * k, 12 * k + 1, 2 * k) for dy in (-2 * k, 0, 2 * k))


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Two wheel titles are the same word."""
    total = max(int(a.sum()), int(b.sum()), 1)
    return int(np.logical_xor(a, b).sum()) / total < 0.25
