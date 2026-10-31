"""Chrono Trigger (Steam): its battle and menu screens read from pixels, no model call.

The profile for glyphread: the canvas (any 16:9 frame is scaled to 2560x1440), the glyph
library learned from live frames, the pointing-hand and target-glove templates, the bracket
drawn round the fighter whose command is up, and the game's words.

Measured on 37 held-out live frames (research/design/battle): screen kind 37/37, the entry
under the hand 33/33, the actor 17/17, party HP 63/63 and MP 63/63; about 100 to 160 ms.

Layout facts (Steam, 2026-11-01): the party box is on the right of the bottom windows and
each ready fighter gets a command window in their party slot to its left (Attack, Tech or
Combo when two or more are ready, Item). The actor's name is yellow; a grey name is critical
HP or knocked out. Tech and Item lists are two columns, three rows in view. The windows were
16 px lower in one session, so nothing here depends on their exact height: lines are found
in a tall box and told apart by what they say.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from .glyphread import GlyphLib, Run, Template, find_template, runs_in, scaled, snap, to_canvas

DATA = Path(__file__).parent / "data" / "ct_steam"
CANVAS = (2560, 1440)
TOP = (390, 16, 2180, 132)  # the message / description bar
BOTTOM = (200, 1150, 2380, 1438)  # the command, list and party windows
PAUSE_BOX = (980, 520, 1580, 720)  # where "Pause" shows
SLOT_TEXT_X = (309, 723, 1138)  # where "Attack" starts in each party slot's command window
COMMANDS = ("Attack", "Tech", "Item")  # row order; "Combo" stands in for Tech
NAMES = ("Crono", "Marle", "Lucca", "Robo", "Frog", "Ayla", "Magus")
ITEMS = ("Potion", "Mid-Potion", "Hi-Potion", "Full Potion", "Ether", "Mid-Ether",
         "Hi-Ether", "Elixir", "Megalixir", "Panacea", "Athenian Water", "Revive", "Lapis",
         "Shelter", "Barrier Sphere", "Shield Sphere", "Heal", "Power Tab", "Magic Tab",
         "Speed Tab")
TECHS = {
    "Crono": ("Cyclone", "Wind Slash", "Lightning", "Cleave", "Lightning II", "Raise",
              "Frenzy", "Luminaire"),
    "Marle": ("Aura", "Allure", "Ice", "Cure", "Haste", "Ice II", "Cure II", "Arise"),
    "Lucca": ("Flamethrower", "Hypnowave", "Fire", "Napalm", "Protect", "Fire II",
              "Mega Bomb", "Flare"),
    "Frog": ("Slurp", "Slurp Slash", "Water", "Heal", "Aerial Strike", "Water II",
             "Cure II", "Frog Squash"),
    "Robo": ("Rocket Punch", "Cure Beam", "Laser Spin", "Robo Tackle", "Heal Beam",
             "Uzzi Punch", "Area Bomb", "Shock"),
    "Ayla": ("Kiss", "Rollo Kick", "Cat Attack", "Rock Throw", "Charm", "Tail Spin",
             "Dino Tail", "Triple Kick"),
    "Magus": ("Lightning II", "Ice II", "Fire II", "Dark Bomb", "Magic Wall", "Dark Mist",
              "Black Hole", "Dark Matter"),
}
DUAL = ("Aura Whirl", "Ice Sword", "Ice Sword II", "Fire Whirl", "Fire Sword", "X-Strike",
        "Sword Stream", "Lightning Rod", "Ice Water", "Frog Launcher", "Bubble Breath",
        "Slurp Kiss", "Bubble Burst", "Red Needle", "Antipode Bomb", "Aura Beam",
        "Ice Tackle", "Cure Touch", "Rocket Roll", "Max Cyclone", "Super Volt",
        "Double Cure", "Twin Charm", "Ice Toss", "Cube Toss", "Flame Kick", "Fire Tackle",
        "Blaze Kick", "Drill Kick", "Volt Bite", "Falcon Hit", "Spin Strike",
        "Line Bomb", "Double Bomb", "[Dual Tech]", "[Triple Tech]")
VOCAB = sorted({*ITEMS, *(t for ts in TECHS.values() for t in ts), *DUAL})
# Field menu words, for the menu focus reader.
MENU_WORDS = (*VOCAB, "Equipment", "Inventory", "Techs", "Party", "Settings", "Bookmark",
              "Save", *NAMES, "Use / Switch", "Sort list", "Battle", "Sound", "Graphics",
              "Controls", "Defaults", "Quit", "Consumables")
NUM = re.compile(r"(\d+)\s*[/t ]\s*(\d+)")
# Words in the top bar after a won fight.
SPOILS = re.compile(r"(?i)\b(earned|exp|tp\b|level|obtained|found|got|received|\d+\s*g\b)")


@cache
def lib() -> GlyphLib:
    return GlyphLib.load(DATA / "glyphs.npz")


@cache
def templates() -> dict[str, list[Template]]:
    hand = cv2.imread(str(DATA / "hand.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(DATA / "hand_mask.png"), cv2.IMREAD_GRAYSCALE)
    br = cv2.imread(str(DATA / "bracket.png"), cv2.IMREAD_GRAYSCALE)
    bm = cv2.imread(str(DATA / "bracket_mask.png"), cv2.IMREAD_GRAYSCALE)
    # The hand was cut from 1280x720 frames; matching runs at a quarter of the canvas.
    return {
        "hand": [scaled(hand, mask, 0.5, "hand")],
        # The target glove is the same hand drawn bigger.
        "glove": [scaled(hand, mask, 0.5 * s, "glove") for s in (1.75, 1.9)],
        "bracket": [Template(br, (bm > 0).astype(np.uint8) * 255, "bracket")],
    }


def fits(image) -> bool:
    """Whether a frame is the Steam port's shape (16:9), not an emulator's 8:7 or 4:3."""
    h, w = (image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1])
    return h > 0 and abs(w / h - 16 / 9) < 0.06


@dataclass
class Member:
    name: str
    hp: int | None
    hp_max: int | None
    mp: int | None
    mp_max: int | None
    colour: str

    @property
    def down(self) -> bool:
        return self.hp == 0

    @property
    def share(self) -> float:
        """HP as a share of the most."""
        return (self.hp or 0) / self.hp_max if self.hp_max else 1.0


@dataclass
class State:
    kind: str = "none"  # command | list | target | none
    paused: bool = False
    columns: list[int] = field(default_factory=list)  # party slots with a command window
    actor_col: int | None = None  # the slot whose window has the hand
    command: str = ""  # Attack, Tech or Item under the hand ("Combo" reads as Tech)
    combo: bool = False  # the second command says Combo
    entries: list[list[str]] = field(default_factory=list)  # list rows in view
    usable_rows: list[list[bool]] = field(default_factory=list)
    entry: str = ""
    entry_rc: tuple[int, int | None] | None = None
    usable: bool | None = None
    party: list[Member] = field(default_factory=list)
    actor: str = ""
    top: str = ""
    gloves: list[tuple[int, int]] = field(default_factory=list)  # glove tips
    hand: tuple[int, int] | None = None  # the menu hand's tip
    bracket: tuple[int, int] | None = None  # centre of the bracket round the actor
    ms: dict = field(default_factory=dict)

    @property
    def in_battle(self) -> bool:
        return bool(self.party) or self.kind in ("list", "target")

    def member(self, name: str) -> Member | None:
        return next((m for m in self.party if m.name.lower() == name.lower()), None)

    def ready(self) -> list[str]:
        """Who has a command window open."""
        return [self.party[c].name for c in self.columns if c < len(self.party)]

    def to_dict(self) -> dict:
        return asdict(self)


def numfix(s: str) -> str:
    """Pixel-font look-alikes of digits, between digits and slashes."""
    t = s.translate(str.maketrans({"l": "1", "I": "1", "i": "1", "|": "1", "O": "0",
                                   "o": "0", "S": "5", "B": "8", "D": "0"}))
    return re.sub(r"(?<=/)\s+|\s+(?=/)", "", t)


def _hp_run(row: list[Run]) -> int | None:
    for i, r in enumerate(row):
        if re.match(r"H[PF]\b|HP", r.text):
            return i
    return None


def is_paused(img: np.ndarray) -> bool:
    """The Start pause: everything dimmed (the bottom windows' white text too) and white
    "Pause" in the middle. Live pause shots: whole-frame 98th percentile 127, bottom text
    127, 2% white in the middle; field menus 183 to 212; unpaused battles 255."""
    if float(np.percentile(img[::8, ::8].max(axis=2), 98)) >= 160:
        return False
    x0, y0, x1, y1 = BOTTOM
    if float(np.percentile(img[y0:y1:2, x0:x1:2].max(axis=2), 99.7)) >= 200:
        return False
    x0, y0, x1, y1 = PAUSE_BOX
    if float((img[y0:y1, x0:x1].min(axis=2) > 200).mean()) <= 0.005:
        return False
    # The battle windows are still there, dimmed (not a dark field with someone in white).
    return sum(len(row) for row in runs_in(lib(), img, BOTTOM)) >= 4


def _slot(x: int) -> int:
    return int(np.argmin([abs(c - x) for c in SLOT_TEXT_X]))


def _near(run: Run, tip: tuple[int, int]) -> bool:
    return abs(run.y - tip[1]) < 30 and -25 <= run.x0 - tip[0] < 60


def _names(row: list[Run]) -> list[Run]:
    """List entries in a row: not counts, MP costs or specks."""
    return [r for r in row if not re.fullmatch(r"[\d\-:.,]+", r.text)
            and not r.text.startswith("MP") and len(r.text.strip("*.,'-")) > 1]


def read(image, full: bool = True) -> State:
    """The battle screen as data. `full` also reads the top bar (about 20 ms more)."""
    t0 = time.perf_counter()
    img = to_canvas(image if isinstance(image, np.ndarray) else _bgr(image), CANVAS)
    st = State()
    tp = templates()
    hs = find_template(img, tp["hand"] + tp["glove"], CANVAS)
    t1 = time.perf_counter()
    st.gloves = [(x + w, y + h // 2) for x, y, w, h, _s, n in hs if n == "glove"]
    menu = [h for h in hs if h[5] == "hand" and h[1] > 1100]
    if menu:
        st.hand = (menu[0][0] + menu[0][2], menu[0][1] + menu[0][3] // 2)
    blank = [h[:4] for h in hs]
    lines = runs_in(lib(), img, BOTTOM, blank)
    t2 = time.perf_counter()
    party_x = None
    for row in lines:
        i = _hp_run(row)
        if i is None or i == 0:
            continue
        name_run = row[i - 1]
        if row[i].x0 - name_run.x1 > 700:
            continue
        party_x = name_run.x0 if party_x is None else min(party_x, name_run.x0)
        nums = NUM.findall(numfix(" ".join(r.text for r in row[i:])))
        hp = nums[0] if nums else (None, None)
        mp = nums[1] if len(nums) > 1 else (None, None)
        who = snap(name_run.text, NAMES, 0.5)
        if who not in NAMES:
            continue
        st.party.append(Member(who,
                               *(int(n) if n is not None else None for n in (*hp, *mp)),
                               name_run.colour))
    st.actor = next((m.name for m in st.party if m.colour == "yellow"), "")
    left = [[r for r in row if party_x is None or r.x1 < party_x - 20] for row in lines]
    left = [row for row in left if row]
    words = ("Attack", "Tech", "Combo", "Item")
    cmd_runs = []
    for row in left:
        for r in row:
            w = snap(r.text, words, 0.6)
            if w in words and abs(len(r.text) - len(w)) <= 1:
                cmd_runs.append((w, r))
    if st.party and cmd_runs:
        st.kind = "command"
        st.combo = any(w == "Combo" for w, _ in cmd_runs)
        st.columns = sorted({_slot(r.x0) for w, r in cmd_runs if w == "Attack"})
        if st.hand:
            cand = [(w, r) for w, r in cmd_runs if _near(r, st.hand)]
            if cand:
                w, r = cand[0]
                st.command = "Tech" if w == "Combo" else w
                st.actor_col = _slot(r.x0)
    elif left and not st.party and st.hand is not None:
        st.kind = "list"
        for ri, row in enumerate(left):
            names = _names(row)
            st.entries.append([snap(r.text, VOCAB) for r in names])
            st.usable_rows.append([r.colour != "grey" for r in names])
            if st.hand:
                for ci, r in enumerate(names):
                    if _near(r, st.hand):
                        st.entry = snap(r.text, VOCAB)
                        st.usable = r.colour != "grey"
                        st.entry_rc = (ri, ci)
    if st.kind == "list" and not any(e in VOCAB and not e.startswith("[")
                                     for row in st.entries for e in row):
        # Not a battle list (a text box with a choice, a field menu): no known names.
        st.kind, st.entries, st.usable_rows, st.entry, st.entry_rc = "none", [], [], "", None
        st.usable = None
    if st.kind == "command" and not st.actor and st.actor_col is not None \
            and st.actor_col < len(st.party):
        st.actor = st.party[st.actor_col].name  # a critical-HP actor is drawn grey
    if st.gloves:
        st.kind = "target"
    if st.kind == "command":
        br = find_template(img, tp["bracket"], CANVAS, thresh=0.6, roi=(0, 0, 2560, 1260),
                           most=1)
        if br:
            x, y, w, h, _s, _n = br[0]
            st.bracket = (x + w // 2, y + h // 2)
    t3 = time.perf_counter()
    if full:
        st.top = " ".join(r.text for row in runs_in(lib(), img, TOP, blank) for r in row)
    st.paused = is_paused(img)
    t4 = time.perf_counter()
    st.ms = {"cursor": round((t1 - t0) * 1e3), "bottom": round((t2 - t1) * 1e3),
             "logic": round((t3 - t2) * 1e3), "top": round((t4 - t3) * 1e3),
             "total": round((t4 - t0) * 1e3)}
    return st


def _bgr(image) -> np.ndarray:
    from .glyphread import bgr

    return bgr(image)


def describe(st: State) -> str:
    """The state in words, for the model and the logs."""
    parts = []
    if st.party:
        who = []
        for m in st.party:
            tag = " (knocked out)" if m.down else (" (low)" if m.share < 0.34 else "")
            mp = f", MP {m.mp}/{m.mp_max}" if m.mp is not None else ""
            who.append(f"{m.name} HP {m.hp}/{m.hp_max}{mp}{tag}")
        parts.append("; ".join(who))
    if st.kind == "command":
        ready = st.ready()
        parts.append(f"{st.actor or 'someone'} is choosing a command"
                     + (f" (also ready: {', '.join(r for r in ready if r != st.actor)})"
                        if len(ready) > 1 else ""))
    elif st.kind == "list":
        rows = " | ".join(", ".join(r) for r in st.entries)
        parts.append(f"a list is open ({rows}), the hand on {st.entry or 'nothing readable'}")
    elif st.kind == "target":
        parts.append("choosing a target" + (f" ({st.top})" if st.top else ""))
    if st.top and st.kind != "target":
        parts.append(f"top bar: {st.top}")
    if st.paused:
        parts.append("paused")
    return ". ".join(parts) + "." if parts else "nothing readable"


# ---------------------------------------------------------------------------------------
# Field menus (the y menu): what the hand points at.

@dataclass
class MenuFocus:
    name: str | None
    colour: str | None
    tip: tuple[int, int]


def menu_read(image) -> tuple[list[MenuFocus], list[list[Run]]]:
    """Every hand on a field menu and what it points at, plus all the text lines."""
    img = to_canvas(image if isinstance(image, np.ndarray) else _bgr(image), CANVAS)
    hs = find_template(img, templates()["hand"], CANVAS)
    lines = runs_in(lib(), img, (0, 0, CANVAS[0], CANVAS[1]), [h[:4] for h in hs],
                    local=True)
    out = []
    for hx, hy, hw, hh, _s, _n in hs:
        cy, x = hy + hh // 2, hx + hw
        cand = [r for row in lines for r in row if abs(r.y - cy) < 34
                and -25 <= r.x0 - x < 140 and len(r.text.strip(" .,'-")) > 1]
        cand.sort(key=lambda r: r.x0)
        name = snap(cand[0].text, MENU_WORDS, 0.6) if cand else None
        if name not in MENU_WORDS:
            # A hand on a character block points at the portrait; the name is up and right.
            near = [r for row in lines for r in row
                    if -90 < r.y - cy < 34 and -25 <= r.x0 - x < 420]
            near = [(snap(r.text, NAMES, 0.6), r) for r in near]
            near = [(n, r) for n, r in near if n in NAMES]
            if near:
                name = min(near, key=lambda t: abs(t[1].y - cy) + 0.1 * abs(t[1].x0 - x))[0]
        out.append(MenuFocus(name, cand[0].colour if cand else None, (x, cy)))
    return out, lines
