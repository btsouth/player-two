"""Chrono Trigger (Steam): what a screen reader would say, read from pixels with no model.

`look` reads one frame into a `Look`: the kind of screen (battle, a menu, a question with
choices, a text box, or none of those) and the parts worth saying (whose turn, what the
cursor is on, text to read, who is in danger). `Reader` turns a stream of looks into lines:
each part is said when it changes, text only once it has stopped typing out, and nothing
twice. Frames the reader can't make sense of are left to the model narrator.

Glyphs come from ctsteam's library of the game's pixel font; a run of text is only trusted
when it reads as words (no strings of outline specks), so a textured field or world map
never turns into made-up text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ctsteam
from .glyphread import Run, find_template, runs_in, snap, to_canvas

CANVAS = ctsteam.CANVAS
# Below this width the font's pixels are two screen pixels or less and glyphs misread (live
# 960x540 previews): such frames are left to the model.
MIN_WIDTH = 1600
SURE = 20.0  # a run whose worst glyph is further than this from the library may be misread
DESC_Y = 1220  # field menus' description bars are below this
DIALOGUE = 0.75  # dialogue is drawn 4/3 the size of menu text: it's read scaled to menu size
BOX_BANDS = ((0, 520), (900, 1440))  # text boxes sit at the top or the bottom
# Words that may stand alone as a menu entry or an option.
OPTIONS = ("Yes", "No", "Empty", "Resume", "New Game", "Load Game", "Extras", "Quit",
           "Use / Switch", "Sort list", "Consumables", "ON", "OFF", "WAIT", "ACTIVE",
           "ACTIONS", "Battle Mode", "Battle Speed", "Battle Message Speed",
           "Battle Cursor Memory", "Help Messages")
VOCAB = tuple(sorted({*ctsteam.MENU_WORDS, *OPTIONS}))
OPTION = re.compile(r"([A-Z][A-Za-z']*(?:[ ,] ?[A-Za-z']+)*)[.!]?\s*$")  # a choice's words
END = re.compile(r"[.?!:\"')\]]$")  # text that ends like a finished sentence


def plausible(r: Run) -> bool:
    """Whether a run reads as words or numbers rather than texture or outline specks."""
    t = r.text.strip()
    if not t or r.worst > 30:
        return False
    body = t.replace(" ", "")
    if re.search(r"['\-,.:]{2,}|^['\-,.]|[\[\]]", t.replace("...", "")) and t not in VOCAB:
        return False
    alnum = sum(c.isalnum() for c in body)
    if alnum / len(body) < 0.75:
        return False
    if re.fullmatch(r"[\d :/G]+", t):
        return True
    if snap(t, VOCAB, 0.8) in VOCAB:
        return True
    if misread(t):
        return False
    words = re.findall(r"[A-Za-z]{3,}", t)
    return any(re.search(r"[aeiouyAEIOUY]", w) for w in words)


def misread(text: str) -> bool:
    """Words that can't be right: digits inside words ("Red0ces", "Com6o", "1ce"), capitals
    after small letters ("SlQp"), or a word of consonants. The font's look-alikes (0 and o,
    1 and l, 5 and S) show up this way when glyphs are small or dimmed."""
    for w in re.findall(r"[A-Za-z0-9']+", text):
        if re.fullmatch(r"\d+(st|nd|rd|th|s|G|HP|MP|TP|EXP)?|[A-Z]{1,4}\d*|\d+[A-Z]", w):
            continue
        if re.search(r"[A-Za-z]", w) and re.search(r"\d", w):
            return True
        if re.search(r"[a-z][A-Z]", w.replace("'", "")) and w not in ("KOd",):
            return True
        if len(w) >= 4 and not re.search(r"[aeiouyAEIOUY]", w) and not w.isupper():
            return True
        if re.search(r"([aeiouAEIOU])\1\1", w):
            return True  # "uuuu": icons read as letters (a drawn-out "Ahhh" is consonants)
    return False


def clean(text: str) -> str:
    """Pixel-font look-alikes in running text."""
    t = " ".join(text.split())
    t = re.sub(r"(?<=[a-z])9(?=$|\s)", "?", t)  # "game9" is "game?"
    # Letters that came apart on a busy battle bar: k's leg ("Attacki"), m's leg ("arri").
    t = re.sub(r"(?<=[a-z]ck)i\b", "", t)
    t = re.sub(r"\barri\b", "arm", t)
    return re.sub(r"\bTes\b", "Yes", t)


@dataclass
class Look:
    kind: str = "none"  # battle | menu | choice | text | none
    parts: dict[str, str] = field(default_factory=dict)  # said in ORDER, each on change
    texts: set[str] = field(default_factory=set)  # parts that are text: said once
    typing: set[str] = field(default_factory=set)  # text parts that may still be typing
    battle: ctsteam.State | None = None
    lines: list[list[Run]] = field(default_factory=list)
    focus: str = ""
    ms: int = 0
    order: tuple[str, ...] = ()
    names: list[str] = field(default_factory=list)  # characters in order (party blocks)
    unsure: bool = False  # the text box doesn't read cleanly: the model reads it

    @property
    def readable(self) -> bool:
        return self.kind != "none"


ORDER = ("pause", "title", "turn", "focus", "question", "top", "text", "desc", "danger")


def _danger_items(st: ctsteam.State) -> list[tuple[str, str]]:
    """Who's in danger, each with what to say: knocked out, or under a third of their HP."""
    out = []
    for m in st.party:
        if m.hp is None or not m.hp_max:
            continue
        if m.hp == 0:
            out.append((m.name, f"{m.name} is down"))
        elif m.hp < m.hp_max / 3:
            out.append((m.name, f"{m.name} {m.hp} HP"))
    return out


def _danger(st: ctsteam.State) -> str:
    return ", ".join(t for _n, t in _danger_items(st))


def battle_look(st: ctsteam.State) -> Look:
    lk = Look("battle", battle=st)
    if st.paused:
        lk.parts["pause"] = "Paused"
    if st.kind == "command" and st.actor:
        lk.parts["turn"] = f"{st.actor}'s turn"
        if st.command:
            lk.parts["focus"] = st.command
    elif st.kind == "list" and st.entry:
        # (Whether it can be used isn't said: a pause dims every entry, usable or not.)
        entry = st.entry if st.entry in ctsteam.VOCAB else snap(st.entry, ctsteam.VOCAB, 0.6)
        if entry in ctsteam.VOCAB or not misread(entry):
            lk.parts["focus"] = entry
    top = clean(st.top)
    if top and plausible(Run(top, 0, 0, 0, "", 0)):
        lk.parts["top"] = top
        lk.texts.add("top")  # the top bar comes up whole, never typed out
    dz = _danger(st)
    if dz:
        lk.parts["danger"] = dz
    lk.names = [m.name for m in st.party]
    lk.focus = lk.parts.get("focus", "")
    # A message ("Combo Attack: Reduces HP to 1") is what just happened, so it comes
    # first; in a list the top bar describes the entry under the hand, so after it.
    lk.order = (("pause", "turn", "focus", "top", "danger") if st.kind == "list" else
                ("pause", "top", "turn", "focus", "danger"))
    return lk


MAIN = ("Equipment", "Inventory", "Techs", "Party", "Settings", "Bookmark", "Save")
SETTINGS = ("Battle", "Sound", "Graphics", "Controls", "Defaults")
TITLE = ("Resume", "New Game", "Load Game", "Extras", "Settings", "Quit")
ORDERS = (MAIN, SETTINGS, TITLE)


def _column_entry(lines: list[list[Run]], tip: tuple[int, int]) -> str | None:
    """A hand's entry in a list whose order is known, counted in rows from the readable
    entries in its column: the one under the hand often reads badly through its
    highlight, and the rows aren't evenly spaced."""
    col = [r for row in lines for r in row if -30 <= r.x0 - tip[0] < 420]
    ys: list[float] = []
    for y in sorted([r.y for r in col] + [tip[1]]):
        if ys and y - ys[-1] < 30:
            continue
        ys.append(y)
    here = min(range(len(ys)), key=lambda j: abs(ys[j] - tip[1]))
    for order in ORDERS:
        votes: dict[str, int] = {}
        for r in col:
            w = snap(r.text, order, 0.75)
            if w not in order or abs(r.y - tip[1]) < 30:
                continue
            j = min(range(len(ys)), key=lambda j: abs(ys[j] - r.y))
            k = order.index(w) + here - j
            if 0 <= k < len(order):
                votes[order[k]] = votes.get(order[k], 0) + 1
        if votes and sum(votes.values()) >= 2:
            best = max(votes, key=votes.get)
            if votes[best] >= 0.6 * sum(votes.values()):
                return best
    return None


def _block_name(lines: list[list[Run]], tip: tuple[int, int]) -> str | None:
    """A hand on a character block points at the portrait; the name is up and right."""
    near = [(snap(r.text, ctsteam.NAMES, 0.6), r) for row in lines for r in row
            if -170 < r.y - tip[1] < 40 and 0 <= r.x0 - tip[0] < 420]
    near = [(n, r) for n, r in near if n in ctsteam.NAMES]
    if not near:
        return None
    return min(near, key=lambda t: abs(t[1].y - tip[1]) + 0.1 * abs(t[1].x0 - tip[0]))[0]


BLOCK_DY = 372  # character blocks in the techs and target lists are this far apart


def _block_by_order(lines: list[list[Run]], tip: tuple[int, int],
                    party: list[str]) -> str | None:
    """The highlighted block's name, counted in blocks from a readable name in the same
    column, in the party's menu order (the highlight makes the name itself unreadable)."""
    votes: dict[str, int] = {}
    for row in lines:
        for r in row:
            if not 0 <= r.x0 - tip[0] < 450:
                continue
            n = snap(r.text, ctsteam.NAMES, 0.75)
            if n not in party:
                continue
            k = party.index(n) + round((tip[1] - 50 - r.y) / BLOCK_DY)
            if 0 <= k < len(party) and party[k] != n:
                votes[party[k]] = votes.get(party[k], 0) + 1
    if not votes:
        return None
    best = max(votes, key=votes.get)
    return best if votes[best] >= 0.6 * sum(votes.values()) else None


def _reread(img: np.ndarray, tip: tuple[int, int]) -> tuple[str, Run | None]:
    """The entry right of a hand, read again on its own: across the whole screen its line
    can share a band with the next panel, and coloured text on bright metal (usable techs
    and items, green) needs a lower bar."""
    box = (tip[0] + 4, max(0, tip[1] - 90), min(CANVAS[0], tip[0] + 1400),
           min(CANVAS[1], tip[1] + 90))
    for kw in ({}, {"margin": 40, "ring_bg": True}):
        for row in runs_in(ctsteam.lib(), img, box, local=True, **kw):
            for r in row:
                if abs(r.y - tip[1]) < 36 and r.x0 - tip[0] < 420:
                    w = snap(r.text, VOCAB, 0.7)
                    if w in VOCAB:
                        return w, r
    return "", None


def _hand_entry(lines: list[list[Run]], tip: tuple[int, int],
                party: list[str] = (), img: np.ndarray | None = None) -> tuple[str, Run | None]:
    """What a hand points at, and the run it was read from (None when inferred)."""
    row = sorted((r for row in lines for r in row if abs(r.y - tip[1]) < 36
                  and -30 <= r.x0 - tip[0] < 1400 and plausible(r)), key=lambda r: r.x0)
    near = [r for r in row if r.x0 - tip[0] < 420]
    if near:
        w = snap(near[0].text, VOCAB, 0.7)
        if w in VOCAB:
            return w, near[0]
    if img is not None:
        w, r = _reread(img, tip)
        if w:
            return w, r
    w = _column_entry(lines, tip) or _block_name(lines, tip) or (
        _block_by_order(lines, tip, list(party)) if party else None)
    if w:
        return w, None
    if row:
        w = snap(row[0].text, VOCAB, 0.7)
        if w in VOCAB or not misread(row[0].text):
            return (w if w in VOCAB else clean(row[0].text)), row[0]
    return "", None


def field_look(img: np.ndarray, party: list[str] = ()) -> Look:
    """Menus, questions and text boxes (not battle)."""
    hands = find_template(img, ctsteam.templates()["hand"], CANVAS)
    lines = _quotes(runs_in(ctsteam.lib(), img, (0, 0, *CANVAS), [h[:4] for h in hands],
                            local=True))
    lk = Look(lines=lines)
    good = [r for row in lines for r in row if plausible(r)]
    blocks = sorted((r.y, n) for r in good if (n := snap(r.text, ctsteam.NAMES, 0.75))
                    in ctsteam.NAMES and 400 < r.x0 < 1700)
    lk.names = [n for _y, n in blocks]
    question = next((r for r in good if clean(r.text).endswith("?")), None)
    tips = [(h[0] + h[2], h[1] + h[3] // 2) for h in hands]
    found = [(tip, *_hand_entry(lines, tip, party, img)) for tip in tips]
    found = [f for f in found if f[1]]
    pool = good  # where a choice's options are read
    talk = False  # a choice in dialogue: its options and question at dialogue size
    if tips and any(r.text.lower().startswith("funds") for row in lines for r in row):
        return _shop_look(lk, tips, lines, good)
    known = set(VOCAB) | set(ctsteam.NAMES) | set(ctsteam.ITEMS)
    if tips and question is None and not any(f[1] in known for f in found):
        # A question in dialogue (the inn's "Would you like to stay for the night?"): the
        # options beside the hand are drawn at dialogue size, and the choice window sits
        # above a box at the bottom, so the question can be below the options.
        rows = [r for row in _dialogue_lines(img) for r in row]
        big = [r for r in rows if plausible(r)]
        asked = next((r for r in big if clean(r.text).endswith("?")), None)
        if asked is not None:
            # The box's last question ("...for a while? Wouldn't that be great?").
            asked = max((r for r in big if clean(r.text).endswith("?")
                         and (r.y < 520) == (asked.y < 520) and abs(r.y - asked.y) < 300),
                        key=lambda r: r.y)
        talk_found: list = []
        said: list[Run] = []
        for tip in tips:
            # The options in the hand's column, read on their own: on a whole line the
            # hand and the window's edge run into the first one ("goreI" for "Sure.").
            col = _choice_column(img, tip)
            here = [r for r in col if abs(r.y - tip[1]) < 45]
            if here:
                talk_found.append((tip, here[0].text, here[0]))
                said = said or col
        if talk_found:
            x = talk_found[0][0][0]
            column = [r for r in said if abs(r.y - talk_found[0][0][1]) < 300]
            if asked is not None:
                # The question is the whole box it ends ("Shelters are quite useful. Do you
                # know about them?"), not just its last line.
                band = [r for r in big if (r.y < 520) == (asked.y < 520)
                        and abs(r.y - asked.y) < 300 and r.y <= asked.y
                        and abs(r.x0 - asked.x0) < 120]
                if band:
                    asked = Run(" ".join(r.text for r in sorted(band, key=lambda r: r.y)),
                                asked.x0, asked.x1, asked.y, asked.colour,
                                max(r.worst for r in band))
                found, talk, question, pool = talk_found, True, asked, said
            elif len(column) >= 2 and len({r.y // 30 for r in column}) >= 2:
                # The question went with the box before ("Take a look." / "No, thanks."
                # under an empty box): the options alone, the hand's first.
                found, talk, pool = talk_found, True, said
                question = Run("", x, x, talk_found[0][0][1], "white", 0.0)
    if question is not None and not talk:
        below = [f for f in found if f[0][1] > question.y and abs(f[0][0] - question.x0) < 900]
        found = below or found
    blocks = [f for f in found if f[1] in ctsteam.NAMES and f[2] is None]
    if len(blocks) >= 2 and len(blocks) == len(found):
        # One hand on each character: the target is all of them.
        names = [f[1] for f in sorted(blocks, key=lambda f: f[0][1])]
        found = [(blocks[0][0], ", ".join(names[:-1]) + " and " + names[-1], None)]
    if found:
        tip, name, run = found[0]
        lk.focus = name
        if question is None and name in ("Yes", "No"):
            # A yes or no whose question didn't read cleanly: the line just above the
            # options, said by the model once it has typed out.
            above = [r for row in lines for r in row if 0 < tip[1] - r.y < 400
                     and abs(r.x0 - tip[0]) < 900 and snap(r.text, VOCAB, 0.7) not in VOCAB]
            if above:
                y = max(r.y for r in above)
                question = Run(" ".join(r.text for r in sorted(above, key=lambda r: r.x0)
                                        if abs(r.y - y) < 30), min(r.x0 for r in above),
                               0, y, "white", 99.0)
                lk.unsure = True
        if question is not None and (question.y < tip[1] or talk):
            lk.unsure = lk.unsure or question.worst > SURE or misread(question.text)
            lk.kind = "choice"
            lk.order = ("question", "focus", "desc")
            if question.text:
                lk.parts["question"] = clean(question.text)
                lk.texts.add("question")
                lk.typing.add("question")
            x = tip[0] if talk or run is None else run.x0
            opts = [r for r in pool if r is not question and abs(r.x0 - x) < 160 and (
                abs(r.y - tip[1]) < 300 if talk else
                r.y > question.y and r.y - question.y < 700)]
            names = []
            for r in sorted(opts, key=lambda r: r.y):
                # (A dialogue option is said as read: "Sure" isn't the tech "Cure".)
                w = r.text if talk else snap(r.text, VOCAB, 0.7)
                names.append(w if w in VOCAB else clean(r.text).rstrip("." if talk else ""))
            others = [n for n in names if n != name]
            lk.parts["focus"] = f"{name} selected" + (f", or {', '.join(others)}"
                                                       if others else "")
        else:
            lk.kind = "menu"
            lk.parts["focus"] = name
            slot = [r for row in lines for r in row if abs(r.y - tip[1]) < 36
                    and r.x1 <= tip[0] and tip[0] - r.x1 < 300 and r.text.strip().isdigit()]
            if slot and run is not None and run.x0 - tip[0] > 420:
                lk.parts["focus"] = f"{slot[0].text.strip()}, {name}"  # a save slot
            if run is not None:
                # A setting's value is centred under it in capitals: "WAIT", "5 (NORMAL)".
                mid = (run.x0 + run.x1) / 2
                val = [r for r in good if 40 < r.y - run.y < 160 and abs((r.x0 + r.x1) / 2 - mid) < 40
                       and re.fullmatch(r"[A-Z0-9][A-Z0-9 ()/%.-]*[A-Z)]", r.text.strip())]
                if val:
                    v = re.sub(r"[()]", "", val[0].text).split()
                    lk.parts["focus"] += ", " + " ".join(w if w.isdigit() else w.capitalize()
                                                         for w in v)
            head = [r for r in good if r.y < 160 and len(r.text) > 12 and finished(r.text)
                    and r.text.strip()[-1] in ".?:"]
            if head:
                lk.parts["title"] = clean(head[0].text)
                lk.texts.add("title")
                lk.typing.add("title")
            if run is not None and name in ctsteam.ITEMS:
                count = [r for row in lines for r in row if abs(r.y - run.y) < 36
                         and 0 < r.x0 - run.x1 < 900 and re.fullmatch(r":?\s*\d+", r.text)]
                if count:
                    lk.parts["focus"] = f"{name}, {count[0].text.lstrip(':').strip()}"
        desc = [r for r in good if r.y > DESC_Y and len(re.findall(r"[A-Za-z]{2,}", r.text)) >= 2
                and not re.search(r"\(C\)|\b(SQUARE|ENIX|STUDIO|SHUEISHA)\b", r.text)]  # title credits
        if desc:
            lk.parts["desc"] = clean(" ".join(r.text for r in sorted(desc, key=lambda r: r.x0)))
            lk.texts.add("desc")
            lk.typing.add("desc")
        if lk.kind == "menu" and name in MAIN:
            # The main page shows the party's HP: who's in danger, as in a fight.
            low = [f"{n} is down" if h == 0 else f"{n} {h} HP"
                   for n, h, top in _menu_hp(lk, list(party)) or [] if h < top / 3]
            if low:
                lk.parts["danger"] = ", ".join(low)
        return lk
    if question is not None:
        lk.kind = "choice"
        lk.parts["question"] = clean(question.text)
        lk.texts.add("question")
        lk.typing.add("question")
        return lk
    # A text box: lines of words across the top or bottom of the screen. Every run on
    # those lines is kept (dropping one would skip words); if any reads badly, the box is
    # unsure, and the model reads it once it has finished typing.
    box = [row for row in lines if row and (row[0].y < 520 or row[0].y > 900)]
    rows = _box_rows(box)
    if any(len(r.text) >= 8 and r.x1 - r.x0 > 300 and len(re.findall(r"[A-Za-z0-9]{2,}", r.text))
           >= 2 for row in box for r in row):
        # Long lines that may be dialogue, which is drawn bigger: read them at its size.
        big = _box_rows(_dialogue_lines(img))
        if _clean_chars(big) > _clean_chars(rows):
            rows, hands = big, []
    if hands:
        return lk  # a menu whose highlighted entry doesn't read: the model sees it
    if rows and sum(len(r.text) for row in rows for r in row) >= 12:
        # Specks on the box's frame, far left or right of the words: not words, however
        # long they read ("-t T" and ".t ." either side of the fair seller's second
        # line). A word that misreads sits among the others.
        ok = [r for row in rows for r in row if plausible(r)]
        lo, hi = min(r.x0 for r in ok), max(r.x1 for r in ok)
        rows = [[r for r in row if plausible(r) or lo - 60 <= r.x0 <= hi + 60]
                for row in rows]
        text = clean(" ".join(r.text for row in rows for r in row))
        lk.kind = "text"
        lk.parts["text"] = text
        lk.texts.add("text")
        lk.typing.add("text")
        lk.unsure = misread(text) or not all(plausible(r) and r.worst <= SURE
                                             for row in rows for r in row)
    return lk


def _shop_look(lk: Look, tips, lines, good) -> Look:
    """A shop screen: what the hand is on with its price (or the count and the total), the
    item's help line, the funds."""
    from .shop import parse

    v = parse(tips, lines)
    lk.kind = "menu"
    if v.kind == "main" and v.on:
        lk.parts["focus"] = v.on
        if v.funds is not None:
            lk.parts["desc"] = f"Funds {v.funds} G"
    elif v.kind == "count" and v.on:
        lk.parts["focus"] = f"{v.on}, {v.count}" + (f", {v.total} G" if v.total else "")
    elif v.kind == "list" and v.on:
        g = next((g for g in v.goods if g.name == v.on), None)
        lk.parts["focus"] = v.on + (f", {g.price} G" if g and g.price else "")
        help_ = [r for r in good if 840 < r.y < 960 and r.x0 < 800
                 and len(re.findall(r"[A-Za-z]{2,}", r.text)) >= 2]
        if help_:
            lk.parts["desc"] = clean(" ".join(r.text for r in help_))
            lk.texts.add("desc")
    lk.focus = lk.parts.get("focus", "")
    if not lk.focus:
        lk.kind = "none"  # the model looks
    return lk


def _quotes(lines: list[list[Run]]) -> list[list[Run]]:
    """A double quote is two ticks, read as two apostrophes."""
    for row in lines:
        for r in row:
            r.text = r.text.replace("''", '"')
    return lines


def _box_rows(lines: list[list[Run]]) -> list[list[Run]]:
    return [row for row in lines if any(plausible(r) and len(re.findall(r"[A-Za-z]{2,}", r.text))
                                        >= 2 for r in row)]


def _clean_chars(rows: list[list[Run]]) -> int:
    return sum(len(r.text) for row in rows for r in row if plausible(r) and r.worst <= SURE)


def _choice_column(img: np.ndarray, tip: tuple[int, int]) -> list[Run]:
    """A dialogue choice's options in the hand's column (the words just right of where
    the hand points, on its row and the rows above and below), read at dialogue size in a
    box of their own. Each Run's text is the option without its full stop."""
    x0, y0 = tip[0] + 4, max(0, tip[1] - 260)
    x1, y1 = min(CANVAS[0], tip[0] + 800), min(CANVAS[1], tip[1] + 260)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=DIALOGUE, fy=DIALOGUE,
                      interpolation=cv2.INTER_AREA)
    out = []
    for row in runs_in(ctsteam.lib(), crop, (0, 0, crop.shape[1], crop.shape[0]), local=True):
        r = row[0]
        x = x0 + round(r.x0 / DIALOGUE)
        if x - tip[0] > 60 or r.worst > SURE:
            continue  # not an option starting just right of the hand's column
        o = OPTION.search(r.text)
        w = o.group(1) if o else ""
        if (snap(w, VOCAB, 0.8) in VOCAB or len(w) >= 3) and not misread(w):
            out.append(Run(w, x, x0 + round(r.x1 / DIALOGUE), y0 + round(r.y / DIALOGUE),
                           r.colour, r.worst))
    return out


def _dialogue_lines(img: np.ndarray) -> list[list[Run]]:
    """Lines in the text box bands read at dialogue size, in canvas coordinates."""
    out = []
    for y0, y1 in BOX_BANDS:
        band = cv2.resize(img[y0:y1], None, fx=DIALOGUE, fy=DIALOGUE, interpolation=cv2.INTER_AREA)
        for row in _quotes(runs_in(ctsteam.lib(), band, (0, 0, band.shape[1], band.shape[0]),
                                   local=True)):
            out.append([Run(r.text, round(r.x0 / DIALOGUE), round(r.x1 / DIALOGUE),
                            y0 + round(r.y / DIALOGUE), r.colour, r.worst) for r in row])
    return out


def sharp(image) -> bool:
    """Whether a frame is big enough to read the game's font from."""
    w = image.shape[1] if isinstance(image, np.ndarray) else image.size[0]
    return w >= MIN_WIDTH


def look(image, party: list[str] = ()) -> Look:
    """`party`: the characters in menu order, as earlier screens showed them (Reader.party)."""
    t0 = time.perf_counter()
    img = to_canvas(image if isinstance(image, np.ndarray) else ctsteam._bgr(image), CANVAS)
    st = ctsteam.read(img)
    lk = battle_look(st) if st.in_battle or st.paused else field_look(img, party)
    lk.ms = round((time.perf_counter() - t0) * 1e3)
    return lk


def finished(text: str) -> bool:
    """Text that ends like a whole sentence or label, not cut off mid-typing (a dash ends
    a box that the next one carries on: "you're going to go see-")."""
    t = text.strip()
    return bool(END.search(t) or re.search(r"[a-z]-$", t)) or t in VOCAB


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "").replace("'", ""))


def covered(text: str, lines: list[str], share: float = 0.75) -> bool:
    """Whether one of these lines says this text: most of its words are in it, in any
    order ("The man says: Shelters are quite useful. Do you know about them? Yes or No?"
    covers "Shelters are quite useful. Do you know about them?")."""
    want = [w for w in _words(text) if len(w) > 1 or w.isdigit()]
    if not want:
        return False
    for line in lines:
        have = set(_words(line))
        if sum(w in have for w in want) >= share * len(want):
            return True
    return False


class Reader:
    """Lines to say from successive looks: each part when it changes, text once it's whole.

    A part of kind text (a description, a message, a question, dialogue) is said once two
    looks agree on it and it ends like a sentence, or three agree whatever its end; the
    rest (whose turn, the cursor) as soon as it's seen. The same text isn't said twice
    within `forget_s`, unless the screen went elsewhere and came back to a new box.
    """

    def __init__(self, forget_s: float = 30.0):
        self.forget_s = forget_s
        self.party: list[str] = []  # characters in menu order, as screens have shown them
        self.active: list[str] = []  # the three in the party now, as a battle showed them
        self.last: list[str] = []  # the parts said by the last update
        self.model_read = False  # a whole text box that doesn't read cleanly: ask the model
        self.fight_started = False  # a battle just came up: the model says what we face
        self.verbosity = "normal"  # the narrator's setting: "quiet" says less
        self.team_up = False  # the game reads its own menus
        self.hp: dict[str, tuple[int, int]] = {}  # party HP as last read steadily
        self.hp_new: dict[str, tuple[int, int]] = {}  # and as the last look read it
        self.text_since: float | None = None  # when a box that doesn't read cleanly came up
        self.text_handed = False  # and whether the model has been asked to read it
        self.kind = "none"
        self.said: dict[str, str] = {}
        self.seen: dict[str, tuple[str, int]] = {}  # part -> (value, looks in a row)
        self.texts: dict[str, float] = {}  # text said -> when
        self.heard = ""  # the last text box read out, for "what did he say?"
        self.danger_at: dict[str, float] = {}  # who's low, as said -> when
        self.battle_at = -1e9  # when a battle was last looked at (a new fight after a gap)
        self.danger_on: set[str] = set()  # danger in this fight said (or covered) and standing
        self.told: list[str] = []  # Player Two's own lines just now (the agent's), and when
        self.told_at = -1e9

    def just_said(self, lines: list[str], now: float | None = None) -> None:
        """Player Two has just said these in its own words (the agent read a box out, or
        asked about it): text on screen they cover isn't said again (live: "Shelters are
        quite useful..." in the agent's question, then from the reader once it finished)."""
        self.told = [t for t in lines if t]
        self.told_at = time.monotonic() if now is None else now

    def _covered(self, value: str, now: float) -> bool:
        if not self.told or now - self.told_at > self.forget_s:
            return False
        return covered(value, self.told)

    def reset(self) -> None:
        self.kind, self.said, self.seen = "none", {}, {}
        self.hp, self.hp_new = {}, {}

    def _hits(self, lk: Look) -> tuple[str, set[str], set[str]]:
        """Party HP that changed since the last steady reading: "Frog took 493, down to 1".
        Only a reading two looks agree on counts (an attack can cover the numbers), only a
        change of a tenth of someone's HP or more, and a knockout is left to danger. Also
        who the line covers (their low HP is said in it) and whose HP is still changing
        (their danger waits a look, to be said with the hit)."""
        st = lk.battle
        if st is None or not st.party or any(m.hp is None or not m.hp_max for m in st.party):
            return "", set(), set()
        now = {m.name: (m.hp, m.hp_max) for m in st.party}
        steady = now == self.hp_new
        self.hp_new = now
        if not steady:
            moving = {n for n, (hp, _t) in now.items() if n in self.hp and self.hp[n][0] != hp}
            return "", set(), moving
        before, self.hp = self.hp, now
        out, covered = [], set()
        for name, (hp, top) in now.items():
            if name not in before:
                continue
            old = before[name][0]
            if hp == old or hp == 0 or abs(hp - old) < top / 10:
                continue
            if old == 0:
                out.append(f"{name} is back up")
                covered.add(name)
            elif hp < old:
                low = hp < top / 3
                out.append(f"{name} took {old - hp}" + (f", down to {hp}" if low else ""))
                if low:
                    covered.add(name)
            else:
                out.append(f"{name} healed {hp - old}")
        return ", ".join(out), covered, set()

    def _danger_news(self, lk: Look, now: float, covered: set[str], moving: set[str]) -> str:
        """Who's newly in danger in a fight, one by one: a knockout at once; low HP once its
        hit is steady (said with it, not before it). Danger that stands isn't said again;
        if the party box drops out and comes back, it's news after 20 s."""
        out, on = [], set()
        for name, text in _danger_items(lk.battle):
            if name in moving and not text.endswith("is down"):
                if text in self.danger_on:
                    on.add(text)
                continue
            on.add(text)
            if text in self.danger_on:
                continue
            if name in covered or now - self.danger_at.get(text, -1e9) < 20 \
                    or self._told_danger(name, text, now):
                self.danger_at.setdefault(text, now)
                continue
            self.danger_at[text] = now
            out.append(text)
        self.danger_on = on
        return ", ".join(out)

    def _told_danger(self, name: str, text: str, now: float) -> bool:
        """Player Two's own line just now gave this already (live 3: "Frog's Slurp healed
        Frog: 101 of 494." then the reader's "Frog 101 HP.")."""
        if not self.told or now - self.told_at > 15:
            return False
        num = re.search(r"\d+", text)
        for t in self.told:
            if name not in t:
                continue
            if text.endswith("is down") and re.search(r"\b(down|knocked out)\b", t, re.IGNORECASE):
                return True
            if num and re.search(rf"\b{num.group()}\b", t):
                return True
        return False

    def prime(self) -> None:
        """Player Two just acted and said what came of it: HP read before that isn't the
        baseline any more ("Crono took 779" after "the combo dropped Crono to 1 HP")."""
        self.hp, self.hp_new = {}, {}

    @property
    def waiting(self) -> bool:
        """Text on screen not said yet: another look may find it whole."""
        return any(v and self.said.get(p) != v and n < 3 for p, (v, n) in self.seen.items())

    def learn_party(self, names: list[str]) -> None:
        """Merge an order of characters seen on screen into the party order."""
        names = list(dict.fromkeys(names))
        if len(names) < 2:
            return
        if all(n in self.party for n in names):
            return
        if all(n in names for n in self.party):
            self.party = names
            return
        for i, n in enumerate(names):
            if n not in self.party:
                before = names[i - 1] if i else None
                at = self.party.index(before) + 1 if before in self.party else len(self.party)
                self.party.insert(at, n)

    def update(self, lk: Look, now: float | None = None, first: bool = False) -> str:
        """What to say for this look ('' for nothing). `first` reads the look as the only
        one there will be: text counts as whole when it ends like a sentence."""
        now = time.monotonic() if now is None else now
        self.learn_party(lk.names)
        if lk.kind == "battle" and len(lk.names) >= 2:
            self.active = list(lk.names)  # a fight shows exactly who is in the party
        self.last = []
        opened = lk.kind != self.kind
        new_fight = False
        if opened:
            self.said, self.seen = {}, {}
            # A new fight, not the same one back after a look that didn't read as one.
            new_fight = lk.kind == "battle" and not lk.parts.get("pause") and \
                now - self.battle_at > 8
            self.fight_started = self.fight_started or new_fight
            if new_fight:
                self.danger_on = set()
            self.kind = lk.kind
            self.hp, self.hp_new = {}, {}
        if lk.kind == "battle":
            self.battle_at = now
        if lk.kind == "none":
            return ""
        hits, covered, moving = self._hits(lk)
        if lk.battle is not None:
            lk.parts["danger"] = self._danger_news(lk, now, covered, moving)
        if self.verbosity == "quiet":
            hits = ""  # quiet: text, decisions and danger only
        if lk.kind == "text" and lk.unsure:
            if self.text_since is None:
                self.text_since = now
            if now - self.text_since > 2.5 and not self.text_handed:
                # It never reads the same twice (misreads vary): the model reads it now,
                # unless Player Two has just said it.
                self.said["text"] = lk.parts.get("text", "") or "?"
                self.text_handed = True
                self.model_read = not self._covered(self.said["text"], now)
                self.heard = ""  # what it says is the model's to tell
        else:
            self.text_since, self.text_handed = None, False
        out = []
        for part in lk.order or ORDER:
            if part == "danger" and hits:
                out.append(hits)
            value = lk.parts.get(part, "")
            if ((self.verbosity == "quiet" and part in ("top", "desc", "title"))
                    or (self.team_up and lk.kind in ("menu", "choice"))):
                value = ""  # the game's own narrator reads its menus (team up)
            old_value, n = self.seen.get(part, ("", 0))
            n = n + 1 if value == old_value else 1
            self.seen[part] = (value, n)
            if not value:
                if part in ("danger", "pause", "top", "text", "question", "title") or (
                        part in ("turn", "focus") and n >= 2):
                    # Gone (a command window for two looks running: the turn is over), so
                    # it's news when it's back: the same fighter's next turn is a new turn.
                    self.said.pop(part, None)
                continue
            if self.said.get(part) == value:
                continue
            if (part == "focus" and lk.kind == "choice" and not lk.unsure and not first
                    and lk.parts.get("question")
                    and self.said.get("question") != lk.parts["question"]):
                continue  # the question first, then where the cursor is, in one line
            if part == "danger" and lk.battle is None and now - self.danger_at.get(
                    value, -1e9) < (20 if lk.kind == "battle" else 120):
                # The party box was hidden a moment, or the menu page shows the same HP
                # again: not news (in a fight, a reminder after 20 s is).
                self.said[part] = value
                continue
            if part in lk.texts:
                whole = finished(value)
                if part in lk.typing and not ((whole and (n >= 2 or first)) or n >= 3
                                              or (first and lk.unsure)):
                    continue
                if now - self.texts.get(value, -1e9) < self.forget_s:
                    self.said[part] = value
                    continue
                self.texts[value] = now
                if self._covered(value, now):
                    # Player Two has just said it (the agent read it out): taken as said.
                    self.said[part] = value
                    if part == "text":
                        self.heard, self.text_handed = value, True
                    continue
                if part in lk.typing and lk.unsure:
                    self.said[part] = value
                    if not self.text_handed:
                        self.model_read = True  # whole now, but read by the model
                        self.text_handed = part == "text"
                        if part == "text":
                            self.heard = ""
                    continue
            before = self.said.get(part, "")
            self.said[part] = value
            if part == "text":
                self.heard = value
            if part == "danger":
                self.danger_at[value] = now
            if part in lk.typing and before and value.startswith(before) and len(value) > len(before):
                # It paused after a sentence, then typed on: only the rest is new.
                value = value[len(before):].strip()
            out.append(value)
        self.texts = {t: w for t, w in self.texts.items() if now - w < self.forget_s}
        self.last = [p for p in (lk.order or ORDER) if self.said.get(p) and lk.parts.get(p) in out]
        if out and lk.kind == "menu" and lk.focus in MAIN and opened:
            out.insert(0, "Menu")
        if new_fight and not first:
            out.insert(0, "Fight")  # the enemies follow, from a look (the model's)
        return " ".join(_ended(s) for s in out)


def _ended(s: str) -> str:
    """A line ending like a sentence when spoken; a box that breaks off with a dash trails
    off."""
    if re.search(r"[a-z]-$", s):
        return s[:-1] + "..."
    return s if s[-1] in ".?!" else s + "."


# ---------------------------------------------------------------------------------------
# Questions the screen answers by itself, in a fraction of a second.

ASK_HP = re.compile(r"\b(my|our|the party'?s?|every(one|body)'?s|the team'?s|(crono|marle|lucca|"
                    r"robo|frog|ayla|magus)'?s?)\s+(hp|health|hit points)\b|\bhow('s| is| are) "
                    r"(every(one|body)|the party|we|my party|my team)\b|^(what'?s |what is )?"
                    r"(the |our )?(party )?(hp|health)\??$")
COMMAND = re.compile(r"\b(if so|and then|then (heal|use|cast|attack|revive)|heal|revive|cure)\b")
ASK_PARTY = re.compile(r"\bwho('s| is| are)? (is |are )?(in|with) (my|our|the) (party|team)\b|"
                       r"\bwho('s| is) with (me|us)\b|\bwho am i with\b")
ASK_SAY = re.compile(r"\bwhat (does|did|do) (that|it|this|he|she|they|the \w+|crono|marle|"
                     r"lucca|robo|frog|ayla|magus) (just )?say\b|\bread (that|it|this|"
                     r"the \w+)( again| out| to me)?\b|\bwhat('s| is| are) (that|the|they|he|she)"
                     r"( \w+)? saying\b")
ASK_ITEMS = re.compile(r"\bwhat (items|stuff|potions|things)\b|\bwhat('s| is) in (my|our|the) "
                       r"(bag|inventory)\b|\bwhat do (i|we) have\b")
ASK_FOE = re.compile(r"\bwhat (are|am) (we|i) fighting\b|\bwho (are|am) (we|i) fighting\b|"
                     r"\bwhat (enemies|monsters|enemy|monster)\b")


def _plain(question: str) -> str:
    q = " ".join(question.lower().replace("’", "'").split()).strip(" .!")
    return re.sub(r"^(hey )?player two,?\s*", "", q)


def asks(question: str) -> bool:
    """Whether this is a question the screen might answer by itself (not a command like
    "heal Frog's HP")."""
    from .ack import QUESTION, VERB

    q = _plain(question)
    if COMMAND.search(q) or any(VERB.match(c.strip()) for c in re.split(r"[,.;?!]| and ", q)):
        return False  # "is Frog's HP low? if so heal him": something to do, for the agent
    return bool(QUESTION.search(q)) and any(p.search(q) for p in (ASK_HP, ASK_PARTY, ASK_SAY,
                                                                  ASK_ITEMS, ASK_FOE))


def _items(lines: list[list[Run]]) -> tuple[list[str], int]:
    """Items in a list with their counts, and how many entries didn't read."""
    items, unread = [], 0
    for row in lines:
        for i, r in enumerate(row):
            w = snap(r.text, ctsteam.ITEMS, 0.8)
            if w not in ctsteam.ITEMS:
                # An entry in the list that didn't read: the answer would leave it out.
                unread += 300 < r.y < 1000 and len(r.text) > 3 and not r.text.startswith(":")
                continue
            n = next((x.text.lstrip(":").strip() for x in row[i + 1:i + 2]
                      if re.fullmatch(r":?\s*\d+", x.text)), "")
            items.append(f"{w} {n}".strip())
    return items, unread


def answer(question: str, lk: Look, party: list[str] = (), heard: str = "",
           image=None) -> str | None:
    """A question answered from the screen as read, or None when the screen doesn't show
    the answer (the agent looks, and may open a menu). `heard`: the last text box read out,
    for "what did he say?" once it's gone. `image`: the frame looked at, to read again."""
    if not asks(question):
        return None
    q = _plain(question)
    st = lk.battle
    if ASK_SAY.search(q):
        if lk.unsure:
            return None  # it doesn't read cleanly here: the agent reads it with the model
        said = [lk.parts[p] for p in ("title", "question", "top", "text", "desc") if lk.parts.get(p)]
        if lk.kind == "choice" and lk.parts.get("focus"):
            said.append(lk.parts["focus"])
        if said:
            return " ".join(s if s[-1] in ".?!" else s + "." for s in said)
        return heard or None
    if ASK_FOE.search(q):
        if st is not None and st.kind == "target" and lk.parts.get("top"):
            top = lk.parts["top"]
            if len(top.split()) <= 3 and not top.endswith("."):
                return f"The target now is {top}."
        return None
    if ASK_PARTY.search(q):
        if st is not None and st.party:
            names = [m.name for m in st.party]
        elif party:
            names = list(party[:3])
        else:
            return None
        return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + "."
    if ASK_HP.search(q):
        who = [n for n in ctsteam.NAMES if re.search(rf"\b{n.lower()}", q)]
        if st is not None and st.party and all(m.hp is not None for m in st.party):
            out = []
            for m in st.party:
                if who and m.name not in who:
                    continue
                out.append(f"{m.name} is knocked out" if m.hp == 0 else
                           f"{m.name} {m.hp} of {m.hp_max} HP")
            return ", ".join(out) + "." if out else None
        rows = [r for r in _menu_hp(lk, list(party)) or [] if not who or r[0] in who]
        if not rows:
            return None
        return ", ".join(f"{n} is knocked out" if h == 0 else f"{n} {h} of {m} HP"
                         for n, h, m in rows) + "."
    if ASK_ITEMS.search(q):
        items, unread = _items(lk.lines)
        if unread and image is not None and lk.kind == "menu":
            # Green (usable) items on bright metal: read the list again with a lower bar.
            img = to_canvas(image if isinstance(image, np.ndarray) else ctsteam._bgr(image), CANVAS)
            hands = [h[:4] for h in find_template(img, ctsteam.templates()["hand"], CANVAS)]
            again = _items(runs_in(ctsteam.lib(), img, (0, 300, CANVAS[0], 1000), hands,
                                   local=True, margin=40, ring_bg=True))
            if not again[1] and len(again[0]) >= len(items):
                items, unread = again
        if st is not None and st.kind == "list":
            items = [e for row in st.entries for e in row if e in ctsteam.ITEMS] or items
        if len(items) >= 2 and (not unread or st is not None):
            return "On screen: " + ", ".join(items) + "."
        return None
    return None


MENU_HP_Y = (252, 668, 1088)  # the HP rows of the three party blocks on the menu's main page


def _menu_hp(lk: Look, party: list[str]) -> list[tuple[str, int, int]] | None:
    """(name, HP, max HP) of the active party on the field menu's main page, top to
    bottom; None if a block's HP or name can't be told. Low HP is right-aligned, apart
    from its "HP :" label; names that don't read come from the party order."""
    if lk.kind != "menu" or lk.focus not in MAIN:
        return None
    hp: dict[int, tuple[int, int]] = {}
    names: dict[int, str] = {}
    for row in lk.lines:
        runs = [r for r in row if 900 < r.x0 < 1500]
        if not runs:
            continue
        y = runs[0].y
        k = min(range(3), key=lambda i: abs(MENU_HP_Y[i] - y))
        m = re.search(r"HP\s*[:;]\s*(\d+)\s*/\s*(\d+)",
                      " ".join(ctsteam.numfix(r.text) for r in runs))
        if m and abs(MENU_HP_Y[k] - y) < 40:
            h, top = int(m.group(1)), int(m.group(2))
            if top > 0 and h <= top:
                hp[k] = (h, top)
            continue
        k = min(range(3), key=lambda i: abs(MENU_HP_Y[i] - 137 - y))
        for r in runs:
            n = snap(r.text, ctsteam.NAMES, 0.75)
            if n in ctsteam.NAMES and abs(MENU_HP_Y[k] - 137 - y) < 40:
                names[k] = n
    slots = range(max(*hp, *names, min(3, len(party)) - 1, -1) + 1) if hp else range(0)
    if not hp or any(k not in hp for k in slots):
        return None
    who = [names.get(k) or (party[k] if k < len(party) else "") for k in slots]
    if not all(who) or len(set(who)) < len(who):
        return None
    return [(n, *hp[k]) for n, k in zip(who, slots, strict=True)]
