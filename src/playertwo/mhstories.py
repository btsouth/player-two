"""Monster Hunter Stories 3 fights: Muse reads the screen and chooses, a short routine presses.

The game shows everything a good player goes by, and all of it is read from the picture:
- a glowing arc from the monster to whoever it's about to hit, coloured by how that one's own
  attack type fares against the monster's move: blue wins, yellow ties, red loses (or the move
  has no type). So the colour and the target's type give the monster's type. Checked against the
  moves the game names when the turn plays out (the ribbons under the monster's name);
- the target wheel: each part's name, which damage type works on it, and its break gauge;
- the party's HP and skill gauges, and Zerk's three weapons with their skills.

Muse reads those and picks (attack type, part, weapon, skill or item) in one look a turn. What
it picked is said from the same fields, so what's said is what's done. The presses are a
fixed routine that checks the screen as it goes: the menu's colour for the attack type, the
tab for the weapon, the wheel's title for the part. No game data ships here beyond the rules.
"""

from __future__ import annotations

import asyncio
import contextlib
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
TYPES = ("Power", "Speed", "Technical")
# The type that wins against each: the triangle on the game's Skills screen.
ANSWER = {"Speed": "Technical", "Technical": "Power", "Power": "Speed"}
LOSES_TO = {v: k for k, v in ANSWER.items()}  # the type each one beats
COLOUR_TYPE = {"red": "Power", "blue": "Speed", "green": "Technical"}

# Requests that this routine plays: a whole fight, or one command.
WHOLE_FIGHT = re.compile(
    r"\b(win (this|the) (fight|battle)|win this|fight for me|finish (this|the) (fight|battle)"
    r"|you (fight|play) (this|it)|take (this|the) (fight|battle)|play (this|the) (fight|battle)"
    r")\b")
ONE_COMMAND = re.compile(r"\b(attack|hit|strike|heal|potion|herb|take your turn|your best move"
                         r"|best move|break)\b")


def is_game(name: str) -> bool:
    return (name or "").lower().startswith(GAME)


def asks(request: str) -> str:
    """"fight", "turn", or "" for a request this routine doesn't play."""
    said = " ".join(request.lower().split())
    return "fight" if WHOLE_FIGHT.search(said) else "turn" if ONE_COMMAND.search(said) else ""


# Reading pixels. Positions are shares of the picture, measured on the game's own screens.

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
    """Which weapon tab the Skills menu is on, from the bright teal of the chosen tab."""
    w, h = img.size
    best, which = 0.0, None
    for i, (a, b) in enumerate(((70, 190), (197, 320), (327, 450))):
        c = np.asarray(img.crop((a * w // 1280, 102 * h // 720, b * w // 1280, 130 * h // 720))
                       .convert("RGB"), dtype=float)
        teal = float(((c[..., 1] > 150) & (c[..., 2] > 150) & (c[..., 0] < 130)).mean())
        if teal > best:
            best, which = teal, i
    return which if best > 0.2 else None


def wheel_title(img: Image.Image) -> np.ndarray:
    """The white letters of the part's name at the top of the target wheel."""
    w, h = img.size
    box = (1150 * w // 2560, 172 * h // 1440, 1510 * w // 2560, 224 * h // 1440)
    return np.asarray(img.crop(box).convert("L")) > 215


def wheel_up(img: Image.Image) -> bool:
    return int(wheel_title(img).sum()) >= 40


def wheel_open(img: Image.Image) -> bool:
    """The target wheel is open: its cyan ring, whatever's behind it."""
    w, h = img.size
    arr = np.asarray(img.crop((1040 * w // 2560, 230 * h // 1440, 1620 * w // 2560,
                               810 * h // 1440)).convert("RGB"), dtype=float)
    cyan = (arr[..., 1] > 190) & (arr[..., 2] > 180) & (arr[..., 0] < 150)
    return float(cyan.mean()) > 0.012


def wheel_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((980 * w // 2560, 170 * h // 1440, 1680 * w // 2560, 770 * h // 1440))


def part_broken(img: Image.Image) -> bool:
    """The wheel's chosen part says Broken: gold letters where its blue gauge would be."""
    w, h = img.size
    c = np.asarray(img.crop((1260 * w // 2560, 566 * h // 1440, 1400 * w // 2560, 596 * h // 1440))
                   .convert("RGB"), dtype=float)
    gold = ((c[..., 0] > 190) & (c[..., 1] > 150) & (c[..., 2] < 130)).mean()
    cyan = ((c[..., 2] > 170) & (c[..., 1] > 140) & (c[..., 0] < 110)).mean()
    return bool(gold > 0.04 and cyan < 0.05)


RIBBON_X = {"Canyne": 192, "Zerk": 512, "Ogden": 819, "Chirpy": 1152}


def ribbon(img: Image.Image, who: str) -> str:
    """The attack type on the ribbon above a party member's name, from its colour ("" when
    it has none: gold, or they're riding)."""
    if who not in RIBBON_X:
        return ""
    w, h = img.size
    x = RIBBON_X[who]
    c = np.asarray(img.crop(((x - 90) * w // 1280, 638 * h // 720, (x + 90) * w // 1280,
                             654 * h // 720)).convert("RGB"), dtype=float)
    r, g, b = np.median(c.reshape(-1, 3), axis=0)
    if r > g * 1.6 and r > b * 1.6 and g < 90:
        return "Power"
    if b > r * 1.4 and b > g * 1.1:
        return "Speed"
    if g > r * 1.3 and g > b * 1.3:
        return "Technical"
    return ""


def party_bar(img: Image.Image) -> Image.Image:
    """The party's names and numbers close up: they're small, and read wrong scaled down."""
    w, h = img.size
    return img.crop((0, 1270 * h // 1440, w, 1420 * h // 1440))


# What Muse is asked

READ_WHEEL = """\
This picture shows a Monster Hunter Stories 3 target wheel, once per part of the monster (left \
to right). For each wheel: the part's name is the word at the top. Inside the wheel are three \
diamonds, always in this order: left is Slash (a sword), middle is Pierce (an arrowhead), right \
is Blunt (a hammer block). A diamond crossed out with a red line doesn't work on this part; \
the one not crossed works. Under them is either a blue gauge (how close the part is to \
breaking) or the word Broken.
JSON: {"parts": [{"name": "...", "works": ["slash"|"pierce"|"blunt", ...], "broken": true|false, \
"gauge": 0 to 100}]}"""

READ_GEAR = """\
These are the three weapon pages of Monster Hunter Stories 3's Skills menu, left to right. For \
each page: the weapon's name is at the top of the right panel, with its damage type icon: a \
sword is slash, an arrowhead is pierce, a hammer block is blunt. Its element is the small icon \
beside Attack. The list on the left has its skills: name, cost, and whether it's greyed out.
JSON: {"weapons": [{"name": "...", "damage": "slash"|"pierce"|"blunt", "skills": [{"name": "...", \
"cost": 0, "greyed": true|false}]}]}"""

DECIDE = """\
You are planning Zerk's next move in a Monster Hunter Stories 3 fight for a player who can't see \
the screen, the way a sharp player would. Zerk is the player's rider.

Rules:
- A part only takes real damage from its working damage type. Choose an unbroken part (the one \
nearest to breaking); when all are broken, the one the current weapon works on. Then use the \
weapon of that damage type. Don't swap weapons without a reason. With no parts listed, the \
monster has no wheel and any weapon works: keep the one in hand and leave part empty.
- A skill of that weapon hits harder than a plain attack: use one when its cost is at most Zerk's \
gauge. Only skills listed under the weapon you choose.
If the player named something (a part, a type, a weapon, a skill), do that.

JSON only: {"do": "attack"|"skill", "part": "one of the parts listed", "weapon": "one of the \
weapons listed", "skill": "name, if do is skill"}"""

READ_BAR = """\
The party's bar in a Monster Hunter Stories 3 fight. Under each name are two numbers: HP first, \
then skill gauge. JSON: {"hp": {"Canyne": 0, "Zerk": 0, "Ogden": 0, "Chirpy": 0}, "zerk_gauge": 0}"""

READ_ARC = """\
A Monster Hunter Stories 3 fight, seen from behind the party, with a round target wheel over \
the monster. The party stand in a row, left to right: far left a purple wolf-like monster \
(Canyne); then a man in dark red and black armor (Zerk); then a man in a pink shirt with a big \
barrel-hammer (Ogden); far right a big green winged bird (Chirpy). Ogden may be riding Chirpy.
Look for a thin, bright glowing streak of light (blue-white, yellow or red-orange) that starts \
at the wheel and runs down to one of them. Trace it to its lower end and name who is there. It \
is not a sword, a wing or a shadow.
JSON: {"colour": "blue"|"yellow"|"red"|"none", "ends_at": "Canyne"|"Zerk"|"Ogden"|"Chirpy"|\
"none"}"""

CHECK = """\
Is this Monster Hunter Stories 3 picture still a fight? JSON: {"screen": "fight", "results" or \
"field"}. "results": a big rank letter and a score bar, or rewards after a win. "field": walking \
around, a round minimap at the bottom right and no party HP bars along the bottom. Anything else, \
close-ups and attacks included, is "fight"."""


@dataclass
class Part:
    name: str
    works: list[str]
    broken: bool = False
    gauge: int = 0


@dataclass
class Weapon:
    name: str
    damage: str
    skills: list[dict] = field(default_factory=list)


def short(weapon: str) -> str:
    """How a weapon is called out loud: what kind it is if its name says, else its name."""
    low = weapon.lower()
    for word in ("hammer", "harp", "sword", "bow", "lance", "blade"):
        if word in low:
            return f"the {word}"
    return weapon.split()[-1] if weapon else "it"


def monster_type(colour: str, target_type: str) -> str:
    """The monster's attack type from its arc and its target's own type: blue, the target's
    type beats it; yellow, the same; red, it beats the target's or has no type: not known."""
    if target_type not in ANSWER:
        return ""
    return {"blue": LOSES_TO[target_type], "yellow": target_type}.get(colour, "")


def arc_line(colour: str, toward: str, kind: str, zerk_type: str) -> str:
    """The first words of a turn, as soon as the arc is read: what's coming at whom. Never a
    type the colour doesn't prove."""
    if colour not in ("blue", "yellow", "red") or toward not in RIBBON_X:
        return ""
    who = "you" if toward == "Zerk" else toward
    if kind:
        verb = "coming at" if toward in ("Zerk", "Canyne") else "going at"
        return f"{kind} attack {verb} {who}."
    if toward == "Zerk" and zerk_type:
        return f"It has the edge on {zerk_type}."
    return f"It's going for {who}."


def plan_line(d: dict, part: Part | None, weapon_changed: bool, weapon: Weapon | None) -> str:
    """What's said of the choice: built from what was picked, so nothing else can slip in."""
    if d.get("do") == "item":
        who, item = str(d.get("on") or "Zerk"), str(d.get("item") or "Potion")
        hp = (d.get("hp") or {}).get(who)
        name = "You're" if who == "Zerk" else f"{who}'s"
        low = f"{name} at {hp}. " if isinstance(hp, int) else ""
        return f"{low}{item}."
    what = (f"{d.get('type')} {d.get('skill')}" if d.get("do") == "skill" and d.get("skill")
            else f"{d.get('type')} attack")
    where = f" on the {part.name.lower()}" if part else ""
    with_ = f", {short(weapon.name)} for {weapon.damage}" if weapon_changed and weapon else ""
    toward = str((d.get("arc") or {}).get("toward") or "")
    lead = "Answering with " if toward == "Zerk" else ""
    return f"{lead}{what}{where}{with_}."


class Fight:
    """One session's fights. `agent` gives grab, controls, the model, speech and cancelling."""

    def __init__(self, agent):
        self.a = agent
        self.parts: dict[str, list[Part]] = {}  # by monster, once read
        self.weapons: list[Weapon] = []
        self.weapon = 0  # the weapon page Zerk is on
        self.most_hp: dict[str, int] = {}
        self.monster = ""
        self.ended = ""
        self.gauge = 0
        self.planned: asyncio.Future | None = None
        self.log: list[str] = []
        self.note: Callable[[str, dict], None] = (
            lambda name, args: agent.on_action and agent.on_action(name, args))

    # Pad, with the screen checked between presses

    async def _press(self, button: str, then: float = 0.0) -> None:
        await self.a.go.wait()
        await asyncio.to_thread(self.a.controls.press, button)
        if then:
            await asyncio.sleep(then)

    async def _flick(self, way: str, then: float = 0.35) -> None:
        await self.a.go.wait()
        await asyncio.to_thread(self.a.controls.stick, "left", way, 160)
        await asyncio.sleep(then)

    async def _shot(self) -> Image.Image:
        return (await self.a.grab()).image

    async def wait_menu(self, most: float = 70.0, after: float = 0.0) -> Image.Image | None:
        """The fight's next turn: the command menu is up (two looks in a row). None when it
        isn't by then, or the fight ended."""
        await asyncio.sleep(after)
        end = time.monotonic() + most
        hits, look_at, over = 0, time.monotonic() + 9.0, 0
        nudge = time.monotonic() + 30.0
        while time.monotonic() < end and not self.a.cancelled:
            img = await self._shot()
            if menu_up(img):
                hits += 1
                if hits >= 2:
                    return img
                continue
            hits = 0
            if wheel_open(img):
                # A press left over from a skill that wanted no target: back to the menu.
                await self._press("b", 0.5)
                continue
            if time.monotonic() >= nudge:
                # This long with no menu and still a fight: a menu may be open that no check
                # knows. Back does nothing while an attack plays.
                nudge = time.monotonic() + 15.0
                await self._press("b", 0.3)
            if time.monotonic() >= look_at:
                look_at = time.monotonic() + 3.0
                seen = await self.screen(img)
                over = over + 1 if seen in ("results", "field") else 0
                self.ended = seen if over else ""
                if over >= 2:
                    return None
            await asyncio.sleep(0.12)
        return None

    async def ask(self, images: list, text: str, tokens: int = 1500, most: float = 18.0,
                  system: str = "") -> dict:
        """One look by the model; a reply cut short or not JSON gets one more try."""
        last: Exception | None = None
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": [*images, {"type": "text", "text": text}]})
        for _ in range(2):
            try:
                return await asyncio.wait_for(self.a.muse.json(
                    msgs, effort="minimal", max_tokens=tokens), most)
            except (ValueError, TimeoutError) as exc:
                last = exc
        raise ValueError(f"no reading: {last}")

    async def screen(self, img: Image.Image) -> str:
        try:
            d = await self.ask([image_part(img, 768)], CHECK, 400, 15)
        except Exception:  # noqa: BLE001  not known: the fight goes on
            return "fight"
        return str(d.get("screen") or "fight")

    # Reading what doesn't change in a fight, once

    async def read_gear(self):
        """The weapons and their skills: three looks at the Skills menu, once a session.
        Returns the model's reading to wait on, so the fight's other reading can go too."""
        await self._press("x", 0.75)
        pages, start = [], None
        for i in range(3):
            img = await self._shot()
            tab = skills_tab(img)
            if start is None:
                start = tab if tab is not None else 0
            w, h = img.size
            pages.append(img.crop((0, 90 * h // 720, 830 * w // 1280, 590 * h // 720)))
            if i < 2:
                await self._press("rb", 0.5)
        tabs = 3
        names = (start or 0)
        # Back to the page it opened on: leaving with b puts the weapon back anyway.
        await self._press("b", 0.6)
        sheet = Image.new("RGB", (sum(p.width for p in pages), pages[0].height))
        x = 0
        for p in pages:
            sheet.paste(p, (x, 0))
            x += p.width
        async def reading() -> None:
            d = await self.ask([image_part(sheet, 2400)], READ_GEAR, 2500)
            got = [Weapon(str(w.get("name") or f"weapon {i}"),
                          str(w.get("damage") or "").lower(), list(w.get("skills") or []))
                   for i, w in enumerate(d.get("weapons") or [])]
            # Pages were read from where it opened: put them in tab order.
            if len(got) == tabs:
                self.weapons = [got[(i - names) % tabs] for i in range(tabs)]
                self.weapon = names
            self.note("mh_gear", {"weapons": [(w.name, w.damage, w.skills)
                                              for w in self.weapons], "on": self.weapon})

        return asyncio.ensure_future(reading())

    async def sync_weapon(self) -> None:
        """A new fight: which weapon Zerk has out, from the page the Skills menu opens on."""
        await self._press("x", 0.65)
        tab = skills_tab(await self._shot())
        if tab is not None:
            self.weapon = tab
        await self._press("b", 0.45)

    async def read_parts(self, monster: str):
        """The monster's parts: round the wheel once, then one look at all of them. A small
        monster has no wheel, only a ring that takes any damage type. Returns the model's
        reading to wait on."""
        await self._press("a", 0.7)
        seen: list[tuple[np.ndarray, Image.Image]] = []
        first = await self._shot()
        if not wheel_up(first) or menu_up(first):
            await self._press("b", 0.5)
            if monster:
                self.parts[monster] = []
            self.note("mh_parts", {"monster": monster, "parts": []})
            return None
        for i in range(5):
            img = await self._shot()
            seen.append((wheel_title(img), wheel_crop(img)))
            if i < 4:
                await self._press("rb", 0.42)
        await self._press("b", 0.5)
        sheet = Image.new("RGB", (len(seen) * 600, 514))
        for i, (_, crop) in enumerate(seen):
            sheet.paste(crop.resize((600, 514)), (i * 600, 0))
        async def reading() -> None:
            d = await self.ask([image_part(sheet, 2400)], READ_WHEEL, 1500)
            parts: list[Part] = []
            for p in d.get("parts") or []:  # round the wheel: a name seen again closes the circle
                name = str(p.get("name") or "?")
                if parts and name.lower() == parts[0].name.lower():
                    break
                parts.append(Part(name, [str(x).lower() for x in p.get("works") or []],
                                  p.get("broken") is True, int(p.get("gauge") or 0)))
            self.parts[monster] = parts
            self.note("mh_parts", {"monster": monster, "parts": [
                (p.name, p.works, p.broken, p.gauge) for p in parts]})

        return asyncio.ensure_future(reading())

    # One turn

    def state(self, monster: str) -> str:
        lines = [f"Monster: {monster or 'see the picture'}."]
        lines += [f"Part {p.name}: {'/'.join(p.works)} works, "
                  + ("broken" if p.broken else f"gauge {p.gauge}%")
                  for p in self.parts.get(monster, [])]
        for i, w in enumerate(self.weapons):
            # What costs nothing isn't an attack (Charge, Power Coating); greyed can't be used.
            skills = "; ".join(f"{s.get('name')} (cost {s.get('cost')})" for s in w.skills
                               if not s.get("greyed") and (s.get("cost") or 0) > 0) or "none"
            lines.append(f"Weapon {w.name}: {w.damage}{' (in hand)' if i == self.weapon else ''}."
                         f" Skills: {skills}.")
        if self.most_hp:
            lines.append("Most HP seen: " + ", ".join(f"{k} {v}" for k, v in self.most_hp.items())
                         + ".")
        if self.log:
            lines.append("Last turns: " + " | ".join(self.log[-3:]))
        return "\n".join(lines)

    async def read_arc(self, img: Image.Image) -> dict:
        """The arc: two looks at once, and both have to agree (a look at a thin streak
        half behind the wheel is sometimes wrong, and a wrong arc is a wrong line)."""
        w, h = img.size
        crop = image_part(img.crop((0, h * 22 // 100, w, h * 86 // 100)), 1400)
        looks = await asyncio.gather(*(self.ask([crop], READ_ARC, 1200, 14) for _ in range(2)),
                                     return_exceptions=True)
        good = [x for x in looks if isinstance(x, dict)]
        arcs = {(str(x.get("colour") or "none").lower(), str(x.get("ends_at") or "none"))
                for x in good}
        colour, toward = arcs.pop() if len(arcs) == 1 and len(good) == 2 else ("none", "unknown")
        self.note("mh_arc", {"looks": [(x.get("colour"), x.get("ends_at")) for x in good],
                             "colour": colour, "toward": toward})
        return {"colour": colour, "toward": toward}

    async def read_title(self, img: Image.Image) -> str:
        """Which part the open wheel is on: the word at its top, close up."""
        names = [p.name for p in self.parts.get(self.monster, [])]
        if not names:
            return ""
        w, h = img.size
        crop = img.crop((1100 * w // 2560, 165 * h // 1440, 1560 * w // 2560, 230 * h // 1440))
        with contextlib.suppress(Exception):
            d = await self.ask([image_part(crop.resize((crop.width * 2, crop.height * 2)), 1024)],
                               f"Which word is written here: {', '.join(names)}? "
                               'JSON: {"word": "..."}', 500, 10)
            return str(d.get("word") or "")
        return ""

    def plan_ahead(self, request: str) -> None:
        """The model's next move, asked for while the turn plays out, so it's ready when the
        menu comes back: the part, the weapon and the skill don't hang on the next arc."""
        text = f"{self.state(self.monster)}\nZerk's gauge: {self.gauge}.\n" \
               f"The player said: \"{request}\"."
        self.planned = asyncio.ensure_future(self.ask([], text, 1500, system=DECIDE))

    async def decide(self, img: Image.Image, request: str, zerk_type: str) -> dict:
        """The arc and the numbers are read at once. What's coming is said the moment the arc
        is read; the choice comes from the plan, and healing goes first."""
        if self.planned is None:
            self.plan_ahead(request)
        bar = asyncio.ensure_future(self.ask([image_part(party_bar(img), 1800)], READ_BAR, 800, 12))
        word = asyncio.ensure_future(self.read_title(img))
        arc = await self.read_arc(img)
        colour, toward = arc["colour"], arc["toward"]
        target = zerk_type if toward == "Zerk" else ribbon(img, toward)
        kind = monster_type(colour, target)
        first = arc_line(colour, toward, kind, zerk_type)
        if first:
            await self.a._speak(first)
        numbers = await bar
        hp = {k: v for k, v in (numbers.get("hp") or {}).items() if isinstance(v, int)}
        if isinstance(numbers.get("zerk_gauge"), int):
            self.gauge = numbers["zerk_gauge"]
        for who, n in hp.items():
            self.most_hp[who] = max(n, self.most_hp.get(who, 0))
        planned = self.planned
        self.planned = None
        d = await planned
        d.update(arc=arc, hp=hp, wheel_part=await word, monster_type=kind, target_type=target,
                 said=first)
        d["type"] = ribbon(img, "Canyne") or zerk_type
        low = [(n, who) for who, n in hp.items()
               if 0 < n and n < max(70 if who == toward else 0, self.most_hp[who] / 3)]
        if low:
            d.update(do="item", item="Potion", on=min(low)[1])
        return d

    def settle_pick(self, d: dict) -> tuple[Part | None, int]:
        """The part and weapon, with the rules held to: the weapon always fits the part."""
        parts = self.parts.get(self.monster, [])
        want = str(d.get("part") or "").lower()
        part = next((p for p in parts if p.name.lower() == want), None)
        if part is None and parts:
            part = next((p for p in parts if not p.broken), parts[0])
        weapon = next((i for i, w in enumerate(self.weapons)
                       if w.name.lower() == str(d.get("weapon") or "").lower()), self.weapon)
        if part is not None and self.weapons and part.works:
            if self.weapons[weapon].damage not in part.works:
                fits = [i for i, w in enumerate(self.weapons) if w.damage in part.works]
                if fits:
                    weapon = self.weapon if self.weapon in fits else fits[0]
        return part, weapon

    def settle_type(self, d: dict, zerk_type: str) -> None:
        """The arc's meaning held to, whatever was picked: an arc at Zerk gets the type that
        wins it."""
        arc = d.get("arc") or {}
        colour, toward = str(arc.get("colour") or "").lower(), str(arc.get("toward") or "")
        d["zerk_type"] = zerk_type
        if toward == "Zerk" and zerk_type in ANSWER and d.get("do") != "item":
            want = {"blue": zerk_type, "yellow": ANSWER[zerk_type],
                    "red": ANSWER[ANSWER[zerk_type]]}.get(colour)
            if want and want != d.get("type"):
                d["model_type"] = d.get("type")
                d["type"] = want

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

    async def turn_to_part(self, part: Part, now: str) -> bool:
        """Round the open wheel to the part: rb steps round in the order it was read."""
        names = [p.name.lower() for p in self.parts.get(self.monster, [])]
        if part.name.lower() not in names or now.lower() not in names:
            return False
        steps = (names.index(part.name.lower()) - names.index(now.lower())) % len(names)
        for _ in range(steps):
            await self._press("rb", 0.38)
        part.broken = part.broken or part_broken(await self._shot())  # told to the next choice
        return True

    async def attack(self, d: dict, part: Part | None, weapon: int, kind_now: str) -> str:
        """From the open wheel: straight on when the type and weapon stay, else back to the
        menu to change them."""
        kind = d.get("type") if d.get("type") in TYPES else kind_now
        typed, skill = True, None
        if d.get("do") == "skill" and self.weapons:
            names = [str(s.get("name") or "").lower() for s in self.weapons[weapon].skills]
            want = str(d.get("skill") or "").lower()
            skill = names.index(want) if want in names else None
        if weapon != self.weapon or skill is not None or kind != kind_now:
            await self._press("b", 0.45)
            typed = await self.set_type(kind, await self._shot())
            if weapon != self.weapon or skill is not None:
                await self._press("x", 0.7)
                for _ in range(3):
                    tab = skills_tab(await self._shot())
                    if tab == weapon or tab is None:
                        break
                    await self._press("rb" if (weapon - tab) % 3 == 1 else "lb", 0.45)
                if skill is not None:
                    for _ in range(skill + 1):
                        await self._press("down", 0.3)
            await self._press("a", 0.6)
        if part is not None and not await self.turn_to_part(part, str(d.get("wheel_part") or "")):
            self.note("mh_note", {"text": f"wheel word {d.get('wheel_part')!r}: can't reach "
                                          f"{part.name}"})
        await self._press("a", 0.1)
        self.weapon = weapon
        return "" if typed else f"menu stayed off {kind}"

    async def use_item(self, d: dict) -> None:
        """Items open with the list: read which row, pick by name, then the ally on the ring."""
        item, who = str(d.get("item") or "Potion"), str(d.get("on") or "Zerk")
        await self._press("y", 0.7)
        order = ("Herb", "Potion")
        rows = order.index(item) if item in order else 1
        img = await self._shot()
        # Row by name, checked with one look at what's highlighted.
        for _ in range(rows):
            await self._press("down", 0.35)
        await self._press("a", 0.7)
        allies = ["Canyne", "Zerk", "Ogden", "Chirpy"]
        steps = allies.index(who) if who in allies else 1
        for _ in range(steps):
            await self._flick("right")
        await self._press("a", 0.1)
        del img

    async def take_turn(self, img: Image.Image, request: str) -> tuple[str, dict]:
        """Read, choose, say it, do it. The picture is taken with the target wheel open: the
        camera holds still there, with the arcs in view, while the menu's own views wander.
        The line goes out as soon as the choice is back."""
        t = time.monotonic()
        kind_now = menu_type(img)
        await self._press("a", 0.55)
        img = await self._shot()
        if os.environ.get("PLAYER_TWO_FRAMES"):  # what each turn was picked from (debugging)
            with contextlib.suppress(OSError):
                img.convert("RGB").save(Path(os.environ["PLAYER_TWO_FRAMES"]).expanduser()
                                        / f"mh-{time.strftime('%H%M%S')}.jpg", quality=85)
        d = await self.decide(img, request, kind_now)
        t_read = time.monotonic() - t
        self.settle_type(d, kind_now)
        part, weapon = self.settle_pick(d)
        if d.get("do") == "skill" and self.weapons:
            # The weapon fits the part, so a skill of another weapon isn't there to use; what's
            # said has to be what's done.
            known = {str(x.get("name") or "").lower(): x.get("cost") or 0
                     for x in self.weapons[weapon].skills}
            cost = known.get(str(d.get("skill") or "").lower())
            if cost is None or cost > self.gauge:
                d["model_skill"], d["do"], d["skill"] = d.get("skill"), "attack", ""
        changed = weapon != self.weapon
        w = self.weapons[weapon] if self.weapons else None
        line = plan_line(d, part, changed, w)
        await self.a._speak(line)
        if d.get("do") == "item":
            await self._press("b", 0.45)
            await self.use_item(d)
        else:
            trouble = await self.attack(d, part, weapon, kind_now)
            if trouble:
                self.note("mh_note", {"text": trouble})
        if os.environ.get("PLAYER_TWO_FRAMES"):  # the moves the game names as the turn plays out
            with contextlib.suppress(OSError):
                for i in range(3):
                    shot = await self._shot()
                    w_, h_ = shot.size
                    shot.crop((0, 0, w_, h_ * 14 // 100)).convert("RGB").save(
                        Path(os.environ["PLAYER_TWO_FRAMES"]).expanduser()
                        / f"mh-{time.strftime('%H%M%S')}-moves{i}.jpg", quality=80)
        self.log.append(line)
        self.plan_ahead(request)
        self.note("mh_turn", {"read_s": round(t_read, 2), "seconds": round(time.monotonic() - t, 2),
                              "line": line, "choice": d, "weapon_now": self.weapon})
        return line, d

    async def monster_name(self, img: Image.Image) -> str:
        w, h = img.size
        crop = img.crop((300 * w // 2560, 20 * h // 1440, 2300 * w // 2560, 90 * h // 1440))
        with contextlib.suppress(Exception):
            d = await self.ask([image_part(crop, 800)],
                               'A monster\'s name is written here. JSON: {"name": "..."}', 300, 12)
            return str(d.get("name") or "")
        return ""

    async def run(self, request: str, mode: str, deadline: float = 240.0) -> str:
        """Plays a turn, or the fight to its end; returns what to tell the player."""
        t0 = time.monotonic()
        for _ in range(3):  # a menu left open by hand: back to the commands
            if menu_up(await self._shot()):
                break
            await self._press("b", 0.5)
        img = await self.wait_menu(most=8.0)
        if img is None:
            return "" if mode == "turn" else "There's no fight to play right now."
        name = asyncio.ensure_future(self.monster_name(img))
        fresh = not self.weapons
        monster = await name
        if fresh or monster not in self.parts:
            await self.a._speak("Checking its weak spots.")
        reads = [await self.read_gear()] if fresh else []
        if not fresh and monster != self.monster:
            await self.sync_weapon()
        self.monster = monster
        if self.monster not in self.parts:
            reads.append(await self.read_parts(self.monster))
        waited = [r for r in reads if r is not None]
        if waited:
            await asyncio.gather(*waited)
            img = await self.wait_menu(most=5.0) or img
        self.planned = None
        turns = 0
        while not self.a.cancelled and time.monotonic() - t0 < deadline:
            await self.take_turn(img, request)
            turns += 1
            if mode == "turn":
                return ""
            img = await self.wait_menu(after=1.0)
            if img is None:
                break
        if self.a.cancelled:
            return ""
        if not self.ended:
            return "I lost track of the fight there. Say describe to hear where things stand."
        self.monster = ""  # the next fight starts fresh, even against the same kind
        await self._press("a", 1.0)  # past the rank screen
        self.planned = None
        return "We won."
