"""Requests done without a model turn: the plain ones the player says most.

Each model turn costs one to three seconds, and a request with a tool call has at least
two (the call, then a sentence from its result), plus a settle look between them. For a
request whose words already say exactly what to do ("buy three potions", "Crono, use
Cyclone", "attack the one on the left", "where am I?"), the tool runs at once and its
result is said from a template: the same facts the model would have phrased, read off
the same screen. Anything the words or the result don't settle goes to the model as
before, with what was tried.

Live 3 (2026-11-01, Steam): buy three potions 18.6 to 26.7 s, Crono's Cyclone 9.5 to
12.2 s, attack the one on the left 6.1 to 9.2 s, where am I 4.6 to 5.6 s; the model's
two turns and the settle look were 3.5 to 5 s of each.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from typing import TYPE_CHECKING

from . import ctsteam
from .battle import (
    HEAL_ITEMS,
    REVIVE_ITEMS,
    SIDES,
    Intent,
    Outcome,
    _hp,
    party_brief,
)
from .ctsteam import State

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent, Run

NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve")
COUNTS = {w: i for i, w in enumerate(NUMBER_WORDS)} | {"a": 1, "an": 1, "a couple of": 2,
                                                      "a pair of": 2}
NAMES = tuple(n.lower() for n in ctsteam.NAMES)
_POLITE = (r"(?:(?:please|can you|could you|would you|i want to|i'd like to|i wanna|let's|"
           r"lets|go ahead and|now)\s+)*")
_TAIL = r"(?:\s+(?:please|now|for me|thanks|thank you))*"


def plain(text: str) -> str:
    """The request in lower case, its words only: "Crono, use Cyclone!" is
    "crono, use cyclone" (the comma after a name kept)."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"^(?:hey |ok |okay )?player (?:two|2),?\s*", "", t.strip())
    t = re.sub(r"[^\w',\- ]+", " ", t)
    return " ".join(t.split()).strip(" ,")


def number_word(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def plural(name: str, n: int) -> str:
    if n == 1 or name.endswith("s"):
        return name
    return name + "s"


# -- buying and resting ----------------------------------------------------------------

BUY = re.compile(rf"^{_POLITE}(?:buy|get|purchase|pick up)(?: me| us)? "
                 rf"(?P<n>\d+|{'|'.join(sorted(COUNTS, key=len, reverse=True))}) "
                 rf"(?:more )?(?P<item>[a-z][a-z' \-]{{1,28}}?){_TAIL}$")
REST = re.compile(rf"^{_POLITE}(?:(?:rest|sleep|stay|get some rest)(?: the night)? (?:at|in) "
                  rf"(?:the|an|this) inn|stay the night|spend the night(?: here)?){_TAIL}$")


def buy_request(text: str) -> tuple[int, str] | None:
    """(how many, which item) for "buy three potions", "get a hi-potion"; None when the
    words don't say both (the model asks), or the item isn't one the shops sell."""
    from .shop import ITEMS, _plain

    m = BUY.match(plain(text))
    if not m:
        return None
    raw = m.group("n")
    n = int(raw) if raw.isdigit() else COUNTS.get(raw)
    if not n or n > 99:
        return None
    want = _plain(m.group("item"))
    item = next((i for i in ITEMS if _plain(i) == want), None)
    return (n, item) if item else None


def rest_request(text: str) -> bool:
    """"Rest at the inn", "stay the night", "sleep here"."""
    return bool(REST.match(plain(text)))


SELLER = re.compile(r"\b(item|items|potion|general|goods)\b.*\b(seller|shop|merchant|"
                    r"shopkeeper|vendor|stall)\b|\b(shopkeeper|merchant)\b", re.IGNORECASE)
INNKEEPER = re.compile(r"\binn ?keeper\b", re.IGNORECASE)


def seller_for(people: list, rest: bool = False) -> str:
    """The one person in a surveyed place who sells items (or keeps the inn), by the
    survey's name for them ("the item seller at the purple tent"); "" when it isn't
    exactly one (the model picks then)."""
    rx = INNKEEPER if rest else SELLER
    found = [p.name for p in people if getattr(p, "kind", "person") == "person"
             and rx.search(p.name) and not re.search(r"\b(sword|armou?r|weapon)\b", p.name,
                                                      re.IGNORECASE)]
    return found[0] if len(found) == 1 else ""


BOUGHT = re.compile(r"Bought (?P<n>\d+) (?P<item>.+?) for (?P<spent>\d+) G \(funds "
                    r"(?P<before>\d+) G to (?P<after>\d+) G\)(?P<note> \(the count read[^)]*\))?")


def price_line(n: int, item: str, total: int | None) -> str:
    """What the count screen shows, before the last a: "Three Potions, 30 gold."."""
    what = f"{number_word(n).capitalize()} {plural(item, n)}"
    return f"{what}, {total} gold." if total is not None else f"{what}."


def bought_line(n: int, item: str, spent: int, after: int, asked: int,
                priced: bool = False) -> str:
    """The purchase, from the gold read before and after: "Got three Potions for 30
    gold. You have 370 left." (priced: the price was said already, a moment ago)."""
    what = f"{number_word(n)} {plural(item, n)}"
    if priced and n == asked:
        line = f"Bought. You have {after} gold left."
    else:
        line = f"Got {what} for {spent} gold. You have {after} left."
    if n < asked:
        line += f" That's all the gold allowed, not {number_word(asked)}."
    return line


def not_sold_line(result: str, item: str) -> str:
    """"no Hi-Potion here; this shop sells: Potion (10 G), ..." in a sentence; "" if the
    result isn't that."""
    m = re.search(r"no (.+?) here; this shop sells: (.+?)\.(?: |$)", result)
    if not m or "nothing I could read" in m.group(2):
        return ""
    goods = [re.sub(r"\s*\((\d+) G\)", r" for \1", g).strip()
             for g in m.group(2).split(", ") if g.strip()]
    if not goods:
        return ""
    listed = goods[0] if len(goods) == 1 else ", ".join(goods[:-1]) + f" and {goods[-1]}"
    return f"This shop doesn't sell {plural(item, 2)}. It has {listed} gold."


# -- fights ----------------------------------------------------------------------------

_ACTOR = rf"(?:(?P<actor>{'|'.join(NAMES)}),? )?"
ATTACK = re.compile(rf"^{_ACTOR}{_POLITE}(?:attack|hit|strike)(?: (?P<rest>.+?))?{_TAIL}$")
USE = re.compile(rf"^{_ACTOR}{_POLITE}(?:use|cast|do)(?: (?:a|an|the|some|your))? "
                 rf"(?P<what>[a-z][a-z' \-]{{1,24}}?)(?: on (?P<on>.+?))?{_TAIL}$")
BARE = re.compile(rf"^(?P<actor>{'|'.join(NAMES)}),? (?P<what>[a-z][a-z' \-]{{1,24}}?)"
                  rf"(?: on (?P<on>.+?))?{_TAIL}$")
# Which enemy, by where it stands: "the one on the left", "the left one", "the enemy
# on the right", "it", "them", "".
_FOE = r"(?:one|enemy|monster|guy|bad guy|imp|thing)"
SIDE = re.compile(rf"^(?:(?:the|that) )?(?:(?P<a>left|right|top|bottom|front|back|far|near)"
                  rf"(?:most)?(?: {_FOE})?|{_FOE} (?:on|to|at|in) the (?P<b>left|right|top|"
                  rf"bottom|front|back)(?: side)?)$")
ANYONE = re.compile(rf"^(?:it|them|him|her|(?:an?|the|any|that) {_FOE}|anyone|anything|"
                    rf"whoever|an enemy|the enemies|enemies)$")
ITEM_NAMES = tuple(dict.fromkeys((*HEAL_ITEMS, *REVIVE_ITEMS, *ctsteam.ITEMS)))


def _target(words: str | None, party: tuple[str, ...] = ctsteam.NAMES) -> str | None:
    """An executor target from the player's words: "" (anyone: the game's pick), a side
    ("left"), an ally's name; None when the words aren't one of those (the model)."""
    w = (words or "").strip()
    if not w or ANYONE.match(w):
        return ""
    if m := SIDE.match(w):
        side = m.group("a") or m.group("b")
        return side if side in SIDES else None
    w = re.sub(r"^(?:on |the )", "", w)
    if w in ("me", "myself", "himself", "herself", "self"):
        return "me"
    return next((n for n in party if n.lower() == w), None)


def _named(what: str, names) -> str | None:
    w = re.sub(r"[-_]", " ", what.lower()).strip()
    w = w[:-1] if w.endswith("s") and not w.endswith("ss") and w[:-1] in {
        re.sub(r"[-_]", " ", n.lower()) for n in names} else w
    return next((n for n in names if re.sub(r"[-_]", " ", n.lower()) == w), None)


def battle_request(text: str) -> Intent | None:
    """One battle command in plain words: "attack the one on the left", "Crono, use
    Cyclone", "Marle, cure on Frog", "use a potion on Crono". None for anything else
    (more than one command, an enemy by name, a combo): the model's."""
    t = plain(text)
    if re.search(r"\b(and|then|after|until|twice|again|everyone attack)\b", t):
        return None
    if m := ATTACK.match(t):
        target = _target(m.group("rest"))
        if target is None:
            return None
        actor = (m.group("actor") or "").capitalize()
        return Intent(actor, "attack", "", target)
    m = USE.match(t) or BARE.match(t)
    if not m:
        return None
    actor = (m.group("actor") or "").capitalize()
    what = m.group("what")
    target = _target(m.group("on"))
    if target is None:
        return None
    if target == "me":
        target = actor or ""
        if not target:
            return None
    techs = ctsteam.TECHS.get(actor, ()) if actor else tuple(
        dict.fromkeys(x for ts in ctsteam.TECHS.values() for x in ts))
    tech = _named(what, techs)
    if tech is not None:
        if not actor:
            owners = [n for n, ts in ctsteam.TECHS.items() if tech in ts]
            if len(owners) != 1:
                return None
            actor = owners[0]
        return Intent(actor, "tech", tech, target)
    item = _named(what, ITEM_NAMES)
    if item is not None and (m.re is not BARE):
        return Intent(actor, "item", item, target)
    return None


def _target_words(target: str) -> str:
    if target in ("left", "right"):
        return f" the one on the {target}"
    if target in ("top", "back", "far"):
        return " the one at the back"
    if target in ("bottom", "front", "near"):
        return " the one at the front"
    return f" {target}" if target else ""


def battle_line(intent: Intent, out: Outcome, before: State | None, after: State | None,
                went_off: bool) -> str:
    """One command done, in a sentence, from what the screen showed: who did what, and
    what happened to the party while it played ("Crono used Cyclone. Frog dropped to 1
    HP."). Enemy HP isn't on the screen, so nothing is said about it."""
    actor = intent.actor or (out.text.split(":")[0].strip() if ":" in out.text else "")
    who = actor or "We"
    cmd = intent.command.lower()
    if cmd == "attack":
        lead = f"{who} attacked{_target_words(intent.target)}."
    elif went_off:
        on = f" on {intent.target}" if intent.target in ctsteam.NAMES else ""
        lead = f"{who} used {intent.name}{on}."
    else:
        lead = (f"{actor + chr(39) + 's ' if actor else ''}{intent.name} is chosen; it goes "
                "off after what's playing now.")
    st = after if after is not None and after.party else out.state
    b, a = _hp(before), _hp(st)
    hit, healed, low = [], [], []
    for m in (st.party if st is not None else []):
        old, hp = b.get(m.name), a.get(m.name)
        if old is None or hp is None:
            continue
        if hp < old:
            hit.append(f"{m.name} is knocked out" if hp == 0 else f"{m.name} dropped to {hp} HP")
        elif hp > old:
            healed.append(f"{m.name} is back up to {hp}" if old == 0 else f"{m.name} is up to {hp}")
        elif m.hp_max and 0 < hp < m.hp_max / 3:
            low.append((m.name, hp))
    parts = [lead]
    news = hit + healed
    if news:
        parts.append(_cap(_join(news)) + ".")
    if low:
        if len({hp for _n, hp in low}) == 1:
            names = _join([n for n, _hp in low])
            verb = "is" if len(low) == 1 else "are"
            parts.append(f"{names} {verb} still at {low[0][1]} HP.")
        else:
            parts.append(_cap(_join([f"{n} is at {hp} HP" for n, hp in low])) + ".")
    return " ".join(parts)


def stopped_line(intent: Intent, out: Outcome) -> str:
    """A stop the player can act on, said plainly ("Crono's knocked out, so he can't use
    Cyclone."); "" for anything else (the model looks at it)."""
    why = out.stopped or out.text
    what = intent.name or ("attack" if intent.command == "attack" else intent.command)
    if m := re.match(r"(\w+) is knocked out$", why):
        return f"{m.group(1)}'s knocked out, so they can't {('use ' + what) if intent.name else what}."
    if m := re.match(r"(\w+) isn't ready for a command yet", why):
        return f"{m.group(1)} isn't ready yet. Ask again in a moment."
    if m := re.match(r"(.+?) is greyed out", why):
        return f"{m.group(1)} is greyed out: not enough MP for it right now."
    if m := re.match(r"(.+?) isn't in the (tech|item) list", why):
        if m.group(2) == "tech" and intent.actor:
            return f"{intent.actor} doesn't have {m.group(1)} yet."
        return f"We don't have any {plural(m.group(1), 2)}."
    if m := re.match(r"(\w+) is at full HP \((\d+)/(\d+)\)", why):
        return f"{m.group(1)} is at full health already, {m.group(2)} of {m.group(3)}."
    if why.startswith("everyone standing is at full HP"):
        return "Everyone standing is at full health already."
    if why.startswith("the fight is won already"):
        return "The fight's already won."
    return ""


def won_line(result: str, st: State | None) -> str:
    """A won fight from the executor's result and the screen after it: "Won the fight:
    648 experience. Everyone's at full health."; "" when it doesn't read as won."""
    if result.startswith("done: the fight ended"):
        line = "The fight's over."
    elif result.startswith("done: won"):
        exp = re.search(r"Earned (\d+) EXP", result, re.IGNORECASE)
        line = "Won the fight" + (f": {exp.group(1)} experience." if exp else ".")
    else:
        return ""
    brief = party_brief(st) if st is not None else ""
    return f"{line} {brief}".strip()


def ran_line(result: str) -> str:
    return "We got away." if result.startswith("done: got away") else ""


def talk_line(who: str, line: str) -> str:
    """Someone's line as said: their name first when the player named them by it
    ("Melchior: Melchior the swordsmith, at your service!"), else the line alone (it
    comes right after "Going to the old man at the lower stalls.")."""
    name = who.strip().split(" ")[0] if who else ""
    if name[:1].isupper() and name.lower() not in ("the", "a", "an", "that", "this"):
        return f"{name}: {line}"
    return line


# -- where am I ------------------------------------------------------------------------

WHERE = re.compile(rf"^{_POLITE}(?:where am i|where are we|where are we now|where am i now|"
                   rf"tell me where (?:i am|we are))(?: right now)?{_TAIL}$")


def where_request(text: str) -> bool:
    return bool(WHERE.match(plain(text)))


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def where_line(m, x: float, y: float, step: float = 8.0, rise: float = 0.0) -> tuple[str, str]:
    """Where a spot on a world map is, as said to the player: "You're on the world map in
    1000 A.D., just north of the east house in Truce. Leene Square is north and Crono's
    House southeast." (the same facts as quests.where_text, in words to say). Also the
    nearest place's name when the character is at it, for the places visited."""
    from .atlas import compass
    from .rooms import plain_place

    px, py = x, y - rise
    era = m.name
    on = (f"on the world map in {era}" if re.search(r"\b(A\.?D|B\.?C)\b|\d", era)
          else f"on the {era} map")
    if not m.places:
        return f"You're {on}.", ""
    near = sorted(m.places.items(), key=lambda kv: (kv[1][0] - px) ** 2 + (kv[1][1] - py) ** 2)
    name, (nx, ny) = near[0]
    steps = ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5 / step
    if steps < 1:
        out = f"You're {on}, right at {plain_place(name)}."
    else:
        side = compass(px - nx, py - ny)
        close = "just " if steps <= 4 else ""
        out = f"You're {on}, {close}{side} of {plain_place(name)}."
    others = []
    for n, (ox, oy) in near[1:3]:
        d = ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5 / step
        if d <= 40:
            others.append((plain_place(n), compass(ox - px, oy - py),
                           "just " if d <= 4 else ""))
    if others:
        first = f"{others[0][0]} is {others[0][2]}{others[0][1]}"
        out += " " + (first[0].upper() + first[1:]) + (
            f" and {others[1][0]} {others[1][1]}." if len(others) > 1 else ".")
    return out, name if steps <= 3 else ""


# -- the agent side ----------------------------------------------------------------------

async def try_fast(agent: Agent, request: str, frame, run: Run, t0: float) -> tuple[bool, str]:
    """Do the request here if its words settle it. (done, what was tried): done means
    said and finished; else what was tried goes to the model with the request."""
    from .agent import Step

    said = " ".join(request.lower().split())
    if where_request(request):
        line = await _where(agent)
        if line:
            await agent._speak(line)
            run.steps.append(Step("where_am_i", {}, time.monotonic() - t0, line))
            run.outcome = line
            return True, ""
        return False, ""
    battle = await agent._battle_on(frame)
    if battle is not None:
        agent._battle = battle
        return await _fight(agent, said, battle, run, t0)
    agent._battle = None
    buy = buy_request(request)
    if buy is not None or rest_request(request):
        return await _shop(agent, buy, run, t0)
    return False, ""


async def _where(agent: Agent) -> str:
    """On a world map the atlas knows: where, from the places round the player."""
    atlas = await agent._the_atlas()
    if atlas is None:
        return ""
    from . import atlas as atlas_mod

    spot = await agent._spot(atlas, (await agent.grab()).image)
    if spot is None or spot.margin < 0.1:
        return ""
    m = atlas.map(spot.map)
    line, at = where_line(m, spot.x, spot.y, getattr(atlas_mod, "CELL", 8),
                          getattr(atlas_mod, "RISE", 0))
    if at:
        progress = agent._progress()
        progress.went(at)
        progress.save(agent.notes.folder, agent.game or "")
    return line


async def _fight(agent: Agent, said: str, battle: State, run: Run,
                 t0: float) -> tuple[bool, str]:
    """A fight: "win this fight", "run", "heal Frog", or one command, done with no
    pause for a model turn first (the fight keeps going while a model thinks, so it's
    paused for one; here nothing thinks)."""
    from .ack import FIGHT, RUN
    from .agent import HEAL, Step

    mode = "win" if FIGHT.search(said) else "run" if RUN.search(said) else ""
    if mode:
        result = await agent._act("fight", {"mode": mode})
        run.steps.append(Step("fight", {"mode": mode}, time.monotonic() - t0, result))
        if agent.cancelled:
            run.outcome = "cancelled"
            return True, ""
        ex = agent.executor()
        fought = getattr(agent, "_fought", None)
        line = won_line(result, fought.state if fought is not None else None) \
            if mode == "win" else ran_line(result)
        if line:
            await agent._speak(line)
            run.outcome = line
            return True, ""
        return False, (f"fight({mode}) was done already: {result}\nTell the player how "
                       "it went in a sentence, from that result, and finish.\n\n")
    heal = HEAL.fullmatch(said.strip(" .!?"))
    if heal:
        line = await agent._heal_now(heal.group("actor") or "", heal.group("who"), run, t0)
        if line is not None:
            run.outcome = line
            return True, ""
        return False, "".join(f"battle({s.args}) was tried already: {s.result}\n"
                              for s in run.steps) + (
            "\n" if run.steps else "The heal wasn't settled from the screen (who, or the "
            "party's HP, didn't read): your call.\n\n")
    intent = battle_request(said)
    if intent is None:
        return False, ""
    ex = agent.executor()
    before = ex.held if battle.paused and ex.held is not None else battle
    ex.after = ex.after_frame = None
    out = await ex.act(intent)
    args = {"actor": intent.actor, "command": intent.command, "name": intent.name,
            "target": intent.target}
    now = ctsteam.describe(out.state) if out.state is not None else "unreadable"
    result = f"{'done' if out.ok else 'stopped'}: {out.text}. Now: {now}"
    run.steps.append(Step("battle", args, time.monotonic() - t0, result))
    if agent.cancelled or "player" in (out.stopped or ""):
        run.outcome = "cancelled"
        return True, ""
    if out.ok:
        if ex.after_frame is not None and intent.command != "item":
            agent.recount(ex.after_frame, [m.name for m in out.state.party]
                          if out.state is not None else [])
        line = battle_line(intent, out, ex.before or before, ex.after, ex.went_off)
    else:
        line = stopped_line(intent, out)
    if line:
        await agent._speak(line)
        run.outcome = line
        return True, ""
    return False, (f"battle({args}) was tried already: {result}\n"
                   "Fix it once if the fix is plain, or tell the player.\n\n")


async def _shop(agent: Agent, buy: tuple[int, str] | None, run: Run,
                t0: float) -> tuple[bool, str]:
    """"Buy three potions" or "rest at the inn" inside a surveyed place with one seller
    (or innkeeper) who fits, or with the shop already open: walked, bought and said
    with no model turn. The price is said from the count screen, and the purchase as
    soon as the gold shows it (the seller's goodbye is pressed on while it's said)."""
    from .agent import Step

    rest = buy is None
    frame = await agent.grab()
    if not agent._ct_steam(frame):
        return False, ""
    seller = ""
    inside = await agent._inside(frame.image)
    if inside is not None:
        rooms, here = inside
        seller = seller_for(rooms.rooms[here.loc].people, rest)
    args: dict = {"action": "rest" if rest else "buy"}
    if seller:
        args["seller"] = seller
    if not rest:
        args |= {"item": buy[1], "count": buy[0]}
    priced = False
    lines: list[asyncio.Task] = []  # said while the shop goes on (no wait for the voice)

    async def price(n: int, item: str, total: int | None) -> None:
        nonlocal priced
        priced = True
        lines.append(asyncio.get_running_loop().create_task(
            agent._speak(price_line(n, item, total))))

    async def bought(n: int, item: str, spent: int, after: int) -> None:
        line = bought_line(n, item, spent, after, buy[0], priced)
        run.outcome = line
        lines.append(asyncio.get_running_loop().create_task(agent._speak(line)))

    agent._shop_hooks = (price, bought) if not rest else None
    try:
        result = await agent._act("shop", args)
    finally:
        agent._shop_hooks = None
        for task in lines:  # in the said list before the request ends
            with contextlib.suppress(Exception):
                await task
    run.steps.append(Step("shop", args, time.monotonic() - t0, result))
    if agent.cancelled:
        run.outcome = "cancelled"
        return True, ""
    if not rest and run.outcome and BOUGHT.search(result):
        return True, ""
    if not rest and (line := not_sold_line(result, buy[1])):
        await agent._speak(line)
        run.outcome = line
        return True, ""
    if run.outcome:  # said already, but the rest didn't read: the model looks
        return False, (f"shop({args}) was done already (the player heard: "
                       f"\"{run.outcome}\"): {result}\n\n")
    return False, f"shop({args}) was done already: {result}\n\n"
