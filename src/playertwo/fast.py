"""Requests done without a model turn: the plain ones the player says most.

Each model turn costs one to three seconds, and a request with a tool call has at least
two (the call, then a sentence from its result), plus a settle look between them. For a
request whose words already say exactly what to do ("buy three potions", "Crono, use
Cyclone", "attack the one on the left"), the tool runs at once and its result is said
from a template: the same facts the model would have phrased, read off the same screen,
in the same words as battle.py's lines. Anything the words or the result don't settle
goes to the model as before, with what was tried. ("Where am I?" is Agent._where_line's,
from the atlas or the room survey.)

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
    act_line,
    heal_line,
    heals,
    low_line,
    read_hp,
    revives,
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


def rested_line(result: str) -> str:
    """A night at the inn that went as it should, from the gold and the menu's HP read
    after it: "Rested for 10 gold, 390 left. Everyone's at full health." ("" when the
    gold or everyone's HP didn't read: the model says what did)."""
    m = re.search(r"Stayed the night \((\d+) G: \d+ to (\d+) G\)\. Now: .*\(everyone full\)",
                  result)
    if not m:
        return ""
    return f"Rested for {m.group(1)} gold, {m.group(2)} left. Everyone's at full health."


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
    # Observed Muse Voice spellings with American male speech. Only the actor slot
    # gets the homophone; "you" counts as "use" only before a known action
    # (the normal parser below still checks the ability/item and target).
    t = re.sub(r"^chrono(?=,? )", "crono", t)
    t = re.sub(rf"^({'|'.join(NAMES)}),? you (?=.)", r"\1, use ", t)
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
    """One command done, in a sentence, from what the screen showed, in the fight's own
    words (battle.act_line, low_line): who did what, and what happened to the party while
    it played ("Crono used Cyclone. Frog and Marle dropped to 1 HP."). Enemy HP isn't on
    the screen, so nothing is said about it. The HP is the newest reading with everyone's
    read: the one when the command had played out, else the paused screen's."""
    actor = intent.actor or (out.text.split(":")[0].strip() if ":" in out.text else "")
    who = actor or "We"
    cmd = intent.command.lower()
    ally = intent.target if intent.target in ctsteam.NAMES else ""
    if cmd == "attack":
        lead = f"{who} attacked{_target_words(intent.target)}."
    elif went_off:
        lead = act_line(Intent(who, cmd, intent.name, ally), done=True)
        if intent.target and not ally:
            lead = f"{lead[:-1]} on{_target_words(intent.target)}."
    else:
        # (As heal_line says a heal that hasn't landed: the executor leaves it paused.)
        lead = f"{who} picked {intent.name}; the fight is paused, so it goes off when we go on."
    st = read_hp(after, out.state)
    b, a = _hp(before), _hp(st)
    down: list[str] = []
    dropped: dict[int, list[str]] = {}
    up: list[str] = []
    for m in (st.party if st is not None else []):
        old, hp = b.get(m.name), a.get(m.name)
        if old is None or hp is None or hp == old:
            continue
        if hp == 0:
            down.append(m.name)
        elif hp < old:
            dropped.setdefault(hp, []).append(m.name)
        else:
            up.append(f"{m.name} is back up, at {hp} HP" if old == 0
                      else f"{m.name} is up to {hp}")
    news = [f"{_join(down)} {'is' if len(down) == 1 else 'are'} down"] if down else []
    news += [f"{_join(names)} dropped to {hp} HP" for hp, names in dropped.items()]
    news += up
    parts = [lead]
    if news:
        parts.append(_cap(_join(news)) + ".")
    changed = {n for n in a if a.get(n) is not None and b.get(n) is not None
               and a[n] != b[n]}
    low = low_line(st, skip=changed) if st is not None else ""
    if low:
        parts.append(low)  # still in danger from before
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
    """A won fight from the executor's result and the screen after it, as the fight's
    lines say it: "Won the fight: 648 experience. Frog's at 1 HP." (who's still low, not
    everyone's HP); "" when it doesn't read as won."""
    if result.startswith("done: the fight ended"):
        line = "The fight's over."
    elif result.startswith("done: won"):
        exp = re.search(r"Earned (\d+) EXP", result, re.IGNORECASE)
        tp = re.search(r"Earned (\d+) TP", result, re.IGNORECASE)
        got = _join([*([f"{exp.group(1)} experience"] if exp else []),
                     *([f"{tp.group(1)} tech points"] if tp else [])])
        line = "Won the fight" + (f": {got}." if got else ".")
        for who in dict.fromkeys(re.findall(r"(\w+)'s level increased", result)):
            line += f" {who} went up a level."
    else:
        return ""
    low = low_line(st) if st is not None and st.party else ""
    return f"{line} {low}".strip()


def ran_line(result: str) -> str:
    return "We got away." if result.startswith("done: got away") else ""


def talk_line(who: str, line: str) -> str:
    """Someone's line as said: explicit attribution when the player named them by it
    ("Melchior says ..."), else an anonymous quotation (it
    comes right after "Going to the old man at the lower stalls.")."""
    name = who.strip().split(" ")[0] if who else ""
    if name[:1].isupper() and name.lower() not in ("the", "a", "an", "that", "this"):
        return f'{name} says, "{line}"'
    return f'They say, "{line}"'


# -- where am I ------------------------------------------------------------------------

WHERE = re.compile(rf"^(?:(?:hey|so|okay|ok|and|um),? )?{_POLITE}(?:where am i|where are we|"
                   rf"tell me where (?:i am|we are))(?: (?:now|right now|exactly|again))?"
                   rf"{_TAIL}$")


def where_request(text: str) -> bool:
    """"Where am I?", "where are we now", "tell me where we are": answered from the atlas
    or the room survey with no model turn (Agent._where_line)."""
    return bool(WHERE.match(plain(text)))


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# -- the agent side ----------------------------------------------------------------------

async def try_fast(agent: Agent, request: str, frame, run: Run, t0: float) -> tuple[bool, str]:
    """Do the request here if its words settle it. (done, what was tried): done means
    said and finished; else what was tried goes to the model with the request."""
    said = " ".join(request.lower().split())
    battle = await agent._battle_on(frame)
    if battle is not None:
        agent._battle = battle
        return await _fight(agent, said, battle, run, t0)
    agent._battle = None
    buy = buy_request(request)
    if buy is not None or rest_request(request):
        return await _shop(agent, buy, run, t0)
    return False, ""


async def _fight(agent: Agent, said: str, battle: State, run: Run,
                 t0: float) -> tuple[bool, str]:
    """A fight: "win this fight", "run", "heal Frog", or one command, done with no
    pause for a model turn first (the fight keeps going while a model thinks, so it's
    paused for one; here nothing thinks)."""
    from .ack import FIGHT, RUN
    from .agent import HEAL, Step

    mode = "win" if FIGHT.search(said) else "run" if RUN.search(said) else ""
    if mode:
        agent._fought = None
        result = await agent._act("fight", {"mode": mode})
        run.steps.append(Step("fight", {"mode": mode}, time.monotonic() - t0, result))
        if agent.cancelled:
            run.outcome = "cancelled"
            return True, ""
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
        # Who needs what, and "Frog is at full health already", from the HP as it is now:
        # the request's frame can be a moment old, and the fight isn't paused here.
        agent._battle = await agent._battle_on(await agent.grab()) or battle
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
    # A heal named outright ("Marle, use Cure on Frog") is watched till it lands, as "heal
    # Frog" is: said as it's put in, not after up to 14 s of silence.
    healing = heals(intent) or revives(intent)
    ex.on_confirm = agent._put_in if healing else None
    try:
        out = await ex.act(intent)
    finally:
        ex.on_confirm = None
    args = {"actor": intent.actor, "command": intent.command, "name": intent.name,
            "target": intent.target}
    now = ctsteam.describe(out.state) if out.state is not None else "unreadable"
    result = f"{'done' if out.ok else 'stopped'}: {out.text}. Now: {now}"
    run.steps.append(Step("battle", args, time.monotonic() - t0, result))
    if agent.cancelled or "player" in (out.stopped or ""):
        run.outcome = "cancelled"
        return True, ""
    if out.ok:
        if ex.after_frame is not None and intent.command != "item" and not healing:
            agent.recount(ex.after_frame, [m.name for m in out.state.party]
                          if out.state is not None else [])
        line = heal_line(intent, out, ex.before or before, ex.after, told=True) if healing \
            else battle_line(intent, out, ex.before or before, ex.after, ex.went_off)
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
    # The seller's goodbye left up over the place (Shop.leave tells it by its window):
    # the model reads it and closes it, after what was said already.
    box_left = "text box is still up" in result
    if not rest and run.outcome and BOUGHT.search(result) and not box_left:
        return True, ""
    if rest and (line := rested_line(result)):
        await agent._speak(line)
        run.outcome = line
        return True, ""
    if not rest and not box_left and (line := not_sold_line(result, buy[1])):
        await agent._speak(line)
        run.outcome = line
        return True, ""
    if run.outcome:  # said already, but the rest didn't read: the model looks
        return False, (f"shop({args}) was done already (the player heard: "
                       f"\"{run.outcome}\"): {result}\n\n")
    return False, f"shop({args}) was done already: {result}\n\n"
