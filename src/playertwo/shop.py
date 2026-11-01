"""Chrono Trigger (Steam) shops and inns, done by a local executor: buy N of an item, or
rest at an inn. The model only picks what to buy and how many; every press is checked
against the menu reader, and the gold is read before and after.

Live facts (2026-11-01, Melchior's stall at the fair): talking to a seller brings up their
line, sometimes with a choice ("Take a look" / "No, thanks"); then the shop screen: Buy,
Sell, Equip top left with the hand on Buy, "Funds: 496 G" under them, and the goods on the
right, one row each (about 78 px apart on the 2560 canvas): an icon, the name, the price
right-aligned. a on Buy puts the hand on the list; a on an item opens the count: "Bronze
Blade < 1 >" with the total ("350 G") under it, right and left change the count (it stops
at what the funds allow). b steps back: count, list, Buy/Sell/Equip, then out (the seller
says goodbye).
"""

from __future__ import annotations

import asyncio
import difflib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from . import ctsteam
from .glyphread import Run, snap

LIST_X = 800  # a hand right of this is in the goods panel
ROW_DY = 78  # rows of goods, apart
MAIN = ("Buy", "Sell", "Equip")
MAIN_Y = (83, 162, 243)  # where Buy, Sell and Equip sit (the hand's tip lines up)
PANEL_Y = 820  # the goods panel ends above this
TAP_MS = 100
GAP_S = 0.15
WAIT_S = 1.5
OPEN = ("Take a look", "Buy", "Yes", "Shop", "Let's see", "Sure")
# Items as the Steam port names them, for snapping a name read off the list.
ITEMS = ("Potion", "Mid-Potion", "Hi-Potion", "Full-Potion", "Ether", "Mid-Ether",
         "Hi-Ether", "Elixir", "Megalixir", "Athenian Water", "Revive", "Panacea", "Heal",
         "Shelter", "Lapis", "Barrier Sphere", "Shield Sphere", "Tonic", "Mid Tonic")


class ShopStop(Exception):
    pass


def number(text: str) -> int | None:
    """A price or a sum as the font reads it ("4.000", "4,000", "496 G", "35O")."""
    t = re.sub(r"(?<=\d)[uUQ]|[uUQ](?=\d)", "0", text.replace("G", "").strip())  # "1u": 10
    t = ctsteam.numfix(t)
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)
    m = re.search(r"\d+", t)
    return int(m.group()) if m else None


def item_name(text: str) -> str:
    """A goods row's name without the icon in front ("k3 Bronze Blade")."""
    words = text.split()
    while len(words) > 1 and (len(words[0]) <= 2 or not re.search(r"[aeiou]", words[0])):
        words = words[1:]
    return " ".join(words)


@dataclass
class Good:
    name: str
    price: int | None
    y: int


@dataclass
class View:
    kind: str = "none"  # main | list | count | none
    hand: tuple[int, int] | None = None
    funds: int | None = None
    goods: list[Good] = field(default_factory=list)
    on: str = ""  # what the hand is on: Buy/Sell/Equip, or a good's name
    count: int | None = None  # in the count view
    total: int | None = None  # its price for that count
    stock: int | None = None  # how many the party has of the good under the hand


def parse(tips: list[tuple[int, int]], lines: list[list[Run]]) -> View:
    v = View()
    rows = [r for row in lines for r in row]
    fund = next((r for r in rows if r.text.lower().startswith(("funds", "fund"))), None)
    if fund is not None:
        after = [r for r in rows if abs(r.y - fund.y) < 30 and r.x0 > fund.x1]
        if after:
            v.funds = number(min(after, key=lambda r: r.x0).text)
    if v.funds is None:
        return v
    stock = next((r for r in rows if r.text.lower().startswith("stock")), None)
    if stock is not None:
        after = [r for r in rows if abs(r.y - stock.y) < 30 and r.x0 > stock.x1]
        if after:
            v.stock = number(after[0].text)
    v.hand = min(tips, key=lambda t: t[1]) if tips else None
    # The goods: rows in the right-hand panel with a name and a price at the right.
    for row in lines:
        names = [r for r in row if LIST_X < r.x0 < 1500 and r.y < PANEL_Y
                 and re.search(r"[A-Za-z]{3}", r.text)]
        if not names:
            continue
        nm = names[0]
        prices = [r for r in row if r.x0 > nm.x1 and r.x1 > 1700 and number(r.text) is not None
                  and not r.text.strip().endswith("G")]
        v.goods.append(Good(snap(item_name(nm.text), ITEMS, 0.8), number(prices[-1].text)
                            if prices else None, nm.y))
    if v.hand is None:
        return v
    hx, hy = v.hand
    if hx < LIST_X:
        v.kind = "main"
        i = min(range(3), key=lambda k: abs(MAIN_Y[k] - hy))
        v.on = MAIN[i] if abs(MAIN_Y[i] - hy) < 40 else ""
        return v
    mine = next((g for g in v.goods if abs(g.y - hy) < 40), None)
    if mine is not None:
        v.on = mine.name
    # The count: the hand's row has the name and a small number, no price; the total
    # ("350 G") on a line under it.
    here = [r for row in lines for r in row if abs(r.y - hy) < 40 and 1450 < r.x0 < 1760]
    total = [r for row in lines for r in row if 60 < r.y - hy < 200 and r.x0 > 1500
             and r.text.strip().endswith("G") and number(r.text) is not None]
    if here and total and (mine is None or mine.price is None):
        v.kind = "count"
        v.count = number(here[0].text)
        v.total = number(total[0].text)
        if not v.on:
            names = [r for row in lines for r in row if abs(r.y - hy) < 40 and LIST_X < r.x0 < 1450]
            v.on = snap(item_name(names[0].text), ITEMS, 0.8) if names else ""
        return v
    v.kind = "list"
    return v


def _plain(name: str) -> str:
    """'Mid-Potion', 'mid potion' and 'Mid Potions' alike."""
    t = " ".join(re.sub(r"[-_]", " ", name.lower()).split()).strip(" .!?")
    return t[:-1] if t.endswith("s") and not t.endswith("ss") else t


KNOWN = {_plain(i) for i in ITEMS}


def same_good(a: str, b: str) -> bool:
    """Whether two names are the same good: the same name (any case, hyphen or space, one
    or many), or, when one of them isn't a name the game uses (the font misread it:
    "Brohze Blade", "Silver Sv)ord"), all but a letter or two alike. Two names the game
    uses never match unless equal: "Hi-Potion" isn't "Potion"."""
    x, y = _plain(a), _plain(b)
    if not x or not y:
        return False
    if x == y:
        return True
    if x in KNOWN and y in KNOWN:
        return False
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.85


def find(goods: list[Good], want: str) -> Good | None:
    """The good the player means, by its name ("Potion", not "Mid-Potion" or "Hi-Potion");
    None when the shop doesn't sell it (the caller lists what it does)."""
    exact = next((g for g in goods if _plain(g.name) == _plain(want)), None)
    if exact is not None:
        return exact
    close = [g for g in goods if same_good(g.name, want)]
    return max(close, key=lambda g: difflib.SequenceMatcher(
        None, _plain(g.name), _plain(want)).ratio(), default=None)


class Shop:
    def __init__(self, press: Callable[[str, int], Awaitable[None]],
                 grab: Callable[[], Awaitable[object]],
                 cancelled: Callable[[], bool] = lambda: False,
                 read: Callable[[object], tuple] = ctsteam.menu_read,
                 dialogue: Callable[[object], object] | None = None,
                 gap_s: float = GAP_S, wait_s: float = WAIT_S):
        self._press = press
        self._grab = grab
        self._read = read
        self._dialogue = dialogue
        self.cancelled = cancelled
        self.gap_s = gap_s
        self.wait_s = wait_s
        self.presses: list[str] = []
        self.frame: object = None
        self.said: list[str] = []  # the seller's lines read on the way in

    async def look(self) -> View:
        self.frame = await self._grab()
        hands, lines = await asyncio.to_thread(self._read, self.frame)
        return parse([h.tip for h in hands], lines)

    async def tap(self, button: str, times: int = 1) -> None:
        for _ in range(times):
            if self.cancelled():
                raise ShopStop("stopped by the player")
            await self._press(button, TAP_MS)
            self.presses.append(button)
            await asyncio.sleep(self.gap_s)

    async def until(self, ok: Callable[[View], bool], wait: float | None = None) -> View:
        end = time.monotonic() + (self.wait_s if wait is None else wait)
        v = await self.look()
        while not ok(v) and time.monotonic() < end:
            await asyncio.sleep(0.08)
            v = await self.look()
        return v

    async def _box(self):
        """The text box or choice on screen (ctscreen's look), or None."""
        if self._dialogue is None:
            from . import ctscreen

            self._dialogue = ctscreen.look
        return await asyncio.to_thread(self._dialogue, self.frame)

    async def open(self) -> View:
        """From the seller's words to the shop screen: lines are pressed on (and kept to
        tell), a choice is answered with its shopping option."""
        v = await self.look()
        for _ in range(8):
            if v.funds is not None:
                return v
            lk = await self._box()
            kind = getattr(lk, "kind", "none")
            parts = getattr(lk, "parts", {}) or {}
            if kind == "choice":
                # "No, thanks selected, or Take a look": only what the hand is on counts.
                focus = parts.get("focus", "")
                on = re.sub(r"[^a-z' ]", "", focus.split(" selected")[0].lower()).strip()
                if on and on not in {o.lower() for o in OPEN}:
                    await self.tap("up")
                    v = await self.look()
                    lk2 = await self._box()
                    if getattr(lk2, "parts", {}).get("focus", "") == focus:
                        await self.tap("down", 2)
                    v = await self.look()
                    continue
                await self.tap("a")
            elif kind == "text":
                line = parts.get("text", "")
                if line and line not in self.said:
                    self.said.append(line)
                await self.tap("a")
            else:
                await asyncio.sleep(0.4)
            v = await self.until(lambda v: v.funds is not None, 1.2)
        if v.funds is None:
            raise ShopStop("no shop came up (talk to the seller first: go_to_exit with talk)")
        return v

    async def to_list(self, v: View) -> View:
        """The Buy list, always gone into from Buy: a list already open may be the Sell
        list (it looks the same), where a sells."""
        for _ in range(3):
            if v.kind not in ("list", "count"):
                break
            back = "list" if v.kind == "count" else "main"
            await self.tap("b")
            v = await self.until(lambda v, back=back: v.kind == back)
        if v.kind in ("list", "count"):
            raise ShopStop("couldn't get back to Buy, Sell and Equip")
        for _ in range(4):
            if v.kind == "main" and v.on == "Buy":
                break
            await self.tap("up")
            v = await self.look()
        if v.kind != "main" or v.on != "Buy":
            raise ShopStop("couldn't put the hand on Buy")
        await self.tap("a")
        v = await self.until(lambda v: v.kind == "list")
        if v.kind != "list":
            raise ShopStop("Buy didn't open the list of goods")
        return v

    async def to_good(self, v: View, want: str) -> tuple[View, Good]:
        good = find(v.goods, want)
        if good is None:
            have = ", ".join(f"{g.name} ({g.price} G)" for g in v.goods if g.name)
            raise ShopStop(f"no {want} here; this shop sells: {have or 'nothing I could read'}")
        for _ in range(3):
            if v.on == good.name:
                return v, good
            rows = round((good.y - v.hand[1]) / ROW_DY) if v.hand else 0
            if rows == 0:
                rows = 1 if (v.goods.index(good) if good in v.goods else 0) > 0 else -1
            await self.tap("down" if rows > 0 else "up", abs(rows))
            v = await self.look()
        if v.on != good.name:
            raise ShopStop(f"couldn't put the hand on {good.name}")
        return v, good

    async def set_count(self, v: View, count: int) -> View:
        """The count view, set to `count` (or as near as the funds allow). Stops when the
        count doesn't read: buying whatever it shows would be a guess."""
        def read(v: View) -> bool:
            return v.kind == "count" and v.count is not None

        await self.tap("a")
        v = await self.until(read)
        if v.kind != "count":
            raise ShopStop("the count didn't come up")
        for _ in range(3):
            if v.count is None or v.count == count:
                break
            step = "right" if v.count < count else "left"
            before = v.count
            await self.tap(step, abs(count - v.count))
            v = await self.until(read, 0.6)
            if v.count == before:
                break  # it won't go further: what the funds allow, or the most there is
        if v.kind != "count" or v.count is None:
            raise ShopStop("the count didn't read, so nothing was bought")
        if v.count > count:
            raise ShopStop(f"the count reads {v.count}, more than the {count} asked for")
        return v

    async def buy(self, item: str, count: int = 1) -> str:
        count = max(1, min(int(count or 1), 99))
        start = len(self.presses)
        try:
            v = await self.open()
            funds0 = v.funds
            v = await self.to_list(v)
            v, good = await self.to_good(v, item)
            have0 = v.stock
            v = await self.set_count(v, count)
            n = v.count
            if n == 0:
                raise ShopStop(f"can't buy any {good.name}: {funds0} G isn't enough "
                               f"({good.price} G each)")
            if not v.on:
                v = await self.until(lambda v: v.kind == "count" and bool(v.on), 0.6)
            if v.kind != "count" or not same_good(v.on, good.name):
                # The last a buys what the count shows: it must be what was asked for.
                raise ShopStop(f"the count shows {v.on or 'a name that does not read'}, "
                               f"not {good.name}")
            await self.tap("a")
            v = await self.until(lambda v: v.funds is not None and v.funds != funds0, 2.5)
            if v.funds == funds0 and v.kind == "count":
                await self.tap("a")  # a confirm step
                v = await self.until(lambda v: v.funds is not None and v.funds != funds0, 2.0)
            if v.funds is None:
                raise ShopStop(f"the funds didn't read after buying (they were {funds0} G), "
                               "so it isn't known what was bought: look at the inventory")
            spent = funds0 - v.funds
            if spent < 0:
                raise ShopStop(f"the funds went up from {funds0} G to {v.funds} G: that "
                               "was a sale, not a purchase")
            if spent == 0:
                raise ShopStop(f"the funds didn't change ({funds0} G): nothing was bought")
            note = ""
            if good.price and spent != n * good.price:
                if spent % good.price == 0:
                    # The gold says how many: the count misread.
                    note = f" (the count read {n}; the gold says {spent // good.price})"
                    n = spent // good.price
                else:
                    note = (f" ({spent} G isn't a whole number of {good.name} at "
                            f"{good.price} G: how many is unsure; look at the inventory)")
            have = v.stock
            out = await self.leave()
            got = (f"Bought {n} {good.name} for {spent} G (funds {funds0} G to {v.funds}"
                   f" G){note}")
            if have is not None and have0 is not None and have == have0 + n:
                got += f"; the party has {have} now"  # (only when it adds up: it misreads)
            if n < count:
                got += f"; only {n} of the {count} asked for: that's what the funds allowed"
            return got + ". " + out
        except ShopStop as exc:
            out = await self.leave()
            pressed = " ".join(self.presses[start:])
            return f"stopped: {exc}. {out}" + (f" (pressed {pressed})" if pressed else "")

    async def leave(self) -> str:
        """Back out of the shop screen (b until it's gone)."""
        v = await self.look()
        for _ in range(5):
            if v.funds is None:
                break
            await self.tap("b")
            v = await self.until(lambda v: v.funds is None, 0.8)
        if v.funds is not None:
            return "The shop screen is still up."
        # The seller's goodbye: read, then pressed on, so no box is left over the place
        # (live: the next "take me outside" stopped on it).
        said: list[str] = []
        end = time.monotonic() + 1.6  # the goodbye comes up a moment after the screen goes
        while time.monotonic() < end:
            lk = await self._box()
            if getattr(lk, "kind", "none") == "text":
                break
            await asyncio.sleep(0.2)
            self.frame = await self._grab()
        for _ in range(4):
            lk = await self._box()
            line = (getattr(lk, "parts", {}) or {}).get("text", "") if lk is not None else ""
            if getattr(lk, "kind", "none") != "text" or not line:
                break
            if line not in said:
                said.append(line)
            await self.tap("a")
            await asyncio.sleep(0.5)
            self.frame = await self._grab()
        if said:
            return "Out of the shop; the seller said: " + " / ".join(f'"{x}"' for x in said) \
                + " (pressed on: no box is up)."
        return "Out of the shop."


INN_Q = re.compile(r"stay|rest|night|lodging|inn\b|sleep", re.IGNORECASE)
NIGHT_S = 10.0  # after Yes, the screen goes dark for the night well within this


def vitals_line(names: list[str], vit: list[tuple[int, int, int, int]]) -> tuple[str, bool]:
    """'Crono 70/70 HP 8/8 MP, Marle ...' and whether everyone read is full."""
    parts, full = [], bool(vit)
    named = len(names) == len(vit)  # the menu's names misread often (Crono's hair)
    for k, (hp, top, mp, mtop) in enumerate(vit[:3]):
        who = names[k] if named else f"the {('first', 'second', 'third')[k]} in the party"
        mp_s = f", {mp}/{mtop} MP" if mp >= 0 else ""
        parts.append(f"{who} {hp}/{top} HP{mp_s}")
        full = full and hp == top and (mp < 0 or mp == mtop)
    return "; ".join(parts), full


class Inn:
    """Resting at an inn: the innkeeper's lines pressed on, Yes to staying, the night
    waited out, then HP, MP and gold read from the menu."""

    def __init__(self, shop: Shop, menu):
        self.shop, self.menu = shop, menu

    async def _dark(self) -> bool:
        import numpy as np

        f = self.shop.frame
        a = np.asarray(f.convert("L") if hasattr(f, "convert") else f)
        return float(a.mean()) < 25

    async def vitals(self):
        """HP, MP and gold from the menu; the morning can hold the controls a moment (the
        party walks out of bed), so the menu gets a few tries."""
        for k in range(4):
            try:
                return await self.menu.vitals()
            except Exception:  # noqa: BLE001  MenuStop: the menu didn't open yet
                if k == 3:
                    return [], None
                await asyncio.sleep(1.5)
        return [], None

    async def rest(self, gold0: int | None = None, vit0=None) -> str:
        sh = self.shop
        start = len(sh.presses)
        heard: list[str] = []
        paid = slept = False
        paid_at = 0.0
        quiet = 0
        end = time.monotonic() + 45
        try:
            while time.monotonic() < end:
                sh.frame = await sh._grab()
                lk = await sh._box()
                kind = getattr(lk, "kind", "none")
                parts = getattr(lk, "parts", {}) or {}
                if kind == "choice" and not paid:
                    q, focus = parts.get("question", ""), parts.get("focus", "")
                    if q and q not in heard:
                        heard.append(q)
                    if not focus.lower().startswith("yes"):
                        await sh.tap("up")
                        continue
                    await sh.tap("a")
                    paid, paid_at = True, time.monotonic()
                    await asyncio.sleep(1.0)
                    continue
                if kind in ("text", "choice"):
                    t = parts.get("text") or parts.get("question", "")
                    if t and t not in heard:
                        heard.append(t)
                    if not paid and t.rstrip().endswith("?") and INN_Q.search(t):
                        # The choice didn't read: Yes is the first option, where the
                        # hand starts.
                        await sh.tap("a")
                        paid, paid_at = True, time.monotonic()
                        await asyncio.sleep(1.0)
                        continue
                    await sh.tap("a")
                    await asyncio.sleep(0.6)
                    quiet = 0
                    continue
                if not paid and any(INN_Q.search(h) or "lodging" in h.lower() for h in heard):
                    hands, _lines = await asyncio.to_thread(sh._read, sh.frame)
                    if hands:
                        # The innkeeper asked; the Yes/No beside a hand didn't read as a
                        # choice (live: the hand's line read "Tut-"). Yes is the first
                        # option, where the hand starts.
                        await sh.tap("a")
                        paid, paid_at = True, time.monotonic()
                        await asyncio.sleep(1.0)
                        continue
                if await self._dark():
                    slept = slept or paid
                    quiet = 0
                    await asyncio.sleep(0.5)
                    continue
                if paid and slept:
                    quiet += 1
                    if quiet >= 3:
                        break  # the room again, no box: morning
                elif paid and time.monotonic() - paid_at > NIGHT_S:
                    break  # no night came (not enough gold?): the gold says, below
                elif not paid and len(heard) == 0 and time.monotonic() > end - 40:
                    await asyncio.sleep(0.4)
                    if time.monotonic() > end - 38:
                        raise ShopStop("the innkeeper said nothing (talk to them first: "
                                       "go_to_exit with talk)")
                await asyncio.sleep(0.4)
            if not paid:
                said = " / ".join(heard)
                raise ShopStop("no question about staying came up" + (f" ({said})" if said
                                                                      else ""))
            vit, gold = await self.vitals()
            line, full = vitals_line(self.menu.party, vit)
            said = (" The innkeeper said: " + " / ".join(f'"{h}"' for h in heard) + "."
                    if heard else "")
            if not slept:
                # No night after Yes (not enough gold, or a fade the looks missed): only
                # the gold going down says they stayed.
                if gold0 is None or gold is None:
                    raise ShopStop("no night came after Yes and the gold didn't read, so "
                                   "it isn't known whether the party stayed. Now: "
                                   f"{line or 'HP not read'}.{said}")
                if gold >= gold0:
                    raise ShopStop(f"didn't stay: no night came after Yes and the gold is "
                                   f"still {gold} G.{said}")
            out = "Stayed the night"
            if gold0 is not None and gold is not None:
                out += f" ({gold0 - gold} G: {gold0} to {gold} G)"
            out += f". Now: {line or 'HP not read'}" + (" (everyone full)." if full else ".")
            return out + said
        except ShopStop as exc:
            pressed = " ".join(sh.presses[start:])
            return f"stopped: {exc}" + (f" (pressed {pressed})" if pressed else "")
