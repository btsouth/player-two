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
    t = ctsteam.numfix(text.replace("G", "").strip())
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


def find(goods: list[Good], want: str) -> Good | None:
    """The good the player means: the same name first ("Potion", not "Mid-Potion")."""
    w = want.strip().lower().rstrip("s") if not want.lower().endswith("ss") else want.lower()
    for g in goods:
        if g.name.lower() == want.strip().lower() or g.name.lower() == w:
            return g
    close = snap(want.strip(), [g.name for g in goods], 0.6)
    return next((g for g in goods if g.name == close), None)


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
                focus = parts.get("focus", "")
                if not any(o.lower() in focus.lower() for o in OPEN) and focus:
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
        if v.kind == "list":
            return v
        if v.kind == "count":
            await self.tap("b")
            return await self.until(lambda v: v.kind == "list")
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
        await self.tap("a")
        v = await self.until(lambda v: v.kind == "count")
        if v.kind != "count":
            raise ShopStop("the count didn't come up")
        for _ in range(3):
            if v.count is None or v.count == count:
                break
            step = "right" if v.count < count else "left"
            before = v.count
            await self.tap(step, abs(count - v.count))
            v = await self.look()
            if v.count == before:
                break  # it won't go further: what the funds allow, or the most there is
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
            n = v.count or count
            if n == 0:
                raise ShopStop(f"can't buy any {good.name}: {funds0} G isn't enough "
                               f"({good.price} G each)")
            await self.tap("a")
            v = await self.until(lambda v: v.funds is not None and v.funds != funds0, 2.5)
            if v.funds == funds0 and v.kind == "count":
                await self.tap("a")  # a confirm step
                v = await self.until(lambda v: v.funds is not None and v.funds != funds0, 2.0)
            spent = (funds0 - v.funds) if funds0 is not None and v.funds is not None else None
            if not spent:
                raise ShopStop(f"the funds didn't change ({funds0} G): nothing was bought")
            have = v.stock
            out = await self.leave()
            got = (f"Bought {n} {good.name} for {spent} G (funds {funds0} G to {funds0 - spent}"
                   f" G)")
            if have is not None:
                got += f"; the party has {have} now" + (f" (had {have0})" if have0 is not None
                                                        else "")
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
        lk = await self._box()
        line = (getattr(lk, "parts", {}) or {}).get("text", "") if lk is not None else ""
        if getattr(lk, "kind", "none") == "text" and line:
            return f'Out of the shop; the seller says: "{line}" (a text box is up).'
        return "Out of the shop."


INN_Q = re.compile(r"stay|rest|night|lodging|inn\b|sleep", re.IGNORECASE)


def vitals_line(names: list[str], vit: list[tuple[int, int, int, int]]) -> tuple[str, bool]:
    """'Crono 70/70 HP 8/8 MP, Marle ...' and whether everyone read is full."""
    parts, full = [], bool(vit)
    for k, (hp, top, mp, mtop) in enumerate(vit[:len(names) or 3]):
        who = names[k] if k < len(names) else f"member {k + 1}"
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

    async def rest(self, gold0: int | None = None, vit0=None) -> str:
        sh = self.shop
        start = len(sh.presses)
        heard: list[str] = []
        paid = slept = False
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
                    paid = True
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
                        paid = True
                        await asyncio.sleep(1.0)
                        continue
                    await sh.tap("a")
                    await asyncio.sleep(0.6)
                    quiet = 0
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
            vit, gold = await self.menu.vitals()
            line, full = vitals_line(self.menu.party, vit)
            out = "Stayed the night"
            if gold0 is not None and gold is not None:
                out += f" ({gold0 - gold} G: {gold0} to {gold} G)"
            out += f". Now: {line or 'HP not read'}" + (" (everyone full)." if full else ".")
            if heard:
                out += " The innkeeper said: " + " / ".join(f'"{h}"' for h in heard) + "."
            return out
        except ShopStop as exc:
            pressed = " ".join(sh.presses[start:])
            return f"stopped: {exc}" + (f" (pressed {pressed})" if pressed else "")
