"""Chrono Trigger (Steam) field menus, done by a local executor: a tech or an item used on
someone, a battle setting changed. Every press is checked against the menu reader.

Live facts (2026-11-01): y opens the menu (Equipment, Inventory, Techs, Party, Settings,
Bookmark, Save). Techs: pick the caster's block on the left (the active party is the first
three), then the tech on the right (green ones work outside battle), then who it's for on
the blocks that come up on the right. Inventory opens on "Use / Switch": a on a green item
uses it, a on any other starts a swap (a second hand shows; b cancels it), so a is never
pressed twice in the list. Settings > Battle: a header per setting with its value below,
changed with left and right (Battle Speed: 1 is fastest, 8 slowest).

The entry under the hand in the tech list reads badly (its highlight), so in lists whose
order is known the hand's entry is told from its neighbours.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable

from . import ctsteam
from .glyphread import Run, snap

MAIN = ("Equipment", "Inventory", "Techs", "Party", "Settings", "Bookmark", "Save")
SETTINGS = ("Battle", "Sound", "Graphics", "Controls", "Defaults")
BATTLE = ("Battle Mode", "Battle Speed", "Battle Message Speed", "Battle Cursor Memory",
          "Help Messages")
MODES = ("WAIT", "ACTIVE")
TAP_MS = 100
GAP_S = 0.12
WAIT_S = 1.5
RIGHT_X = 1150  # hands right of this are in the right-hand panel


class MenuStop(Exception):
    pass


def same(a: str, b: str) -> bool:
    return bool(a) and snap(a, (b,), 0.75) == b


class FieldMenu:
    def __init__(self, press: Callable[[str, int], Awaitable[None]],
                 grab: Callable[[], Awaitable[object]],
                 cancelled: Callable[[], bool] = lambda: False,
                 read: Callable[[object], tuple] = ctsteam.menu_read,
                 gap_s: float = GAP_S, wait_s: float = WAIT_S, party: list[str] | None = None):
        self._press = press
        self._grab = grab
        self._read = read
        self.cancelled = cancelled
        self.gap_s = gap_s
        self.wait_s = wait_s
        self.presses: list[str] = []
        self.party: list[str] = list(party or [])  # the active party, as battles read it

    async def look(self) -> tuple[list[tuple[int, int]], list[list[Run]]]:
        hands, lines = await asyncio.to_thread(self._read, await self._grab())
        return [h.tip for h in hands], lines

    async def tap(self, button: str, times: int = 1) -> None:
        for _ in range(times):
            if self.cancelled():
                raise MenuStop("stopped by the player")
            await self._press(button, TAP_MS)
            self.presses.append(button)
            await asyncio.sleep(self.gap_s)

    async def until(self, ok: Callable[[list, list], bool]) -> tuple[list, list]:
        end = time.monotonic() + self.wait_s
        tips, lines = await self.look()
        while not ok(tips, lines) and time.monotonic() < end:
            await asyncio.sleep(0.05)
            tips, lines = await self.look()
        return tips, lines

    # -- what the hand is on -----------------------------------------------------------

    @staticmethod
    def on(lines, tip, vocab, span: int = 420) -> str | None:
        """The vocabulary word on the hand's line, just right of it."""
        runs = sorted((r for row in lines for r in row
                       if abs(r.y - tip[1]) < 34 and -25 <= r.x0 - tip[0] < span),
                      key=lambda r: r.x0)
        for r in runs:
            w = snap(r.text, vocab, 0.6)
            if w in vocab:
                return w
        return None

    @classmethod
    def in_order(cls, lines, tip, order, span: int = 420) -> str | None:
        """The hand's entry in a list whose order is known: read, or told from the
        nearest readable entry above or below (the hand's own reads badly)."""
        got = cls.on(lines, tip, order, span)
        if got:
            return got
        col = [(r.y, snap(r.text, order, 0.75)) for row in lines for r in row
               if -25 <= r.x0 - tip[0] < span]
        col = [(y, w) for y, w in col if w in order and abs(y - tip[1]) > 30]
        if not col:
            return None
        y, w = min(col, key=lambda c: abs(c[0] - tip[1]))
        rows = round((tip[1] - y) / 80)  # tech rows are about 80 px apart
        i = order.index(w) + rows
        return order[i] if 0 <= i < len(order) else None

    @staticmethod
    def block_name(lines, tip) -> str | None:
        """A hand on a character block points at the portrait; the name is up and right."""
        near = [(snap(r.text, ctsteam.NAMES, 0.6), r) for row in lines for r in row
                if -170 < r.y - tip[1] < 40 and -25 <= r.x0 - tip[0] < 420]
        near = [(n, r) for n, r in near if n in ctsteam.NAMES]
        if not near:
            return None
        return min(near, key=lambda t: abs(t[1].y - tip[1]) + 0.1 * abs(t[1].x0 - tip[0]))[0]

    def who(self, lines, tip) -> str | None:
        """The block's name, or, when its highlight makes it unreadable, the one member of
        the party (as the main menu showed it) whose name isn't readable in that column."""
        got = self.block_name(lines, tip)
        if got or not self.party:
            return got
        # Blocks are about 385 px apart, in party order: count from a readable one.
        col = [(r.y, snap(r.text, ctsteam.NAMES, 0.75)) for row in lines for r in row
               if 0 <= r.x0 - tip[0] < 450]
        col = [(y, n) for y, n in col if n in self.party]
        if not col:
            return None
        y, n = min(col, key=lambda c: abs(c[0] - (tip[1] - 80)))
        k = self.party.index(n) + round((tip[1] - 80 - y) / 385)
        return self.party[k] if 0 <= k < len(self.party) else None

    # -- moving in lists ----------------------------------------------------------------

    async def walk(self, order, want: str, pick=lambda tips: tips[0], span: int = 420,
                   wrap_ok: bool = False) -> tuple:
        """Move the hand down a list whose order is known to `want`."""
        for _ in range(12):
            tips, lines = await self.look()
            if not tips:
                raise MenuStop("the hand went away")
            tip = pick(tips)
            cur = self.in_order(lines, tip, order, span)
            if cur == want:
                return tip, lines
            if cur is None:
                await self.tap("down")
                continue
            d = order.index(want) - order.index(cur)
            await self.tap("down" if d > 0 else "up", abs(d))
        raise MenuStop(f"couldn't put the hand on {want}")

    async def blocks(self, want: str, pick) -> tuple:
        """Move the hand over character blocks to `want`'s."""
        last = None
        for _ in range(8):
            tips, lines = await self.look()
            tip = pick(tips)
            if tip is None:
                raise MenuStop("the hand went away")
            cur = self.who(lines, tip)
            if cur == want:
                return tip, lines
            at = [r.y for row in lines for r in row if snap(r.text, (want,), 0.75) == want
                  and abs(r.x0 - tip[0]) < 500]
            way = "down" if not at or at[0] > tip[1] else "up"
            if last == (cur, tip):
                way = "up" if way == "down" else "down"
            last = (cur, tip)
            await self.tap(way)
        raise MenuStop(f"couldn't put the hand on {want}")

    async def open(self) -> None:
        tips, lines = await self.look()
        # Already somewhere inside the menu (a sub-screen): back out to its main list.
        for _ in range(4):
            if not tips or self.on(lines, tips[0], MAIN):
                break
            await self.tap("b")
            await asyncio.sleep(0.2)
            tips, lines = await self.look()
        if not (tips and self.on(lines, tips[0], MAIN)):
            await self.tap("y")
            tips, lines = await self.until(lambda t, ls: bool(t) and bool(
                self.on(ls, t[0], MAIN)))
            if not tips:
                raise MenuStop("the menu didn't open (y)")
        # The active party, top to bottom, from the blocks beside the main list.
        names = sorted((r.y, snap(r.text, ctsteam.NAMES, 0.6)) for row in lines for r in row
                       if r.x0 > 600)
        seen = list(dict.fromkeys(n for _y, n in names if n in ctsteam.NAMES))
        if len(seen) > len(self.party):
            self.party = seen

    async def close(self) -> None:
        for _ in range(5):
            tips, _lines = await self.look()
            if not tips:
                return
            await self.tap("b")
            await asyncio.sleep(0.15)

    async def to_main(self, item: str) -> None:
        await self.open()
        await self.walk(MAIN, item, lambda tips: min(tips, key=lambda t: t[0]))
        await self.tap("a")

    # -- the requests --------------------------------------------------------------------

    async def use(self, kind: str, name: str, user: str, target: str) -> str:
        try:
            text = await (self._tech(name, user, target) if kind == "tech"
                          else self._item(name, target))
            await self.close()
            return text
        except MenuStop as exc:
            if "player" not in str(exc):
                await self.close()
            return f"stopped: {exc}. The menu is closed."

    async def _tech(self, name: str, user: str, target: str) -> str:
        user = snap(user, ctsteam.NAMES, 0.6)
        order = ctsteam.TECHS.get(user)
        if not order:
            raise MenuStop(f"say who casts it ({', '.join(ctsteam.NAMES)})")
        tech = snap(name, order, 0.6)
        if tech not in order:
            raise MenuStop(f"{user} has no tech called {name} ({', '.join(order)})")
        await self.to_main("Techs")
        await self.until(lambda t, ls: bool(t))
        left = lambda tips: min(tips, key=lambda t: t[0]) if tips else None
        right = lambda tips: max(tips, key=lambda t: t[0]) if tips else None
        await self.blocks(user, left)
        await self.tap("a")
        await self.until(lambda t, ls: any(x > RIGHT_X for x, _ in t))
        tip, lines = await self.walk(order, tech, right)
        runs = [r for row in lines for r in row if abs(r.y - tip[1]) < 34 and r.x0 > tip[0]]
        if runs and runs[0].colour not in ("green", ""):
            raise MenuStop(f"{tech} can't be used outside battle (it isn't green)")
        return await self._on_whom(tech, target)

    async def _on_whom(self, what: str, target: str) -> str:
        await self.tap("a")
        tips, lines = await self.until(lambda t, ls: any(
            x > RIGHT_X and self.who(ls, (x, y)) for x, y in t))
        on_blocks = [t for t in tips if t[0] > RIGHT_X and self.who(lines, t)]
        if len(on_blocks) >= 2:  # everyone at once
            await self.tap("a")
            return f"used {what} on everyone"
        if not on_blocks:
            return f"used {what} (no one to pick: it went off at once)"
        who = snap(target, ctsteam.NAMES, 0.6)
        if who not in ctsteam.NAMES:
            raise MenuStop(f"say who {what} is for")
        tip, lines = await self.blocks(
            who, lambda tips: max(tips, key=lambda t: t[0]) if tips else None)
        before = self.hp_near(lines, tip)
        await self.tap("a")
        await asyncio.sleep(0.6)
        _tips, lines = await self.look()
        after = self.hp_near(lines, tip)
        seen = f": {who} HP {before} to {after}" if before and after and before != after else ""
        return f"used {what} on {who}{seen}"

    @staticmethod
    def hp_near(lines, tip) -> str | None:
        for row in lines:
            for r in row:
                if r.text.startswith("HP") and -60 < r.y - tip[1] < 160 \
                        and abs(r.x0 - tip[0]) < 500:
                    m = re.search(r"(\d+)\s*/\s*(\d+)", ctsteam.numfix(r.text))
                    if m:
                        return f"{m.group(1)}/{m.group(2)}"
        return None

    async def _item(self, name: str, target: str) -> str:
        item = snap(name, ctsteam.ITEMS, 0.6)
        await self.to_main("Inventory")
        tips, lines = await self.until(lambda t, ls: bool(t) and bool(
            self.on(ls, t[0], ("Use / Switch", "Sort list"))))
        await self.tap("a")
        await self.until(lambda t, ls: bool(t) and bool(self.on(ls, t[0], ctsteam.ITEMS)))
        way, turned, last = "down", 0, None
        for _ in range(30):
            tips, lines = await self.look()
            if len(tips) >= 2:
                await self.tap("b")  # a swap is pending: cancel it
                raise MenuStop("a swap started in the list; cancelled it")
            if not tips:
                raise MenuStop("the hand went away")
            tip = tips[0]
            cur = self.on(lines, tip, ctsteam.ITEMS, 200)
            if cur == item:
                runs = [r for row in lines for r in row
                        if abs(r.y - tip[1]) < 34 and 0 <= r.x0 - tip[0] < 200]
                if runs and runs[0].colour != "green":
                    raise MenuStop(f"{item} can't be used here (it isn't green)")
                return await self._on_whom(item, target)
            at = [r for row in lines for r in row
                  if snap(r.text, ctsteam.ITEMS, 0.75) == item]
            if at:
                r = at[0]
                if abs(r.y - tip[1]) > 34:
                    await self.tap("down" if r.y > tip[1] else "up")
                else:
                    await self.tap("right" if r.x0 > tip[0] else "left")
                continue
            if last == (cur, tip):
                turned += 1
                if turned >= 2:
                    raise MenuStop(f"no {item} in the inventory")
                way = "up" if way == "down" else "down"
            last = (cur, tip)
            await self.tap(way)
        raise MenuStop(f"couldn't reach {item} in the inventory")

    async def setting(self, name: str, value: str) -> str:
        """Settings > Battle: Battle Mode (Wait, Active), Battle Speed (1 fastest to 8
        slowest), Battle Cursor Memory."""
        row = snap(name if name.lower().startswith("battle") else f"Battle {name}", BATTLE, 0.5)
        if row not in BATTLE:
            return f"stopped: no battle setting called {name} ({', '.join(BATTLE)})"
        try:
            await self.to_main("Settings")
            await self.until(lambda t, ls: bool(t) and bool(self.on(ls, t[0], SETTINGS)))
            await self.walk(SETTINGS, "Battle", lambda tips: min(tips, key=lambda t: t[0]))
            await self.tap("a")
            await self.until(lambda t, ls: any(x > 900 for x, _ in t))
            right = lambda tips: max(tips, key=lambda t: t[0])
            tip, lines = await self.walk(BATTLE, row, right, span=900)
            now = self.value(lines, tip)
            flipped = False
            for _ in range(10):
                step = self.step_toward(row, now, value)
                if step is None:
                    break
                if flipped:
                    step = "left" if step == "right" else "right"
                await self.tap(step)
                _t, lines = await self.look()
                got = self.value(lines, tip)
                if got == now:
                    if flipped or row != "Battle Mode":
                        break  # the end of the range
                    flipped = True  # this way doesn't go round: the other way
                now = got
            await self.close()
            return f"{row} is now {now or 'unreadable'}"
        except MenuStop as exc:
            await self.close()
            return f"stopped: {exc}. The menu is closed."

    @staticmethod
    def value(lines, tip) -> str:
        runs = [r for row in lines for r in row if 40 < r.y - tip[1] < 170 and r.x0 > tip[0]]
        if not runs:
            return ""
        t = min(runs, key=lambda r: r.y).text.upper()
        m = re.match(r"\s*(\d)", ctsteam.numfix(t))
        if m:
            return m.group(1)
        return snap(t, (*MODES, "ACTIONS", "OFF", "ON"), 0.5)

    @staticmethod
    def step_toward(row: str, now: str, want: str) -> str | None:
        w = want.strip().lower()
        if row in ("Battle Speed", "Battle Message Speed"):
            goal = 8 if "slow" in w else 1 if "fast" in w else int(w) if w.isdigit() else None
            if goal is None or not now.isdigit():
                return None
            n = int(now)
            return None if n == goal else ("right" if goal > n else "left")
        if row == "Battle Mode":
            goal = "ACTIVE" if w.startswith("act") else "WAIT" if w.startswith("wait") else ""
            return None if not goal or now == goal else "right"
        return None if same(now, want.upper()) else "right"
