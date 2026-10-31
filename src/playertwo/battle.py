"""Chrono Trigger (Steam) battles, played by a local executor instead of model presses.

The model says what to do once ("Crono, Cyclone on the one on the left"); this does it:
it unpauses, picks the actor's command window, the command, the list entry and the
target, each press checked against the reader (ctsteam.read, about 0.1 s a look), confirms,
watches it play out, and pauses again. Anything unexpected (the hand gone, the entry not
in the list, the actor knocked out, the wrong actor's menu) stops it, paused, with the
reader's state in words.

"Win this fight" is a loop over the same steps: at each turn a standing plan (attack while
everyone is healthy) or, on a surprise, a decision callback (Muse, given the state in
words, while the game is paused) gives the next command.

Live facts (Steam, 2026-11-01): Start pauses a battle and the menus freeze while paused;
Battle Mode is Wait (time stops in the Tech and Item lists only); Battle Cursor Memory is
on, so a list opens on its last pick; 60 ms taps are ignored, 100 ms work; lists are two
columns, three rows in view, and down on the last row's left entry moves right.
What isn't settled live is in TUNING, so it can be flipped without code changes.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from . import ctsteam
from .ctsteam import COMMANDS, SPOILS, State


@dataclass
class Tuning:
    tap_ms: int = 100  # Steam ignores 60 ms taps
    after_tap_s: float = 0.1  # between taps (live: 100 to 150 ms after a 100 ms hold)
    after_start_s: float = 0.05  # extra after the unpausing Start, before the next tap
    poll_s: float = 0.04  # between looks while waiting (a look itself takes ~0.1 s)
    use_pause: bool = True  # pause with Start while the model thinks
    pause_in: tuple[str, ...] = ("command", "list", "target", "none")  # where Start pauses
    hold_paused: bool = True  # leave the fight paused after a command, until the next ask
    pause_wait_s: float = 1.2  # how long a pause or unpause may take to show
    ready_wait_s: float = 8.0  # for anyone's command window
    actor_wait_s: float = 6.0  # for the named actor's window
    next_actor: str = "right"  # moves the hand to the next ready actor's window
    prev_actor: str = "left"
    target_wait_s: float = 1.2  # for the glove after choosing a command or entry
    playout_min_s: float = 1.0
    playout_quiet_s: float = 0.7  # party HP and the top bar unchanged this long: done
    playout_max_s: float = 3.5
    run_chunk_ms: int = 2500  # L and R held this long between looks
    run_chunks: int = 3
    spoils_wait_s: float = 1.2  # a won fight's message left this long gets an a
    win_max_s: float = 240.0
    low_share: float = 0.34  # under this share of most HP, someone needs healing


TUNING = Tuning()
# A callback (a model call) failing this way means "no answer", not a crash.
CALLBACK_ERRORS = (ValueError, TypeError, KeyError, OSError, RuntimeError, TimeoutError)

# Healing and reviving the standing plan knows, by who casts them (Steam names), and MP.
HEAL_TECHS = {"Marle": [("Cure", 2), ("Aura", 1)], "Frog": [("Slurp", 1)],
              "Robo": [("Cure Beam", 2)], "Ayla": [("Kiss", 1)], "Lucca": []}
HEAL_ALL_TECHS = {"Frog": ("Heal", 2), "Robo": ("Heal Beam", 3), "Marle": ("Cure II", 5)}
REVIVE_TECHS = {"Crono": ("Raise", 10)}
HEAL_ITEMS = ("Mid-Potion", "Hi-Potion", "Potion")
REVIVE_ITEMS = ("Athenian Water", "Revive")


@dataclass
class Intent:
    actor: str = ""  # "" is whoever's command is up
    command: str = "attack"  # attack, tech or item
    name: str = ""  # the tech or item
    target: str = ""  # an ally's name, "left"/"right"/"top"/"bottom", an enemy's name, ""

    def __str__(self) -> str:
        what = self.command if not self.name else f"{self.command} {self.name}"
        return (f"{self.actor or 'whoever is up'}: {what}"
                + (f" on {self.target}" if self.target else ""))


@dataclass
class Outcome:
    ok: bool
    text: str
    state: State | None = None
    stopped: str = ""  # why it stopped, when it did
    presses: list[str] = field(default_factory=list)


class Stop(Exception):
    """Something unexpected: stop, pause, and tell the model."""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def same_name(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85)


def find_entry(entries: list[list[str]], name: str) -> tuple[int, int] | None:
    for r, row in enumerate(entries):
        for c, e in enumerate(row):
            if not e.startswith("[") and same_name(e, name):
                return r, c
    return None


SIDES = {"left": ("x", -1, "left"), "right": ("x", 1, "right"),
         "top": ("y", -1, "up"), "back": ("y", -1, "up"), "far": ("y", -1, "up"),
         "bottom": ("y", 1, "down"), "front": ("y", 1, "down"), "near": ("y", 1, "down")}
ANY_TARGET = {"", "default", "any", "anyone", "all", "everyone", "enemy", "the enemy",
              "enemies", "all enemies", "the party", "party", "allies", "all allies",
              "nearest", "closest", "whoever"}


def victory(st: State) -> bool:
    return bool(st.top) and bool(SPOILS.search(st.top))


def merge_rows(order: list[tuple[str, ...]], view: list[list[str]]) -> list[tuple[str, ...]]:
    """A list's rows seen so far, in order, with a new view of up to three rows merged in."""
    rows = [tuple(r) for r in view if r]
    if not rows:
        return order
    if not order:
        return rows
    for k in range(len(rows), 0, -1):
        if order[-k:] == rows[:k]:  # the view goes on past the end
            return order + rows[k:]
        if rows[-k:] == order[:k]:  # it starts before the beginning
            return rows[:-k] + order
    if all(r in order for r in rows):
        return order
    return order + [r for r in rows if r not in order]


class Executor:
    """One fight's executor. `press(button, ms)` and `grab()` are async; `grab` returns a
    frame (BGR array or PIL image) for `read`."""

    def __init__(self, press: Callable[[str, int], Awaitable[None]],
                 grab: Callable[[], Awaitable[object]],
                 read: Callable[[object], State] = ctsteam.read,
                 tuning: Tuning = TUNING,
                 identify: Callable[[object, tuple[int, int], list[str]],
                                    Awaitable[str | None]] | None = None,
                 cancelled: Callable[[], bool] = lambda: False,
                 chord: Callable[[list[str], int], Awaitable[None]] | None = None,
                 log: Callable[[str], None] | None = None):
        self._press = press
        self._grab = grab
        self._read = read
        self.t = tuning
        self.identify = identify
        self.cancelled = cancelled
        self._chord = chord
        self._log = log
        self.presses: list[str] = []
        self.frame: object = None  # the last frame looked at
        self.ally_at: dict[str, tuple[int, int]] = {}  # bracket centres, by name
        self.lists: dict[str, list[tuple[str, ...]]] = {}  # rows seen, per list
        self.notes: list[str] = []
        self.held: State | None = None  # the screen as it was when paused
        self._unpause_first = False

    # -- seeing and pressing ---------------------------------------------------------------

    def log(self, text: str) -> None:
        self.notes.append(text)
        if self._log:
            self._log(text)

    async def look(self, full: bool = True) -> State:
        self.frame = await self._grab()
        st = await asyncio.to_thread(self._read, self.frame, full)
        if st.kind == "command" and st.actor and st.bracket:
            self.ally_at[st.actor] = st.bracket
        return st

    async def tap(self, button: str, times: int = 1, force: bool = False) -> None:
        for _ in range(times):
            if self.cancelled() and not force:
                raise Stop("stopped by the player")
            await self._press(button, self.t.tap_ms)
            self.presses.append(button)
            await asyncio.sleep(self.t.after_tap_s)

    async def until(self, ok: Callable[[State], bool], secs: float,
                    st: State | None = None) -> State:
        end = time.monotonic() + secs
        st = st or await self.look()
        while not ok(st) and time.monotonic() < end:
            if self.cancelled():
                raise Stop("stopped by the player")
            await asyncio.sleep(self.t.poll_s)
            st = await self.look()
        return st

    async def pause(self, st: State | None = None) -> State:
        st = st or await self.look()
        if not self.t.use_pause or st.paused or not st.in_battle or victory(st) \
                or st.kind not in self.t.pause_in:
            return st
        # Menus freeze while paused, so this read stays true until the unpause: the first
        # presses after it can be planned from it (a paused screen reads badly).
        held = st
        await self.tap("start", force=True)  # a stop from the player pauses too
        end = time.monotonic() + self.t.pause_wait_s
        while not st.paused and time.monotonic() < end:
            await asyncio.sleep(self.t.poll_s)
            st = await self.look()
        if st.paused:
            self.held = held
        else:
            self.log("start didn't pause")
        return st

    async def unpause(self, st: State | None = None) -> State:
        st = st or await self.look()
        if not st.paused:
            return st
        self.held = None
        await self.tap("start")
        st = await self.until(lambda s: not s.paused, self.t.pause_wait_s)
        if st.paused:
            await self.tap("start")  # a dropped tap
            st = await self.until(lambda s: not s.paused, self.t.pause_wait_s)
        if st.paused:
            raise Stop("the game stayed paused after Start")
        return st

    async def burst(self, buttons: list[str]) -> None:
        """Presses with no look between; Start first when the game is paused on a known
        screen. A Start that doesn't take drops the rest too (the game drops presses
        while paused), so a burst never lands half-planned on a frozen screen."""
        if self._unpause_first:
            self._unpause_first = False
            self.held = None
            buttons = ["start", *buttons]
        for b in buttons:
            await self.tap(b)
            if b == "start":
                await asyncio.sleep(self.t.after_start_s)

    async def _fresh(self) -> State:
        """A look after a burst; if the game is still paused (Start was dropped), unpause
        the slow way: nothing else landed."""
        st = await self.look()
        if st.paused:
            st = await self.unpause(st)
        return st

    async def back_out(self, st: State) -> State:
        """Back to the command windows from a list or the target glove."""
        for _ in range(3):
            if st.kind not in ("list", "target"):
                return st
            await self.tap("b")
            st = await self.until(lambda s: s.kind not in ("list", "target"), 0.6)
        return st

    # -- one command -----------------------------------------------------------------------

    async def act(self, intent: Intent, watch: bool = True, pause_after: bool = True,
                  st: State | None = None) -> Outcome:
        """Do one command. Never raises: a stop comes back as Outcome(ok=False)."""
        start = len(self.presses)
        before: State | None = None
        self._unpause_first = False
        try:
            st = st or await self.look()
            if st.paused and self.held is not None and self.held.kind == "command":
                # Plan from what was there when it paused; Start goes with the first burst,
                # so the fight runs only while the presses do.
                st, self._unpause_first = self.held, True
            else:
                st = await self.unpause(st)
            if not st.in_battle:
                raise Stop("not in a battle: no party health on screen")
            if victory(st):
                raise Stop("the fight is won already")
            st = await self.back_out(st)
            st = await self.until(lambda s: s.kind == "command", self.t.ready_wait_s, st)
            if st.kind != "command":
                raise Stop("no one's command came up")
            before = st
            st = await self._pick_actor(st, intent.actor)
            before = st
            actor = st.actor
            st = await self._pick_command(st, intent.command)
            if intent.command.lower() in ("tech", "item"):
                st = await self._pick_entry(st, intent, actor)
            st = await self._pick_target(st, intent)
            st = await self._confirm(st)
            text = str(Intent(actor, intent.command, intent.name, intent.target))
            if watch:
                st, seen = await self._watch(before, intent)
                text += ". " + seen
            if pause_after and self.t.hold_paused:
                st = await self._pause_after(st)
            return Outcome(True, text, st, presses=self.presses[start:])
        except Stop as exc:
            why = str(exc)
            self.log(f"stopped: {why}")
            st = await self._settle_stopped(pause_after)
            return Outcome(False, why, st, stopped=why, presses=self.presses[start:])
        finally:
            self._unpause_first = False

    async def _settle_stopped(self, pause: bool = True) -> State | None:
        if self.cancelled():
            # The player said stop: no more presses than it takes to pause.
            try:
                return await self.pause()
            except Stop:
                return None
        try:
            st = await self.look()
            if st.kind == "target":
                st = await self.back_out(st)
            return await self.pause(st) if pause else st
        except Stop:
            return None

    async def _pause_after(self, st: State) -> State:
        try:
            return await self.pause(st)
        except Stop:
            return st

    async def _pick_actor(self, st: State, actor: str) -> State:
        if not actor:
            return st
        slot = next((i for i, m in enumerate(st.party) if same_name(m.name, actor)), None)
        if slot is None:
            names = ", ".join(m.name for m in st.party)
            raise Stop(f"no one called {actor} in the party ({names})")
        if st.party[slot].down:
            raise Stop(f"{st.party[slot].name} is knocked out")
        name = st.party[slot].name
        if slot not in st.columns:
            if self._unpause_first:
                await self.burst([])  # their turn needs the fight to run
                st = await self._fresh()
            st = await self.until(lambda s: s.kind != "command" or slot in s.columns
                                  or (slot < len(s.party) and s.party[slot].down),
                                  self.t.actor_wait_s, st)
            if st.kind != "command" or slot not in st.columns:
                ready = ", ".join(st.ready()) or "no one"
                raise Stop(f"{name} isn't ready for a command yet (ready: {ready})")
        for _ in range(3):
            if st.actor_col == slot:
                return st
            if st.actor_col is None or st.actor_col not in st.columns:
                raise Stop("the hand isn't on a command window")
            d = st.columns.index(slot) - st.columns.index(st.actor_col)
            await self.burst([self.t.next_actor if d > 0 else self.t.prev_actor] * abs(d))
            st = await self._fresh()
            if st.kind != "command":
                raise Stop("the command windows went away")
        if st.actor_col != slot:
            raise Stop(f"couldn't move the hand to {name}'s commands")
        return st

    async def _pick_command(self, st: State, command: str) -> State:
        """Rows and the confirm in one burst, checked once after: three quick taps rather
        than three looks while the fight runs."""
        want = {"attack": "Attack", "tech": "Tech", "techs": "Tech", "combo": "Tech",
                "item": "Item", "items": "Item"}.get(command.lower().strip())
        if want is None:
            raise Stop(f"{command!r} isn't a command; use attack, tech or item")
        goal = "target" if want == "Attack" else "list"
        actor_col = st.actor_col
        for _ in range(3):
            if st.kind != "command" or st.actor_col != actor_col:
                raise Stop("the command windows changed under the hand")
            if st.command not in COMMANDS:
                raise Stop("can't read which command the hand is on")
            d = COMMANDS.index(want) - COMMANDS.index(st.command)
            await self.burst(["down" if d > 0 else "up"] * abs(d) + ["a"])
            st = await self.until(lambda s: s.kind == goal or s.paused, self.t.target_wait_s)
            if st.paused:
                st = await self.unpause(st)
            if st.kind == goal:
                return st
            if st.kind in ("list", "target"):  # the wrong one opened
                st = await self.back_out(st)
        raise Stop(f"couldn't open {want}")

    async def _pick_entry(self, st: State, intent: Intent, actor: str) -> State:
        if st.kind != "list":
            raise Stop(f"the {intent.command} list didn't open")
        name = intent.name.strip()
        if not name:
            raise Stop(f"say which {intent.command.lower()} to use")
        key = "item" if intent.command.lower().startswith("item") else f"tech:{actor}"
        order = self.lists.get(key, [])
        way = "down"
        hit = next((i for i, r in enumerate(order) for e in r if same_name(e, name)), None)
        first = next((tuple(r) for r in st.entries if r), None)
        if hit is not None and first:
            here = next((i for i, r in enumerate(order) if r == first), None)
            if here is not None and hit < here:
                way = "up"
        turned = 0
        for _ in range(48):
            order = merge_rows(order, st.entries)
            self.lists[key] = order
            if st.kind != "list":
                raise Stop("the list closed")
            if st.entry and same_name(st.entry, name):
                if st.usable is False:
                    st = await self.back_out(st)
                    raise Stop(f"{name} is greyed out: not enough MP, or it can't be used now")
                await self.tap("a")
                st = await self.until(lambda s: s.kind != "list", self.t.target_wait_s)
                if st.kind == "list":
                    st = await self.back_out(st)
                    raise Stop(f"picking {name} did nothing: maybe no one it can be used "
                               "on (a revive with no one knocked out)")
                return st
            at = find_entry(st.entries, name)
            if st.entry_rc is None:
                raise Stop("can't see the hand in the list")
            hr, hc = st.entry_rc
            if at is not None:
                r, c = at
                if r != hr:
                    await self.tap("down" if r > hr else "up", abs(r - hr))
                elif hc is not None and c != hc:
                    await self.tap("right" if c > hc else "left")
                else:
                    raise Stop(f"the hand won't go to {name}")
                st = await self.look()
                continue
            view = (st.entries, st.entry)
            await self.tap(way)
            st = await self.look()
            if (st.entries, st.entry) == view:  # the end of the list that way
                turned += 1
                if turned >= 2:
                    seen = ", ".join(e for r in order for e in r if not e.startswith("["))
                    st = await self.back_out(st)
                    raise Stop(f"{name} isn't in the {intent.command.lower()} list "
                               f"(it has {seen})")
                way = "up" if way == "down" else "down"
        raise Stop(f"couldn't reach {name} in the list")

    async def who_at(self, st: State) -> str | None:
        """Which ally the glove is on: by where each was bracketed on their turn, by
        elimination, or by asking `identify`."""
        if not st.gloves:
            return None
        gx, gy = st.gloves[0]
        names = [m.name for m in st.party] or list(self.ally_at)
        near = sorted((abs(gx - (bx - 8)) + abs(gy - (by + 60)), n)
                      for n, (bx, by) in self.ally_at.items())
        if near and near[0][0] < 170:
            return near[0][1]
        unknown = [n for n in names if n not in self.ally_at]
        if len(unknown) == 1 and len(names) > 1:
            return unknown[0]
        if self.identify is not None:
            try:
                got = await self.identify(self.frame, st.gloves[0], names)
            except CALLBACK_ERRORS as exc:  # any failure is "don't know"
                self.log(f"identify failed: {exc}")
                return None
            who = next((n for n in names if got and same_name(n, got)), None)
            if who:
                self.ally_at[who] = (gx + 8, gy - 60)  # known from now on
            return who
        return None

    async def _pick_target(self, st: State, intent: Intent) -> State:
        if st.kind != "target":
            st = await self.until(lambda s: s.kind == "target", self.t.target_wait_s, st)
        if st.kind != "target":
            if st.kind in ("list",):
                raise Stop("the list is still open: the pick didn't take")
            return st  # no target to choose (it went straight off)
        if len(st.gloves) > 1:
            return st  # it hits everyone on that side
        want = _norm(intent.target)
        want = re.sub(r"^(the|on|at) ", "", want)
        if want in ANY_TARGET:
            return st
        names = [m.name for m in st.party]
        ally = next((n for n in names if _norm(n) in want.split() or same_name(n, want)),
                    None)
        if want in ("me", "self", "myself", "himself", "herself", "themselves"):
            ally = intent.actor or None
        if ally:
            return await self._target_ally(st, ally)
        side = next((SIDES[w] for w in want.split() if w in SIDES), None)
        if side:
            return await self._target_side(st, side)
        return await self._target_named(st, intent.target)

    async def _target_ally(self, st: State, ally: str) -> State:
        member = st.member(ally) if st.party else None
        who = await self.who_at(st)
        if who and same_name(who, ally):
            return st
        if who is None and member is not None and member.down \
                and sum(m.down for m in st.party) == 1:
            return st  # the glove opens on the one knocked out
        tried = []
        for button in self._ally_moves(st, ally):
            await self.tap(button)
            tried.append(button)
            st = await self.look()
            if st.kind != "target":
                raise Stop("the target glove went away")
            who = await self.who_at(st)
            if who and same_name(who, ally):
                return st
        raise Stop(f"couldn't put the glove on {ally} (tried {' '.join(tried)})")

    def _ally_moves(self, st: State, ally: str) -> list[str]:
        known = next((p for n, p in self.ally_at.items() if same_name(n, ally)), None)
        if known and st.gloves:
            dx, dy = known[0] - st.gloves[0][0], known[1] + 60 - st.gloves[0][1]
            first = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else \
                ("down" if dy > 0 else "up")
            return [first, first, "right", "down", "left", "up", "right", "down"]
        return ["right", "right", "down", "left", "left", "up", "right", "down"]

    async def _target_side(self, st: State, side: tuple[str, int, str]) -> State:
        axis, sign, button = side
        k = 0 if axis == "x" else 1
        back = {"left": "right", "right": "left", "up": "down", "down": "up"}[button]
        best = st.gloves[0][k]
        for _ in range(6):
            await self.tap(button)
            nxt = await self.look()
            if nxt.kind != "target" or not nxt.gloves:
                raise Stop("the target glove went away")
            v = nxt.gloves[0][k]
            if (v - best) * sign > 12:
                best, st = v, nxt
                continue
            if abs(v - best) > 12:  # it wrapped round: one back
                await self.tap(back)
                st = await self.look()
            else:
                st = nxt
            break
        return st

    async def _target_named(self, st: State, name: str) -> State:
        for _ in range(8):
            top = st.top.strip()
            if top and (same_name(top, name) or _norm(name) in _norm(top)
                        or _norm(top) in _norm(name)):
                return st
            await self.tap("right")
            st = await self.look()
            if st.kind != "target":
                raise Stop("the target glove went away")
        raise Stop(f"no enemy called {name} came up under the glove (the bar says "
                   f"{st.top or 'nothing'})")

    async def _confirm(self, st: State) -> State:
        if st.kind != "target":
            return st
        await self.tap("a")
        st = await self.until(lambda s: s.kind != "target", 1.0)
        if st.kind == "target":
            raise Stop("the confirm didn't take: the glove is still up")
        return st

    async def _watch(self, before: State | None, intent: Intent) -> tuple[State, str]:
        """Look until the command has played out; what changed, in words."""
        t0 = time.monotonic()
        st = await self.look()
        msgs: list[str] = []
        last_change = t0
        hp = _hp(st) or (_hp(before) if before else {})
        target = next((m.name for m in (before.party if before else [])
                       if _norm(m.name) in _norm(intent.target).split()), None)
        while time.monotonic() - t0 < self.t.playout_max_s:
            if st.top and st.top not in msgs and st.kind != "target":
                msgs.append(st.top)
                last_change = time.monotonic()
            now = _hp(st)
            if now and now != hp:
                if target and before and now.get(target) != _hp(before).get(target):
                    hp = now
                    break  # the heal or revive landed
                hp, last_change = now, time.monotonic()
            if victory(st):
                break
            el = time.monotonic() - t0
            if el >= self.t.playout_min_s and time.monotonic() - last_change \
                    >= self.t.playout_quiet_s and not st.top:
                break
            await asyncio.sleep(self.t.poll_s)
            st = await self.look()
        return st, changes(before, st, msgs)

    # -- whole fights ----------------------------------------------------------------------

    async def run_away(self) -> Outcome:
        start = len(self.presses)
        try:
            st = await self.unpause()
            if not st.in_battle:
                return Outcome(False, "not in a battle", st, "not in a battle")
            st = await self.back_out(st)
            for i in range(self.t.run_chunks):
                if self.cancelled():
                    raise Stop("stopped by the player")
                if self._chord is None:
                    raise Stop("can't hold two buttons at once here")
                await self._chord(["lb", "rb"], self.t.run_chunk_ms)
                self.presses.append(f"lb+rb {self.t.run_chunk_ms} ms")
                st = await self.look()
                if not st.in_battle:
                    st2 = await self.until(lambda s: s.in_battle, 0.5)
                    if not st2.in_battle:
                        return Outcome(True, f"got away (held L and R {i + 1} times)", st2,
                                       presses=self.presses[start:])
            st = await self._pause_after(st)
            return Outcome(False, "couldn't run: some fights can't be run from", st,
                           "can't run", self.presses[start:])
        except Stop as exc:
            return Outcome(False, str(exc), await self._settle_stopped(), str(exc),
                           self.presses[start:])

    async def spoils(self, st: State) -> tuple[State, list[str]]:
        """A won fight's messages, each moved on with a when it waits for one."""
        said: list[str] = []
        for _ in range(10):
            if not victory(st):
                break
            said.append(st.top)
            shown = st.top
            st = await self.until(lambda s: s.top != shown, self.t.spoils_wait_s, st)
            if st.top == shown:
                await self.tap("a")
                st = await self.until(lambda s: s.top != shown, self.t.spoils_wait_s)
                if st.top == shown:
                    break  # a doesn't move it on: leave it
        return st, said

    async def win(self, decide: Callable[[State, str], Awaitable[Intent | None]] | None = None,
                  say: Callable[[str], Awaitable[None]] | None = None) -> Outcome:
        """Fight until it's won, lost or the player stops it."""
        start = len(self.presses)
        t0 = time.monotonic()
        turns: list[str] = []
        gone = 0.0
        try:
            st = await self.unpause()
            if not st.in_battle:
                return Outcome(False, "not in a battle", st, "not in a battle")
            while time.monotonic() - t0 < self.t.win_max_s:
                if self.cancelled():
                    raise Stop("stopped by the player")
                if victory(st):
                    st, said = await self.spoils(st)
                    return Outcome(True, f"won after {len(turns)} commands. "
                                   + " ".join(said), st, presses=self.presses[start:])
                if not st.in_battle:
                    gone = gone or time.monotonic()
                    if time.monotonic() - gone > 1.5:
                        return Outcome(True, f"the fight ended after {len(turns)} commands",
                                       st, presses=self.presses[start:])
                else:
                    gone = 0.0
                if st.party and all(m.down for m in st.party):
                    return Outcome(False, "everyone is knocked out", st, "party down",
                                   self.presses[start:])
                if st.kind in ("list", "target"):
                    st = await self.back_out(st)
                    continue
                if st.kind == "command" and st.actor:
                    intent = plan(st, self.t)
                    why = ""
                    if intent is None and decide is not None:
                        why = surprise(st, self.t)
                        seen = st
                        await self.pause(st)
                        try:
                            intent = await decide(seen, why)
                        except CALLBACK_ERRORS as exc:  # fall back to the plan
                            self.log(f"decide failed: {exc}")
                            intent = None
                        st = await self.look()  # paused: act plans from what it held
                        if not st.paused and (st.kind != "command" or not st.actor):
                            continue
                    else:
                        seen = st
                    missed: set[str] = set()  # targets the glove couldn't reach
                    for option in [intent, *fallbacks(seen, self.t)]:
                        if option is None or (option.target and option.target in missed):
                            continue
                        option.actor = seen.actor
                        out = await self.act(option, watch=False, pause_after=False, st=st)
                        st = out.state or await self.look()
                        if out.ok:
                            turns.append(str(option))
                            break
                        self.log(f"{option} didn't go: {out.stopped}")
                        if "glove" in out.stopped:
                            missed.add(option.target)
                        if self.cancelled() or "player" in out.stopped:
                            raise Stop("stopped by the player")
                        st = await self.unpause(st) if st else await self.look()
                        if st.kind != "command":
                            break
                    continue
                await asyncio.sleep(self.t.poll_s)
                st = await self.look()
            st = await self._pause_after(st)
            return Outcome(False, f"still fighting after {round(self.t.win_max_s)} s", st,
                           "too long", self.presses[start:])
        except Stop as exc:
            return Outcome(False, f"{exc} after {len(turns)} commands",
                           await self._settle_stopped(), str(exc), self.presses[start:])


def _hp(st: State | None) -> dict[str, int | None]:
    return {m.name: m.hp for m in st.party} if st and st.party else {}


def changes(before: State | None, after: State, msgs: list[str]) -> str:
    """What the screen showed changing, in words: only what was seen."""
    out = []
    b, a = _hp(before), _hp(after)
    for name, hp in a.items():
        old = b.get(name)
        if old is not None and hp is not None and old != hp:
            word = "revived" if old == 0 else ("knocked out" if hp == 0 else "")
            out.append(f"{name} HP {old} to {hp}" + (f" ({word})" if word else ""))
    if msgs:
        out.append("the bar said: " + "; ".join(msgs))
    if victory(after):
        out.append("the fight is won")
    return ("Seen: " + ", ".join(out)) if out else "No party HP changed and no message showed"


def plan(st: State, t: Tuning = TUNING) -> Intent | None:
    """The standing plan: attack while no one is knocked out or low. None is a surprise."""
    if surprise(st, t):
        return None
    return Intent(st.actor, "attack")


def surprise(st: State, t: Tuning = TUNING) -> str:
    down = [m.name for m in st.party if m.down]
    low = [m.name for m in st.party if not m.down and m.share < t.low_share]
    if down:
        return f"{', '.join(down)} knocked out"
    if low:
        return f"{', '.join(low)} low on HP"
    return ""


def fallbacks(st: State, t: Tuning = TUNING) -> list[Intent]:
    """What to try, in order, when there's a surprise and no decision (or it failed)."""
    actor = st.member(st.actor)
    mp = (actor.mp or 0) if actor else 0
    down = [m for m in st.party if m.down]
    low = sorted((m for m in st.party if not m.down and m.share < t.low_share),
                 key=lambda m: m.share)
    out: list[Intent] = []
    if down:
        who = down[0].name
        tech = REVIVE_TECHS.get(st.actor)
        if tech and mp >= tech[1]:
            out.append(Intent(st.actor, "tech", tech[0], who))
        out += [Intent(st.actor, "item", it, who) for it in REVIVE_ITEMS]
    if low:
        who = low[0].name
        group = HEAL_ALL_TECHS.get(st.actor)
        if group and len(low) >= 2 and mp >= group[1]:
            out.append(Intent(st.actor, "tech", group[0], ""))
        out += [Intent(st.actor, "tech", n, who) for n, cost in HEAL_TECHS.get(st.actor, [])
                if mp >= cost]
        out += [Intent(st.actor, "item", it, who) for it in HEAL_ITEMS]
    out.append(Intent(st.actor, "attack"))
    return out


def parse_intent(d: dict, actor: str = "") -> Intent | None:
    """An intent from a model's JSON ({"command", "name", "target"})."""
    if not isinstance(d, dict):
        return None
    cmd = str(d.get("command") or "").lower().strip()
    if cmd not in ("attack", "tech", "item"):
        return None
    return Intent(str(d.get("actor") or actor), cmd, str(d.get("name") or ""),
                  str(d.get("target") or ""))
