"""Voice commands: look at the screen, press buttons, look again, until it's done."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
import openai
from PIL import Image

from .areamap import AreaMap
from .controls import BUTTONS, DIRECTIONS, Controls
from .explore import ANGLES, Trail, find_me, keep_walking, words
from .muse import Muse, image_part
from .notes import GameNotes
from .prompts import AGENT, CONFIRM, GAME_CONTEXT, LANGUAGE
from .screen import Frame, thumb
from .walk import Walker, small

# Never the guide button: it opens the Steam overlay or the system menu, not the game.
AGENT_BUTTONS = [b for b in BUTTONS if b != "guide"]

TOOLS = [
    {"type": "function", "function": {
        "name": "press",
        "description": "Press controller buttons in order. d-pad is up/down/left/right.",
        "parameters": {"type": "object", "properties": {
            "buttons": {"type": "array", "items": {"type": "string", "enum": AGENT_BUTTONS},
                        "description": "Buttons to press one after another."},
            "gap_ms": {"type": "integer", "description": "Pause between presses, default 150."},
            "say": {"type": "string", "description": "Said out loud before pressing: the "
                    "dialogue line on screen (who, then the line word for word) when a "
                    "moves it on, so it's one step."},
            "skipping": {"type": "boolean", "description": "true only when the player asked "
                         "to skip or fast-forward text or a cutscene: then a can be pressed "
                         "several times. Otherwise any a after another press waits for you to "
                         "look first."},
        }, "required": ["buttons"]}}},
    {"type": "function", "function": {
        "name": "hold",
        "description": "Hold a button or d-pad direction for a while (2D games walk with "
                       "the d-pad), or push a stick in a direction (3D games walk with the "
                       "left stick).",
        "parameters": {"type": "object", "properties": {
            "input": {"type": "string", "enum": ["left_stick", "right_stick", *AGENT_BUTTONS]},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "ms": {"type": "integer", "description": "How long, in milliseconds (max 5000)."},
        }, "required": ["input", "ms"]}}},
    {"type": "function", "function": {
        "name": "walk_to",
        "description": "Walk the player's character to something you can see on screen: "
                       "a door, stairs, a person, a chest, an exit. For games seen from "
                       "above (2D, top-down or angled); it steers by itself, goes around "
                       "what's in the way, and stops next to it, facing it. To talk to "
                       "it, open or search it, use talk_to instead. Not for 3D games with the camera behind "
                       "the character.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "What to walk to, described so "
                       "it can be found on this screen: \"the woman in the green dress\", "
                       "\"the stairs at the top left\"."},
            "me": {"type": "string", "description": "Which character on screen is the "
                   "player's, so it isn't mixed up with others: \"the boy with spiky red "
                   "hair\"."},
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "talk_to",
        "description": "Walk to a person (or a sign, a chest, a door) you can see on "
                       "screen and press a when there, in games seen from above: walk_to "
                       "and the press in one. Look at the screen after to read what they "
                       "say.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Who or what, described so it "
                       "can be found on this screen."},
            "me": {"type": "string", "description": "Which character is the player's, "
                   "if it's unclear."},
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "keep_walking",
        "description": "Walk on in one direction toward somewhere not on screen yet, "
                       "in games seen from above. It watches as it goes and stops at a "
                       "wall, at a new area (a door, an exit) or when something comes up, "
                       "then tells you how far it went. Better than holding the d-pad "
                       "blind.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": list(ANGLES)},
            "seconds": {"type": "number", "description": "At most how long, 1 to 8; "
                        "default 3."},
        }, "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "go_to_place",
        "description": "On a world map, go to a named place with the game's atlas (its "
                       "maps, if the player has them): it knows exactly where you are and "
                       "where the place is, walks the way there around water and forest, "
                       "and goes in. Use it first for any 'take me to X' on a world map; "
                       "with no atlas or no such place it says so and lists the places.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string", "description": "The place's name, as the "
                      "player said it."},
        }, "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "guide",
        "description": "Look something up in this game's walkthrough, if the player has "
                       "one: where a place is and how to get there, what to do next, how a "
                       "puzzle works. Search with names of places, people and things "
                       "('Guardia Forest castle', 'Zenan Bridge'). It follows the story in "
                       "order, so check which part fits where you are.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "travel",
        "description": "Cross a big map (a world map, a large field) toward a place that "
                       "isn't on screen yet, in games seen from above: it heads that way, "
                       "slides around coasts, trees and walls in the way, and looks every "
                       "few seconds for the place. It stops when the place comes into "
                       "sight (then walk_to it), at a new area, or when that way is shut. "
                       "One call goes far; better than many keep_walking.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": list(ANGLES)},
            "target": {"type": "string", "description": "What the place looks like on "
                       "the map, to know it when it comes into sight ('a grey stone "
                       "castle with towers')."},
            "seconds": {"type": "number", "description": "At most how long, 10 to 90; "
                        "default 45."},
        }, "required": ["direction", "target"]}}},
    {"type": "function", "function": {
        "name": "exits",
        "description": "List the ways out of the place on screen (doors, stairs, paths off "
                       "the edge, gaps in fences), nearest first, how far each is from your "
                       "character, and which one you probably came in by (2D games). Use it "
                       "to look for something that isn't on screen.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "area_map",
        "description": "See a map of this area so far, stitched from everything on "
                       "screen while walking (2D games): exits, gaps, paths and people "
                       "that have scrolled off. You get it with the next screen.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "walk_to_map",
        "description": "Walk to a spot on the area map you were shown last, even if it's "
                       "off screen now: x and y on that map image, 0 to 1000 each way "
                       "(0, 0 the top left). Stops next to it, at a wall or at a new area.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "label": {"type": "string", "description": "What's there, in a few words."},
        }, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Wait without pressing anything: loading screens, fades, movies, "
                       "animations. You get a new screenshot after.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "number", "description": "How long, 0.5 to 8."},
        }, "required": ["seconds"]}}},
    {"type": "function", "function": {
        "name": "say",
        "description": "Say something short to the player out loud.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "ask_player",
        "description": "Ask the player a short question out loud and wait for their spoken "
                       "answer. Use it to confirm a plan or clear up what they meant.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Keep a short fact about this game for later sessions, once you've "
                       "seen it hold: which button opens the menu or skips text, how "
                       "battles work, where a place or person is and how to get there, "
                       "what the player's character looks like. Not things that change "
                       "(health, what's on screen now).",
        "parameters": {"type": "object", "properties": {
            "note": {"type": "string"},
            "instead_of": {"type": "string", "description": "An earlier note this one "
                           "corrects, as it was written, so the wrong one goes."},
        }, "required": ["note"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "The request is done (or can't be done). Tell the player in one "
                       "short sentence.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
]


FINISH_ONLY = [t for t in TOOLS if t["function"]["name"] == "finish"]
# Out of steps but not done: the player can say to keep going (that adds steps).
FINISH_OR_ASK = [t for t in TOOLS if t["function"]["name"] in ("finish", "ask_player")]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
MOVES = {"up", "down", "left", "right", "lb", "rb"}
# Something broke while walking (not the model's arguments): the model hears it failed.
WALK_ERRORS = (ValueError, TypeError, IndexError, OSError, RuntimeError, openai.APIError)
WALK_FAILED = ("blocked", "timeout", "lost", "not_found", "walking failed")
# A text box or a cutscene in the way: not the way being cut off, nor a walk that got there.
WALK_PAUSED = ("didnt_move", "interrupted")


def screen_change(a: Frame, b: Frame, cell: float = 0.06) -> float:
    """Fraction of the screen that differs between two frames (small thumbnails)."""
    return float((abs(thumb(a.image) - thumb(b.image)) > cell).mean())


def skip_grid(f: Frame) -> np.ndarray:
    """A grey grid fine enough to tell one line of text from the next."""
    img = f.image.convert("L").resize((192, 108), Image.Resampling.BILINEAR)
    return np.asarray(img, np.float32) / 255


def skip_on(text: tuple | None, before: np.ndarray, after: np.ndarray) -> tuple[tuple | None, str]:
    """Whether a skip through text should go on after a press (grids from skip_grid).

    Lines of text change the same box on screen each time. When a press changes
    somewhere outside it (a menu, a Yes/No, a battle's commands) or the whole screen,
    the text is over and another a would pick something, so the skip stops there.
    Returns the box the text has used so far (top, bottom, left, right) and why to stop
    ("" to go on).
    """
    moved = abs(after - before) > 0.06
    if moved.mean() > 0.35:
        return text, "the whole screen changed"
    ys, xs = np.nonzero(moved)
    if not len(ys):
        return text, ""
    if text is not None:
        y0, y1, x0, x1 = text
        pad = 4
        inside = np.zeros_like(moved)
        inside[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1] = True
        if (moved & ~inside).mean() > 0.004:
            return text, "something new showed up outside the text"
        ys, xs = np.r_[ys, y0, y1], np.r_[xs, x0, x1]
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())), ""


def way(angle: float | None) -> str:
    """The d-pad direction nearest a stick angle (0 up, 90 right), or ""."""
    if angle is None:
        return ""
    return ("up", "right", "down", "left")[round(angle / 90) % 4]


def split_before_confirm(buttons: list[str]) -> tuple[list[str], str]:
    """Cut a press list just before an `a` that follows any other press.

    Pressing and confirming blind is where wrong picks come from (a menu opened with
    start, a Yes/No that defaults to Yes, a second `a` that skips a line of dialogue the
    player never heard), so the agent looks at the screen before every confirm. An `a`
    first, or on its own, is fine.
    """
    for i, b in enumerate(buttons):
        if b == "a" and i > 0:
            return buttons[:i], "a"
    return buttons, ""


@dataclass
class Step:
    tool: str
    args: dict
    t: float
    result: str = ""


@dataclass
class Run:
    request: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = ""
    seconds: float = 0.0
    # Per turn: seconds in the model, acting (presses, waits), and getting the next screen.
    turns: list[dict] = field(default_factory=list)


# Screenshots older than this many turns are dropped from what the model sees again: they
# make every later turn slower and cost more, and the latest screen is what matters.
KEEP_IMAGES = 2


def trim_images(msgs: list[dict], keep: int = KEEP_IMAGES) -> list[dict]:
    """A copy of the conversation with all but the newest `keep` screenshots replaced."""
    seen = 0
    out: list[dict] = []
    for m in reversed(msgs):
        content = m.get("content")
        if isinstance(content, list) and any(c.get("type") == "image_url" for c in content):
            seen += 1
            if seen > keep:
                m = {**m, "content": [c if c.get("type") != "image_url"
                                      else {"type": "text", "text": "(an earlier screen)"}
                                      for c in content]}
        out.append(m)
    return out[::-1]


class Agent:
    def __init__(self, muse: Muse, controls: Controls, grab: Callable[[], Awaitable[Frame]],
                 speak: Callable[[str], Awaitable[None]], game: str = "",
                 max_turns: int = 32, max_settle_s: float = 1.5,
                 ask: Callable[[str], Awaitable[str]] | None = None,
                 confirm: str = "important", language: str = "English",
                 on_action: Callable[[str, dict], None] | None = None,
                 look_before_confirm: bool = True, notes: GameNotes | None = None):
        self.muse = muse
        self.controls = controls
        self.grab = grab
        self.speak = speak
        self.game = game
        self.max_turns = max_turns
        self.max_settle_s = max_settle_s
        self.ask = ask
        self.on_action = on_action
        self.look_before_confirm = look_before_confirm
        self.confirm = confirm
        self.language = language
        self.history: list[str] = []
        self.said: list[str] = []
        self.cancelled = False
        self._walker: Walker | None = None
        self.trail = Trail()  # a rough map of the area from where walks went
        self.area = AreaMap()  # the area as seen while walking, stitched together
        self._map_image = None  # the map as last shown to the model
        self._map_shown = -1    # which map that was (_map_gen when shown)
        self._map_gen = 0       # counts new maps: one per area
        self._map_fed = 0.0     # when a frame last went on it
        self._show_map = False
        self.notes = notes or GameNotes()
        # Cleared while the player holds the talk key: no presses or speech until it's
        # known whether they said stop, asked something else, or just wanted a repeat.
        self.go = asyncio.Event()
        self.go.set()

    def hold(self) -> None:
        self.go.clear()

    def resume(self) -> None:
        self.go.set()

    def cancel(self) -> None:
        self.cancelled = True
        self.go.set()

    async def _speak(self, text: str) -> None:
        await self.go.wait()
        if not self.cancelled and text.strip():
            self.said.append(text)
            await self.speak(text)

    def _already_said(self, text: str) -> bool:
        """Whether the last say already said it (a finish that restates it). Only a say
        since the last action counts: "Saved the game" after "I'll save the game" and a
        press is news. Never when the numbers or a "not" differ either: "I couldn't open
        the map" after "I'll open the map" is news."""
        if len(self.said) <= self._said_before_act:
            return False
        said = self.said[-1]

        def words(t: str) -> set[str]:
            return set(re.findall(r"[a-z0-9']+", t.lower().replace("\u2019", "'")))

        def negations(w: set[str]) -> set[str]:
            return {x for x in w if x in ("not", "no", "never", "nothing", "none", "cannot")
                    or x.endswith("n't")}

        now, before = words(text), words(said)
        if not now or re.findall(r"\d+", said) != re.findall(r"\d+", text):
            return False
        if negations(now) != negations(before):
            return False
        return now <= before or len(now & before) / len(now | before) >= 0.5

    def system(self) -> str:
        parts = [AGENT, CONFIRM.get(self.confirm, CONFIRM["important"])]
        parts.append(LANGUAGE.format(language=self.language))
        if self.game:
            parts.append(GAME_CONTEXT.format(game=self.game))
            learned = self.notes.prompt(self.game)
            if learned:
                parts.append(learned)
        return "\n".join(parts)

    async def _noise(self) -> float:
        """How much the screen changes by itself (fog, idle animation) between two looks."""
        a = await self.grab()
        await asyncio.sleep(0.05)
        return screen_change(a, await self.grab())

    async def _settled(self, noise: float | None) -> Frame:
        """The screen once it has stopped changing after an action, or after max_settle_s.

        Most menus redraw within a frame or two, so this is usually much quicker than a
        fixed pause; slow transitions get the time they need.
        """
        await asyncio.sleep(0.08)
        still = max(0.004, (noise or 0.0) * 1.3 + 0.002)
        seen = [(time.monotonic(), await self.grab())]
        end = time.monotonic() + self.max_settle_s
        calm = 0
        while time.monotonic() < end:
            await asyncio.sleep(0.05)
            now, cur = time.monotonic(), await self.grab()
            # Against a frame a quarter second back, not the last one: text typing out
            # a few letters at a time looks still from one frame to the next.
            old = [f for t, f in seen if now - t >= 0.25]
            seen.append((now, cur))
            if not old:
                continue
            calm = calm + 1 if screen_change(old[-1], cur) <= still else 0
            if calm >= 2:
                return cur
        return seen[-1][1]

    async def _walk_said(self, walker: Walker) -> None:
        """A line the walk started saying ("Something's in the way") ends before the next
        step, so it isn't taken for something said after the walk; stopped, it's cut."""
        talk = getattr(walker, "_talk", None)
        if talk is None or talk.done():
            return
        if self.cancelled:
            talk.cancel()
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(talk), 8)

    async def _press_a_and_see(self) -> float:
        """Press a; how much of the screen changed in the next second (a text box takes
        a moment to open)."""
        before = await self.grab()
        await asyncio.to_thread(self.controls.press, "a")
        changed = 0.0
        for _ in range(4):
            await asyncio.sleep(0.25)
            changed = max(changed, screen_change(before, await self.grab()))
            if changed > 0.04:
                break
        return changed

    async def _act(self, name: str, a: dict) -> str:
        if name == "press":
            asked = a.get("buttons") or []
            if isinstance(asked, str):
                asked = [asked]  # one button given bare, not a list of letters
            asked = [str(b) for b in asked]
            buttons = [b for b in asked if b in AGENT_BUTTONS][:12]
            ignored = [b for b in asked if b not in AGENT_BUTTONS]
            if not buttons:
                return (f"nothing pressed: {', '.join(ignored) or 'no buttons'} isn't a "
                        f"button; use {', '.join(AGENT_BUTTONS)}")
            if str(a.get("say") or "").strip():
                await self._speak(str(a["say"]))
            held_back = ""
            skipping = a.get("skipping") is True and set(buttons) <= {"a", "b", "start"}
            if self.look_before_confirm and not skipping:
                buttons, held_back = split_before_confirm(buttons)
            gap = max(60, min(int(float(a.get("gap_ms") or 150)), 1000))
            if self.on_action:
                self.on_action(name, {**a, "buttons": buttons})
            # A skip watches each press land, and stops when the text is over.
            watch = skipping and self.look_before_confirm and len(buttons) > 1
            if watch:
                noise = await self._noise()
                before, box = skip_grid(await self.grab()), None
            # One press at a time, so "stop" takes effect between presses.
            pressed, stopped = [], ""
            for i, b in enumerate(buttons):
                await self.go.wait()
                if self.cancelled:
                    return f"stopped by the player after {' '.join(pressed) or 'no presses'}"
                await asyncio.to_thread(self.controls.press, b)
                pressed.append(b)
                if i == len(buttons) - 1:
                    break
                if not watch:
                    await asyncio.sleep(gap / 1000)
                    continue
                after = skip_grid(await self._settled(noise))
                box, stopped = skip_on(box, before, after)
                if stopped:
                    break
                before = after
            done = f"pressed {' '.join(pressed)}"
            if stopped:
                done += (f" and stopped skipping there, {len(pressed)} of {len(buttons)}: "
                         f"{stopped}. Look before pressing anything else")
            if ignored:
                done += f" (ignored {', '.join(ignored)}: not buttons)"
            if buttons == ["a"] and not str(a.get("say") or "").strip():
                # The player can't read along: a text box skipped is story lost.
                done += (". If a new line of dialogue shows now, say it word for word "
                         "(who, then the line) before pressing again.")
            if held_back:
                done += (". Stopped before confirming: look at what is highlighted now. If it "
                         "is what the player wants, press a on its own; if not, fix it first.")
            return done
        if self.on_action and name != "talk_to":  # its walk_to says it
            self.on_action(name, a)
        if name == "wait":
            secs = max(0.5, min(float(a.get("seconds") or 2), 8.0))
            end = time.monotonic() + secs
            while time.monotonic() < end and not self.cancelled:
                await asyncio.sleep(0.05)  # "stop" ends the wait at once
            return f"waited {secs:g} s"
        if name == "hold":
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            ms = max(50, min(int(float(a.get("ms") or 500)), 5000))
            inp = a.get("input") or "left_stick"
            if not (inp in ("left_stick", "right_stick") or inp in AGENT_BUTTONS):
                return f"can't hold {inp!r}"
            if inp.endswith("_stick"):
                d = a.get("direction", "up")
                await asyncio.to_thread(self.controls.stick, inp.split("_")[0], d, ms)
                return f"held {inp} {d} for {ms} ms"
            if inp in ANGLES and ms >= 400 and self._walker is not None \
                    and self._walker.memory is not None and await self._still_me(self._walker):
                # Walking by d-pad where it has walked and the character is in view:
                # watch it, so the model learns how far it went or what stopped it.
                return await self._act("keep_walking", {"direction": inp,
                                                        "seconds": ms / 1000})
            if inp in ANGLES:
                # Held as a steer, so "stop" lets go at once instead of after the hold.
                t_hold = time.monotonic()
                end = t_hold + ms / 1000
                try:
                    self.controls.steer(ANGLES[inp])
                    while time.monotonic() < end and not self.cancelled and self.go.is_set():
                        await asyncio.sleep(0.02)
                finally:
                    self.controls.steer(None)
                held = round((time.monotonic() - t_hold) * 1000)
                if held < ms - 60:
                    why = "the player stopped it" if self.cancelled else "the player spoke"
                    return f"held {inp} for only {held} ms of {ms}: {why}"
                return f"held {inp} for {ms} ms"
            if inp == "a" and ms > 400 and self.look_before_confirm:
                return ("not held: holding a confirms and skips text the player can't read; "
                        "press a once and look")
            await asyncio.to_thread(self.controls.press, inp, ms)
            return f"held {inp} for {ms} ms"
        if name not in TOOL_NAMES:
            return f"no tool called {name}; use one of {', '.join(sorted(TOOL_NAMES))}"
        if name == "talk_to":
            walked = await self._act("walk_to", a)
            if not walked.startswith("arrived"):
                return walked
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            changed = await self._press_a_and_see()
            facing = getattr(self, "_facing", None)
            if changed <= 0.04 and facing is not None and not self.cancelled:
                # Small things on the ground answer only from right beside them: one
                # step closer the way it was facing, and try again.
                self.controls.steer(facing)
                await asyncio.sleep(0.15)
                self.controls.steer(None)
                await asyncio.sleep(0.2)
                changed = await self._press_a_and_see()
                if changed > 0.04:
                    walked += "; stepped one closer"
            if changed <= 0.04:
                return (f"{walked}; pressed a, but nothing came up on screen: they may "
                        "have nothing to say, or it isn't facing them. Look before saying "
                        "anything happened.")
            return (f"{walked}; pressed a, and something came up. If they speak, say the "
                    "line word for word (who, then the line) before pressing again.")
        if name == "walk_to":
            target = str(a.get("target") or "").strip()
            if not target:
                return "say what to walk to"
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            me = str(a.get("me") or "").strip()
            walker = self.walker()
            try:
                await self._find_me(walker)
                walked = await walker.walk(target, me)
                await self._walk_said(walker)
            except WALK_ERRORS as exc:
                # Not the model's arguments: something broke while walking.
                self.controls.steer(None)
                return f"walking failed ({exc}); walk with the d-pad instead"
            steps = walker.travel / walker.size if walker.size else None
            self.trail.add(f"walk_to {target}", walked.outcome, steps)
            return walked.text()
        if name == "exits":
            from .locate import exits_text, find_exits
            from .walk import WIDTH

            img = await self.walker_image()
            found, me = await find_exits(self.muse, img)
            th = img.height * WIDTH / img.width
            step = (16 / WIDTH, 16 / th)  # about a character, when none is tracked
            walker = self._walker
            if walker is not None and walker.memory is not None \
                    and await self._still_me(walker):
                _look, at, (bw, _bh) = walker.memory
                me = (float(at[0]) / WIDTH, float(at[1]) / th)
                step = (bw / WIDTH, bw / th)
            return exits_text(found, me, step, getattr(self, "_came_in", ""))
        if name == "area_map":
            await self.walker_image()  # the screen now: maybe somewhere new since
            if self.area.frames < 2:
                return "no map yet: nothing has been walked in this area"
            self._map_image = self.area.image(marks=None)
            self._map_shown = self._map_gen
            self._show_map = True
            return "the map comes with the next screen"
        if name == "walk_to_map":
            if self._map_image is None:
                return "look at area_map first"
            await self.walker_image()
            if self._map_shown != self._map_gen:
                return "this is a different area from the map you saw: look at area_map again"
            try:
                x = max(0.0, min(float(a.get("x")), 1000.0))
                y = max(0.0, min(float(a.get("y")), 1000.0))
            except (TypeError, ValueError):
                return "x and y must be numbers from 0 to 1000"
            label = str(a.get("label") or "that spot on the map")
            spot = self.area.from_image((x / 1000 * self._map_image.width,
                                         y / 1000 * self._map_image.height))
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            walker = self.walker()
            gen = self._map_gen

            def where():
                # Gone once the map starts over (a door, the view lost): its points with it.
                return tuple(self.area.to_screen(spot)) if self._map_gen == gen else None

            try:
                await self._find_me(walker)
                walked = await walker.walk_toward(where, label)
                await self._walk_said(walker)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc})"
            steps = walker.travel / walker.size if walker.size else None
            self.trail.add(f"walk_to_map {label}", walked.outcome, steps)
            return walked.text()
        if name == "go_to_place":
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            try:
                return await self._go_to_place(str(a.get("place") or ""))
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc})"
        if name == "guide":
            from .guide import look_up

            return look_up(self.game or "", str(a.get("query") or ""))
        if name == "travel":
            direction = str(a.get("direction") or "")
            if direction not in ANGLES:
                return f"direction must be one of {', '.join(ANGLES)}"
            secs = max(10.0, min(float(a.get("seconds") or 45), 90.0))
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            try:
                return await self._travel(direction, str(a.get("target") or ""), secs)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc}); hold the d-pad instead"
        if name == "keep_walking":
            direction = str(a.get("direction") or "")
            if direction not in ANGLES:
                return f"direction must be one of {', '.join(ANGLES)}"
            secs = max(0.4, min(float(a.get("seconds") or 3), 8.0))
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            walker = self.walker()
            try:
                await self._find_me(walker)
                went = await keep_walking(self.walker_image, self.controls.steer, direction,
                                          secs, self.walker_keep_going, walker.memory,
                                          walker.log)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc}); hold the d-pad instead"
            if went.at is None:
                walker.memory = None
            elif walker.memory is not None:
                walker.memory = (walker.memory[0], went.at, walker.memory[2])
                self.area.character_at(went.at)
            if went.steps.any() or went.outcome not in ("didnt_move", "unsure"):
                self.trail.add(f"keep_walking {direction}", went.outcome, went.steps)
            return went.text(direction.replace("_", " and "))
        return "ok"

    async def _go_to_place(self, place: str, limit: float = 90.0) -> str:
        """Walk to a named place on a world map by the game's atlas (see atlas.walk_to)."""
        from .atlas import Atlas, walk_to

        atlas = Atlas.for_game(self.game or "")
        if atlas is None:
            return "no atlas for this game: use travel and the guide instead"
        found = atlas.find_place(place)
        if found is None:
            names = sorted({n for m in atlas.maps for n in m.places})
            return f"no place called {place!r} in the atlas; places: {', '.join(names)}"
        m, name = found
        walker = self.walker()

        async def shoot():
            return (await self.grab()).image

        async def keep_going():
            return await self.walker_keep_going()

        return await walk_to(atlas, m, name, shoot, self.controls.steer,
                             self.controls.press, keep_going, walker.log, limit)

    async def _travel(self, direction: str, target: str, secs: float) -> str:
        """Head `direction` across a scrolling map until `target` is in sight. The view
        follows the character there, so how far the scenery scrolls says how far it went
        (no need to pick the character out), and a way that's shut is slid along."""
        walker = self.walker()
        walker.memory = None  # the character is where the view keeps it: the middle
        t0 = time.monotonic()
        angle = ANGLES[direction]
        unit = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
        # The two ways off the line, to slide along whatever blocks it.
        sides = [d for d, v in ANGLES.items() if abs(((v - angle + 180) % 360) - 180) == 90]
        total, stalls, looks = np.zeros(2), 0, 0

        def far() -> str:
            return f"went {words(total)} in all ({time.monotonic() - t0:.0f} s)"

        while time.monotonic() - t0 < secs:
            if self.cancelled:
                return "stopped by the player"
            went = await keep_walking(self.walker_image, self.controls.steer, direction,
                                      3.0, self.walker_keep_going, None, walker.log)
            total += went.steps
            if went.outcome in ("new_area", "interrupted", "stopped"):
                return went.text(direction.replace("_", " and ")) + f"; {far()}"
            if float(went.steps @ unit) < 1:
                stalls += 1
                if stalls > 2 * len(sides) + 1:
                    return (f"blocked: the way {direction.replace('_', ' ')} is shut here "
                            f"(slid both ways); {far()}. Try another direction or a path")
                side = sides[(stalls - 1) % len(sides)]
                slid = await keep_walking(self.walker_image, self.controls.steer, side,
                                          1.5 + 0.5 * stalls, self.walker_keep_going, None,
                                          walker.log)
                total += slid.steps
                if slid.outcome in ("new_area", "interrupted", "stopped"):
                    return slid.text(side.replace("_", " and ")) + f"; {far()}"
                continue
            stalls = 0
            if target:
                looks += 1
                seen = await self._sighted(target)
                if seen:
                    return (f"in_sight: {target} is on screen now, {seen}; walk_to it. "
                            f"{far()}")
        return f"walked: no {target or 'sign of it'} in sight yet; {far()}"

    async def _sighted(self, target: str) -> str:
        """Where `target` is on the screen now, in a few words, or "" if it isn't there:
        one quick look, strict about near misses."""
        from .muse import image_part

        img = await self.walker_image()
        try:
            data = await self.muse.json([{"role": "user", "content": [
                image_part(img, 1024),
                {"type": "text", "text": (
                    f'Is this on the screen: "{target}"? Only if it clearly matches that '
                    "description; something merely like it (a town for a castle, a "
                    "house for a shop) doesn't count. Answer JSON: {\"there\": true or "
                    "false, \"where\": \"where on screen, relative to the middle, a "
                    "few words\"}")}]}], effort="minimal", max_tokens=600)
        except (ValueError, OSError):
            return ""
        return str(data.get("where") or "in view") if data.get("there") is True else ""

    async def _still_me(self, walker: Walker) -> bool:
        """Whether the character last walked is still in view near where it was (it may
        have been moved by hand, or gone through a door); updates where it is."""
        rgb = small(await self.walker_image())[0]
        if walker.memory is None:  # a new place, seen just now
            return False
        look, at, (bw, bh) = walker.memory
        x, y, score = look.find(rgb, (at[0], at[1]), 3 * max(bw, bh))
        walker.memory = (look, np.array([x, y]), (bw, bh)) if score > 0.6 else None
        return walker.memory is not None

    async def _find_me(self, walker: Walker) -> None:
        """New to this area: which character is the player's, by moving it a little. The
        model often can't tell (two people with red hair, a mother at the bedside)."""
        if walker.memory is not None:
            await self._still_me(walker)
        if walker.memory is None:
            walker.memory = await find_me(self.walker_image, self.controls.steer,
                                          self.walker_keep_going, walker.log)
        if walker.memory is not None:
            self.area.character_at(walker.memory[1])

    async def walker_image(self):
        img = (await self.grab()).image
        # Every frame a walk looks at goes on the map; it starts over in a new place.
        rgb, g = small(img)
        gap = time.monotonic() - self._map_fed > 0.3
        self._map_fed = time.monotonic()
        try:
            if gap:
                # The view may have moved meanwhile (buttons pressed, a door): find it.
                fresh = await asyncio.to_thread(self.area.resume, rgb, g)
            else:
                # Off the event loop, so listening and speech keep up while walking.
                fresh = await asyncio.to_thread(self.area.track, rgb, g)
        except (ValueError, IndexError) as exc:
            self.area.reset()  # never let the map stop a walk
            fresh = True
            if self._walker is not None:
                self._walker.log(f"area map reset ({exc})")
        if fresh:
            self._map_gen += 1  # a new place: points on the old map mean nothing
            # Somewhere else now: where the character was and the walks so far belong
            # to the last place (a walk through a door also says how it got here).
            if self._walker is not None:
                self._walker.memory = None
            self.trail.clear("entered this area")
        now = self._walker._now if self._walker is not None else None
        if now is not None:
            self.area.character_at(now[1])
        return img

    async def walker_keep_going(self) -> bool:
        if not self.go.is_set():
            self.controls.steer(None)  # let go while the player is talking
            await self.go.wait()
        return not self.cancelled

    def walker(self) -> Walker:
        """One walker for the whole session, so what it learns about the camera stays."""
        if self._walker is None:
            def note(text: str) -> None:
                if self.on_action:
                    self.on_action("walk_note", {"text": text})

            async def exact(img, phrase):
                from .locate import segment

                return [m.box for m in await segment(self.muse, img, phrase) if m.box]

            def steer(angle):
                if angle is not None:
                    self._facing = angle  # the way it last walked, to step closer by
                self.controls.steer(angle)

            self._walker = Walker(self.muse, self.walker_image, steer,
                                  self.walker_keep_going, note, say=self._speak, exact=exact)
        return self._walker

    def _check_walk(self, name: str, result: str) -> str:
        """Two walks in a row that didn't get there: say plainly that way is cut off, so
        the model stops rewording the same target."""
        if name not in ("walk_to", "talk_to", "walk_to_map"):
            return result
        if result.startswith(WALK_PAUSED):
            return result
        if not result.startswith(WALK_FAILED):
            self._failed_walks = 0
            return result
        self._failed_walks += 1
        if self._failed_walks >= 2:
            result += (". That's two walks in a row that didn't get there: that way is cut "
                       "off from here. Don't aim at it again, however you name it; look at "
                       "area_map, or keep_walking a direction you haven't tried, or tell the "
                       "player.")
        return result

    async def run(self, request: str, frame: Frame | None = None) -> Run:
        t0 = time.monotonic()
        self._failed_walks = 0
        run = Run(request)
        self.cancelled = False
        self.said: list[str] = []
        self._said_before_act = 0  # how many lines were said before the last action
        frame = frame or await self.grab()
        recent = "\n".join(f"- {h}" for h in self.history[-4:])
        trail = self.trail.notes()
        msgs: list[dict] = [
            {"role": "system", "content": self.system()},
            {"role": "user", "content": [
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": (f"Earlier requests:\n{recent}\n\n" if recent else "")
                 + (f"{trail}\n\n" if trail else "")
                 + f'The player says: "{request}"'},
            ]},
        ]
        # The player's answer to a question sets a new course: it gets turns of its own.
        budget, _turn, out_of_turns = self.max_turns, -1, True
        while _turn + 1 < budget:
            _turn += 1
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            t_model = time.monotonic()
            last_turn = _turn == budget - 1
            msg = await self.muse.tools(trim_images(msgs), TOOLS if not last_turn else
                                        FINISH_OR_ASK if self.ask else FINISH_ONLY)
            turn = {"model_s": round(time.monotonic() - t_model, 2)}
            run.turns.append(turn)
            await self.go.wait()
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            calls = msg.tool_calls or []
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump() for c in calls] or None})
            if not calls:
                if msg.content:
                    await self._speak(msg.content)
                    run.outcome = msg.content
                    out_of_turns = False
                    break
                # Nothing at all (its thinking ran out of room): ask again rather than
                # end in silence.
                msgs.append({"role": "user", "content": "You replied with nothing. Call a "
                             "tool; if you're done, finish with what to tell the player."})
                continue
            acted = False
            finished = None
            noise = None
            if any(c.function.name in ("press", "hold", "walk_to", "talk_to", "walk_to_map",
                                       "keep_walking", "travel")
                   for c in calls):
                noise = await self._noise()
            t_act = time.monotonic()
            for c in calls:
                if self.cancelled:
                    break
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                step = Step(c.function.name, args, time.monotonic() - t0)
                run.steps.append(step)
                if finished is not None:
                    result = "ignored: it came after finish"
                elif last_turn and c.function.name not in ("finish", "ask_player"):
                    result = ("not done: you're out of steps; ask_player whether to keep "
                              "going, or finish" if self.ask else
                              "not done: only finish is left now")
                elif c.function.name == "finish" and acted:
                    # It hasn't seen what its presses did: no telling the player yet.
                    result = "not finished: look at the screen after those actions first"
                elif c.function.name == "finish" and not str(args.get("text") or "").strip() \
                        and len(self.said) <= self._said_before_act and not last_turn:
                    result = "not finished: give the text to tell the player what happened"
                elif c.function.name == "say" and acted:
                    # Nor say what happened before seeing it.
                    result = "not said: look at the screen after those actions first"
                elif c.function.name == "say":
                    text = str(args.get("text") or "")
                    if self._already_said(text):
                        result = "you just said that; don't say it again"
                    else:
                        await self._speak(text)
                        result = "said"
                elif c.function.name == "ask_player":
                    question = args.get("question", "")
                    if self.ask is None:
                        await self._speak(question)
                        result = "The player can't answer right now. Don't do it; finish."
                    else:
                        answer = await self.ask(question)
                        result = f'The player said: "{answer}"' if answer else (
                            "No answer. Don't do it; finish and say you'll wait.")
                        if answer:
                            budget = max(budget, _turn + 1 + self.max_turns // 2)
                elif c.function.name == "remember":
                    result = self.notes.add(self.game, str(args.get("note") or ""),
                                            str(args.get("instead_of") or ""))
                elif c.function.name == "finish":
                    finished = (args.get("text") or "").strip()
                    result = "finished"
                else:
                    try:
                        result = await self._act(c.function.name, args)
                    except (TypeError, ValueError, KeyError) as exc:
                        result = f"bad arguments ({exc}); fix them and try again"
                    acted = True
                    self._said_before_act = len(self.said)
                    result = self._check_walk(c.function.name, result)
                    if result.startswith("new_area"):
                        # Which way it was going when it got here: the way back is behind.
                        self._came_in = (str(args.get("direction") or "")
                                         if c.function.name in ("keep_walking", "travel")
                                         else way(getattr(self, "_facing", None)))
                step.result = result
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            if finished is not None:
                if not finished and last_turn and len(self.said) <= self._said_before_act:
                    finished = ("I stopped there after a lot of steps. Say describe to hear "
                                "where things stand.")
                if finished and not self._already_said(finished):
                    await self._speak(finished)
                # Kept for the next request's history even when say already said it.
                run.outcome = finished or (self.said[-1] if self.said else "done")
                out_of_turns = False
                break
            if last_turn and self.ask is not None and not self.cancelled and not any(
                    c.function.name == "ask_player" for c in calls):
                # Out of steps and not done: the player says whether it's worth more.
                answer = await self.ask("This is taking a lot of steps and I'm not there "
                                        "yet. Should I keep going?")
                if answer and not re.match(r"\s*(no|nope|stop|don't|do not|cancel)\b",
                                           answer.lower()):
                    budget = max(budget, _turn + 1 + self.max_turns // 2)
                    msgs.append({"role": "user", "content": (
                        "You were out of steps, so the player was asked whether to keep "
                        f'going. They said: "{answer}". Carry on.')})
            turn["act_s"] = round(time.monotonic() - t_act, 2)
            t_look = time.monotonic()
            # After a say too: the model may be waiting for text still typing out.
            frame = await self._settled(noise)
            turn["look_s"] = round(time.monotonic() - t_look, 2)
            last = _turn == budget - 2
            shown = []
            if self._show_map and self._map_image is not None:
                self._show_map = False
                shown = [image_part(self._map_image, 1024), {"type": "text", "text": (
                    "The map of this area so far: dark grey is not seen yet, the yellow box "
                    "is the screen now, the cyan line is where you walked and its dot is "
                    "you. People shown may have moved since. For walk_to_map, x and y are "
                    "on this map, 0 to 1000 each way.")}]
            msgs.append({"role": "user", "content": [
                *shown,
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": "The screen now." + (
                    " You're out of steps. Done: finish. Not done: "
                    + ("ask_player, saying in a sentence how far you got and asking whether "
                       "to keep going (a yes gives you more steps); if it can't be done, "
                       if self.ask else "")
                    + "finish, telling the player plainly how far you got, where things "
                    "stand on screen, and what they could ask next." if last else "")},
            ]})
        if out_of_turns:
            await self._speak("I couldn't finish that one. Want me to try another way?")
            run.outcome = "gave up"
        run.seconds = time.monotonic() - t0
        self.history.append(f"{request} -> {run.outcome}")
        return run
