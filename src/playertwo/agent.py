"""Voice commands: look at the screen, press buttons, look again, until it's done."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .controls import BUTTONS, DIRECTIONS, Controls
from .muse import Muse, image_part
from .prompts import AGENT, CONFIRM, GAME_CONTEXT, LANGUAGE
from .screen import Frame, thumb
from .walk import Walker

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
        }, "required": ["buttons"]}}},
    {"type": "function", "function": {
        "name": "hold",
        "description": "Hold a button, or push a stick in a direction, for a while. Use "
                       "the left stick to walk.",
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
                       "what's in the way, and stops next to it, facing it. Then press a "
                       "to talk to it or open it. Not for 3D games with the camera behind "
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
        "name": "finish",
        "description": "The request is done (or can't be done). Tell the player in one "
                       "short sentence.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
]


MOVES = {"up", "down", "left", "right", "lb", "rb"}


def screen_change(a: Frame, b: Frame, cell: float = 0.06) -> float:
    """Fraction of the screen that differs between two frames (small thumbnails)."""
    return float((abs(thumb(a.image) - thumb(b.image)) > cell).mean())


def split_before_confirm(buttons: list[str]) -> tuple[list[str], str]:
    """Cut a press list just before an `a` that follows cursor movement.

    Moving and confirming blind is where wrong picks come from, so the agent has to look
    at the highlight before every confirm. A lone `a` (advancing dialogue) is fine.
    """
    moved = False
    for i, b in enumerate(buttons):
        if b in MOVES:
            moved = True
        elif b == "a" and moved:
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
                 look_before_confirm: bool = True):
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
        """Whether the last say already said it (a finish that restates it). Never when
        the numbers or a "not" differ: "I couldn't open the map" after "I'll open the
        map" is news."""
        if not self.said:
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
        prev = await self.grab()
        end = time.monotonic() + self.max_settle_s
        while time.monotonic() < end:
            await asyncio.sleep(0.05)
            cur = await self.grab()
            if screen_change(prev, cur) <= still:
                return cur
            prev = cur
        return prev

    async def _act(self, name: str, a: dict) -> str:
        if name == "press":
            buttons = [b for b in a.get("buttons", []) if b in AGENT_BUTTONS][:12]
            held_back = ""
            if self.look_before_confirm:
                buttons, held_back = split_before_confirm(buttons)
            gap = max(60, min(int(float(a.get("gap_ms") or 150)), 1000))
            if self.on_action:
                self.on_action(name, {**a, "buttons": buttons})
            # One press at a time, so "stop" takes effect between presses.
            pressed = []
            for i, b in enumerate(buttons):
                await self.go.wait()
                if self.cancelled:
                    return f"stopped by the player after {' '.join(pressed) or 'no presses'}"
                await asyncio.to_thread(self.controls.press, b)
                pressed.append(b)
                if i < len(buttons) - 1:
                    await asyncio.sleep(gap / 1000)
            done = f"pressed {' '.join(buttons)}"
            if buttons == ["a"]:
                # The player can't read along: a text box skipped is story lost.
                done += (". If a new line of dialogue shows now, say it word for word "
                         "(who, then the line) before pressing again.")
            if held_back:
                done += (". Stopped before confirming: look at what is highlighted now. If it "
                         "is what the player wants, press a on its own; if not, fix it first.")
            return done
        if self.on_action:
            self.on_action(name, a)
        if name == "wait":
            secs = max(0.5, min(float(a.get("seconds") or 2), 8.0))
            await asyncio.sleep(secs)
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
            await asyncio.to_thread(self.controls.press, inp, ms)
            return f"held {inp} for {ms} ms"
        if name == "walk_to":
            target = str(a.get("target") or "").strip()
            if not target:
                return "say what to walk to"
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            me = str(a.get("me") or "").strip()
            try:
                return (await self.walker().walk(target, me)).text()
            except (ValueError, TypeError, IndexError, OSError, RuntimeError) as exc:
                # Not the model's arguments: something broke while walking.
                self.controls.steer(None)
                return f"walking failed ({exc}); walk with the d-pad instead"
        return "ok"

    def walker(self) -> Walker:
        """One walker for the whole session, so what it learns about the camera stays."""
        if self._walker is None:
            async def image():
                return (await self.grab()).image

            async def keep_going() -> bool:
                if not self.go.is_set():
                    self.controls.steer(None)  # let go while the player is talking
                    await self.go.wait()
                return not self.cancelled

            def note(text: str) -> None:
                if self.on_action:
                    self.on_action("walk_note", {"text": text})

            self._walker = Walker(self.muse, image, self.controls.steer, keep_going, note,
                                  say=self._speak)
        return self._walker

    async def run(self, request: str, frame: Frame | None = None) -> Run:
        t0 = time.monotonic()
        run = Run(request)
        self.cancelled = False
        self.said: list[str] = []
        frame = frame or await self.grab()
        recent = "\n".join(f"- {h}" for h in self.history[-4:])
        msgs: list[dict] = [
            {"role": "system", "content": self.system()},
            {"role": "user", "content": [
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": (f"Earlier requests:\n{recent}\n\n" if recent else "")
                 + f'The player says: "{request}"'},
            ]},
        ]
        for _turn in range(self.max_turns):
            if self.cancelled:
                run.outcome = "cancelled"
                break
            t_model = time.monotonic()
            msg = await self.muse.tools(trim_images(msgs), TOOLS)
            turn = {"model_s": round(time.monotonic() - t_model, 2)}
            run.turns.append(turn)
            await self.go.wait()
            if self.cancelled:
                run.outcome = "cancelled"
                break
            calls = msg.tool_calls or []
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump() for c in calls] or None})
            if not calls:
                if msg.content:
                    await self._speak(msg.content)
                run.outcome = msg.content or "done"
                break
            acted = False
            finished = None
            noise = None
            if any(c.function.name in ("press", "hold", "walk_to") for c in calls):
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
                if c.function.name == "say":
                    await self._speak(args.get("text", ""))
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
                elif c.function.name == "finish":
                    finished = (args.get("text") or "").strip()
                    result = "finished"
                else:
                    try:
                        result = await self._act(c.function.name, args)
                    except (TypeError, ValueError, KeyError) as exc:
                        result = f"bad arguments ({exc}); fix them and try again"
                    acted = True
                step.result = result
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if self.cancelled:
                run.outcome = "cancelled"
                break
            if finished is not None:
                if finished and not self._already_said(finished):
                    await self._speak(finished)
                run.outcome = finished
                break
            turn["act_s"] = round(time.monotonic() - t_act, 2)
            t_look = time.monotonic()
            frame = await self._settled(noise) if acted else await self.grab()
            turn["look_s"] = round(time.monotonic() - t_look, 2)
            last = _turn == self.max_turns - 2
            msgs.append({"role": "user", "content": [
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": "The screen now." + (
                    " You're out of steps: call finish now, telling the player plainly how "
                    "far you got, where things stand on screen, and what they could ask "
                    "next." if last else "")},
            ]})
        else:
            await self._speak("I couldn't finish that one. Want me to try another way?")
            run.outcome = "gave up"
        run.seconds = time.monotonic() - t0
        self.history.append(f"{request} -> {run.outcome}")
        return run
