"""Voice commands: look at the screen, press buttons, look again, until it's done."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .controls import BUTTONS, DIRECTIONS, Controls
from .muse import Muse, image_part
from .prompts import AGENT, CONFIRM, GAME_CONTEXT, LANGUAGE
from .screen import Frame

TOOLS = [
    {"type": "function", "function": {
        "name": "press",
        "description": "Press controller buttons in order. d-pad is up/down/left/right.",
        "parameters": {"type": "object", "properties": {
            "buttons": {"type": "array", "items": {"type": "string", "enum": list(BUTTONS)},
                        "description": "Buttons to press one after another."},
            "gap_ms": {"type": "integer", "description": "Pause between presses, default 150."},
        }, "required": ["buttons"]}}},
    {"type": "function", "function": {
        "name": "hold",
        "description": "Hold a button, or push a stick in a direction, for a while. Use "
                       "the left stick to walk.",
        "parameters": {"type": "object", "properties": {
            "input": {"type": "string", "enum": ["left_stick", "right_stick", *BUTTONS]},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "ms": {"type": "integer", "description": "How long, in milliseconds (max 5000)."},
        }, "required": ["input", "ms"]}}},
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


@dataclass
class Run:
    request: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = ""
    seconds: float = 0.0


class Agent:
    def __init__(self, muse: Muse, controls: Controls, grab: Callable[[], Awaitable[Frame]],
                 speak: Callable[[str], Awaitable[None]], game: str = "",
                 max_turns: int = 10, settle_s: float = 0.45,
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
        self.settle_s = settle_s
        self.ask = ask
        self.on_action = on_action
        self.look_before_confirm = look_before_confirm
        self.confirm = confirm
        self.language = language
        self.history: list[str] = []
        self.cancelled = False

    def system(self) -> str:
        parts = [AGENT, CONFIRM.get(self.confirm, CONFIRM["important"])]
        if self.language.lower() not in ("english", "british english"):
            parts.append(LANGUAGE.format(language=self.language))
        if self.game:
            parts.append(GAME_CONTEXT.format(game=self.game))
        return "\n".join(parts)

    async def _act(self, name: str, a: dict) -> str:
        if name == "press":
            buttons = [b for b in a.get("buttons", []) if b in BUTTONS][:12]
            held_back = ""
            if self.look_before_confirm:
                buttons, held_back = split_before_confirm(buttons)
            gap = int(a.get("gap_ms", 150))
            if self.on_action:
                self.on_action(name, {**a, "buttons": buttons})
            await asyncio.to_thread(self.controls.sequence, buttons, max(60, min(gap, 1000)))
            done = f"pressed {' '.join(buttons)}"
            if held_back:
                done += (". Stopped before confirming: look at what is highlighted now. If it "
                         "is what the player wants, press a on its own; if not, fix it first.")
            return done
        if self.on_action:
            self.on_action(name, a)
        if name == "hold":
            ms = max(50, min(int(a.get("ms", 500)), 5000))
            inp = a.get("input", "left_stick")
            if inp.endswith("_stick"):
                d = a.get("direction", "up")
                await asyncio.to_thread(self.controls.stick, inp.split("_")[0], d, ms)
                return f"held {inp} {d} for {ms} ms"
            await asyncio.to_thread(self.controls.press, inp, ms)
            return f"held {inp} for {ms} ms"
        return "ok"

    async def run(self, request: str, frame: Frame | None = None) -> Run:
        t0 = time.monotonic()
        run = Run(request)
        self.cancelled = False
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
            msg = await self.muse.tools(msgs, TOOLS)
            calls = msg.tool_calls or []
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump() for c in calls] or None})
            if not calls:
                if msg.content:
                    await self.speak(msg.content)
                run.outcome = msg.content or "done"
                break
            acted = False
            finished = None
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                run.steps.append(Step(c.function.name, args, time.monotonic() - t0))
                if c.function.name == "say":
                    await self.speak(args.get("text", ""))
                    result = "said"
                elif c.function.name == "ask_player":
                    question = args.get("question", "")
                    if self.ask is None:
                        await self.speak(question)
                        result = "The player can't answer right now. Don't do it; finish."
                    else:
                        answer = await self.ask(question)
                        result = f'The player said: "{answer}"' if answer else (
                            "No answer. Don't do it; finish and say you'll wait.")
                elif c.function.name == "finish":
                    finished = args.get("text", "Done.")
                    result = "finished"
                else:
                    result = await self._act(c.function.name, args)
                    acted = True
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if finished is not None:
                await self.speak(finished)
                run.outcome = finished
                break
            if acted:
                await asyncio.sleep(self.settle_s)
            frame = await self.grab()
            msgs.append({"role": "user", "content": [
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": "The screen now."},
            ]})
        else:
            await self.speak("I couldn't finish that one. Want me to try another way?")
            run.outcome = "gave up"
        run.seconds = time.monotonic() - t0
        self.history.append(f"{request} -> {run.outcome}")
        return run
