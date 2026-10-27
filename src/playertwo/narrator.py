"""Proactive narration: speaks up when a new screen settles."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .muse import Muse, image_part
from .prompts import DESCRIBE, GAME_CONTEXT, LANGUAGE, NARRATOR, TEAM_UP, VERBOSITY
from .screen import Frame


@dataclass
class Line:
    text: str
    t: float
    latency_s: float
    frame_label: str = ""


class Narrator:
    def __init__(self, muse: Muse, verbosity: str = "normal", game: str = "",
                 language: str = "English", team_up: bool = False):
        self.muse = muse
        self.verbosity = verbosity
        self.game = game
        self.language = language
        self.team_up = team_up
        self.recent: deque[str] = deque(maxlen=4)
        self.log: list[Line] = []

    def system(self) -> str:
        parts = [NARRATOR, VERBOSITY.get(self.verbosity, "")]
        if self.language.lower() not in ("english", "british english"):
            parts.append(LANGUAGE.format(language=self.language))
        if self.team_up:
            parts.append(TEAM_UP)
        if self.game:
            parts.append(GAME_CONTEXT.format(game=self.game))
        return "\n".join(p for p in parts if p)

    def messages(self, frame: Frame, extra: str = "") -> list[dict]:
        said = "\n".join(f"- {r}" for r in self.recent) or "- (nothing yet)"
        content = [
            image_part(frame.image, self.muse.s.image_max_side),
            {"type": "text", "text": f"You said recently:\n{said}\n\n{extra}".strip()},
        ]
        return [{"role": "system", "content": self.system()},
                {"role": "user", "content": content}]

    async def narrate(self, frame: Frame) -> AsyncIterator[str]:
        """Stream what to say about a newly settled frame. Yields nothing when silent."""
        t = time.monotonic()
        text = ""
        head = ""
        silent = None
        async for delta in self.muse.stream(self.messages(frame)):
            text += delta
            if silent is None:
                head += delta
                stripped = head.lstrip()
                if len(stripped) >= 8 or not "<silent>".startswith(stripped):
                    silent = stripped.startswith("<silent>")
                    if not silent:
                        yield head
                continue
            if not silent:
                yield delta
        text = text.strip()
        if silent is None:
            silent = text.startswith("<silent>") or not text
            if not silent:
                yield head
        if not silent:
            self.recent.append(text)
        self.log.append(Line("" if silent else text, time.time(), time.monotonic() - t,
                             frame.label))

    async def describe(self, frame: Frame) -> AsyncIterator[str]:
        """A full spoken tour of the screen, on request."""
        msgs = self.messages(frame, DESCRIBE)
        text = ""
        async for delta in self.muse.stream(msgs, effort="low"):
            text += delta
            yield delta
        self.recent.append(text.strip())
