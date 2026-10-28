"""Proactive narration: speaks up when a new screen settles."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .muse import Muse, image_part
from .prompts import DESCRIBE, GAME_CONTEXT, LANGUAGE, NARRATOR, TEAM_UP, VERBOSITY
from .screen import Frame

_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)]*(\s|$)")


def _sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[start : m.end()].strip())
        start = m.end()
    return out


def _opening(text: str) -> tuple[str, int] | None:
    """Enough of the reply to tell whether it repeats itself, and where it ends: the first
    sentence, or the first two when the first is short ("Your turn.")."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if not ends:
        return None
    first = text[:ends[0]].strip()
    if len(_words(first)) >= 4:
        return first, ends[0]
    if len(ends) >= 2:
        return " ".join(text[:ends[1]].split()), ends[1]
    return None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


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

    def repeats(self, opening: str) -> bool:
        """True when this opening says what the last narration opened with.

        Idle animations (a character bobbing, drifting fog) can make a screen look new when
        nothing changed, and the model doesn't always answer <silent> then. A reworded
        opening ("on the TV World stage" vs "of the TV World stage") still counts.
        """
        if not self.recent:
            return False
        n = len(_sentences(opening)) or 1
        before = " ".join(_sentences(self.recent[-1])[:n]) or self.recent[-1]
        # A changed number is news ("Chie has 40 HP" after "60 HP"), however alike the rest.
        if re.findall(r"\d+", opening) != re.findall(r"\d+", before):
            return False
        new = set(_words(opening))
        old = set(_words(before))
        if not new or not old:
            return False
        return len(new & old) / len(new | old) >= 0.75

    def news(self, text: str) -> str:
        """The sentences of `text` that aren't filler or something said recently."""
        said = [s for r in self.recent for s in _sentences(r) or [r]]
        keep = []
        for s in _sentences(text) or ([text] if text.strip() else []):
            if re.match(r"(still|same|nothing (new|changed))\b", s.strip().lower()):
                continue
            if any(self._alike(s, old) for old in said):
                continue
            keep.append(s)
        return " ".join(keep)

    @staticmethod
    def _alike(a: str, b: str) -> bool:
        if re.findall(r"\d+", a) != re.findall(r"\d+", b):
            return False
        x, y = set(_words(a)), set(_words(b))
        return bool(x and y) and len(x & y) / len(x | y) >= 0.75

    async def narrate(self, frame: Frame) -> AsyncIterator[str]:
        """Stream what to say about a newly settled frame. Yields nothing when silent."""
        t = time.monotonic()
        text = ""  # what was said (a repeated opening left out)
        head = ""
        decided = False
        silent = False
        repeated = False  # the opening repeated: hold the rest and keep only what's new
        stream = self.muse.stream(self.messages(frame))
        try:
            async for delta in stream:
                if decided:
                    text += delta
                    if not repeated:
                        yield delta
                    continue
                head += delta
                stripped = head.lstrip()
                if stripped.startswith("<silent>"):
                    silent = True
                    break
                if len(stripped) < 8 and "<silent>".startswith(stripped):
                    continue
                opening = _opening(stripped)
                if opening is None:
                    continue
                decided = True
                said, end = opening
                # A repeated opening ("Still in the Velvet Room.") is dropped, but anything
                # new after it is still said.
                repeated = self.repeats(said)
                text = stripped[end:] if repeated else stripped
                if not repeated and text.strip():
                    yield text
        finally:
            await stream.aclose()
        if repeated:
            text = self.news(text)
            if text:
                yield text
        if not decided and not silent:
            # A short reply that never completed an opening ("Bath 1.").
            stripped = head.strip()
            silent = not stripped or stripped.startswith("<silent>") or self.repeats(stripped)
            text = "" if silent else stripped
            if text:
                yield text
        text = text.strip()
        silent = silent or not text or text.startswith("<silent>")
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
