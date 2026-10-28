"""Where is it? SAM 3.1 finds every match for a phrase, then we say and play where.

Muse Spark is good at describing a scene; SAM 3.1 is exact about pixels. For "where's
the door?" the player gets a spoken direction plus a short sound panned to where each
match is, so they can hear the layout.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from PIL import Image

from .muse import Muse, image_part

BOX = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")


@dataclass
class Match:
    cx: float  # 0..1 across the screen
    cy: float  # 0..1 down the screen
    area: float  # fraction of the screen covered

    @property
    def pan(self) -> float:
        return round(self.cx * 2 - 1, 2)


def parse_boxes(text: str) -> list[Match]:
    out = []
    for m in BOX.finditer(text):
        x1, y1, x2, y2, w, h = (int(v) for v in m.groups())
        if w <= 0 or h <= 0:
            continue
        out.append(Match(((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                         max(0, (x2 - x1) * (y2 - y1)) / (w * h)))
    return out


def where_words(m: Match) -> str:
    side = ("on the far left" if m.cx < 0.2 else "on the left" if m.cx < 0.4
            else "in the middle" if m.cx < 0.6 else "on the right" if m.cx < 0.8
            else "on the far right")
    height = "near the top" if m.cy < 0.3 else "near the bottom" if m.cy > 0.7 else ""
    size = "close" if m.area > 0.08 else "far off" if m.area < 0.004 else ""
    return " ".join(p for p in (side, height, ", " + size if size else "") if p).replace(
        " , ", ", ")


def describe(phrase: str, matches: list[Match]) -> str:
    if not matches:
        return f"I can't find any {phrase} on screen."
    ordered = sorted(matches, key=lambda m: m.cx)
    n = len(ordered)
    if n == 1:
        return f"Found it: {where_words(ordered[0])}."
    if n <= 3:
        return f"{n} {phrase}: " + ", ".join(f"one {where_words(m)}" for m in ordered) + "."
    thirds = [sum(1 for m in ordered if lo <= m.cx < hi)
              for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))]
    parts = [f"{k} {where}" for k, where in zip(thirds, ("on the left", "in the middle",
                                                        "on the right"), strict=True) if k]
    return f"{n} {phrase}: " + ", ".join(parts) + "."


POINT = """\
Point to every "{phrase}" in this game screen. JSON only, on a 0 to 1000 grid of the image:
{{"points": [[x, y], ...]}}. Return {{"points": []}} if there are none."""


async def point(muse: Muse, img: Image.Image, phrase: str) -> list[Match]:
    """Fallback when SAM doesn't know the concept (a game's own monsters, say): Muse
    Spark knows what the game calls things and can point at them."""
    data = await muse.json([{"role": "user", "content": [
        image_part(img, 1280), {"type": "text", "text": POINT.format(phrase=phrase)}]}])
    out = []
    for p in data.get("points", []):
        try:
            x, y = float(p[0]) / 1000, float(p[1]) / 1000
        except (TypeError, ValueError, IndexError):
            continue
        out.append(Match(min(max(x, 0), 1), min(max(y, 0), 1), 0.02))
    return out


async def locate(muse: Muse, img: Image.Image, phrase: str) -> tuple[list[Match], float]:
    """Run SAM 3.1 on a frame. Returns the matches and how long it took."""
    t = time.monotonic()
    part = image_part(img, 1280)
    text = ""
    async with muse.client.responses.stream(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_image", "image_url": part["image_url"]["url"]},
        ]}],
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
    return parse_boxes(text), time.monotonic() - t


WHERE = re.compile(
    r"^(?:hey,? )?(?:where(?:'s| is| are)|find|how many)\s+(?:the |a |an |my |all )?"
    r"(?P<what>[\w' -]{2,40}?)\??$", re.IGNORECASE)
# Words that make it a request to do something ("find a potion and use it"), which is the
# agent's job, not a question about where things are on screen.
ACTION_WORDS = {"and", "then", "use", "open", "go", "attack", "heal", "buy", "sell", "equip",
                "talk", "pick", "give", "cast", "hit", "to", "with", "for", "me", "out"}


def where_request(text: str) -> str | None:
    """The noun phrase in "where's the door?", or None if it isn't that kind of question."""
    text = text.strip().rstrip(".!")
    m = WHERE.match(text)
    if not m:
        return None
    # "How many enemies are there" is about the screen; "how many potions do I have"
    # needs the menus.
    if text.lower().startswith("how many") and not re.search(
            r"\b(on (the )?screen|are there|here|do you see|can you see)\b", text, re.IGNORECASE):
        return None
    what = re.sub(r"\b(on (the )?screen|here|now|right now|at)\b", "", m.group("what"), flags=re.IGNORECASE)
    what = what.strip(" ,")
    if set(re.findall(r"[a-z']+", what.lower())) & ACTION_WORDS:
        return None
    return what or None
