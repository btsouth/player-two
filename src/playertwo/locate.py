"""Where is it? SAM 3.1 finds every match for a phrase, then we say and play where.

Muse Spark is good at describing a scene; SAM 3.1 outlines each thing to the pixel, and
the masks it sends are kept on each Match (decoded only when asked for). For "where's the
door?" the player gets a spoken direction plus a short sound panned to where each match
is, so they can hear the layout.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from .muse import Muse, image_part

BOX = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")
# The mask token that follows a box. Its payload holds no "|" (see the SAM 3 protocol).
BOX_MASK = re.compile(BOX.pattern + r"(?:<\|mask;(?P<mask>[^|]*)\|>)?")
MASK_DATA = re.compile(r"(?:^|;)\s*data=(\d+),(\d+),([~!][^;]*)")


@dataclass(frozen=True)
class Mask:
    """One SAM mask as sent: a raster the size of its box, placed at the box's corner."""

    height: int
    width: int
    payload: str  # with its encoding marker: "~" lossless, "!" one bit
    left: int  # the box's corner in the frame SAM saw
    top: int
    frame_w: int
    frame_h: int

    @cached_property
    def _decoded(self) -> np.ndarray:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster

        raw = decode_mask_to_raster(SegmentationMask(
            encoding="lossless" if self.payload.startswith("~") else "one_bit",
            payload=self.payload, width=self.width, height=self.height))
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width)

    def raster(self) -> np.ndarray:
        """height x width array of 0 and 1. Slow the first time for a big mask (it's pure
        Python, a few microseconds a pixel): use a thread when live."""
        return self._decoded

    def full_frame(self) -> np.ndarray:
        """frame_h x frame_w array of 0 and 1, the mask where it is on the frame."""
        out = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        part = self.raster()[:max(0, self.frame_h - self.top), :max(0, self.frame_w - self.left)]
        out[self.top:self.top + part.shape[0], self.left:self.left + part.shape[1]] = part
        return out


def parse_mask(fields: str | None, left: int, top: int, w: int, h: int) -> Mask | None:
    """The mask token's fields, or None when there's no usable payload (a box is still kept)."""
    m = MASK_DATA.search(fields or "")
    if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
        return None
    return Mask(int(m.group(1)), int(m.group(2)), m.group(3), left, top, w, h)


@dataclass
class Match:
    cx: float  # 0..1 across the screen
    cy: float  # 0..1 down the screen
    area: float  # fraction of the screen covered
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2, 0..1
    mask: Mask | None = None  # SAM's own outline, when it came from SAM

    @property
    def pan(self) -> float:
        return round(self.cx * 2 - 1, 2)


def wide_pan(pan: float) -> float:
    """Spread a screen position for the ear, so off-centre is obvious on room speakers."""
    return round(math.copysign(abs(pan) ** 0.5, pan), 2)


def parse_boxes(text: str) -> list[Match]:
    out = []
    for m in BOX_MASK.finditer(text):
        x1, y1, x2, y2, w, h = (int(v) for v in m.groups()[:6])
        if w <= 0 or h <= 0:
            continue
        out.append(Match(((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                         max(0, (x2 - x1) * (y2 - y1)) / (w * h),
                         (x1 / w, y1 / h, x2 / w, y2 / h),
                         parse_mask(m.group("mask"), x1, y1, w, h)))
    return out


@dataclass(frozen=True)
class Shape:
    """A thing's measured place on the frame: what its pixels cover, not its box."""

    cx: float  # the middle of its pixels, 0..1
    cy: float
    area: float  # fraction of the screen its pixels cover
    x1: float  # how far its pixels reach, 0..1
    y1: float
    x2: float
    y2: float

    @property
    def pan(self) -> float:
        return round(self.cx * 2 - 1, 2)


def measure(m: Match) -> Shape:
    """Decode a match's mask and measure it. A match without one is measured by its box."""
    if m.mask is not None:
        grid = m.mask.full_frame()
        ys, xs = np.nonzero(grid)
        if len(xs):
            w, h = m.mask.frame_w, m.mask.frame_h
            return Shape(float(xs.mean() + 0.5) / w, float(ys.mean() + 0.5) / h,
                         len(xs) / (w * h), int(xs.min()) / w, int(ys.min()) / h,
                         (int(xs.max()) + 1) / w, (int(ys.max()) + 1) / h)
    x1, y1, x2, y2 = m.box or (m.cx, m.cy, m.cx, m.cy)
    return Shape(m.cx, m.cy, m.area, x1, y1, x2, y2)


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
    # "3 doors", but never "2 oil rig": a phrase asked in the singular isn't repeated.
    head = f"{n} {phrase}" if phrase.endswith("s") else f"I see {n}"
    if n <= 3:
        return f"{head}: " + ", ".join(f"one {where_words(m)}" for m in ordered) + "."
    thirds = [sum(1 for m in ordered if lo <= m.cx < hi)
              for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))]
    parts = [f"{k} {where}" for k, where in zip(thirds, ("on the left", "in the middle",
                                                        "on the right"), strict=True) if k]
    return f"{head}: " + ", ".join(parts) + "."


def from_you(m: Match, me: tuple[float, float], step: tuple[float, float]) -> str:
    """Where a match is from the player's character, in steps (a character's width):
    "3 steps up and 2 to your left". `me` and `step` are fractions of the screen."""
    dx, dy = (m.cx - me[0]) / step[0], (m.cy - me[1]) / step[1]
    x, y = round(dx), round(dy)
    if abs(dx) < 1.2 and abs(dy) < 1.2:
        return "right next to you"
    parts = []
    if y:
        parts.append(f"{abs(y)} {'step' if abs(y) == 1 else 'steps'} {'up' if y < 0 else 'down'}")
    if x:
        side = f"to your {'left' if x < 0 else 'right'}"
        parts.append(f"{abs(x)} {side}" if parts else
                     f"{abs(x)} {'step' if abs(x) == 1 else 'steps'} {side}")
    return " and ".join(parts)


def describe_from(phrase: str, matches: list[Match], me: tuple[float, float],
                  step: tuple[float, float]) -> str:
    """describe(), but from where the player's character stands, nearest first."""
    if not matches:
        return f"I can't find any {phrase} on screen."
    ordered = sorted(matches, key=lambda m: math.hypot((m.cx - me[0]) / step[0],
                                                       (m.cy - me[1]) / step[1]))
    if len(ordered) == 1:
        return f"Found it: {from_you(ordered[0], me, step)}."
    near = [from_you(m, me, step) for m in ordered[:3]]
    more = f", and {len(ordered) - 3} more" if len(ordered) > 3 else ""
    return (f"{len(ordered)} {phrase}. The nearest is {near[0]}"
            + "".join(f"; then {w}" for w in near[1:]) + more + ".")


def pan_from(m: Match, me: tuple[float, float]) -> float:
    """Left or right of the player's character, for a sound: -1 to 1."""
    return round(max(-1.0, min(1.0, (m.cx - me[0]) * 2.5)), 2)


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
    return await segment(muse, img, phrase), time.monotonic() - t


async def segment(muse: Muse, img: Image.Image, phrase: str, side: int = 1280) -> list[Match]:
    """Every match SAM 3.1 finds for one phrase on the frame, with its box and mask. It
    takes one phrase a call: "horse. tree. rock" comes back empty. `side` is the longest
    side of the picture sent, and so of the mask that comes back."""
    part = image_part(img, side)
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
    return parse_boxes(text)


FIND_MAX = 5  # phrases in one find
PING_MAX = 8  # things pinged after one find
SAM_LOOK_S = 8.0  # a phrase SAM hasn't answered by then is left out, so the player isn't kept
# The picture find sends SAM. Masks decode in pure Python, a few microseconds a pixel on one
# core: at 1280 a big tree took over a second, with the player waiting on it. Position and
# area don't need more. (For pictures, scripts/sam_masks.py asks for more on its own.)
FIND_SIDE = 960


@dataclass
class Found:
    """What one phrase turned up on a frame."""

    phrase: str
    matches: list[Match] = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)  # one per match, measured
    error: str = ""


def find_phrases(value) -> list[str]:
    """The model's phrases as it wrote them, one of each."""
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    seen, out = set(), []
    for item in items:
        phrase = str(item).strip()
        if phrase and phrase.lower() not in seen:
            seen.add(phrase.lower())
            out.append(phrase)
    return out


async def find_all(muse: Muse, img: Image.Image, phrases: list[str]) -> list[Found]:
    """SAM 3.1 on one frame for each phrase, all at once (it takes a phrase a call), with
    every mask measured. A phrase that fails is reported, and the others still count."""

    async def one(phrase: str) -> Found:
        try:
            async with asyncio.timeout(SAM_LOOK_S):
                matches = await segment(muse, img, phrase, side=FIND_SIDE)
            # In a thread: a big mask takes a moment, and the game's audio shouldn't wait.
            return Found(phrase, matches, await asyncio.to_thread(lambda: [
                measure(m) for m in matches]))
        except TimeoutError:
            return Found(phrase, error="no answer in time")
        except Exception as exc:  # noqa: BLE001  the other phrases still count
            return Found(phrase, error=str(exc)[:120] or type(exc).__name__)

    found = list(await asyncio.gather(*(one(p) for p in phrases)))
    await asyncio.to_thread(drop_repeats, found)
    return found


def same_thing(a: Match, b: Match) -> bool:
    """Whether two masks outline one thing: nearly the same pixels. A door in a wall's mask
    isn't the wall."""
    if a.mask is None or b.mask is None:
        return False
    x, y = a.mask.full_frame() > 0, b.mask.full_frame() > 0
    if x.shape != y.shape:
        return False
    return (x & y).sum() >= 0.7 * max(1, (x | y).sum())


def drop_repeats(found: list[Found]) -> None:
    """A thing two phrases both outlined ("large rock", "boulder") counts under the first."""
    for i, later in enumerate(found):
        keep = [k for k, m in enumerate(later.matches)
                if not any(same_thing(m, o) for f in found[:i] for o in f.matches)]
        later.matches = [later.matches[k] for k in keep]
        later.shapes = [later.shapes[k] for k in keep]


def screen_words(s: Shape) -> str:
    """Where a thing is on the screen, in words. Nothing about how far away: see found_text."""
    across = ("far left" if s.cx < 0.2 else "left" if s.cx < 0.4 else "middle" if s.cx < 0.6
              else "right" if s.cx < 0.8 else "far right")
    down = "top" if s.cy < 1 / 3 else "bottom" if s.cy > 2 / 3 else "middle"
    return f"{across} of the screen, {down} height" if across != "middle" or down != "middle" \
        else "the middle of the screen"


def found_text(found: list[Found]) -> str:
    """find's result for the model: what's outlined, and where on the screen."""
    lines = [("Outlined by SAM on one frame. Each is measured by its pixels: x runs 0 to 100 "
              "left to right, y 0 to 100 top to bottom, and area is the share of the screen it "
              "covers. These say where it is on screen and how big it looks there, not how far "
              "away it is.")]
    for f in found:
        if f.error:
            lines.append(f"{f.phrase}: couldn't look ({f.error})")
            continue
        if not f.shapes:
            lines.append(f"{f.phrase}: none found")
            continue
        biggest = sorted(f.shapes, key=lambda s: -s.area)
        each = [f"{screen_words(s)}, centre x {s.cx * 100:.0f} y {s.cy * 100:.0f}, spans x "
                f"{s.x1 * 100:.0f}-{s.x2 * 100:.0f} y {s.y1 * 100:.0f}-{s.y2 * 100:.0f}, area "
                f"{s.area * 100:.1f}%" for s in biggest[:6]]
        more = f"; and {len(biggest) - 6} smaller" if len(biggest) > 6 else ""
        lines.append(f"{f.phrase}: {len(biggest)} found, biggest first: " + "; ".join(each) + more)
    return "\n".join(lines)


# Things there's some of, not one of.
MASS = re.compile(r"\b(grass|water|sand|mud|snow|dirt|smoke|lava|ice|fog|gravel|hay)$",
                  re.IGNORECASE)


def _plural(phrase: str) -> str:
    for one, many in (("person", "people"), ("woman", "women"), ("man", "men")):
        if phrase.endswith(one):
            return phrase[:-len(one)] + many
    if phrase.endswith("s") and not phrase.endswith(("ss", "us")):
        return phrase
    if phrase.endswith(("ss", "us", "sh", "ch", "x")):
        return phrase + "es"
    if phrase.endswith("y") and phrase[-2:-1] not in "aeiou":
        return phrase[:-1] + "ies"
    return phrase + "s"


def found_line(found: list[Found]) -> str:
    """What was outlined, in a sentence, in the order the pings come. SAM's counts run high on
    small things, so more than three is "several"."""
    parts = []
    for f in found:
        n = len(f.matches)
        name = re.sub(r"^(a|an|the|some)\s+", "", f.phrase, flags=re.IGNORECASE)
        if n and MASS.search(name):
            parts.append(f"some {name}")
        elif n == 1:
            parts.append(f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}")
        elif n > 1:
            parts.append(f"{ {2: 'two', 3: 'three'}.get(n, 'several') } {_plural(name)}")
    if not parts:
        return "I can't pick out any of those on screen."
    return "I can see " + (parts[0] if len(parts) == 1
                           else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."


def ping_order(found: list[Found], limit: int = PING_MAX) -> list[tuple[str, float]]:
    """(phrase, pan) to ping, a phrase at a time and left to right within it. Past `limit`
    the biggest of each phrase take turns, so every phrase found gets heard."""
    queues = [sorted(f.shapes, key=lambda s: -s.area) for f in found]
    picked: list[list[Shape]] = [[] for _ in found]
    left = limit
    while left > 0 and any(queues):
        for queue, chosen in zip(queues, picked, strict=True):
            if queue and left > 0:
                chosen.append(queue.pop(0))
                left -= 1
    return [(f.phrase, s.pan) for f, chosen in zip(found, picked, strict=True)
            for s in sorted(chosen, key=lambda s: s.cx)]


def found_record(found: list[Found], folder: Path | None = None, stem: str = "") -> list[dict]:
    """find's results for a log, the masks as PNGs of their boxes in `folder` when given,
    so a log line stays a log line."""
    out = []
    for f in found:
        items = []
        for i, (m, s) in enumerate(zip(f.matches, f.shapes, strict=True)):
            item = {"box": [round(v, 4) for v in m.box or ()], "centre": [round(s.cx, 4),
                    round(s.cy, 4)], "pan": s.pan, "area": round(s.area, 5)}
            if m.mask is not None:
                item["mask_size"] = [m.mask.height, m.mask.width]
                item["frame_size"] = [m.mask.frame_h, m.mask.frame_w]
                item["mask_at"] = [m.mask.left, m.mask.top]
                if folder is not None:
                    slug = re.sub(r"[^a-z0-9]+", "-", f.phrase.lower()).strip("-") or "thing"
                    name = f"{stem}{slug}-{i + 1}.png"
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(m.mask.raster() * 255).convert("1").save(folder / name)
                    item["mask"] = name
            items.append(item)
        out.append({"phrase": f.phrase, "count": len(f.matches), "error": f.error,
                    "items": items})
    return out


WHERE = re.compile(
    r"^(?:hey,? )?(?:where(?:'s| is| are)|find|how many)\s+(?:the |a |an |my |all )?"
    r"(?P<what>[\w' -]{2,40}?)\??$", re.IGNORECASE)
# Words that make it a request to do something ("find a potion and use it"), which is the
# agent's job, not a question about where things are on screen.
ACTION_WORDS = {"and", "then", "use", "open", "go", "attack", "heal", "buy", "sell", "equip",
                "talk", "pick", "give", "cast", "hit", "to", "with", "for", "me", "out"}


OPEN_REQUEST = re.compile(
    r"^(?:hey,? |so,? |okay,? |ok,? )?(?:what|what's|whats|anything|is there|are there|is anyone"
    r"|is anybody|can i|could i|where can|where could)\b.*\b(?:around|nearby|near|cover|hide"
    r"|behind|ahead|out there|in front|next to|beside|see from|see here)\b", re.IGNORECASE)


def open_request(text: str) -> bool:
    """An open question about what's in view ("what's around me?"), for find, where the
    model's first move is plain and shouldn't take long."""
    return bool(OPEN_REQUEST.match(text.strip().replace("\u2019", "'")))


def where_request(text: str) -> str | None:
    """The noun phrase in "where's the door?", or None if it isn't that kind of question."""
    # Speech to text ends a phrase how it likes ("where's my horse,") and may curl the "'s".
    text = text.strip().replace("\u2019", "'").rstrip(".!,;: ")
    m = WHERE.match(text)
    if not m:
        return None
    # "How many enemies are there" is about the screen; "how many potions do I have"
    # needs the menus.
    if text.lower().startswith("how many") and not re.search(
            r"\b(on (the )?screen|are there|here|do you see|can you see)\b", text, re.IGNORECASE):
        return None
    what = re.sub(r"\b(on (the )?screen|here|now|right now|at|are there|is there|do you see"
                  r"|can you see|there)\b", "", m.group("what"), flags=re.IGNORECASE)
    what = re.sub(r"\s+(are|is)$", "", what.strip(" ,"), flags=re.IGNORECASE)
    words = re.findall(r"[a-z']+", what.lower())
    if set(words) & ACTION_WORDS:
        return None
    # "Where are we?", "where is it going?": about the game, not something to point at.
    if words and words[0] in {"we", "i", "us", "you", "it", "they", "this", "that"}:
        return None
    return what or None


EXITS = """This is a 2D game screen seen from above; the player's character is on it. List \
every way out of this place you can see: doors and doorways, stairs, ladders, paths or \
gaps that lead off the edge of the screen, openings in fences, hedges or walls. Also any \
screen edge where open ground (floor, grass, a plaza, a path) runs right up to it with no \
wall or trees along it: the place goes on that way ("plaza floor running off the left \
edge"). Only ways a character could walk through: not windows, not pictures. Answer JSON: {"exits": \
[{"label": "a few words, e.g. 'stairs going up at the top middle'", "x": 0-1000, "y": \
0-1000, "leads": "up, down, left, right or in"}], "player": [x, y] or null}"""

OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass
class Exit:
    label: str
    at: Match
    leads: str


async def find_exits(muse: Muse, img: Image.Image
                     ) -> tuple[list[Exit], tuple[float, float] | None]:
    """The ways out of the place on screen, and where the player's character is (both as
    fractions of the screen), by Muse Spark."""
    data = await muse.json([{"role": "user", "content": [
        image_part(img, 1280), {"type": "text", "text": EXITS}]}], effort="minimal")
    out = []
    for e in data.get("exits") or []:
        try:
            x, y = float(e["x"]) / 1000, float(e["y"]) / 1000
        except (TypeError, ValueError, KeyError):
            continue
        leads = str(e.get("leads") or "").lower()
        out.append(Exit(str(e.get("label") or "a way out")[:80],
                        Match(min(max(x, 0), 1), min(max(y, 0), 1), 0.0),
                        leads if leads in (*OPPOSITE, "in") else "in"))
    me = None
    try:
        p = data.get("player")
        me = (float(p[0]) / 1000, float(p[1]) / 1000)
    except (TypeError, ValueError, IndexError):
        pass
    return out, me


def exits_text(exits: list[Exit], me: tuple[float, float] | None,
               step: tuple[float, float], came: str = "") -> str:
    """The exits for the agent, nearest first, from the character, the way it came in
    (the nearest exit leading back the way it was walking when it arrived) marked."""
    if not exits:
        return "no ways out that I can see on this screen"
    if me is not None:
        exits = sorted(exits, key=lambda e: math.hypot((e.at.cx - me[0]) / step[0],
                                                       (e.at.cy - me[1]) / step[1]))
    back = OPPOSITE.get(came)
    came_in = next((e for e in exits if back and e.leads == back), None)
    lines = []
    for e in exits:
        where = from_you(e.at, me, step) if me is not None else where_words(e.at)
        mark = " (probably the way you came in)" if e is came_in else ""
        lines.append(f"- {e.label}, leading {e.leads}: {where}{mark}")
    return "Ways out on this screen, nearest first:\n" + "\n".join(lines)
