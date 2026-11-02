"""Walk me to it: find a thing by ear, in any game with a view of the world.

Each look finds the thing with SAM, then plays one ping panned to where it truly is on screen,
higher and more often as its box grows, like a parking sensor. Words come only when something
changes. No game knowledge here: a phrase, a screen and the size of a box. A box says how big
a thing looks, not how far it is, so nothing here claims a distance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .locate import Match

REQUEST = re.compile(
    r"^(?:(?:hey|so|okay|ok|please|can you|could you|would you),? )*"
    r"(?:walk|guide|lead) me (?:over )?(?:to|towards?)\s+(?:the |a |an |my |that |this )?"
    r"(?P<what>[\w' -]{2,40}?)(?:,? please)?$", re.IGNORECASE)


def guide_request(text: str) -> str | None:
    """The thing in "walk me to my horse", or None if it isn't that kind of request."""
    text = text.strip().replace("’", "'").rstrip(".!?,;: ")
    m = REQUEST.match(text)
    return m.group("what").strip() if m else None


DRIVE = re.compile(
    r"^(?:(?:hey|so|okay|ok|please|can you|could you|would you),? )*"
    r"(?:take|bring) me (?:over )?to\s+(?:the |a |an |my |that |this )?"
    r"(?P<what>[\w' -]{2,40}?)(?:,? please)?$", re.IGNORECASE)
# The one line a game's notes need for Player Two to walk it there itself.
WALK_NOTE = re.compile(r"(?im)^\s*[-*]?\s*walk\s*:\s*turn with the mouse,\s*walk with (\w)\s*\.?\s*$")


def drive_request(text: str) -> str | None:
    """The thing in "take me to my horse", or None."""
    m = DRIVE.match(text.strip().replace("\u2019", "'").rstrip(".!?,;: "))
    return m.group("what").strip() if m else None


def walk_key(notes: str) -> str | None:
    """The walking key, if the game's notes say how to walk this way, or None."""
    m = WALK_NOTE.search(notes or "")
    return m.group(1).lower() if m else None


LEFT, AHEAD, RIGHT = "Left.", "Straight ahead.", "Right."
CLOSE = "Getting close."
THERE = "You're there."
LOST = "I don't see it. Turn slowly."
MANY = "I see more than one. Going to the biggest."
UNSURE = "Not sure that's the same one."
STOPPED = "Okay, stopped."
START = "Okay. Listen for the ping."
TAKING = "Okay, taking you there. Say stop to stop."
TIMED_OUT = "That's two minutes, and we didn't get there, so I've stopped."
NEVER_SEEN = "That's two minutes, and I never found it, so I've stopped."

# Straight ahead within this much of the middle (-1 to 1 across the screen), and it stays so
# until the thing is this far out: no flipping between words at the edge.
AHEAD_PAN = 0.16
AHEAD_LEAVE = 0.26
LOW_HZ, HIGH_HZ = 660.0, 1760.0
# The box's size, as a fraction of the screen, from far off to right there.
FAR_AREA, CLOSE_AREA, THERE_AREA = 0.002, 0.03, 0.10
THERE_HEIGHT = 0.42  # or this tall: a thing end-on has a narrow box however near
THERE_LOOKS = 2  # looks in a row at that size, since one box can be a fluke
SIDE_LOOKS = 2  # looks on a new side before saying so
# A locked box this much bigger in one look isn't believed until the next look agrees: live,
# one look's box sat on the player's character and said "Getting close." for nothing.
JUMP_AREA = 2.5
# So far off from where it was that it's another one, unless it stays.
OTHER_DIST = 0.7
STUCK_SEEN, STUCK_SPOT, STUCK_CHANGE = 3, 0.03, 0.04
LOST_S = 4.0  # not seen for this long: say so
LOST_AGAIN_S = 15.0
TIMEOUT_S = 120.0


@dataclass
class Step:
    """What one look comes to."""

    target: Match | None
    pan: float | None  # where to ping, -1 to 1; None when it isn't in frame
    hz: float | None
    beeps: int
    wait_s: float  # to the next look, at least
    say: str  # "" when nothing has changed
    side: str
    end: str = ""  # "there" or "timeout"


def size_of(area: float) -> float:
    """0 (far off) to 1 (right there) for a box's share of the screen, by its log: each
    halving of the distance about quadruples the area, so this moves evenly as you walk."""
    if area <= FAR_AREA:
        return 0.0
    return min(1.0, math.log(area / FAR_AREA) / math.log(THERE_AREA / FAR_AREA))


def pick(matches: list[Match], last: Match | None) -> Match | None:
    """Which match to follow: the one nearest where it was, or at the start the biggest,
    which is most likely the nearest. SAM finds the kind of thing, not whose it is."""
    if not matches:
        return None
    if last is None:
        return max(matches, key=lambda m: m.area)
    return min(matches, key=lambda m: distance(m, last))


def distance(m: Match, last: Match) -> float:
    return (abs(m.cx - last.cx) + abs(m.cy - last.cy)
            + abs(math.log(max(m.area, 1e-5) / max(last.area, 1e-5))) * 0.1)


def stuck_to_camera(seen: list[tuple[Match, float]]) -> bool:
    """A box that stays put on screen while the view changes around it is on something the
    camera carries, the player's own character or the game's display, not out in the world.
    `seen`: the looks' boxes in this spot, each with how much the picture changed."""
    return (len(seen) >= STUCK_SEEN
            and sum(c for _, c in seen[-STUCK_SEEN:]) / STUCK_SEEN >= STUCK_CHANGE
            and all(abs(m.cx - seen[-1][0].cx) < STUCK_SPOT
                    and abs(m.cy - seen[-1][0].cy) < STUCK_SPOT
                    for m, _ in seen[-STUCK_SEEN:]))


def picture_change(a, b) -> float:
    """How different two small greyscale frames are, 0 to 1."""
    if a is None or b is None:
        return 0.0
    import numpy as np

    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean()
                 / 255.0)


class Guide:
    """The words and pings for a run of looks. Time comes in, so this can be tested."""

    def __init__(self, started: float, timeout_s: float = TIMEOUT_S):
        self.started = started
        self.timeout_s = timeout_s
        self.target: Match | None = None
        self.side = ""  # the side as last said
        self.seen_side = ""  # the side as last seen
        self.side_looks = 0
        self.close_said = False
        self.there_looks = 0
        self.seen_at: float | None = None
        self.lost_said_at: float | None = None
        self.ever_seen = False
        self.many_said = False
        self.doubt: Match | None = None  # a box too big or too far to believe yet
        self.unsure = False
        self.spots: list[list[tuple[Match, float]]] = []  # boxes staying put on screen
        self.camera_spots: list[Match] = []  # those that stayed while the view changed

    def _on_camera(self, matches: list[Match], change: float) -> list[Match]:
        """The matches, less any carried along with the camera (see stuck_to_camera)."""
        keep = []
        for m in matches:
            if any(abs(m.cx - c.cx) < STUCK_SPOT * 2 and abs(m.cy - c.cy) < STUCK_SPOT * 2
                   for c in self.camera_spots):
                continue
            spot = next((s for s in self.spots
                         if abs(s[-1][0].cx - m.cx) < STUCK_SPOT
                         and abs(s[-1][0].cy - m.cy) < STUCK_SPOT), None)
            if spot is None:
                spot = []
                self.spots.append(spot)
            spot.append((m, change))
            if stuck_to_camera(spot):
                self.camera_spots.append(m)
                continue
            keep.append(m)
        self.spots = self.spots[-12:]
        return keep

    def _follow(self, matches: list[Match]) -> tuple[Match | None, str]:
        """The box to follow this look, and anything to say about which it is."""
        target = pick(matches, self.target)
        if target is None or self.target is None:
            self.doubt = None
            return target, ""
        far = distance(target, self.target) > OTHER_DIST
        big = target.area > self.target.area * JUMP_AREA
        if not far and not big:
            self.doubt = None
            return target, ""
        if self.doubt is None or distance(target, self.doubt) > OTHER_DIST:
            self.doubt = target
            return None, ""  # one look more before believing it
        self.doubt = None
        return target, UNSURE if far else ""

    def _side(self, pan: float) -> str:
        limit = AHEAD_LEAVE if self.seen_side == AHEAD else AHEAD_PAN
        if abs(pan) < limit:
            return AHEAD
        return LEFT if pan < 0 else RIGHT

    def look(self, matches: list[Match], now: float, change: float = 0.0) -> Step:
        """One look's matches, and how much the picture changed since the last look. `say`
        isn't taken as said until `said` is called with it."""
        if now - self.started >= self.timeout_s:
            line = TIMED_OUT if self.ever_seen else NEVER_SEEN
            return Step(None, None, None, 0, 0.0, line, self.seen_side, end="timeout")
        matches = self._on_camera(matches, change)
        had = self.target
        target, note = self._follow(matches)
        if target is None and had is not None and self.doubt is not None:
            # Waiting a look on a box hard to believe: no ping, and it isn't lost yet.
            return Step(None, None, None, 0, 0.3, "", "")
        if target is None:
            self.there_looks = 0
            self.target = None  # gone: the biggest again when it's back
            since = now - (self.seen_at if self.seen_at is not None else self.started)
            say = ""
            if since >= LOST_S and (self.lost_said_at is None
                                    or now - self.lost_said_at >= LOST_AGAIN_S):
                say = LOST
            return Step(None, None, None, 0, 0.6, say, "")
        back = self.target is None
        self.target, self.seen_at, self.ever_seen = target, now, True
        pan = target.pan
        side = self._side(pan)
        self.side_looks = self.side_looks + 1 if side == self.seen_side else 1
        self.seen_side = side
        size = size_of(target.area)
        hz = LOW_HZ * (HIGH_HZ / LOW_HZ) ** size
        beeps = 3 if size >= 0.85 else 2 if size >= 0.55 else 1
        tall = target.box is not None and target.box[3] - target.box[1] >= THERE_HEIGHT
        self.there_looks = self.there_looks + 1 if target.area >= THERE_AREA or tall else 0
        wait_s = 1.0 - 0.7 * size
        if self.there_looks >= THERE_LOOKS:
            return Step(target, pan, hz, beeps, 0.0, THERE, side, end="there")
        say = ""
        if len(matches) > 1 and not self.many_said and had is None:
            say = MANY
        elif note:
            say = note
        elif side != self.side and (back or not self.side or self.side_looks >= SIDE_LOOKS):
            say = side
        elif not self.close_said and target.area >= CLOSE_AREA:
            say = CLOSE
        return Step(target, pan, hz, beeps, wait_s, say, side)

    def said(self, line: str, now: float) -> None:
        """That line was really said."""
        if line == MANY:
            self.many_said = True
        elif line in (LEFT, RIGHT, AHEAD):
            self.side = line
        elif line == CLOSE:
            self.close_said = True
        elif line == LOST:
            self.lost_said_at = now
            self.side = ""  # say the side again when it's back


# Walking it for the player: turn toward the thing, walk when it's about ahead. The same few
# rules a player would use by ear. The gain is mouse counts for the width of half the screen,
# low on purpose: it comes right over a few looks whatever the game's mouse speed.
TURN_GAIN = 700
TURN_PAN = 0.22
WALK_PAN = 0.55
SWEEP = 420
STUCK_LOOKS = 5  # looks walking with the box no bigger: something's in the way


@dataclass
class Move:
    turn: int  # mouse counts
    walk: bool  # hold the key
    step_s: float  # or just a step
    around: bool = False  # something's in the way: turn off and walk a little


class Driver:
    """What to do with the keys after each look."""

    def __init__(self):
        self.sizes: list[float] = []

    def after(self, step: Step) -> Move:
        if step.end:
            return Move(0, False, 0.0)
        if step.pan is None:
            self.sizes.clear()
            return Move(SWEEP, False, 0.0)
        turn = round(TURN_GAIN * step.pan) if abs(step.pan) >= TURN_PAN else 0
        near = step.beeps >= 3
        ok = abs(step.pan) < WALK_PAN
        self.sizes.append(step.target.area if step.target else 0.0)
        recent = self.sizes[-STUCK_LOOKS:]
        if ok and len(recent) == STUCK_LOOKS and max(recent) <= recent[0] * 1.15:
            self.sizes.clear()
            return Move(turn, False, 0.0, around=True)
        return Move(turn, ok and not near, 0.35 if ok and near else 0.0)
