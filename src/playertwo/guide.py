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


LEFT, AHEAD, RIGHT = "Left.", "Straight ahead.", "Right."
CLOSE = "Getting close."
THERE = "You're there."
LOST = "I don't see it. Turn slowly."
STOPPED = "Okay, stopped."
START = "Okay. Listen for the ping."
TIMED_OUT = "That's two minutes, and we didn't get there, so I've stopped."
NEVER_SEEN = "That's two minutes, and I never found it, so I've stopped."

# Straight ahead within this much of the middle (-1 to 1 across the screen), and it stays so
# until the thing is this far out: no flipping between words at the edge.
AHEAD_PAN = 0.16
AHEAD_LEAVE = 0.26
LOW_HZ, HIGH_HZ = 660.0, 1760.0
# The box's size, as a fraction of the screen, from far off to right there.
FAR_AREA, CLOSE_AREA, THERE_AREA = 0.002, 0.03, 0.08
THERE_LOOKS = 2  # looks in a row at that size, since one box can be a fluke
SIDE_LOOKS = 2  # looks on a new side before saying so
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
    return min(matches, key=lambda m: abs(m.cx - last.cx) + abs(m.cy - last.cy)
               + abs(math.log(max(m.area, 1e-5) / max(last.area, 1e-5))) * 0.1)


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

    def _side(self, pan: float) -> str:
        limit = AHEAD_LEAVE if self.seen_side == AHEAD else AHEAD_PAN
        if abs(pan) < limit:
            return AHEAD
        return LEFT if pan < 0 else RIGHT

    def look(self, matches: list[Match], now: float) -> Step:
        """One look's matches. `say` isn't taken as said until `said` is called with it."""
        if now - self.started >= self.timeout_s:
            line = TIMED_OUT if self.ever_seen else NEVER_SEEN
            return Step(None, None, None, 0, 0.0, line, self.seen_side, end="timeout")
        target = pick(matches, self.target)
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
        self.there_looks = self.there_looks + 1 if target.area >= THERE_AREA else 0
        wait_s = 1.0 - 0.7 * size
        if self.there_looks >= THERE_LOOKS:
            return Step(target, pan, hz, beeps, 0.0, THERE, side, end="there")
        say = ""
        if side != self.side and (back or not self.side or self.side_looks >= SIDE_LOOKS):
            say = side
        elif not self.close_said and target.area >= CLOSE_AREA:
            say = CLOSE
        return Step(target, pan, hz, beeps, wait_s, say, side)

    def said(self, line: str, now: float) -> None:
        """That line was really said."""
        if line in (LEFT, RIGHT, AHEAD):
            self.side = line
        elif line == CLOSE:
            self.close_said = True
        elif line == LOST:
            self.lost_said_at = now
            self.side = ""  # say the side again when it's back
