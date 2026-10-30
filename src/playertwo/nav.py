"""Finding a way around things in a 2D top-down game, by bumping into them.

Image math can't tell what's solid: a rug and a railing look alike to it, and asking
Muse Spark for a way around takes seconds and is often wrong about the side. So the
walker learns the room the way a blind player would: where the character stops against
something, the cell ahead of it is marked blocked. A* over character-sized cells then
plans the shortest way to the aim through every cell not known to be blocked (unknown
is free), and the walker follows its corners, in the four directions, until the next
bump.

Cells are fixed to the world: the grid's origin moves with the measured scroll, the
same way the walker's points do, so a cell marked once stays on the same spot.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

Cell = tuple[int, int]

# Stick angles (0 up, 90 right) for the four ways a cell steps.
STEPS: dict[Cell, float] = {(0, -1): 0.0, (1, 0): 90.0, (0, 1): 180.0, (-1, 0): 270.0}


def major(v) -> Cell:
    """The one of the four ways closest to v (the longer of its two parts)."""
    dx, dy = float(v[0]), float(v[1])
    if abs(dx) >= abs(dy):
        return (1, 0) if dx >= 0 else (-1, 0)
    return (0, 1) if dy > 0 else (0, -1)


def astar(start: Cell, goal: Cell, blocked: set[Cell],
          box: tuple[int, int, int, int], first: Cell | None = None,
          turn: float = 0.3, limit: int = 6000,
          extra: dict[Cell, float] | None = None) -> list[Cell] | None:
    """The shortest 4-way path from start to goal (both included), or None. Cells in
    `blocked` can't be walked; the goal always can (it's often solid: a chest, a
    person). Only cells inside `box` (x0, y0, x1, y1, inclusive) are searched. Each
    turn costs a little, so of equally long paths the one with fewest corners wins,
    and going `first` way at the start is free of that cost. `extra` costs more to
    step onto: cells probably blocked, not known to be."""
    extra = extra or {}
    x0, y0, x1, y1 = box
    if not (x0 <= start[0] <= x1 and y0 <= start[1] <= y1):
        return None

    def h(c: Cell) -> int:
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    tie = 0
    heap: list = [(h(start), 0.0, tie, start, first)]
    cost = {(start, first): 0.0}
    came: dict = {}
    seen = 0
    while heap:
        _f, back, _t, c, d = heapq.heappop(heap)
        g = -back
        if c == goal:
            path, key = [c], (c, d)
            while key in came:
                key = came[key]
                path.append(key[0])
            return path[::-1]
        if g > cost.get((c, d), math.inf):
            continue
        seen += 1
        if seen > limit:
            return None
        for step in STEPS:
            n = (c[0] + step[0], c[1] + step[1])
            if not (x0 <= n[0] <= x1 and y0 <= n[1] <= y1):
                continue
            if n in blocked and n != goal:
                continue
            ng = g + 1 + extra.get(n, 0.0) + (turn if d is not None and step != d else 0.0)
            if ng < cost.get((n, step), math.inf):
                cost[(n, step)] = ng
                came[(n, step)] = (c, d)
                tie += 1
                # Ties go to the one further along, then the older.
                heapq.heappush(heap, (ng + h(n), -ng, tie, n, step))
    return None


def corners(path: list[Cell]) -> list[tuple[Cell, Cell]]:
    """Where a path turns: each corner cell with the way the path came into it."""
    out = []
    for a, b, c in zip(path, path[1:], path[2:], strict=False):
        come = (b[0] - a[0], b[1] - a[1])
        if (c[0] - b[0], c[1] - b[1]) != come:
            out.append((b, come))
    return out


class Grid:
    """Character-sized cells over the world, in screen pixels: what's known blocked."""

    def __init__(self, cell: float, at) -> None:
        self.cell = float(cell)
        # Cell (0, 0) has its middle at `at`, where the character stands.
        self.origin = np.array(at, float) - self.cell / 2
        self.blocked: set[Cell] = set()
        # Next to a blocked cell, across the way it was walked into: probably the same
        # wall or railing going on, so a plan tries a cell further along first rather
        # than bumping along it a cell at a time. Only next to it: costlier further
        # out, plans swing wide and come back to fill in what they skipped.
        self.likely: dict[Cell, float] = {}

    def block(self, c: Cell, step: Cell) -> None:
        self.blocked.add(c)
        across = (step[1], step[0])
        for s in (1, -1):
            self.likely[(c[0] + s * across[0], c[1] + s * across[1])] = 8.0

    def shift(self, scroll) -> None:
        self.origin += scroll

    def at(self, p) -> Cell:
        q = (np.asarray(p, float) - self.origin) / self.cell
        return (math.floor(q[0]), math.floor(q[1]))

    def middle(self, c: Cell) -> np.ndarray:
        return self.origin + (np.array(c, float) + 0.5) * self.cell

    def path(self, p, q, margin: int = 6) -> list[Cell] | None:
        """The way from screen point p to q, cell by cell, or None. The search keeps
        within a few cells of where the detour began (cell 0, 0) and the aim: further
        off is another room's worth of guessing, better asked about, and a box that
        moved with the character would let it follow a wall forever."""
        a, b = self.at(p), self.at(q)
        xs, ys = (0, b[0]), (0, b[1])
        box = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        return astar(a, b, self.blocked - {a}, box, major(np.subtract(q, p)),
                     extra=self.likely)


class Detour:
    """A way around, planned on a Grid and followed corner to corner, one of the four
    ways at a time: diagonals cut corners, and a bump going straight says which cell
    is solid."""

    def __init__(self, cell: float, at) -> None:
        self.grid = Grid(cell, at)
        self.corners: list[tuple[Cell, Cell]] = []

    def shift(self, scroll) -> None:
        self.grid.shift(scroll)

    def bump(self, player, stick: float | None) -> Cell | None:
        """Stuck going `stick`: the cell ahead is solid. A diagonal says nothing about
        which cell, so only the next plan's straight ways will find out."""
        for step, angle in STEPS.items():
            if stick is not None and abs((stick - angle + 180) % 360 - 180) < 1:
                here = self.grid.at(player)
                ahead = (here[0] + step[0], here[1] + step[1])
                self.grid.block(ahead, step)
                return ahead
        return None

    def plan(self, player, aim) -> bool:
        path = self.grid.path(player, aim)
        self.corners = corners(path) if path is not None else []
        return path is not None

    def corner(self, player) -> float | None:
        """The way to go for the next corner, or None once past the last one (then
        `toward` the aim). Off the line between corners by more than a third of a
        cell, it steps back onto it first, or it clips what the plan goes past."""
        cell = self.grid.cell
        while self.corners:
            c, leg = self.corners[0]
            off = self.grid.middle(c) - player
            along = float(off[0] * leg[0] + off[1] * leg[1])
            across = off - along * np.array(leg, float)
            if along < 0.35 * cell and np.hypot(*across) < 0.5 * cell:
                self.corners.pop(0)
                continue
            if np.hypot(*across) > 0.35 * cell and not self._known(player, major(across)):
                return STEPS[major(across)]
            return STEPS[leg]
        return None

    def toward(self, player, aim) -> float:
        """Straight for the aim, the longer way first, or the other if that's known
        blocked."""
        off = np.subtract(aim, player)
        d = major(off)
        other = (0, 1 if off[1] > 0 else -1) if d[1] == 0 else (1 if off[0] > 0 else -1, 0)
        if self._known(player, d, aim) and not self._known(player, other, aim):
            d = other
        return STEPS[d]

    def _known(self, player, d: Cell, aim=None) -> bool:
        """Whether the cell next to the character's, way d, is known blocked."""
        here = self.grid.at(player)
        n = (here[0] + d[0], here[1] + d[1])
        return n in self.grid.blocked and (aim is None or n != self.grid.at(aim))
