"""The bump-and-replan grid planner, without a game."""

from itertools import pairwise

import numpy as np

from playertwo.nav import Detour, Grid, astar, corners


def test_astar_goes_around_what_is_known_and_through_what_isnt():
    box = (-10, -10, 20, 20)
    # Nothing known: straight, the longer way first, one turn.
    path = astar((0, 0), (3, 5), set(), box, first=(0, 1))
    assert path[0] == (0, 0) and path[-1] == (3, 5) and len(path) == 9
    assert corners(path) == [((0, 5), (0, 1))]
    # A railing below, open on the left: down the left side, then right.
    railing = {(x, 2) for x in range(6)} | {(6, y) for y in range(2, 6)}
    path = astar((1, 0), (3, 4), railing, box, first=(0, 1))
    assert path is not None and not railing & set(path)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairwise(path))
    assert min(x for x, _ in path) == -1  # round the left end
    # The goal may be solid (a chest): it's still where the path ends.
    assert astar((0, 0), (2, 0), {(2, 0)}, box)[-1] == (2, 0)


def test_astar_gives_up_when_cut_off_or_out_of_its_box():
    box = (-3, -3, 3, 3)
    pen = {(x, y) for x in range(-1, 2) for y in range(-1, 2)} - {(0, 0)}
    assert astar((0, 0), (3, 0), pen, box) is None
    assert astar((0, 0), (3, 0), set(), (1, -3, 5, 3)) is None  # starts outside
    # A wall right across the box: the way round would leave it.
    wall = {(1, y) for y in range(-3, 4)}
    assert astar((0, 0), (3, 0), wall, box) is None


def test_grid_cells_stay_on_the_world_as_the_screen_scrolls():
    g = Grid(10, (100, 50))
    assert g.at((100, 50)) == (0, 0) and g.at((106, 50)) == (1, 0)
    g.shift(np.array([-30.0, 0.0]))  # the view moved right: the world goes left
    assert g.at((70, 50)) == (0, 0)
    assert np.allclose(g.middle((1, 0)), (80, 50))


def test_a_detour_learns_from_bumps_and_steers_four_ways():
    d = Detour(10, (0, 0))
    goal = (40, 0)
    assert d.bump((0, 0), 135) is None  # a diagonal says nothing about which cell
    assert d.bump((0, 0), 90) == (1, 0)
    assert d.plan(np.array([0.0, 0.0]), goal) and d.corners
    first = d.corner(np.array([0.0, 0.0]))
    assert first in (0.0, 180.0)  # up or down to get round it, not into it
    # Stuck again the way it went: that's known too, and the next plan goes elsewhere.
    d.bump((0, 0), first)
    assert d.plan(np.array([0.0, 0.0]), goal)
    assert d.corner(np.array([0.0, 0.0])) == (180.0 if first == 0.0 else 0.0)
    # Past the last corner, straight for the aim; the other way if that's known blocked.
    d = Detour(10, (0, 0))
    assert d.toward(np.array([0.0, 0.0]), (0, 40)) == 180.0
    d.grid.blocked.add((0, 1))
    assert d.toward(np.array([0.0, 0.0]), (5, 40)) == 90.0
