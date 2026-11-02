"""Rooms: finding the screen on a surveyed place's picture, and the way out of it."""

import json

import numpy as np
from PIL import Image

from playertwo.rooms import Rooms, centre, pick_words


def make(folder, loc, size, walk, exits, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = np.kron(img[::8, ::8], np.ones((8, 8, 1), np.uint8))[:size[1], :size[0]]
    Image.fromarray(img).save(folder / f"loc-{loc:03x}.png")
    (folder / f"loc-{loc:03x}.json").write_text(json.dumps({
        "loc": loc, "name": "", "tile": 16, "walk": walk, "wall": [], "exits": exits,
        "cams": [[0, 0], [size[0] - 256, size[1] - 224]]}))
    return img


def test_rooms_way_out(tmp_path):
    walk = [[r, c] for r in range(2, 12) for c in range(2, 20)]
    up = make(tmp_path, 2, (512, 320), walk,
              [{"from": [11, 5], "dir": [0, 1], "to": 1}], 1)
    make(tmp_path, 1, (256, 224), [[r, c] for r in range(2, 12) for c in range(2, 14)],
         [{"from": [2, 5], "dir": [0, -1], "to": 2},
          {"from": [11, 8], "dir": [0, 1], "to": 496}], 2)
    (tmp_path / "names.json").write_text(json.dumps({
        "names": {"1": "the house, downstairs", "2": "the house, upstairs",
                  "496": "outside (the world map)"}, "worlds": [496]}))
    rooms = Rooms(tmp_path)
    assert rooms.hops(2, 496) == [2, 1, 496]
    assert rooms.reachable(2) == {"the house, downstairs": (1, 1),
                                  "outside (the world map)": (496, 2)}
    path = rooms.route(rooms.rooms[2], (3, 15), {(11, 5)})
    assert path[0] == (3, 15) and path[-1] == (11, 5)
    assert all(t in rooms.rooms[2].walk for t in path)
    # The screen is a window onto the picture: found there, in the right place.
    view = up[40:264, 100:356]
    here = rooms.locate(Image.fromarray(view))
    assert here is not None and here.loc == 2 and here.cam == (100, 40)
    assert centre((2, 3)) == (56, 45)


def test_pick_words():
    choices = {"outside (the world map)": "outside (the world map)",
               "the house, upstairs": "the house, upstairs"}
    assert pick_words("take me outside", choices) == "outside (the world map)"
    assert pick_words("the house, upstairs", choices) == "the house, upstairs"
    assert pick_words("go find mom", choices) is None
    choices["the innkeeper (in here)"] = "the innkeeper (in here)"
    choices["the cat (the inn, downstairs)"] = "the cat (the inn, downstairs)"
    assert pick_words("talk to the innkeeper", choices) == "the innkeeper (in here)"
    assert pick_words("take me to the cat.", choices) == "the cat (the inn, downstairs)"
    assert pick_words("talk to the old man", choices) is None
    assert pick_words("go upstairs", choices) == "the house, upstairs"
    assert pick_words("go downstairs", choices) is None


def test_world_exits_named_by_their_end(tmp_path):
    # A forest with ways out at both ends: each named by the nearest place but itself.
    walk = [[r, c] for r in range(1, 13) for c in range(2, 14)]
    make(tmp_path, 19, (256, 224), walk,
         [{"from": [1, 5], "dir": [0, -1], "to": 496, "at": [280, 240]},
          {"from": [12, 5], "dir": [0, 1], "to": 496, "at": [280, 280]}], 3)
    (tmp_path / "names.json").write_text(json.dumps({"names": {"19": "Guardia Forest"},
                                                     "worlds": [496]}))

    class World:
        def __init__(self):
            self.world = 496
            self.places = {"Guardia Forest": (280, 268), "Guardia Castle": (300, 205),
                           "Truce Market": (250, 340)}

        def nearest(self, x, y):
            import math
            n = min(self.places, key=lambda k: math.dist(self.places[k], (x, y)))
            return n, 0.0, ""

    class Atlas:
        def __init__(self):
            self.maps = [World()]

    rooms = Rooms(tmp_path, Atlas())
    assert set(rooms.reachable(19)) == {"outside, toward Guardia Castle",
                                        "outside, toward Truce Market",
                                        "outside, the nearest way out"}
    assert pick_words("take me outside", {k: k for k in rooms.reachable(19)}) == \
        "outside, the nearest way out"


def test_places_named_alike_get_plain_names(tmp_path):
    from types import SimpleNamespace
    from typing import ClassVar

    walk = [[r, c] for r in range(2, 12) for c in range(2, 14)]
    for loc in (15, 16, 17, 39):
        make(tmp_path, loc, (256, 224), walk,
             [{"from": [11, 5], "dir": [0, 1], "to": 496, "at": [100, 100]}], loc)
    (tmp_path / "names.json").write_text(json.dumps({"names": {
        "15": "Residence", "16": "Residence", "17": "Market", "39": "Market",
        "12": "Truce Inn", "116": "Truce Inn"}, "worlds": [496]}))

    class Map:
        world = 496
        places: ClassVar = {
            "Truce Residence (west)": (366, 251), "Truce Residence (east)": (424, 252),
            "Truce Market": (348, 283), "Medina Market": (980, 411), "Truce Inn": (426, 304)}
        doors: ClassVar = {(1, 1): 16, (2, 2): 15, (3, 3): 17, (4, 4): 39}
        cells: ClassVar = {
            "Truce Residence (west)": {(1, 1)}, "Truce Residence (east)": {(2, 2)},
            "Truce Market": {(3, 3)}, "Medina Market": {(4, 4)}}

        def entrance(self, name):
            return self.cells.get(name)

        def nearest(self, x, y):
            return "Truce Residence (east)", 0.0, ""

    rooms = Rooms(tmp_path, SimpleNamespace(maps=[Map()]))
    assert rooms.name(15) == "the east house in Truce"
    assert rooms.name(16) == "the west house in Truce"
    assert rooms.name(17) == "Truce Market" and rooms.name(39) == "Medina Market"
    assert rooms.name(12) == "Truce Inn" and rooms.name(116) == "Truce Inn"  # one town
    assert rooms.rooms[15].name == "the east house in Truce"
    assert rooms.rooms[17].exits[0].label == "outside, by the east house in Truce"


def with_people(folder, loc, people):
    d = json.loads((folder / f"loc-{loc:03x}.json").read_text())
    d["people"] = people
    (folder / f"loc-{loc:03x}.json").write_text(json.dumps(d))


def two_floors(tmp_path):
    """Upstairs (2) with an innkeeper behind a counter and a wandering cat; downstairs
    (1) with a shopkeeper; out to the world map (496) from downstairs."""
    # Upstairs: rows 2-11, cols 2-19, but the counter (row 5, cols 8-10) and the
    # innkeeper's tile behind it (row 4, col 9) can't be walked.
    walk = [[r, c] for r in range(2, 12) for c in range(2, 20)
            if not (r == 5 and 8 <= c <= 10) and (r, c) != (4, 9) and (r, c) != (8, 15)]
    up = make(tmp_path, 2, (512, 320), walk, [{"from": [11, 5], "dir": [0, 1], "to": 1}], 1)
    make(tmp_path, 1, (256, 224), [[r, c] for r in range(2, 12) for c in range(2, 14)],
         [{"from": [2, 5], "dir": [0, -1], "to": 2},
          {"from": [11, 8], "dir": [0, 1], "to": 496}], 2)
    with_people(tmp_path, 2, [
        {"obj": 8, "at": [152.0, 77.9], "name": "the innkeeper", "kind": "person"},
        {"obj": 9, "at": [248.0, 141.9], "name": "the cat", "kind": "animal", "moves": True},
        {"obj": 10, "at": [40.0, 40.0], "name": "", "kind": "person"},  # not named: left out
    ])
    with_people(tmp_path, 1, [{"obj": 8, "at": [104.0, 77.9], "name": "the shopkeeper",
                               "kind": "person"}])
    (tmp_path / "names.json").write_text(json.dumps({
        "names": {"1": "the inn, downstairs", "2": "the inn, upstairs",
                  "496": "outside (the world map)"}, "worlds": [496]}))
    return Rooms(tmp_path), up


def test_people_and_where_to_stand(tmp_path):
    from playertwo.rooms import Here, describe

    rooms, _ = two_floors(tmp_path)
    room = rooms.rooms[2]
    assert [p.name for p in room.people] == ["the innkeeper", "the cat"]
    # Across the counter from below (two tiles off, nothing walkable between), or
    # beside them on the left, right and above: each facing them.
    ends = rooms.stand_by(room, (152.0, 77.9), True)
    assert ends == {(6, 9): (0, -1), (3, 9): (0, 1), (4, 8): (1, 0), (4, 10): (-1, 0)}
    # A thing (the counter): the nearest walked tiles, facing it.
    ends = rooms.stand_by(room, (152.0, 88.0), False)
    assert {(4, 8): (1, 0), (4, 10): (-1, 0), (6, 9): (0, -1)}.items() <= ends.items()
    assert len(ends) <= 4
    here = Here(2, (0, 0), 152.0, 173.0, 0.9)  # row 10, col 9: below the counter
    choices, targets, hints = rooms.options(here)
    assert targets["the innkeeper (in here)"].person
    assert targets["the shopkeeper (the inn, downstairs)"].loc == 1
    assert choices["the shopkeeper (the inn, downstairs)"] == (1, 1)
    assert choices["the innkeeper (in here)"] == (2, 0)
    assert hints["the innkeeper (in here)"] == "6 steps north"
    assert hints["the inn, downstairs"].endswith("west") or "south" in \
        hints["the inn, downstairs"]
    assert "way out" in hints["the shopkeeper (the inn, downstairs)"]
    said = describe(rooms, here)
    assert "the innkeeper (6 steps north)" in said and "walks about" in said


class Sim:
    """A game to walk in: the place's picture, the view following the character (a
    block of one colour, its feet at x, y), the d-pad moving it 160 pixels a second
    over the walkable tiles only."""

    def __init__(self, img, walk, x, y, clock=None):
        import time

        self.clock = clock or time.monotonic
        self.img, self.walk, self.x, self.y = img, walk, x, y
        self.angle = None
        self.t = None
        self.pressed = []
        self.facing = None

    def update(self):
        from playertwo.controls import steer_dirs

        now = self.clock()
        dt = 0.0 if self.t is None else min(0.1, now - self.t)
        self.t = now
        ds = steer_dirs(self.angle)
        dx = ("right" in ds) - ("left" in ds)
        dy = ("down" in ds) - ("up" in ds)
        if dx or dy:
            self.facing = (dx, dy)
        speed = 320 if self.dashing else 160
        for mx, my in ((dx, 0), (0, dy)):
            nx, ny = self.x + mx * speed * dt, self.y + my * speed * dt
            if (int(ny) // 16, int(nx) // 16) in self.walk:
                self.x, self.y = nx, ny

    def steer(self, angle, strength=1.0):
        self.update()
        self.angle = angle

    dashing = False
    dashes = 0

    def dash(self, on):
        self.update()
        self.dashes += on and not self.dashing
        self.dashing = on

    def press(self, button, hold_ms=100):
        assert not self.dashing  # the run button let go before any press
        self.pressed.append((button, self.x, self.y, self.facing))
        self.talking = self.talking or (button == "a" and self.answers is not None and
                                        self.answers(self.x, self.y, self.facing))

    talking = False
    answers = None  # (x, y, facing) -> whether someone's there to talk

    async def shoot(self):
        self.update()
        h, w = self.img.shape[:2]
        cx = int(min(max(0, self.x - 128), w - 256))
        cy = int(min(max(0, self.y - 138), h - 224))
        f = self.img[cy:cy + 224, cx:cx + 256].copy()
        px, py = int(self.x) - cx, int(self.y) - cy
        f[max(0, py - 24):py, max(0, px - 6):px + 6] = (255, 0, 255)
        if self.talking:  # a text box over the top of the screen
            f[8:80, 8:248] = (40, 40, 160)
        if self.box_below:  # a scene's text box over the bottom
            f[168:216, 8:248] = (40, 40, 160)
        return Image.fromarray(f)

    box_below = False


async def test_walk_to_talk_across_a_counter(tmp_path):
    from playertwo.rooms import Target, walk_out

    rooms, img = two_floors(tmp_path)
    room = rooms.rooms[2]
    sim = Sim(img, room.walk, 280.0, 173.0)  # row 10, col 17
    target = Target(2, (152.0, 77.9), True, "the innkeeper")
    ends = rooms.stand_by(room, target.at, True)
    sim.answers = lambda x, y, f: ends.get((int(y) // 16, int(x) // 16)) == f
    out = await walk_out(rooms, 2, sim.shoot, sim.steer, lambda: _true(), lambda _t: None,
                         20.0, spot=target, press=sim.press, talk=True)
    assert out.startswith("arrived"), out
    assert len(sim.pressed) == 1
    _, x, y, facing = sim.pressed[0]
    # By the innkeeper (below the counter, beside or above them), lined up, facing them.
    tile = (int(y) // 16, int(x) // 16)
    ends = rooms.stand_by(room, target.at, True)
    assert tile in ends, (x, y)
    assert facing == ends[tile]


async def test_recheck_after_facing_before_confirming_across_a_counter(tmp_path):
    """A facing tap can carry feet beyond the tile seen before it. Don't press A
    from the stale position, or try a second confirm to compensate for that miss."""
    from playertwo.rooms import Target, walk_out

    rooms, img = two_floors(tmp_path)
    room = rooms.rooms[2]
    sim = Sim(img, room.walk, 172.0, 77.0)
    target = Target(2, (152.0, 77.9), True, "the innkeeper")
    ends = rooms.stand_by(room, target.at, True)
    sim.answers = lambda x, y, f: ends.get((int(y) // 16, int(x) // 16)) == f
    original = sim.steer
    slipped = False

    def delayed_stop(angle, strength=1.0):
        nonlocal slipped
        if angle == 270 and not slipped:
            sim.update()
            # Residual motion after the last captured frame, at the counter's edge.
            sim.y = 80.25
            slipped = True
        original(angle, strength)

    out = await walk_out(rooms, 2, sim.shoot, delayed_stop, _true, lambda _t: None,
                         20.0, spot=target, press=sim.press, talk=True)
    assert slipped
    assert "pressed a to talk" in out, out
    assert len(sim.pressed) == 1, sim.pressed
    _, x, y, facing = sim.pressed[0]
    assert ends[(int(y) // 16, int(x) // 16)] == facing


async def test_running_on_the_straight_and_walking_to_line_up(tmp_path):
    """b held on a straight run (twice the speed), let go for turns, lining up and the
    press: the walk still ends lined up across the counter (SNES box: 6/6 trips with
    it, 20 to 26% quicker to the fair's sellers)."""
    from playertwo.rooms import Target, walk_out

    rooms, img = two_floors(tmp_path)
    room = rooms.rooms[2]
    sim = Sim(img, room.walk, 296.0, 77.0)  # row 4, col 18: eight tiles straight across
    target = Target(2, (152.0, 77.9), True, "the innkeeper")
    ends = rooms.stand_by(room, target.at, True)
    sim.answers = lambda x, y, f: ends.get((int(y) // 16, int(x) // 16)) == f
    out = await walk_out(rooms, 2, sim.shoot, sim.steer, lambda: _true(), lambda _t: None,
                         20.0, spot=target, press=sim.press, talk=True, dash=sim.dash)
    assert out.startswith("arrived"), out
    assert sim.dashes >= 1 and not sim.dashing
    _, x, y, facing = sim.pressed[0]
    tile = (int(y) // 16, int(x) // 16)
    assert tile in ends and facing == ends[tile], (x, y)


def test_straight_ahead():
    from playertwo.rooms import straight_ahead

    assert straight_ahead([(5, 5)]) == 0
    assert straight_ahead([(5, 5), (5, 6), (5, 7), (5, 8), (6, 8)]) == 3
    assert straight_ahead([(5, 5), (6, 5), (6, 6)]) == 1


async def _true():
    return True


class SimPad:
    """Controls driving a Sim: the stick steers it, buttons are noted."""

    def __init__(self, sim):
        self.sim = sim
        self.log = []

    def steer(self, angle, strength=1.0):
        self.sim.steer(angle)

    def press(self, button, hold_ms=100):
        self.sim.press(button)
        self.log.append(button)

    def __getattr__(self, name):  # anything else a walk might use: nothing
        return lambda *a, **k: None


async def test_agent_talks_to_someone_by_the_room_map(tmp_path, monkeypatch):
    """talk_to (or go_to_exit) inside a surveyed place goes by its map: the survey's
    person picked, walked to straight, faced and a pressed; spoken as it sets off."""
    from playertwo import rooms as rooms_mod
    from playertwo.agent import Agent
    from playertwo.screen import Frame

    rooms, img = two_floors(tmp_path)
    sim = Sim(img, rooms.rooms[2].walk, 280.0, 173.0)
    ends = rooms.stand_by(rooms.rooms[2], (152.0, 77.9), True)
    sim.answers = lambda x, y, f: ends.get((int(y) // 16, int(x) // 16)) == f
    said = []

    async def grab():
        return Frame(await sim.shoot(), 0.0)

    async def speak(text):
        said.append(text)

    picks = []

    async def pick(muse, game, wanted, here, choices, hints=None):
        picks.append((wanted, dict(choices), dict(hints or {})))
        return "the innkeeper (in here)", "talk: they asked to"

    monkeypatch.setattr(rooms_mod, "pick_exit", pick)
    agent = Agent(None, SimPad(sim), grab, speak, max_settle_s=0.2)
    agent._rooms = rooms
    out = await agent._act("talk_to", {"target": "the innkeeper"})
    assert out.startswith("arrived") and "pressed a" in out, out
    assert said == ["Going to the innkeeper."]
    assert sim.pressed and sim.pressed[-1][0] == "a"
    _, x, y, facing = sim.pressed[-1]
    ends = rooms.stand_by(rooms.rooms[2], (152.0, 77.9), True)
    assert ends.get((int(y) // 16, int(x) // 16)) == facing
    assert not picks  # a name: no model needed (people.resolve)
    # Words that fit no one the survey knows: the model picks.
    sim.pressed.clear()
    sim.talking = False
    out = await agent._act("talk_to", {"target": "the one behind the desk"})
    assert out.startswith("arrived"), out
    _wanted, choices, hints = picks[0]
    # Only people and things for talk_to; each with which way it is.
    assert set(choices) == {"the innkeeper (in here)", "the cat (in here)",
                            "the shopkeeper (the inn, downstairs)"}
    assert all(hints[k] for k in choices)
    # go_to_exit with the player's own words: talks when they say so.
    sim.pressed.clear()
    sim.talking = False
    out = await agent._go_to_exit("talk to the innkeeper")
    assert out.startswith("arrived"), out
    assert sim.pressed and sim.pressed[-1][0] == "a"


async def test_a_text_box_stops_the_walk(tmp_path):
    from playertwo.rooms import walk_out

    rooms, img = two_floors(tmp_path)
    room = rooms.rooms[2]
    sim = Sim(img, room.walk, 280.0, 173.0)
    sim.box_below = True  # the place still shows above it
    here = rooms.locate(await sim.shoot())
    assert here is not None and here.loc == 2
    assert rooms.boxed(await sim.shoot(), here)
    sim.box_below = False
    assert not rooms.boxed(await sim.shoot(), here)
    sim.box_below = True
    out = await walk_out(rooms, 496, sim.shoot, sim.steer, lambda: _true(), lambda _t: None,
                         10.0)
    assert out.startswith("interrupted: a text box"), out


def test_line_up_with_someone_off_the_tiles_middle():
    from playertwo.rooms import line_up

    # A shopkeeper standing on a tile's edge (x 896, between columns 55 and 56): from
    # column 56 below, as near their x as the tile allows, not its middle (904; the
    # game answers only within about 10 px).
    assert line_up((9, 56), (896.0, 118.0), (0, -1)) == 899
    assert line_up((9, 56), (904.0, 118.0), (0, -1)) == 904
    assert line_up((7, 54), (896.0, 118.0), (1, 0)) == 118.0


async def test_a_save_point_is_stood_on_not_pressed_at(tmp_path):
    from playertwo.rooms import Person, Target, walk_out

    rooms, img = two_floors(tmp_path)
    room = rooms.rooms[2]
    room.people.append(Person("the save point", "save point", (232.0, 125.0)))
    sim = Sim(img, room.walk, 72.0, 173.0)
    target = Target(2, (232.0, 125.0), False, "the save point", "save point")
    out = await walk_out(rooms, 2, sim.shoot, sim.steer, lambda: _true(), lambda _t: None,
                         20.0, spot=target, press=sim.press, talk=True)
    assert out.startswith("arrived: standing on the save point"), out
    assert "y opens it" in out and "Don't press a" in out
    assert not sim.pressed  # a on a save point does nothing; beside a chain it climbs it
    assert (int(sim.y) // 16, int(sim.x) // 16) == (7, 14)
    # The survey's kind goes with it as a target.
    with_people(tmp_path, 2, [{"obj": 8, "at": [232.0, 125.0], "name": "the save point",
                               "kind": "save point"}])
    targets = Rooms(tmp_path).targets(2)
    assert targets["the save point (in here)"].kind == "save point"
    assert not targets["the save point (in here)"].person
