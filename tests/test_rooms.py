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

    def __init__(self, img, walk, x, y):
        self.img, self.walk, self.x, self.y = img, walk, x, y
        self.angle = None
        self.t = None
        self.pressed = []
        self.facing = None

    def update(self):
        import time

        from playertwo.controls import steer_dirs

        now = time.monotonic()
        dt = 0.0 if self.t is None else min(0.1, now - self.t)
        self.t = now
        ds = steer_dirs(self.angle)
        dx = ("right" in ds) - ("left" in ds)
        dy = ("down" in ds) - ("up" in ds)
        if dx or dy:
            self.facing = (dx, dy)
        for mx, my in ((dx, 0), (0, dy)):
            nx, ny = self.x + mx * 160 * dt, self.y + my * 160 * dt
            if (int(ny) // 16, int(nx) // 16) in self.walk:
                self.x, self.y = nx, ny

    def steer(self, angle, strength=1.0):
        self.update()
        self.angle = angle

    def press(self, button, hold_ms=100):
        self.pressed.append((button, self.x, self.y, self.facing))

    async def shoot(self):
        self.update()
        h, w = self.img.shape[:2]
        cx = int(min(max(0, self.x - 128), w - 256))
        cy = int(min(max(0, self.y - 138), h - 224))
        f = self.img[cy:cy + 224, cx:cx + 256].copy()
        px, py = int(self.x) - cx, int(self.y) - cy
        f[max(0, py - 24):py, max(0, px - 6):px + 6] = (255, 0, 255)
        return Image.fromarray(f)


async def test_walk_to_talk_across_a_counter(tmp_path):
    from playertwo.rooms import Target, walk_out

    rooms, img = two_floors(tmp_path)
    room = rooms.rooms[2]
    sim = Sim(img, room.walk, 280.0, 173.0)  # row 10, col 17
    target = Target(2, (152.0, 77.9), True, "the innkeeper")
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
    wanted, choices, hints = picks[0]
    # Only people and things for talk_to; each with which way it is.
    assert set(choices) == {"the innkeeper (in here)", "the cat (in here)",
                            "the shopkeeper (the inn, downstairs)"}
    assert all(hints[k] for k in choices)
    # go_to_exit with the player's own words: talks when they say so.
    sim.pressed.clear()
    out = await agent._go_to_exit("talk to the innkeeper")
    assert out.startswith("arrived"), out
    assert sim.pressed and sim.pressed[-1][0] == "a"
