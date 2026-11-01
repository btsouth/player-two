"""Rooms: walking where the survey's tiles and the game's own ground differ (the feet
reach the edge of a tile they can't stand on: the top of a fence's row, the side of a
stall), and the party following the player (it never blocks the way)."""

import json

from test_rooms import Sim, make

from playertwo.rooms import Rooms, Target, centre, on_tile, walk_out


async def _true():
    return True


def fenced(tmp_path, fence=True):
    """A place 512x320: rows 2-17, columns 2-29. A fence along row 9, columns 10-16 (a
    gap at column 9), not walked in the survey when `fence`; someone at (240, 116)
    (row 7, column 15), their own tile not walked."""
    walk = [[r, c] for r in range(2, 18) for c in range(2, 30)
            if (r, c) != (7, 15) and not (fence and r == 9 and 10 <= c <= 16)]
    img = make(tmp_path, 2, (512, 320), walk, [{"from": [17, 5], "dir": [0, 1], "to": 496}], 4)
    d = json.loads((tmp_path / "loc-002.json").read_text())
    d["people"] = [{"obj": 8, "at": [240.0, 116.0], "name": "the item seller",
                    "kind": "person"}]
    (tmp_path / "loc-002.json").write_text(json.dumps(d))
    (tmp_path / "names.json").write_text(json.dumps({
        "names": {"2": "the fair", "496": "outside (the world map)"}, "worlds": [496]}))
    return Rooms(tmp_path), img


class Ground(Sim):
    """A Sim whose ground isn't whole tiles: `ok(x, y)` says where the feet may be."""

    def __init__(self, img, ok, x, y):
        super().__init__(img, set(), x, y)
        self.ok = ok

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
            if self.ok(nx, ny):
                self.x, self.y = nx, ny


def test_feet_on_a_fences_edge_count_on_the_near_side(tmp_path):
    rooms, _ = fenced(tmp_path)
    room = rooms.rooms[2]
    # Feet 6 px into the fence's row from above (the game lets them, by the seller):
    # the tile above it, not the one beyond the fence, though both are a row away.
    assert (9, 15) not in room.walk
    assert on_tile(room, 248.0, 150.0) == (8, 15)
    assert on_tile(room, 248.0, 150.0, last=(10, 15)) == (8, 15)
    # Halfway (as near either): the one it was on.
    assert on_tile(room, 248.0, 157.0, last=(10, 15)) == (10, 15)
    assert on_tile(room, 248.0, 157.0, last=(8, 15)) == (8, 15)
    # On a walked tile: that one; far off the survey: the nearest by rows and columns.
    assert on_tile(room, 100.0, 100.0) == (6, 6)
    assert on_tile(room, 8.0, 100.0) == (6, 2)
    assert centre((8, 15)) == (248.0, 141.0)


async def test_walk_from_a_fences_edge_goes_the_near_way(tmp_path):
    """Live (take A2, the fair): the feet on the edge of the fence's row in front of the
    seller were taken for the tile below the fence, and the walk pushed into the fence
    for 6 s. Starting there, it walks straight to the seller and talks."""
    rooms, img = fenced(tmp_path)
    room = rooms.rooms[2]

    def ok(x, y):
        t = (int(y) // 16, int(x) // 16)
        if t[0] == 9 and 10 <= t[1] <= 16:  # the fence: its top 8 px from above
            return y - 144 < 8
        return t in room.walk

    sim = Ground(img, ok, 200.0, 150.0)  # row 9, column 12: on the fence's edge
    target = Target(2, (240.0, 116.0), True, "the item seller")
    ends = rooms.stand_by(room, target.at, True)
    sim.answers = lambda x, y, f: ends.get((int(y) // 16, int(x) // 16)) == f
    notes = []
    out = await walk_out(rooms, 2, sim.shoot, sim.steer, _true, notes.append, 20.0,
                         spot=target, press=sim.press, talk=True)
    assert out.startswith("arrived") and "pressed a" in out, out
    assert not [n for n in notes if "stuck" in n or "no nearer" in n], notes
    assert sim.y < 144  # never went round below the fence


async def test_a_spot_by_someone_that_cant_be_stood_on_is_given_up(tmp_path):
    """Live (the fair, 63 s): the survey walked the edge of the tile beside the item
    seller (their stall's side), the walk pushed into it 15 s (a stand tile was never
    found shut), then pressed a from the wrong spot. Now it's shut like any other and
    the next best is used."""
    rooms, img = fenced(tmp_path, fence=False)
    room = rooms.rooms[2]
    side = (7, 16)  # right of the seller: walked in the survey, can't be stood on
    assert side in room.walk

    def ok(x, y):
        t = (int(y) // 16, int(x) // 16)
        return t in room.walk and t != side

    sim = Ground(img, ok, 440.0, 125.0)  # row 7, column 27: coming from the right
    target = Target(2, (240.0, 116.0), True, "the item seller")
    ends = rooms.stand_by(room, target.at, True)
    assert side in ends
    sim.answers = lambda x, y, f: (int(y) // 16, int(x) // 16) != side and \
        ends.get((int(y) // 16, int(x) // 16)) == f
    notes = []
    out = await walk_out(rooms, 2, sim.shoot, sim.steer, _true, notes.append, 25.0,
                         spot=target, press=sim.press, talk=True)
    assert out.startswith("arrived") and "pressed a" in out, out
    assert side in room.learned
    _, x, y, facing = sim.pressed[-1]
    assert ends[(int(y) // 16, int(x) // 16)] == facing


def test_to_view():
    from playertwo.rooms import to_view

    follow = (128, 138)
    assert to_view((130, 140), follow, (True, True)) == (130, 140)
    assert to_view((128, 154), follow, (True, True)) == (128, 138)  # 16 px behind: Marle
    assert to_view((128, 154), follow, (True, False)) == (128, 154)  # the view stopped
    assert to_view((106, 160), follow, (True, False)) == (128, 160)
    assert to_view((60, 20), follow, (False, False)) == (60, 20)


async def test_the_follower_over_the_player_isnt_taken_for_it(tmp_path):
    """Live (take A2, the fair) and on the SNES with RAM: Marle standing close behind
    or drawn over Crono was taken for him, 10 to 22 px (a tile) off; the walk then
    headed the wrong way or locked onto her. While the view follows, the player is
    where it keeps it."""
    import numpy as np
    from PIL import Image

    rooms, img = fenced(tmp_path, fence=False)
    sim = Sim(img, rooms.rooms[2].walk, 250.0, 180.0)
    frame = np.asarray(await sim.shoot()).copy()
    # The view follows (cam 122, 42 is inside the place's range); the player's feet at
    # (128, 138) on the screen. Someone in other colours drawn over its lower half,
    # their feet 16 px lower (walking up into a wall, she comes up behind).
    frame[126:154, 121:135] = (250, 200, 40)
    here = rooms.locate(Image.fromarray(frame))
    assert here is not None and here.cam == (122, 42)
    assert abs(here.x - 250) <= 3 and abs(here.y - 180) <= 3, (here.x, here.y)
