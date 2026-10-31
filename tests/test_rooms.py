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
