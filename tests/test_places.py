from types import SimpleNamespace

import pytest

from playertwo.agent import nearest_of_kind, way_round
from playertwo.atlas import Atlas
from playertwo.rooms import Rooms

M = SimpleNamespace(places={"Truce Inn": (420, 300), "Porre Inn": (410, 830),
                            "Truce Mayor's Manor": (400, 330),
                            "Porre Mayor's Manor": (330, 850), "Leene Square": (424, 214)})


def test_nearest_of_kind_takes_the_one_here():
    assert nearest_of_kind(M, "Porre Inn", "take me to the inn", (440, 280)) == "Truce Inn"
    assert nearest_of_kind(M, "Porre Mayor's Manor", "the mayor's house",
                           (440, 280)) == "Truce Mayor's Manor"


def test_nearest_of_kind_keeps_the_town_named():
    assert nearest_of_kind(M, "Porre Inn", "the inn in Porre", (440, 280)) == "Porre Inn"
    assert nearest_of_kind(M, "Truce Inn", "the inn in Porre", (440, 280)) == "Porre Inn"
    assert nearest_of_kind(M, "Leene Square", "the fair", (440, 900)) == "Leene Square"


def test_find_place_prefers_the_exact_name(tmp_path):
    a = Atlas.__new__(Atlas)
    a.maps = [SimpleNamespace(name="x", places={"Last Village": (0, 0),
                                                "Last Village Hut (west)": (8, 8)})]
    assert a.find_place("Last Village Hut (west)")[1] == "Last Village Hut (west)"
    assert a.find_place("last village")[1] == "Last Village"


def test_find_place_prefers_the_map_the_player_is_on():
    from playertwo.atlas import Atlas

    a = Atlas.__new__(Atlas)

    class M:
        def __init__(self, name):
            self.name, self.places = name, {"Truce Inn": (1, 1)}

    a.maps = [M("1000 A.D."), M("600 A.D.")]
    assert a.find_place("Truce Inn")[0].name == "1000 A.D."
    assert a.find_place("the Truce Inn", "600 A.D.")[0].name == "600 A.D."


def test_a_step_found_not_to_go_is_routed_round_and_kept(tmp_path):
    import numpy as np

    from playertwo.atlas import CELL, Atlas, WorldMap

    m = WorldMap("M", np.zeros((8 * CELL, 8 * CELL, 3), np.uint8), {})
    m.blocked = np.zeros((8, 8), bool)
    m.walk = {(r, c) for r in range(3) for c in range(3)}
    m.steps_file = tmp_path / "steps-m.json"
    a = Atlas.__new__(Atlas)
    start, goal = (0, CELL), (2 * CELL, CELL)  # (x, y): cell (1, 0) to cell (1, 2)
    assert a.route(m, start, goal) == [(0, CELL), (CELL, CELL), (2 * CELL, CELL)]
    m.learn_step((1, 0), (1, 1))
    path = a.route(m, start, goal)
    assert path[0] == (0, CELL) and path[1] != (CELL, CELL)
    assert m.steps_file.exists()


def test_others_of_kind_for_a_shut_way_unless_the_town_was_named():
    from playertwo.agent import others_of_kind

    assert others_of_kind(M, "Truce Inn", "take me to the inn", (440, 280)) == ["Porre Inn"]
    assert others_of_kind(M, "Porre Inn", "the inn in Porre", (440, 280)) == []
    assert others_of_kind(M, "Leene Square", "the fair", (440, 280)) == []


@pytest.mark.skipif(Atlas.folder_for("Chrono Trigger") is None
                    or Rooms.folder_for("Chrono Trigger") is None,
                    reason="no Chrono Trigger atlas and room survey here")
def test_from_the_castles_side_the_way_back_is_through_the_forest():
    """Live 2026-11-01: after "take me to Guardia Castle" the party stands on the castle's
    side of Guardia Forest, which open ground doesn't leave: every trip from there said
    "no way" or went for Medina Square on another continent."""
    a = Atlas.for_game("Chrono Trigger")
    m, r = a.map("1000 A.D."), Rooms.for_game("Chrono Trigger")
    castle_side = (304.0, 224.0)  # where the guards leave the party, by the atlas
    assert way_round(a, m, r, "Leene Square", castle_side) == (
        "Guardia Forest", (280, 240), "outside, toward Truce Market")
    assert way_round(a, m, r, "Guardia Forest", castle_side)[:2] == (
        "Guardia Forest", (280, 240))  # into the forest by its north door
    assert way_round(a, m, r, "Guardia Castle", castle_side) is None  # walkable
    assert way_round(a, m, r, "Leene Square", (441.0, 262.0)) is None  # from Crono's house
