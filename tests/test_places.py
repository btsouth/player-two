from types import SimpleNamespace

from playertwo.agent import nearest_of_kind
from playertwo.atlas import Atlas

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
