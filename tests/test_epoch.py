"""The Epoch's era names and landing spots (no game needed)."""

import numpy as np

from playertwo.atlas import WorldMap
from playertwo.epoch import Hangar, era_named, era_of, era_only, landing_spots, reachable


def test_era_names():
    assert era_named("take me to 600 AD") == "600 A.D."
    assert era_named("the middle ages") == "600 A.D."
    assert era_named("go to the future") == "2300 A.D."
    assert era_named("65,000,000 BC") == "65,000,000 B.C."
    assert era_named("the End of Time") == "End of Time"
    assert era_named("12000 B.C.") == "12,000 B.C."
    assert era_named("Medina Village") is None
    assert era_named("Bangor Dome") is None
    assert era_of("12,000 B.C. Last Village") == "12,000 B.C."
    assert era_of("1000 A.D.") == "1000 A.D."


def test_era_only():
    assert era_only("600 AD") == "600 A.D."
    assert era_only("the future") == "2300 A.D."
    assert era_only("back to the present") == "1000 A.D."
    assert era_only("Truce Inn in 600 AD") is None
    assert era_only("Medina") is None


def test_landing_spots(tmp_path):
    img = np.zeros((160, 160, 3), np.uint8)
    walk = {(r, c) for r in range(2, 18) for c in range(2, 10)}
    m = WorldMap("1000 A.D.", img, {"Inn": (40, 40)}, walk=walk,
                 doors={(5, 5): 12}, exits={"3": {(5, 5)}}, world=496)
    h = Hangar(tmp_path)
    spots = landing_spots(m, {(5, 5)}, h)
    assert spots and all(max(abs(r - 5), abs(c - 5)) >= 2 for r, c in spots)
    h.learn("1000 A.D.", spots[0], False)
    assert spots[0] not in landing_spots(m, {(5, 5)}, Hangar(tmp_path))
    h.learn("1000 A.D.", (15, 8), True)
    assert landing_spots(m, {(5, 5)}, Hangar(tmp_path))[0] != (15, 8)  # too far to win
    assert reachable(m, (24, 120), {(5, 5)})
    assert not reachable(m, (150, 150), {(5, 5)})
