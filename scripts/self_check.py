"""Offline install smoke check. No key, desktop, audio device or controller is opened."""
from importlib.metadata import version

import cv2
import numpy as np

from playertwo import ctsteam
from playertwo.nav import astar


def main() -> None:
    print(f"Player Two {version('player-two')}")
    for name in ("hand.png", "hand_mask.png", "bracket.png", "bracket_mask.png"):
        if cv2.imread(str(ctsteam.DATA / name)) is None:
            raise RuntimeError(f"Missing reader template: {name}")
    print("PASS: packaged reader templates load")
    ctsteam.read(np.zeros((1440, 2560, 3), dtype=np.uint8))
    print("PASS: pixel reader processes a blank frame")
    wall = {(1, y) for y in range(-1, 2)}
    route = astar((0, 0), (3, 0), wall, (-4, -4, 4, 4))
    if not route or route[-1] != (3, 0) or wall.intersection(route):
        raise RuntimeError("Navigation did not avoid the wall")
    print(f"PASS: A* routes around a wall ({len(route) - 1} steps)")
    print("Offline checks passed. Live narration still needs a key and a supported desktop.")


if __name__ == "__main__":
    main()
