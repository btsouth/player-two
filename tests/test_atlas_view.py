"""Atlas.locate on both kinds of screen, on a made-up map: the game's own picture in a
4:3 box (an emulator) and redrawn art filling a wide window at its own scale (the Steam
port), which is measured once per window size."""

import json

import cv2
import numpy as np
from PIL import Image

from playertwo.atlas import Atlas


def make_atlas(tmp_path):
    rng = np.random.default_rng(7)
    img = np.zeros((640, 768, 3), np.float32)
    for sigma, weight in ((24, 1.0), (8, 0.6), (3, 0.3)):
        layer = cv2.GaussianBlur(rng.random((640, 768, 3), np.float32), (0, 0), sigma)
        layer = (layer - layer.mean()) / (layer.std() + 1e-6)
        img += weight * layer
    img = np.clip(128 + 40 * img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "w.png"), img)
    (tmp_path / "atlas.json").write_text(json.dumps(
        {"native": [256, 224], "maps": [{"name": "World", "image": "w.png"}]}))
    return Atlas(tmp_path), img


def test_emulator_box(tmp_path):
    atlas, img = make_atlas(tmp_path)
    cx, cy = 400, 300
    view = img[cy - 112:cy + 112, cx - 128:cx + 128, ::-1]
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, 240:1680] = cv2.resize(view, (1440, 1080), interpolation=cv2.INTER_NEAREST)
    s = atlas.locate(Image.fromarray(frame))
    assert s is not None and (s.x, s.y) == (cx, cy) and s.tap == 0.06
    assert atlas._scales() == {}


def test_full_window_art(tmp_path):
    atlas, img = make_atlas(tmp_path)
    vw, vh = 304, 191  # what the Steam port shows of the map, across and down

    def frame(cx, cy, size):
        x0, y0 = cx - vw // 2, cy - (vh + 1) // 2
        view = img[y0:y0 + vh, x0:x0 + vw, ::-1]
        return Image.fromarray(cv2.resize(view, size, interpolation=cv2.INTER_LINEAR))

    s = atlas.locate(frame(380, 320, (1280, 720)))
    assert s is not None and s.tap > 0.06
    assert abs(s.x - 380) <= 1 and abs(s.y - atlas.FULL_DROP - 320) <= 1
    got = atlas._scales()[(1280, 720)]
    assert abs(got[0] - vw) <= 6 and abs(got[1] - vh) <= 3
    # Kept beside the atlas, and reused for the same shape at another size.
    assert "1280x720" in json.loads((tmp_path / "view-scales.json").read_text())
    s2 = Atlas(tmp_path).locate(frame(300, 250, (1920, 1080)), near=None)
    assert s2 is not None and abs(s2.x - 300) <= 1
    near = atlas.locate(frame(388, 320, (1280, 720)), near=s)
    assert near is not None and abs(near.x - 388) <= 1
    # Not the map: nothing.
    rng = np.random.default_rng(1)
    noise = cv2.GaussianBlur(rng.random((720, 1280, 3), np.float32), (0, 0), 6)
    assert atlas.locate(Image.fromarray((noise * 255).astype(np.uint8))) is None
