#!/usr/bin/env python
"""One transparent PNG per SAM 3.1 mask, plus a JSON index, from an image and phrases.

    python scripts/sam_masks.py frame.png -o out/ horse person tree rock

Each thing found gets a flat fill and a thin edge, at 1920x1080 in out/1080/ and at the
image's own size in out/source/. One SAM call per phrase, one after another. index.json has
phrase, box, centre, pan and area for each, all as fractions of the frame except pixel
boxes, and preview.png shows them all on the frame.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from playertwo.config import Settings
from playertwo.locate import measure, segment, wide_pan
from playertwo.muse import Muse

COLORS = [(242, 193, 78), (90, 200, 250), (250, 110, 110), (120, 230, 140), (200, 140, 250),
          (250, 160, 90), (110, 230, 220), (240, 130, 200)]
FILL_ALPHA = 150
EDGE_PX_AT_1080 = 2


def scaled(grid: np.ndarray, w: int, h: int) -> np.ndarray:
    """A 0/1 mask at another size, with a smooth outline rather than stair steps."""
    if (grid.shape[1], grid.shape[0]) == (w, h):
        return grid.astype(bool)
    soft = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return soft >= 0.5


def layer(mask: np.ndarray, color: tuple[int, int, int], edge_px: int) -> Image.Image:
    """Flat fill plus a thin edge, transparent everywhere else."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px * 2 + 1, edge_px * 2 + 1))
    inner = cv2.erode(mask.astype(np.uint8), kernel, borderValue=0).astype(bool)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, FILL_ALPHA)
    rgba[mask & ~inner] = (*color, 255)
    return Image.fromarray(rgba, "RGBA")


async def run(image: Path, phrases: list[str], out: Path, side: int) -> list[dict]:
    img = Image.open(image).convert("RGB")
    sw, sh = img.size
    out.mkdir(parents=True, exist_ok=True)
    (out / "1080").mkdir(exist_ok=True)
    (out / "source").mkdir(exist_ok=True)
    muse = Muse(Settings.load())
    index, layers = [], []
    for n, phrase in enumerate(phrases):
        color = COLORS[n % len(COLORS)]
        found = await segment(muse, img, phrase, side=side)
        print(f"{phrase}: {len(found)}", flush=True)
        slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-") or "thing"
        for i, m in enumerate(found):
            if m.mask is None:
                continue
            grid, s = m.mask.full_frame(), measure(m)
            name = f"{slug}-{i + 1}.png"
            layer(scaled(grid, 1920, 1080), color, EDGE_PX_AT_1080).save(out / "1080" / name)
            big = layer(scaled(grid, sw, sh), color, max(1, round(EDGE_PX_AT_1080 * sh / 1080)))
            big.save(out / "source" / name)
            layers.append((s.area, big))
            x1, y1, x2, y2 = m.box
            index.append({
                "phrase": phrase, "file": name, "color": list(color),
                "box": [round(v, 4) for v in (x1, y1, x2, y2)],
                "box_px_source": [round(x1 * sw), round(y1 * sh), round(x2 * sw), round(y2 * sh)],
                "box_px_1080": [round(x1 * 1920), round(y1 * 1080),
                                round(x2 * 1920), round(y2 * 1080)],
                "centre": [round(s.cx, 4), round(s.cy, 4)],
                "pan": s.pan, "wide_pan": wide_pan(s.pan), "area": round(s.area, 5),
            })
    (out / "index.json").write_text(json.dumps({
        "image": str(image), "source_size": [sw, sh], "sam_input_side": side,
        "phrases": phrases, "masks": index}, indent=1))
    preview = img.convert("RGBA")
    for _area, big in sorted(layers, key=lambda p: -p[0]):  # small things stay on top
        preview.alpha_composite(big)
    preview.convert("RGB").save(out / "preview.png")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("phrases", nargs="+")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--side", type=int, default=1280,
                    help="longest side of the picture sent to SAM (default 1280)")
    a = ap.parse_args()
    print(f"{len(asyncio.run(run(a.image, a.phrases, a.out, a.side)))} masks in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
