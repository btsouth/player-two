"""Add labelled glyphs to the Chrono Trigger (Steam) glyph library from frames whose text
was read by eye.

    python scripts/ct_glyph_learn.py LABELS.json [--save]

LABELS.json: [{"path": frame, "lines": [[y, "text near canvas row y", x0?, x1?], ...],
               "box": [x0, y0, x1, y1] (optional), "local": true}]
A line is learned only when its glyph count matches the label's (spaces aside), so a misread
cut never teaches the library a wrong shape. Prints what it read before and after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playertwo import ctsteam  # noqa: E402
from playertwo.glyphread import (  # noqa: E402
    GlyphLib, find_template, glyphs_in_line, text_lines, text_mask, to_canvas)


def lines_of(img, box, blank, local):
    x0, y0, x1, y1 = box
    m = text_mask(img[y0:y1, x0:x1], local=local)
    for hx, hy, hw, hh in blank:
        m[max(0, hy - y0 - 4):max(0, hy - y0 + hh + 4), max(0, hx - x0 - 4):max(0, hx - x0 + hw + 4)] = False
    out = []
    for a, b in text_lines(m):
        gl = glyphs_in_line(m, a, b)
        gl = [g for g in gl if not (g.x1 - g.x0 < 12 and len(gl) == 1)]
        if gl:
            out.append((y0 + (a + b) // 2, [g for g in gl]))
    return out


def main() -> None:
    labels = json.load(open(sys.argv[1]))
    lib = ctsteam.lib()
    new = GlyphLib(lib.X.copy(), list(lib.y))
    added, skipped = 0, 0
    for item in labels:
        img = to_canvas(cv2.imread(item["path"]), ctsteam.CANVAS)
        hs = find_template(img, ctsteam.templates()["hand"], ctsteam.CANVAS)
        box = item.get("box") or [0, 0, *ctsteam.CANVAS]
        found = lines_of(img, box, [h[:4] for h in hs], item.get("local", True))
        for spec in item["lines"]:
            y, text = spec[0], spec[1]
            cand = [(abs(fy - y), fy, gl) for fy, gl in found if abs(fy - y) < 30]
            if not cand:
                print(f"  NO LINE  {Path(item['path']).name} y={y} {text!r}")
                skipped += 1
                continue
            _d, fy, gl = min(cand, key=lambda c: c[0])
            chars = text.replace(" ", "")
            xs = list(gl)
            if len(spec) == 4:  # only the glyphs in this x span
                xs = [g for g in gl if spec[2] <= g.x0 < spec[3]]
            if len(xs) != len(chars):
                got, _w = lib.read(xs)
                print(f"  COUNT    {Path(item['path']).name} y={fy} {len(xs)} glyphs vs "
                      f"{len(chars)} in {text!r}; reads {got!r}")
                skipped += 1
                continue
            before, worst = lib.read(xs)
            for g, ch in zip(xs, chars, strict=True):
                new.add(g, ch)
                added += 1
            print(f"  ok       {Path(item['path']).name} y={fy} {before!r} -> {text!r}")
    print(f"added {added} glyphs, skipped {skipped} lines; library {len(lib.y)} -> {len(new.y)}")
    print("chars:", "".join(sorted(set(new.y))))
    if "--save" in sys.argv:
        new.save(ctsteam.DATA / "glyphs.npz")
        print("saved", ctsteam.DATA / "glyphs.npz")


if __name__ == "__main__":
    main()
