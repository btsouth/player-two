#!/usr/bin/env python
"""The real agent on a saved frame: one open request, nothing pressed.

    python scripts/vision_ask.py frame.png "What's around me?" -o out/ --game "Some Game"

Writes out/prompts.txt, out/run.json (request, every tool call and result, what was said and when, the
model's turns), out/masks/ (one PNG per mask, by box) and out/overlay.png.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from playertwo.agent import Agent
from playertwo.config import Settings
from playertwo.locate import found_record, ping_order, wide_pan
from playertwo.muse import Muse
from playertwo.notes import GameNotes
from playertwo.prompts import OPEN_ANSWER, OPEN_LOOK
from playertwo.screen import Frame

COLORS = [(242, 193, 78), (90, 200, 250), (250, 110, 110), (120, 230, 140), (200, 140, 250)]


class NoControls:
    """Nothing reaches a game: a press is only written down."""

    def __init__(self):
        self.calls: list = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append([name, repr(a)[:80]])


def overlay(img: Image.Image, found: list) -> Image.Image:
    size = (img.width, img.height)
    out = np.asarray(img.convert("RGB")).astype(float)
    for n, f in enumerate(found):
        for m in f.matches:
            if m.mask is None:
                continue
            grid = np.asarray(Image.fromarray(m.mask.full_frame() * 255).resize(size)) > 127
            out[grid] = out[grid] * 0.45 + np.array(COLORS[n % len(COLORS)]) * 0.55
    return Image.fromarray(out.astype("uint8"))


async def main(a) -> int:
    img = Image.open(a.image).convert("RGB")
    a.out.mkdir(parents=True, exist_ok=True)
    muse = Muse(Settings.load())
    t0 = time.monotonic()
    said: list[dict] = []
    finds: list[dict] = []
    last: list = []

    async def speak(text: str) -> None:
        said.append({"t": round(time.monotonic() - t0, 2), "text": text})

    async def grab() -> Frame:
        return Frame(img, time.monotonic(), label=str(a.image))

    def on_action(name: str, args: dict) -> None:
        if name == "found":
            last[:] = args["found"]
            finds.append({"t": round(time.monotonic() - t0, 2), "frame_t": args["frame_t"],
                          "phrases": args["phrases"], "seconds": args["seconds"],
                          "found": found_record(args["found"], a.out / "masks",
                                                f"{len(finds) + 1}-")})

    controls = NoControls()
    with tempfile.TemporaryDirectory() as notes:
        agent = Agent(muse, controls, grab, speak, game=a.game, on_action=on_action,
                      notes=GameNotes(Path(notes)))

        async def sound() -> None:
            # As long as saying the found line and the pings take in the app.
            await asyncio.sleep(len(said[-1]["text"]) * 0.065 if said else 0)
            await asyncio.sleep(0.28 * len(ping_order(last)) + 0.25 * len(last))

        agent.ping_found = sound
        (a.out / "prompts.txt").write_text(f"{OPEN_LOOK}\n\n{OPEN_ANSWER}\n\n{agent.system()}")
        run = await agent.run(a.request, await grab())
    (a.out / "run.json").write_text(json.dumps({
        "image": str(a.image), "request": a.request, "game": a.game, "model": muse.model,
        "answer_started_s": said[0]["t"] if said else None, "seconds": round(run.seconds, 2),
        "said": said, "outcome": run.outcome, "turns": run.turns,
        "steps": [{"tool": s.tool, "args": s.args, "t": round(s.t, 2), "result": s.result}
                  for s in run.steps],
        "finds": finds, "presses": controls.calls,
        "pings": [[p, wide_pan(pan)] for p, pan in ping_order(last)],
    }, indent=1))
    if last:
        overlay(img, last).save(a.out / "overlay.png")
    print(f"started {said[0]['t'] if said else None} s; total {run.seconds:.1f} s; "
          f"tools {[s.tool for s in run.steps]}")
    for f in finds:
        print(f"find at {f['t']} s took {f['seconds']} s:",
              {x["phrase"]: x["count"] for x in f["found"]})
    print("SAID:", " | ".join(s["text"] for s in said))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("request")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--game", default="")
    sys.exit(asyncio.run(main(ap.parse_args())))
