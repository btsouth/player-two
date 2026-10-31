"""Read the place names drawn on a world-map rip with Muse Spark and turn them into atlas
places: Muse builds the game pack, a person checks it.

Fan-made map rips (VGMaps and the like) write each place's name in small text next to its
icon. Pass 1 cuts the rip into overlapping tiles, enlarges each so the pixel font is
legible, asks Muse for every label it can read and the box of the icon it names, maps the
boxes back to rip pixels and merges what several tiles saw into one candidate per place.
Pass 2 shows Muse a few close-ups around each candidate and takes the median icon centre
(on Chrono Trigger's rips this took icons within 8 px from 39% to 72%; every label was
read).

    python scripts/label_atlas.py RIP.png --height 1024 -o candidates.json
    python scripts/label_atlas.py RIP.png --from-raw candidates.json --truth checked.json

The output is a candidate list to check by eye: the name as written on the rip, the icon
centre in rip pixels, how many tiles agreed, and Muse's raw answers. Same-named labels get
a number ("Residence (2)"); naming them for the player is the checker's job. Any game's
rip works; nothing here is Chrono Trigger specific.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

PROMPT = """This is part of a fan-made world map of a video game, enlarged {scale}x so the \
small text is readable. The image is {w} x {h} pixels.

Place names are written on the map as short text labels (light text with a dark outline). \
Each label sits just above, below or beside the picture of the place it names: a town, \
house, castle, cave, dome, mountain, tower, ruin, bridge or similar icon.

List every label you can read completely. Skip text cut off by the image edge, and skip \
anything that is not a place label (credits, logos). For each label give:
- "name": the label text exactly as written (keep capitals, apostrophes, numbers)
- "label": the box around the text, [x0, y0, x1, y1] in pixels of this image
- "icon": the box around the icon the label names (not the text), [x0, y0, x1, y1] in \
pixels of this image. The icon is the nearest building or landmark the text points at, \
usually directly below or above the text.

Answer only JSON: {{"places": [{{"name": "...", "label": [..], "icon": [..]}}, ...]}}. \
If there are no labels, answer {{"places": []}}."""


@dataclass
class Seen:
    name: str
    x: float  # icon centre, rip pixels
    y: float
    label: tuple[float, float, float, float]  # text box, rip pixels
    tile: tuple[int, int]


@dataclass
class Candidate:
    name: str
    x: float
    y: float
    votes: int
    seen: list[Seen] = field(default_factory=list)


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower().replace("’", "'"))


def tiles(w: int, h: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Overlapping tile boxes covering w x h: every label (shorter than `overlap`) lies
    wholly inside at least one tile."""
    step = size - overlap

    def starts(n: int) -> list[int]:
        if n <= size:
            return [0]
        return sorted({*range(0, n - size, step), n - size})

    return [(x, y, min(w, x + size), min(h, y + size)) for y in starts(h) for x in starts(w)]


def png_part(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def box(v) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(a) for a in v)
    except (TypeError, ValueError):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def to_rip(
    answer: dict, tile: tuple[int, int, int, int], scale: float, edge: float = 2.0
) -> list[Seen]:
    """Muse's answer for one enlarged tile, in rip pixels. Labels touching the tile's
    edge are dropped (another tile sees them whole)."""
    tx, ty, tx1, ty1 = tile
    out = []
    for p in answer.get("places") or []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        ib, lb = box(p.get("icon")), box(p.get("label"))
        if not name or ib is None:
            continue
        ib = tuple(v / scale for v in ib)
        lb = tuple(v / scale for v in lb) if lb else ib
        if lb[0] < edge or lb[1] < edge or lb[2] > tx1 - tx - edge or lb[3] > ty1 - ty - edge:
            continue
        out.append(
            Seen(
                name,
                tx + (ib[0] + ib[2]) / 2,
                ty + (ib[1] + ib[3]) / 2,
                (tx + lb[0], ty + lb[1], tx + lb[2], ty + lb[3]),
                (tx, ty),
            )
        )
    return out


def overlap(a: tuple, b: tuple) -> float:
    """Intersection over the smaller of two boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / max(small, 1e-6)


def same(a: Seen, b: Seen, near: float) -> bool:
    """Two sightings of one label: the same name with the labels (or icons) close, or one
    name a part of the other ("Inn" for "Porre Inn") on overlapping text."""
    na, nb = norm(a.name), norm(b.name)
    if na == nb:
        ca = ((a.label[0] + a.label[2]) / 2, (a.label[1] + a.label[3]) / 2)
        cb = ((b.label[0] + b.label[2]) / 2, (b.label[1] + b.label[3]) / 2)
        return math.dist(ca, cb) <= near or math.dist((a.x, a.y), (b.x, b.y)) <= near
    return (na in nb or nb in na) and overlap(a.label, b.label) > 0.5


def merge(seen: list[Seen], near: float = 24.0) -> list[Candidate]:
    """One candidate per place from what overlapping tiles saw. Same-named places far
    apart (two "Residence" labels) stay separate."""
    groups: list[list[Seen]] = []
    for s in seen:
        for g in groups:
            if any(same(s, o, near) for o in g):
                g.append(s)
                break
        else:
            groups.append([s])
    out = []
    for g in groups:
        names = [s.name for s in g]
        # The reading most tiles agree on; on a tie the longer (a cut-off read is shorter).
        name = max(set(names), key=lambda n: (names.count(n), len(n)))
        out.append(Candidate(name, median([s.x for s in g]), median([s.y for s in g]), len(g), g))
    return sorted(out, key=lambda c: (c.y, c.x))


def median(v: list[float]) -> float:
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def unique_names(cands: list[Candidate]) -> dict[str, list[float]]:
    """A places dict: repeated names (two "Residence" labels) get a number, top to
    bottom, for a person to rename."""
    by: dict[str, list[Candidate]] = {}
    for c in cands:
        by.setdefault(c.name, []).append(c)
    places: dict[str, list[float]] = {}
    for name, cs in by.items():
        if len(cs) == 1:
            places[name] = [round(cs[0].x), round(cs[0].y)]
            continue
        for i, c in enumerate(sorted(cs, key=lambda c: (c.y, c.x)), 1):
            places[f"{name} ({i})"] = [round(c.x), round(c.y)]
    return places


class Asker:
    """Muse Spark on Meta Model API, for images, with retries and a usage count."""

    def __init__(self, model: str | None = None, concurrency: int = 4, log=print):
        from playertwo.config import Settings
        from playertwo.muse import Muse

        s = Settings.load()
        s.brain = "meta"  # images go to Muse Spark on Meta Model API
        if model:
            s.model = model
        self.muse = Muse(s)
        self.model = s.model
        self.client = self.muse.client.with_options(timeout=180.0, max_retries=2)
        self.sem = asyncio.Semaphore(concurrency)
        self.log = log

    async def __call__(self, text: str, img: Image.Image, what: str) -> dict:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}, png_part(img)]}]
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        max_completion_tokens=4000,
                        response_format={"type": "json_object"},
                        reasoning_effort="low",
                    )
                    self.muse.usage.add(r.usage, 0.0)
                    return json.loads(r.choices[0].message.content or "{}")
                except Exception as exc:  # noqa: BLE001  network, bad JSON: try again
                    self.log(f"{what}: {type(exc).__name__}: {exc}"[:200])
                    await asyncio.sleep(2 * (attempt + 1))
        return {}

    def stats(self) -> dict:
        u = self.muse.usage
        return {
            "calls": u.calls,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "cost_usd": round(u.cost_usd(), 4),
        }


def enlarge(img: Image.Image, scale: int) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


async def read_tiles(
    rip: Image.Image, ask: Asker, *, size: int = 320, overlap: int = 112, scale: int = 3
) -> list[dict]:
    """Pass 1: every tile's labels, as Muse answered (tile box and answer)."""
    raw: list[dict] = []

    async def one(t: tuple[int, int, int, int]) -> None:
        big = enlarge(rip.crop(t), scale)
        ans = await ask(PROMPT.format(scale=scale, w=big.width, h=big.height), big, f"tile {t}")
        raw.append({"tile": list(t), "scale": scale, "answer": ans})
        ask.log(f"tile {t[:2]}: {', '.join(to_names(ans)) or '-'}")

    await asyncio.gather(*(one(t) for t in tiles(rip.width, rip.height, size, overlap)))
    return raw


def to_names(ans: dict) -> list[str]:
    return [str(p.get("name", "")) for p in ans.get("places") or [] if isinstance(p, dict)]


def from_raw(raw: list[dict]) -> list[Candidate]:
    seen: list[Seen] = []
    for r in raw:
        seen.extend(to_rip(r["answer"], tuple(r["tile"]), r.get("scale", 3)))
    return merge(seen)


REFINE = """This is a close-up of a fan-made video game world map, enlarged {scale}x; it is {w} x {h} pixels. The place label "{name}" is written in it as small light text, near the middle. Find the icon that label names: the picture of the place (building, town, castle, cave mouth, mountain, tower, bridge, ruin...) just below, above or beside the text. Answer only JSON: {{"found": true, "icon": [x0, y0, x1, y1]}} with the box around that icon in pixels of this image, not around the text. If the label isn't there, {{"found": false}}."""


JITTER = [(0, 0), (-14, -10), (14, 10), (14, -10), (-14, 10)]


async def refine(
    rip: Image.Image,
    cands: list[Candidate],
    ask: Asker,
    *,
    size: int = 112,
    scale: int = 6,
    samples: int = 3,
) -> list[dict]:
    """Pass 2: close-ups around each candidate (a few, each framed a little
    differently), so Muse places the icon more exactly than it can on a whole tile; the
    median of its answers wins. Moves each candidate in place; returns the answers."""
    raw: list[dict] = []

    async def look(c: Candidate, cx: float, cy: float) -> tuple[float, float] | None:
        x0 = int(min(max(0, cx - size / 2), max(0, rip.width - size)))
        y0 = int(min(max(0, cy - size / 2), max(0, rip.height - size)))
        box_ = (x0, y0, min(rip.width, x0 + size), min(rip.height, y0 + size))
        big = enlarge(rip.crop(box_), scale)
        ans = await ask(
            REFINE.format(scale=scale, w=big.width, h=big.height, name=c.name),
            big,
            f"refine {c.name}",
        )
        raw.append({"name": c.name, "was": [c.x, c.y], "box": list(box_), "answer": ans})
        b = box(ans.get("icon")) if ans.get("found", True) else None
        if b is None:
            return None
        p = (x0 + (b[0] + b[2]) / 2 / scale, y0 + (b[1] + b[3]) / 2 / scale)
        # A jump that far is another icon, or the text.
        return p if math.dist(p, (c.x, c.y)) <= size / 2 else None

    async def one(c: Candidate) -> None:
        labels = [s.label for s in c.seen]
        lx = median([(b[0] + b[2]) / 2 for b in labels])
        ly = median([(b[1] + b[3]) / 2 for b in labels])
        cx, cy = (lx + c.x) / 2, (ly + c.y) / 2
        got = await asyncio.gather(*(look(c, cx + dx, cy + dy) for dx, dy in JITTER[:samples]))
        got = [g for g in got if g is not None]
        if got:
            c.x, c.y = median([g[0] for g in got]), median([g[1] for g in got])

    await asyncio.gather(*(one(c) for c in cands))
    return raw


def score(cands: list[Candidate], truth: list, tol: float = 8.0) -> dict:
    """How Muse did against hand-checked places. truth rows are [label text as on the
    rip, place name, [x, y] icon centre, optional [x, y] way in (a cave's mouth, a
    castle's gate), optional "zone"]. Each is matched to the nearest unused candidate
    reading the same text (case and punctuation aside): right within tol px of the icon
    centre or the way in, off when further, missed when nothing near reads it. A "zone"
    (a label over an area with no icon: a forest, a ruin field) only counts as read."""
    free = list(cands)
    rows = []
    for text, name, (x, y), *rest in truth:
        spots = [(x, y)] + [tuple(e) for e in rest[:1] if e]
        zone = "zone" in rest[1:]

        def far(c: Candidate, spots=spots) -> float:
            return min(math.dist((c.x, c.y), p) for p in spots)

        pool = [c for c in free if norm(c.name) == norm(text)]
        best = min(pool, key=far, default=None)
        if best is None or far(best) > (48 if zone else 128):
            rows.append({"place": name, "result": "missed"})
            continue
        d = far(best)
        free.remove(best)
        result = "read" if zone else "right" if d <= tol else "off"
        rows.append(
            {
                "place": name,
                "result": result,
                "px": round(d, 1),
                "muse": [round(best.x), round(best.y)],
            }
        )
    count = {k: sum(r["result"] == k for r in rows) for k in ("right", "off", "missed", "read")}
    return {
        **count,
        "total": len(rows) - count["read"],
        "extra": [f"{c.name} @{round(c.x)},{round(c.y)}" for c in free],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rip", type=Path)
    ap.add_argument("--height", type=int, help="crop the rip to this height (legend off)")
    ap.add_argument("-o", "--out", type=Path, help="candidates JSON (default: stdout)")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--overlap", type=int, default=112)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--no-refine", action="store_true", help="skip the close-up pass")
    ap.add_argument("--samples", type=int, default=3, help="close-ups per place")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model")
    ap.add_argument(
        "--from-raw",
        type=Path,
        help="reuse the tile answers in an earlier output (no pass-1 calls)",
    )
    ap.add_argument(
        "--truth", type=Path, help="score against hand-checked rows [label text, name, [x, y]]"
    )
    a = ap.parse_args()

    rip = Image.open(a.rip).convert("RGB")
    if a.height:
        rip = rip.crop((0, 0, rip.width, a.height))

    def log(m: str) -> None:
        print(m, file=sys.stderr)

    ask = Asker(a.model, a.concurrency, log)

    async def run() -> dict:
        if a.from_raw:
            raw = json.loads(a.from_raw.read_text())["raw"]
        else:
            raw = await read_tiles(rip, ask, size=a.tile, overlap=a.overlap, scale=a.scale)
        cands = from_raw(raw)
        out = {"rip": str(a.rip), "size": list(rip.size), "raw": raw}
        first = [Candidate(c.name, c.x, c.y, c.votes, c.seen) for c in cands]
        if not a.no_refine:
            out["refine_raw"] = await refine(rip, cands, ask, samples=a.samples)
        out["places"] = unique_names(cands)
        out["candidates"] = [
            {
                "name": c.name,
                "xy": [round(c.x), round(c.y)],
                "tiles_xy": [round(f.x), round(f.y)],
                "votes": c.votes,
            }
            for c, f in zip(cands, first, strict=True)
        ]
        out["stats"] = ask.stats()
        if a.truth:
            truth = json.loads(a.truth.read_text())
            out["score_tiles"] = score(first, truth)
            if not a.no_refine:
                out["score"] = score(cands, truth)
            for k in ("score_tiles", "score"):
                if k in out:
                    sc = out[k]
                    log(
                        f"{k}: right {sc['right']}/{sc['total']}, off {sc['off']}, "
                        f"missed {sc['missed']}, zones read {sc['read']}, "
                        f"extra {len(sc['extra'])}"
                    )
        return out

    text = json.dumps(asyncio.run(run()), indent=1)
    if a.out:
        a.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
