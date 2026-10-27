"""Fast warnings for real-time games.

Muse Spark looks at the HUD once and says where the health and stamina bars are. From
then on a few lines of pixel math read the bars on every frame, with no model call, so
"health low" arrives in a fraction of a second even in an action game.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .muse import Muse, image_part

CALIBRATE = """\
Find the player's own status meters on this game screen: health, stamina, mana or focus, \
and similar bars. Ignore enemy and boss bars. For each meter, return the box around the \
whole bar track, including any empty part, on a 0 to 1000 grid of the image, the colour \
of the filled part, and which way it fills. JSON only:
{"meters": [{"name": "health", "box": [x0, y0, x1, y1], "fill_rgb": [r, g, b], \
"fills": "left_to_right", "fraction": 0.5}]}
"fraction" is how full the meter looks right now, 0 to 1.
Use "fills" values left_to_right, right_to_left or bottom_to_top. If there are no \
meters, return {"meters": []}."""


REFINE = """\
This is a zoomed-in crop of a game HUD. For each meter listed, give the exact box of its \
full track, from the very start of the bar to the very end of the empty part (not \
decorations or end caps), on a 0 to 1000 grid of this crop. JSON only:
{"meters": [{"name": "...", "box": [x0, y0, x1, y1]}]}
Meters: """


@dataclass
class Meter:
    name: str
    box: tuple[float, float, float, float]  # 0..1 fractions of the frame
    fill_rgb: tuple[int, int, int]
    fills: str = "left_to_right"
    estimate: float = 0.0  # the model's reading at calibration
    full_px: float = 0.0  # length of a full bar, in pixels of the frame
    history: list[float] = field(default_factory=list)

    def _hsv(self):
        r, g, b = (c / 255 for c in self.fill_rgb)
        return colorsys.rgb_to_hsv(r, g, b)

    def learn_color(self, img: Image.Image) -> None:
        """Take the fill colour from the bar itself: its first few percent are filled
        unless the meter is empty, and real bars are often darker than described."""
        w, h = img.size
        x0, y0, x1, y1 = self.box
        if self.fills == "bottom_to_top":
            region = (x0, y1 - (y1 - y0) * 0.08, x1, y1)
        elif self.fills == "right_to_left":
            region = (x1 - (x1 - x0) * 0.08, y0, x1, y1)
        else:
            region = (x0 + (x1 - x0) * 0.02, y0, x0 + (x1 - x0) * 0.08, y1)
        crop = img.crop((int(region[0] * w), int(region[1] * h),
                         max(int(region[2] * w), int(region[0] * w) + 1),
                         max(int(region[3] * h), int(region[1] * h) + 1))).convert("RGB")
        px = np.asarray(crop, np.float32).reshape(-1, 3)
        if not len(px):
            return
        sat = px.max(axis=1) - px.min(axis=1)
        vivid = px[sat >= np.percentile(sat, 60)]
        r, g, b = np.median(vivid, axis=0)
        th, _, _ = self._hsv()
        h2, s2, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s2 > 0.15 and min(abs(h2 - th), 1 - abs(h2 - th)) < 0.12:
            self.fill_rgb = (int(r), int(g), int(b))

    def _mask(self, img: Image.Image, box) -> np.ndarray:
        crop = np.asarray(img.crop(box).convert("HSV"), np.float32) / 255.0
        hue, sat, val = crop[..., 0], crop[..., 1], crop[..., 2]
        th, ts, tv = self._hsv()
        dh = np.minimum(np.abs(hue - th), 1 - np.abs(hue - th))
        return (dh < 0.07) & (sat > max(0.15, ts * 0.5)) & (val > max(0.08, tv * 0.5))

    def _px(self, img: Image.Image, box=None) -> tuple[int, int, int, int]:
        w, h = img.size
        x0, y0, x1, y1 = box or self.box
        return (int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 2),
                max(int(y1 * h), int(y0 * h) + 2))

    def locate(self, img: Image.Image) -> None:
        """Tighten the model's rough box to the rows the fill colour actually occupies."""
        self.learn_color(img)
        x0, y0, x1, y1 = self.box
        bh = y1 - y0
        loose = (x0, max(0.0, y0 - bh * 0.6), x1, min(1.0, y1 + bh * 0.6))
        px = self._px(img, loose)
        mask = self._mask(img, px)
        rows = np.where(mask.mean(axis=1) > 0.15)[0]
        if len(rows):
            h = img.size[1]
            top, bottom = px[1] + rows.min(), px[1] + rows.max() + 1
            self.box = (x0, float(top / h), x1, float(bottom / h))
        if self.estimate > 0.25:
            run = self._run(img)
            if run:
                self.full_px = run / min(1.0, self.estimate)

    def _run(self, img: Image.Image) -> int:
        px = self._px(img)
        mask = self._mask(img, px)
        if self.fills == "bottom_to_top":
            profile = mask.mean(axis=1)[::-1]
        else:
            profile = mask.mean(axis=0)
            if self.fills == "right_to_left":
                profile = profile[::-1]
        filled = profile > 0.3
        if not filled.any():
            return 0
        lead = int(np.argmax(filled))
        if lead > len(filled) * 0.15:
            return 0
        end, gap = lead, 0
        for i, f in enumerate(filled[lead:], start=lead):
            if f:
                end, gap = i + 1, 0
            else:
                gap += 1
                if gap > max(2, len(filled) // 40):
                    break
        return end - lead

    def measure(self, img: Image.Image) -> float | None:
        """Fraction of the bar that is filled, or None before it has ever been seen."""
        run = self._run(img)
        self.full_px = max(self.full_px, run)  # a fuller bar than we knew: that's closer to full
        if self.full_px <= 0:
            return None
        frac = min(1.0, run / self.full_px)
        self.history = [*self.history[-29:], frac]
        return frac


@dataclass
class Alert:
    meter: str
    level: str  # "low", "critical", "recovered", "empty"
    value: float


class HudWatch:
    """Turns meter readings into a few clear warnings, without nagging."""

    def __init__(self, meters: list[Meter], low: float = 0.3, critical: float = 0.15):
        self.meters = meters
        self.low, self.critical = low, critical
        self.state: dict[str, str] = {}
        self.last_alert_t: dict[str, float] = {}

    def update(self, img: Image.Image, t: float) -> list[Alert]:
        alerts = []
        for m in self.meters:
            v = m.measure(img)
            if v is None:
                continue
            prev = self.state.get(m.name, "ok")
            is_health = "health" in m.name.lower() or m.name.lower() in ("hp", "life")
            if is_health:
                if v <= self.critical:
                    level = "critical"
                elif v <= self.low:
                    level = "low"
                elif v >= self.low + 0.15:
                    level = "ok"
                else:
                    level = prev if prev in ("low", "critical") else "ok"
            else:
                level = "empty" if v <= 0.05 else ("ok" if v >= 0.25 else prev)
            repeat = level == "critical" and t - self.last_alert_t.get(m.name, -99) > 4.0
            if level != prev or repeat:
                if level != "ok" or prev in ("low", "critical"):
                    alerts.append(Alert(m.name, "recovered" if level == "ok" else level, v))
                    self.last_alert_t[m.name] = t
                self.state[m.name] = level
        return alerts


async def calibrate(muse: Muse, img: Image.Image) -> list[Meter]:
    data = await muse.json([
        {"role": "user", "content": [image_part(img, 1280), {"type": "text", "text": CALIBRATE}]}
    ], effort="low")
    meters = []
    for m in data.get("meters", []):
        try:
            x0, y0, x1, y1 = (max(0.0, min(1.0, float(v) / 1000)) for v in m["box"])
            rgb = tuple(int(c) for c in m["fill_rgb"])[:3]
        except (KeyError, TypeError, ValueError):
            continue
        if x1 - x0 < 0.01 or y1 - y0 < 0.002:
            continue
        try:
            est = float(m.get("fraction", 0) or 0)
        except (TypeError, ValueError):
            est = 0.0
        meters.append(Meter(str(m.get("name", "meter")), (x0, y0, x1, y1), rgb,
                            m.get("fills", "left_to_right"), estimate=est))
    for m in meters:
        m.locate(img)
    return meters


async def refine(muse: Muse, img: Image.Image, meters: list[Meter]) -> None:
    """Second look, zoomed in on the HUD, to pin down where each track starts and ends."""
    pad = 0.02
    cx0 = max(0.0, min(m.box[0] for m in meters) - pad)
    cy0 = max(0.0, min(m.box[1] for m in meters) - pad)
    cx1 = min(1.0, max(m.box[2] for m in meters) + pad * 2)
    cy1 = min(1.0, max(m.box[3] for m in meters) + pad)
    w, h = img.size
    crop = img.crop((int(cx0 * w), int(cy0 * h), int(cx1 * w), int(cy1 * h)))
    scale = 1280 / max(crop.size)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)),
                       Image.Resampling.LANCZOS)
    names = ", ".join(m.name for m in meters)
    data = await muse.json([{"role": "user", "content": [
        image_part(crop, 1280), {"type": "text", "text": REFINE + names}]}], effort="low")
    by_name = {str(d.get("name", "")).lower(): d for d in data.get("meters", [])}
    for m in meters:
        d = by_name.get(m.name.lower())
        try:
            bx0, by0, bx1, by1 = (max(0.0, min(1.0, float(v) / 1000)) for v in d["box"])
        except (TypeError, KeyError, ValueError):
            continue
        if bx1 - bx0 < 0.02:
            continue
        m.box = (cx0 + bx0 * (cx1 - cx0), cy0 + by0 * (cy1 - cy0),
                 cx0 + bx1 * (cx1 - cx0), cy0 + by1 * (cy1 - cy0))
