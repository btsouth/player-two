"""Screen capture and change detection.

The watcher samples the screen a few times a second and decides when something new
has appeared and stopped moving, so narration fires once per new screen instead of
on every animation frame.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

THUMB = (96, 54)


@dataclass
class Frame:
    image: Image.Image
    t: float
    label: str = ""


class Capture:
    def grab(self) -> Frame:
        raise NotImplementedError


class HyprlandCapture(Capture):
    """Captures the monitor that has focus (or a named output) with grim."""

    def __init__(self, output: str | None = None):
        self.output = output

    def _focused_output(self) -> str | None:
        try:
            mons = json.loads(subprocess.check_output(["hyprctl", "-j", "monitors"], timeout=2))
            for m in mons:
                if m.get("focused"):
                    return m["name"]
        except (subprocess.SubprocessError, OSError, ValueError):
            return None
        return None

    def grab(self) -> Frame:
        out = self.output or self._focused_output()
        cmd = ["grim", "-t", "ppm"] + (["-o", out] if out else []) + ["-"]
        data = subprocess.check_output(cmd, timeout=3)
        return Frame(Image.open(io.BytesIO(data)), time.monotonic())


class MssCapture(Capture):
    """Windows and X11."""

    def __init__(self, monitor: int = 1):
        import mss

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]

    def grab(self) -> Frame:
        shot = self.sct.grab(self.monitor)
        return Frame(Image.frombytes("RGB", shot.size, shot.rgb), time.monotonic())


class FileCapture(Capture):
    """Replays a folder of screenshots, one per grab. Used for tests and the eval set."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.i = 0

    def grab(self) -> Frame:
        p = self.paths[min(self.i, len(self.paths) - 1)]
        self.i += 1
        return Frame(Image.open(p).convert("RGB"), time.monotonic(), label=p.name)


def video_frames(path: Path, fps: float) -> Iterator[Frame]:
    """Decode a gameplay recording at fps, as if it were the live screen."""
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", str(path)]
    )
    st = json.loads(probe)["streams"][0]
    w, h = st["width"], st["height"]
    scale = min(1.0, 1280 / max(w, h))
    w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    n = 0
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield Frame(Image.frombytes("RGB", (w, h), buf), n / fps, label=f"{n / fps:.2f}s")
        n += 1
    proc.wait()


def auto_capture() -> Capture:
    if sys.platform == "win32":
        return MssCapture()
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("grim"):
        return HyprlandCapture()
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("grim"):
        return HyprlandCapture()
    return MssCapture()


def thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(THUMB, Image.Resampling.BILINEAR), np.float32) / 255


class ChangeWatcher:
    """Turns a stream of frames into "a new screen has settled" events.

    Cells that are always moving (a flickering TV, rain, an idle animation) are learned
    and ignored, so they neither trigger narration nor stop the screen from settling.
    """

    def __init__(self, change: float = 0.02, still: float = 0.004, settle_s: float = 0.3,
                 max_wait_s: float = 1.6, cell_delta: float = 0.06):
        self.change, self.still = change, still
        self.settle_s, self.max_wait_s = settle_s, max_wait_s
        self.cell_delta = cell_delta
        self.base: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.motion = np.zeros(THUMB[::-1], np.float32)
        self.pending_since: float | None = None
        self.still_since: float | None = None

    def _frac(self, a: np.ndarray, b: np.ndarray) -> float:
        moving = np.abs(a - b) > self.cell_delta
        mask = self.motion < 0.5
        return float((moving & mask).sum() / max(mask.sum(), 1))

    def force(self) -> None:
        """Treat the next settled frame as new, e.g. after the player asks for a re-read."""
        self.base = None

    def update(self, frame: Frame) -> bool:
        """Feed a frame. Returns True when it is a new screen that has settled."""
        th = thumb(frame.image)
        now = frame.t
        if self.prev is None:
            self.prev = th
            self.pending_since = now
            self.still_since = now
            return False
        step = np.abs(th - self.prev) > self.cell_delta
        self.motion = 0.97 * self.motion + 0.03 * step
        moved = self._frac(th, self.prev)
        self.prev = th
        changed = self.base is None or self._frac(th, self.base) > self.change
        if changed and self.pending_since is None:
            self.pending_since = now
        if self.pending_since is None:
            return False
        if moved > self.still:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now
        settled = self.still_since is not None and now - self.still_since >= self.settle_s
        timed_out = now - self.pending_since >= self.max_wait_s
        if settled or timed_out:
            self.base = th
            self.pending_since = None
            self.still_since = None
            return True
        return False
