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


OWN_TITLES = ("Player Two",)


def hypr_clients() -> list[dict]:
    try:
        return json.loads(subprocess.check_output(["hyprctl", "-j", "clients"], timeout=2))
    except (subprocess.SubprocessError, OSError, ValueError):
        return []


def hypr_focus(address: str) -> None:
    """Give a window keyboard focus (Lua config first, then the classic dispatcher)."""
    lua = f'hl.dispatch(hl.dsp.focus({{ window = "address:{address}" }}))'
    r = subprocess.run(["hyprctl", "eval", lua], capture_output=True, text=True, timeout=2, check=False)
    if r.returncode != 0 or "ok" not in r.stdout:
        subprocess.run(["hyprctl", "dispatch", "focuswindow", f"address:{address}"],
                       capture_output=True, timeout=2, check=False)


class HyprlandCapture(Capture):
    """Captures the game's own window with grim, even when something covers it.

    The target is the window whose class or title contains `window`, or else the most
    recently focused window that isn't Player Two's. Falls back to the whole monitor.
    """

    def __init__(self, window: str = "", output: str | None = None):
        self.want = window.lower()
        self.output = output
        self.target: dict | None = None
        self._checked = 0.0

    def pick(self) -> dict | None:
        clients = [c for c in hypr_clients()
                   if c.get("mapped", True) and c.get("title") not in OWN_TITLES]
        if self.want:
            clients = [c for c in clients if self.want in c.get("class", "").lower()
                       or self.want in c.get("title", "").lower()]
        clients.sort(key=lambda c: c.get("focusHistoryID", 99))
        return clients[0] if clients else None

    def grab(self) -> Frame:
        now = time.monotonic()
        if self.target is None or now - self._checked > 2.0:
            self.target = self.pick()
            self._checked = now
        if self.target is not None:
            try:
                data = subprocess.check_output(
                    ["grim", "-t", "ppm", "-T", str(self.target["stableId"]), "-"],
                    timeout=3, stderr=subprocess.DEVNULL)
                if data:
                    return Frame(Image.open(io.BytesIO(data)), now,
                                 label=self.target.get("title", ""))
            except (subprocess.SubprocessError, OSError, KeyError):
                pass
            self.target = None
        cmd = ["grim", "-t", "ppm"] + (["-o", self.output] if self.output else []) + ["-"]
        data = subprocess.check_output(cmd, timeout=3)
        return Frame(Image.open(io.BytesIO(data)), now)


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


def auto_capture(window: str = "") -> Capture:
    if sys.platform == "win32":
        return MssCapture()
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("grim"):
        return HyprlandCapture(window)
    return MssCapture()


def thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(THUMB, Image.Resampling.BILINEAR), np.float32) / 255


class ChangeWatcher:
    """Turns a stream of frames into "a new screen has settled" events.

    Cells that are always moving (a flickering TV, rain, an idle animation) are learned
    and ignored, so they neither trigger narration nor stop the screen from settling.
    """

    # Tuned on Persona 4 Golden's TV World, where fog drifts over half the screen: small
    # frame-to-frame steps (motion_step) are what reveal slow, constant motion.
    def __init__(self, change: float = 0.02, still: float = 0.004, settle_s: float = 0.3,
                 max_wait_s: float = 1.6, cell_delta: float = 0.06,
                 motion_step: float = 0.02, motion_rate: float = 0.1,
                 motion_cut: float = 0.25, motion_period_s: float = 0.5):
        self.change, self.still = change, still
        self.settle_s, self.max_wait_s = settle_s, max_wait_s
        self.cell_delta = cell_delta
        self.motion_step, self.motion_rate, self.motion_cut = motion_step, motion_rate, motion_cut
        # Motion is measured over a fixed time, not frame to frame, so slow drift counts the
        # same whether we capture at 2 or 10 frames a second.
        self.motion_period_s = motion_period_s
        self.motion_ref: tuple[float, np.ndarray] | None = None
        self.base: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.motion = np.zeros(THUMB[::-1], np.float32)
        self.pending_since: float | None = None
        self.still_since: float | None = None
        # How different the last settled screen was from the one before (0 to 1).
        self.last_change = 0.0

    def _frac(self, a: np.ndarray, b: np.ndarray) -> float:
        moving = np.abs(a - b) > self.cell_delta
        mask = self.motion < self.motion_cut
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
        if self.motion_ref is None:
            self.motion_ref = (now, th)
        elif now - self.motion_ref[0] >= self.motion_period_s:
            step = np.abs(th - self.motion_ref[1]) > self.motion_step
            self.motion = (1 - self.motion_rate) * self.motion + self.motion_rate * step
            self.motion_ref = (now, th)
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
            self.last_change = 1.0 if self.base is None else self._frac(th, self.base)
            self.base = th
            self.pending_since = None
            self.still_since = None
            return True
        return False
