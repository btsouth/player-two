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


class NoGame(Exception):
    """No window that looks like a game is open yet."""


class GameHidden(NoGame):
    """The game is open but can't be seen right now (minimized, or covered on Windows)."""


BLANK = ("Your game looks solid black to me. If it's in exclusive fullscreen, switch it "
         "to borderless or windowed in its display settings, and I'll see it.")


class Capture:
    # Something to tell the player about the capture itself, once (the watch loop says it).
    notice = ""

    def grab(self) -> Frame:
        raise NotImplementedError

    def game(self) -> str:
        """The game's name, when the capture knows which game it is looking at."""
        return ""

    def game_key(self) -> str:
        """Which game this is, steady while its window title changes."""
        return self.game()

    def focus(self) -> None:
        """Give the game keyboard focus (for keyboard controls). Best effort."""


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

    The target is the window whose class or title contains `window`, or else the game
    window games.pick_game finds (a Steam game first). With no game open, grab raises
    NoGame, unless `whole_screen` asks for the monitor instead.
    """

    def __init__(self, window: str = "", output: str | None = None,
                 whole_screen: bool = False):
        self.want = window
        self.output = output
        self.whole_screen = whole_screen
        self.target: dict | None = None
        self._checked = 0.0

    def pick(self) -> dict | None:
        from .games import pick_game

        return pick_game(hypr_clients(), self.want)

    def game(self) -> str:
        from .games import game_name

        return game_name(self.target) if self.target else ""

    def game_key(self) -> str:
        from .games import game_key

        return game_key(self.target) if self.target else ""

    def focus(self) -> None:
        if self.target and self.target.get("address"):
            hypr_focus(self.target["address"])

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
            # Older Hyprland can't capture one window: show the monitor it's on.
            if self.output is None and self.target.get("monitor") is not None:
                self.output = monitor_name(self.target["monitor"])
        elif not self.whole_screen:
            raise NoGame
        cmd = ["grim", "-t", "ppm"] + (["-o", self.output] if self.output else []) + ["-"]
        data = subprocess.check_output(cmd, timeout=3)
        return Frame(Image.open(io.BytesIO(data)), now)


def monitor_name(monitor_id: int) -> str | None:
    try:
        mons = json.loads(subprocess.check_output(["hyprctl", "-j", "monitors"], timeout=2))
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return next((m["name"] for m in mons if m.get("id") == monitor_id), None)


class MssCapture(Capture):
    """Windows and X11."""

    def __init__(self, monitor: int = 1):
        import mss

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]

    def grab(self) -> Frame:
        shot = self.sct.grab(self.monitor)
        return Frame(Image.frombytes("RGB", shot.size, shot.rgb), time.monotonic())

    def game(self) -> str:
        from .games import running_steam_game

        return running_steam_game()


class WindowsCapture(Capture):
    """Captures the game's window on Windows.

    The game is found the way HyprlandCapture finds it (games.pick_game over win32's
    window list). In front, its part of the screen is copied; behind another window, its
    own pixels are asked for, and if Windows can't give them, grab raises GameHidden so
    Player Two never describes some other program as the game.

    Exclusive fullscreen games can't be copied by any of this and come out black. When a
    game has shown nothing but black for a while, `notice` says how to fix it, once per
    game window (a loading screen in a game that has shown pictures isn't that).
    """

    BLANK_S = 20.0

    def __init__(self, window: str = "", whole_screen: bool = False):
        import mss

        from . import win32

        win32.dpi_aware()
        self.w = win32
        self.sct = mss.mss()
        self.want = window
        self.whole_screen = whole_screen
        self.target: dict | None = None
        self._checked = 0.0
        self._dark_since: float | None = None
        self._seen: set[int] = set()  # windows that have shown a picture, or been warned

    def pick(self) -> dict | None:
        from .games import pick_game

        return pick_game(self.w.clients(), self.want)

    def game(self) -> str:
        from .games import game_name

        return game_name(self.target) if self.target else ""

    def game_key(self) -> str:
        from .games import game_key

        return game_key(self.target) if self.target else ""

    def focus(self) -> None:
        if self.target and self.w.alive(self.target["hwnd"]):
            self.w.focus(self.target["hwnd"])

    def _region(self, rect: tuple[int, int, int, int]) -> Image.Image | None:
        screen = self.sct.monitors[0]
        left, top = max(rect[0], screen["left"]), max(rect[1], screen["top"])
        right = min(rect[2], screen["left"] + screen["width"])
        bottom = min(rect[3], screen["top"] + screen["height"])
        if right - left < 16 or bottom - top < 16:
            return None
        shot = self.sct.grab({"left": left, "top": top, "width": right - left,
                              "height": bottom - top})
        return Image.frombytes("RGB", shot.size, shot.rgb)

    def _blank_check(self, img: Image.Image, hwnd: int, now: float) -> None:
        if hwnd in self._seen:
            return
        if img.convert("L").resize((64, 36)).getextrema()[1] > 12:
            self._seen.add(hwnd)
            self._dark_since = None
            return
        self._dark_since = self._dark_since or now
        if now - self._dark_since > self.BLANK_S:
            self._seen.add(hwnd)
            self.notice = BLANK

    def grab(self) -> Frame:
        now = time.monotonic()
        t = self.target
        alive = t is not None and self.w.alive(t["hwnd"])
        # A minimized game stays the game: minimized, it no longer covers its screen,
        # which may be all that made it look like one.
        if not alive or (now - self._checked > 2.0 and not self.w.minimized(t["hwnd"])):
            picked = self.pick()
            self._checked = now
            if (picked and picked["hwnd"]) != (t and t["hwnd"]):
                self._dark_since = None
            self.target = t = picked
        if t is None:
            if not self.whole_screen:
                raise NoGame
            shot = self.sct.grab(self.sct.monitors[1])
            return Frame(Image.frombytes("RGB", shot.size, shot.rgb), now)
        hwnd = t["hwnd"]
        if self.w.minimized(hwnd):
            raise GameHidden
        img = None
        if self.w.foreground() == hwnd:
            rect = self.w.frame_rect(hwnd)
            img = self._region(rect) if rect else None
        else:
            img = self.w.print_window(hwnd)
            if img is None or img.convert("L").getextrema()[1] <= 12:
                raise GameHidden
        if img is None:
            raise GameHidden
        self._blank_check(img, hwnd, now)
        return Frame(img, now, label=t.get("title", ""))


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


def auto_capture(window: str = "", capture: str = "auto") -> Capture:
    if sys.platform == "win32":
        return WindowsCapture(window, whole_screen=capture == "screen")
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("grim"):
        return HyprlandCapture(window, whole_screen=capture == "screen")
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

    def accept(self, frame: Frame) -> None:
        """Count this frame as already seen: only a change from it is news."""
        th = thumb(frame.image)
        self.base = self.prev = th
        self.pending_since = None
        self.still_since = None

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
