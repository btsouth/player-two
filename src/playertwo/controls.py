"""Hands for Player Two: a virtual controller plugged in as player two.

On Linux it is a uinput Xbox 360 pad, which SDL, Steam Input and Proton already know.
On Windows it uses ViGEm through vgamepad. A keyboard backend covers PC-only games, and
a dry-run backend records presses for tests.
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from dataclasses import dataclass, field
from typing import ClassVar

BUTTONS = ("a", "b", "x", "y", "lb", "rb", "lt", "rt", "start", "back", "guide", "ls", "rs",
           "up", "down", "left", "right")
DIRECTIONS = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
    "up_left": (-0.707, -0.707), "up_right": (0.707, -0.707),
    "down_left": (-0.707, 0.707), "down_right": (0.707, 0.707),
}


class Controls:
    # Set when this isn't what the player asked for (no controller, keys instead).
    problem = ""
    kind = "base"

    def press(self, button: str, hold_ms: int = 100) -> None:
        raise NotImplementedError

    def stick(self, which: str, direction: str, ms: int, strength: float = 1.0) -> None:
        raise NotImplementedError

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        """Hold the left stick at `angle` (0 up, 90 right) until the next call; None lets
        go. For walking somewhere in a loop that keeps correcting the direction."""
        raise NotImplementedError

    def sequence(self, buttons: list[str], gap_ms: int = 140, hold_ms: int = 100) -> None:
        for i, b in enumerate(buttons):
            self.press(b, hold_ms)
            if i < len(buttons) - 1:
                time.sleep(gap_ms / 1000)

    def close(self) -> None:
        pass


@dataclass
class DryRun(Controls):
    kind = "dry-run"
    log: list[str] = field(default_factory=list)

    def press(self, button: str, hold_ms: int = 100) -> None:
        self.log.append(button)

    def stick(self, which: str, direction: str, ms: int, strength: float = 1.0) -> None:
        self.log.append(f"{which}:{direction}:{ms}")

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        self.log.append("steer:off" if angle is None else f"steer:{round(angle)}")


def steer_dirs(angle: float | None) -> list[str]:
    """The d-pad directions closest to a stick angle (0 up, 90 right): one or two."""
    if angle is None:
        return []
    sector = int(((angle % 360) + 22.5) // 45) % 8
    return [["up"], ["up", "right"], ["right"], ["down", "right"], ["down"],
            ["down", "left"], ["left"], ["up", "left"]][sector]


class LinuxPad(Controls):
    kind = "gamepad"

    def __init__(self):
        from evdev import AbsInfo, UInput
        from evdev import ecodes as e

        self.e = e
        stick = AbsInfo(0, -32768, 32767, 16, 128, 0)
        trig = AbsInfo(0, 0, 255, 0, 0, 0)
        hat = AbsInfo(0, -1, 1, 0, 0, 0)
        caps = {
            e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y, e.BTN_TL, e.BTN_TR, e.BTN_SELECT,
                       e.BTN_START, e.BTN_MODE, e.BTN_THUMBL, e.BTN_THUMBR],
            e.EV_ABS: [(e.ABS_X, stick), (e.ABS_Y, stick), (e.ABS_RX, stick),
                       (e.ABS_RY, stick), (e.ABS_Z, trig), (e.ABS_RZ, trig),
                       (e.ABS_HAT0X, hat), (e.ABS_HAT0Y, hat)],
        }
        # Same ids as a wired Xbox 360 pad, so games pick the right button layout.
        self.ui = UInput(caps, name="Microsoft X-Box 360 pad", vendor=0x045E,
                         product=0x028E, version=0x110, bustype=e.BUS_USB)
        self.keys = {"a": e.BTN_A, "b": e.BTN_B, "x": e.BTN_X, "y": e.BTN_Y, "lb": e.BTN_TL,
                     "rb": e.BTN_TR, "back": e.BTN_SELECT, "start": e.BTN_START,
                     "guide": e.BTN_MODE, "ls": e.BTN_THUMBL, "rs": e.BTN_THUMBR}
        self.hats = {"up": (e.ABS_HAT0Y, -1), "down": (e.ABS_HAT0Y, 1),
                     "left": (e.ABS_HAT0X, -1), "right": (e.ABS_HAT0X, 1)}
        self.triggers = {"lt": e.ABS_Z, "rt": e.ABS_RZ}

    def _syn(self) -> None:
        self.ui.syn()

    def press(self, button: str, hold_ms: int = 100) -> None:
        e = self.e
        if button in self.keys:
            self.ui.write(e.EV_KEY, self.keys[button], 1)
            self._syn()
            time.sleep(hold_ms / 1000)
            self.ui.write(e.EV_KEY, self.keys[button], 0)
        elif button in self.hats:
            axis, v = self.hats[button]
            self.ui.write(e.EV_ABS, axis, v)
            self._syn()
            time.sleep(hold_ms / 1000)
            self.ui.write(e.EV_ABS, axis, 0)
        elif button in self.triggers:
            self.ui.write(e.EV_ABS, self.triggers[button], 255)
            self._syn()
            time.sleep(hold_ms / 1000)
            self.ui.write(e.EV_ABS, self.triggers[button], 0)
        else:
            raise ValueError(f"unknown button {button}")
        self._syn()

    def stick(self, which: str, direction: str, ms: int, strength: float = 1.0) -> None:
        e = self.e
        ax, ay = (e.ABS_X, e.ABS_Y) if which == "left" else (e.ABS_RX, e.ABS_RY)
        dx, dy = DIRECTIONS[direction]
        v = 32767 * max(0.1, min(1.0, strength))
        self.ui.write(e.EV_ABS, ax, int(dx * v))
        self.ui.write(e.EV_ABS, ay, int(dy * v))
        self._syn()
        time.sleep(ms / 1000)
        self.ui.write(e.EV_ABS, ax, 0)
        self.ui.write(e.EV_ABS, ay, 0)
        self._syn()

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        e = self.e
        x = y = 0
        if angle is not None:
            v = 32767 * max(0.1, min(1.0, strength))
            r = math.radians(angle)
            x, y = int(math.sin(r) * v), int(-math.cos(r) * v)
        self.ui.write(e.EV_ABS, e.ABS_X, x)
        self.ui.write(e.EV_ABS, e.ABS_Y, y)
        # The d-pad too: many 2D games walk only with it and ignore the stick.
        hat = {e.ABS_HAT0X: 0, e.ABS_HAT0Y: 0}
        for d in steer_dirs(angle):
            axis, v = self.hats[d]
            hat[axis] = v
        for axis, v in hat.items():
            self.ui.write(e.EV_ABS, axis, v)
        self._syn()

    def close(self) -> None:
        self.ui.close()


class WindowsPad(Controls):
    kind = "gamepad"

    def __init__(self):
        import vgamepad as vg

        self.vg = vg
        self.pad = vg.VX360Gamepad()
        B = vg.XUSB_BUTTON
        self.keys = {"a": B.XUSB_GAMEPAD_A, "b": B.XUSB_GAMEPAD_B, "x": B.XUSB_GAMEPAD_X,
                     "y": B.XUSB_GAMEPAD_Y, "lb": B.XUSB_GAMEPAD_LEFT_SHOULDER,
                     "rb": B.XUSB_GAMEPAD_RIGHT_SHOULDER, "back": B.XUSB_GAMEPAD_BACK,
                     "start": B.XUSB_GAMEPAD_START, "guide": B.XUSB_GAMEPAD_GUIDE,
                     "ls": B.XUSB_GAMEPAD_LEFT_THUMB, "rs": B.XUSB_GAMEPAD_RIGHT_THUMB,
                     "up": B.XUSB_GAMEPAD_DPAD_UP, "down": B.XUSB_GAMEPAD_DPAD_DOWN,
                     "left": B.XUSB_GAMEPAD_DPAD_LEFT, "right": B.XUSB_GAMEPAD_DPAD_RIGHT}

    def press(self, button: str, hold_ms: int = 100) -> None:
        if button in ("lt", "rt"):
            setter = self.pad.left_trigger if button == "lt" else self.pad.right_trigger
            setter(255)
            self.pad.update()
            time.sleep(hold_ms / 1000)
            setter(0)
        else:
            self.pad.press_button(self.keys[button])
            self.pad.update()
            time.sleep(hold_ms / 1000)
            self.pad.release_button(self.keys[button])
        self.pad.update()

    def stick(self, which: str, direction: str, ms: int, strength: float = 1.0) -> None:
        dx, dy = DIRECTIONS[direction]
        s = max(0.1, min(1.0, strength))
        move = self.pad.left_joystick_float if which == "left" else self.pad.right_joystick_float
        move(dx * s, -dy * s)
        self.pad.update()
        time.sleep(ms / 1000)
        move(0.0, 0.0)
        self.pad.update()

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        if angle is None:
            self.pad.left_joystick_float(0.0, 0.0)
        else:
            s = max(0.1, min(1.0, strength))
            r = math.radians(angle)
            self.pad.left_joystick_float(math.sin(r) * s, math.cos(r) * s)
        # The d-pad too: many 2D games walk only with it and ignore the stick.
        held = steer_dirs(angle)
        for d in ("up", "down", "left", "right"):
            if d in held:
                self.pad.press_button(button=self.keys[d])
            else:
                self.pad.release_button(button=self.keys[d])
        self.pad.update()


class WaylandKeys(Controls):
    """Keyboard through the compositor's virtual keyboard (wtype). Safe in a nested session."""

    kind = "keyboard"
    MAP: ClassVar[dict[str, str]] = {"a": "Return", "b": "Escape", "x": "BackSpace", "y": "Tab", "start": "Escape",
           "back": "Tab", "up": "Up", "down": "Down", "left": "Left", "right": "Right",
           "lb": "q", "rb": "e", "lt": "z", "rt": "c", "ls": "Shift_L", "rs": "v", "guide": "F1"}
    STICK: ClassVar[dict[str, str]] = {"up": "w", "down": "s", "left": "a", "right": "d"}

    def __init__(self, keymap: dict[str, str] | None = None, focus=None):
        self.map = {**self.MAP, **(keymap or {})}
        # Keys go to whatever has focus, so hand focus back to the game first.
        self.focus = focus

    def _run(self, *args: str) -> None:
        import subprocess

        if self.focus:
            self.focus()
        try:
            subprocess.run(["wtype", *args], check=True, timeout=5, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ControlsError(f"keyboard keys didn't go through ({exc})") from exc

    @staticmethod
    def works() -> bool:
        """Whether wtype is there and this desktop takes a virtual keyboard (GNOME and
        X11 don't). Presses nothing."""
        import shutil
        import subprocess

        if shutil.which("wtype") is None:
            return False
        try:
            return subprocess.run(["wtype", "-s", "0"], check=False, capture_output=True,
                                  timeout=3).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def press(self, button: str, hold_ms: int = 100) -> None:
        key = self.map[button]
        self._run("-P", key, "-s", str(hold_ms), "-p", key)

    def stick(self, which: str, direction: str, ms: int, strength: float = 1.0) -> None:
        keys = [self.STICK[d] for d in direction.split("_")]
        args = [a for k in keys for a in ("-P", k)] + ["-s", str(ms)]
        args += [a for k in keys for a in ("-p", k)]
        self._run(*args)

    _held: tuple[list[str], object] | None = None

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        import subprocess

        # The d-pad keys (arrows): 2D games walk with those, and some use WASD for
        # other things.
        keys = [self.map[d] for d in steer_dirs(angle)]
        if self._held and self._held[0] == keys and self._held[1].poll() is None:
            return
        if self._held:
            old, proc = self._held
            self._held = None
            proc.terminate()
            proc.wait(timeout=2)
            # Let go explicitly too, in case the compositor keeps keys of a closed
            # virtual keyboard down.
            self._run(*[a for k in old for a in ("-p", k)])
        if keys:
            if self.focus:
                self.focus()
            args = [a for k in keys for a in ("-P", k)] + ["-s", "60000"]
            try:
                self._held = (keys, subprocess.Popen(["wtype", *args]))
            except OSError as exc:
                raise ControlsError(f"keyboard keys didn't go through ({exc})") from exc

    def close(self) -> None:
        """Let go of held keys: the wtype holding them outlives Player Two otherwise."""
        if self._held is None:
            return
        old, proc = self._held
        self._held = None
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            proc.kill()
        with contextlib.suppress(ControlsError):
            self._run(*[a for k in old for a in ("-p", k)])


class HyprlandKeys(WaylandKeys):
    """Keys through Hyprland's send_key_state, with the keyboard's real key codes. For
    apps that read key codes rather than key names (RetroArch), where wtype's own keymap
    lands on the wrong keys. PLAYER_TWO_KEYS=hyprland picks it."""

    def _state(self, key: str, state: str) -> None:
        import subprocess

        lua = f'hl.dsp.send_key_state({{ mods = "", key = "{key}", state = "{state}" }})'
        try:
            subprocess.run(["hyprctl", "dispatch", lua], check=True, timeout=5,
                           capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ControlsError(f"keyboard keys didn't go through ({exc})") from exc

    def _run(self, *args: str) -> None:
        """wtype's -P KEY (down), -p KEY (up) and -s MS (pause), done one by one."""
        if self.focus:
            self.focus()
        for flag, arg in zip(args[::2], args[1::2], strict=False):
            if flag == "-s":
                time.sleep(int(arg) / 1000)
            else:
                self._state(arg, "down" if flag == "-P" else "up")

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        keys = [self.map[d] for d in steer_dirs(angle)]
        old = self._held[0] if self._held else []
        if keys == old:
            return
        if self.focus:
            self.focus()
        for k in old:
            if k not in keys:
                self._state(k, "up")
        for k in keys:
            if k not in old:
                self._state(k, "down")
        self._held = (keys, None) if keys else None

    def close(self) -> None:
        if self._held:
            keys, self._held = self._held[0], None
            for k in keys:
                with contextlib.suppress(ControlsError):
                    self._state(k, "up")


class WindowsKeys(Controls):
    """Keyboard on Windows through SendInput with scan codes, which games read."""

    kind = "keyboard"
    # Scan codes; 0x100 marks an extended key (the arrows).
    SCAN: ClassVar[dict[str, int]] = {
        "Return": 0x1C, "Escape": 0x01, "BackSpace": 0x0E, "Tab": 0x0F, "Up": 0x148,
        "Down": 0x150, "Left": 0x14B, "Right": 0x14D, "q": 0x10, "e": 0x12, "z": 0x2C,
        "c": 0x2E, "Shift_L": 0x2A, "v": 0x2F, "F1": 0x3B, "w": 0x11, "a": 0x1E, "s": 0x1F,
        "d": 0x20}

    def __init__(self, keymap: dict[str, str] | None = None, focus=None):
        import ctypes
        from ctypes import wintypes

        # Keys go to whatever has focus, so hand focus back to the game first.
        self.focus = focus

        class KEYBDINPUT(ctypes.Structure):
            _fields_: ClassVar = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                                  ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                                  ("dwExtraInfo", ctypes.c_size_t)]

        class INPUT(ctypes.Structure):
            # The union's largest member (MOUSEINPUT) sets the size; pad to it.
            _fields_: ClassVar = [("type", wintypes.DWORD), ("ki", KEYBDINPUT),
                                  ("_pad", ctypes.c_byte * 8)]

        self.ctypes, self.INPUT, self.KEYBDINPUT = ctypes, INPUT, KEYBDINPUT
        self.send = ctypes.windll.user32.SendInput
        self.map = {**WaylandKeys.MAP, **(keymap or {})}

    def _key(self, name: str, up: bool) -> None:
        code = self.SCAN[name]
        flags = 0x0008 | (0x0002 if up else 0) | (0x0001 if code & 0x100 else 0)
        inp = self.INPUT(type=1, ki=self.KEYBDINPUT(0, code & 0xFF, flags, 0, 0))
        # 0 sent: Windows blocked it, as it does for keys aimed at a game running as
        # administrator when Player Two isn't.
        if not self.send(1, self.ctypes.byref(inp), self.ctypes.sizeof(inp)):
            raise ControlsError("SendInput sent no keys", say=BLOCKED)

    def press(self, button: str, hold_ms: int = 100) -> None:
        key = self.map[button]
        if self.focus:
            self.focus()
        self._key(key, False)
        time.sleep(hold_ms / 1000)
        self._key(key, True)

    def stick(self, which: str, direction: str, ms: int, strength: float = 1.0) -> None:
        keys = [WaylandKeys.STICK[d] for d in direction.split("_")]
        if self.focus:
            self.focus()
        for k in keys:
            self._key(k, False)
        time.sleep(ms / 1000)
        for k in keys:
            self._key(k, True)

    _held: ClassVar[list[str]] = []

    def steer(self, angle: float | None, strength: float = 1.0) -> None:
        keys = [self.map.get(d, WaylandKeys.MAP[d]) for d in steer_dirs(angle)]
        if keys == self._held:
            return
        if keys and self.focus:
            self.focus()
        for k in self._held:
            if k not in keys:
                self._key(k, True)
        for k in keys:
            if k not in self._held:
                self._key(k, False)
        self._held = keys

    def close(self) -> None:
        self.steer(None)


class ControlsError(RuntimeError):
    """Buttons or keys couldn't be pressed. `say` is how to tell the player, if known."""

    def __init__(self, *args, say: str = ""):
        super().__init__(*args)
        self.say = say


BLOCKED = ("Windows is blocking my key presses. That happens when the game runs as "
           "administrator: start the game normally, or start me as administrator too.")


def keymap_from_env() -> dict[str, str]:
    """Keys of your own for some buttons, e.g. an emulator's: PLAYER_TWO_KEYMAP="start=F5,
    back=F6" (button=key, key names as wtype or Windows take them)."""
    import os

    pairs = (p.split("=", 1) for p in os.environ.get("PLAYER_TWO_KEYMAP", "").split(",")
             if "=" in p)
    return {b.strip(): k.strip() for b, k in pairs if b.strip() in WaylandKeys.MAP}


def keyboard(focus=None) -> Controls:
    keymap = keymap_from_env()
    import os

    if sys.platform == "win32":
        return WindowsKeys(keymap, focus=focus)
    if os.environ.get("PLAYER_TWO_KEYS") == "hyprland":
        return HyprlandKeys(keymap, focus=focus)
    return WaylandKeys(keymap, focus=focus)


def pad_problem(exc: BaseException) -> str:
    """Why the virtual controller couldn't be plugged in, said to the player."""
    if sys.platform == "win32":
        import platform

        if platform.machine().upper() == "ARM64":
            return ("Virtual controllers don't work on ARM Windows computers, so I'll press "
                    "keyboard keys instead.")
        return ("I couldn't plug in a virtual controller. It needs the ViGEmBus driver; "
                "run the Player Two installer again to add it.")
    if isinstance(exc, PermissionError) or "permission" in str(exc).lower():
        return ("I couldn't plug in a virtual controller because I don't have permission "
                "to. Run player-two setup to fix that.")
    return "I couldn't plug in a virtual controller."


def open_controls(kind: str = "auto", focus=None) -> Controls:
    """The controller, or keyboard keys when it can't be plugged in.

    A keyboard stand-in records why in `problem`, so the session can say so.
    """
    if kind == "dry-run":
        return DryRun()
    if kind == "keyboard":
        return keyboard(focus)
    try:
        return WindowsPad() if sys.platform == "win32" else LinuxPad()
    except Exception as exc:
        if kind == "gamepad":
            raise
        try:
            keys = keyboard(focus)
            if isinstance(keys, WaylandKeys) and not WaylandKeys.works():
                raise ControlsError("no virtual keyboard on this desktop")
        except Exception:  # noqa: BLE001  no keyboard either: record presses only
            keys = DryRun()
            keys.problem = (pad_problem(exc) + " I can't press keyboard keys on this desktop "
                            "either, so for now I can only watch and talk.")
            return keys
        keys.problem = pad_problem(exc) + " I'll press keyboard keys instead."
        return keys
