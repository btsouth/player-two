"""Fitting into the desktop: the companion window, the talk key and the app launcher.

The companion window is the local web page in a Chromium-family browser's app mode
(no tabs or address bar), with a profile of its own so Player Two can close it again.
On Hyprland it opens without taking focus from the game, on another monitor when there
is one. The talk key is a hold bind in the compositor (Omarchy or plain Hyprland) or, on
Windows, watched by Player Two itself.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path
from typing import ClassVar

from .config import CACHE_DIR

TALK_KEY = "F8"
BROWSERS = ("chromium", "google-chrome-stable", "google-chrome", "brave", "brave-browser",
            "microsoft-edge-stable", "microsoft-edge", "vivaldi-stable", "helium")


def browser() -> str | None:
    if sys.platform == "win32":
        for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES")):
            if base:
                edge = Path(base) / "Microsoft/Edge/Application/msedge.exe"
                if edge.exists():
                    return str(edge)
        return shutil.which("msedge")
    return next((p for b in BROWSERS if (p := shutil.which(b))), None)


def hypr(*args: str) -> str:
    try:
        return subprocess.run(["hyprctl", *args], capture_output=True, text=True,
                              timeout=2, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def lua_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def window_workspace(game: dict | None) -> str | None:
    """Where the companion window should open: the active workspace of a monitor the game
    isn't on, else none (the current workspace, behind a fullscreen game)."""
    import json

    try:
        monitors = json.loads(hypr("-j", "monitors") or "[]")
    except ValueError:
        return None
    game_mon = game.get("monitor") if game else None
    others = [m for m in monitors if m.get("id") != game_mon and not m.get("disabled")]
    if game is None or not others:
        return None
    others.sort(key=lambda m: not m.get("focused"))
    return str(others[0]["activeWorkspace"]["id"])


# Chromium names app windows after the address it opened: the host for a web page
# ("chrome-127.0.0.1__-Default"), the path for the start page ("..._player-two-window...").
PAGE = "player-two-window.html"
APP_CLASS = r".*(127\.0\.0\.1__|player-two-window).*"


def private_dir() -> Path:
    """A folder only this user can read: the runtime dir, else one under the cache."""
    run = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(run) / "player-two" if run and os.path.isdir(run) else CACHE_DIR / "private"
    base.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        base.chmod(0o700)
    return base


def start_page(url: str) -> Path:
    """A page that goes straight to `url`, for opening the window by.

    The address holds this run's secret, and a browser's command line can be read by
    every user of the PC; this file can only be read by the player.
    """
    import html
    import json

    path = private_dir() / PAGE
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write('<!doctype html><meta charset="utf-8"><title>Player Two</title>'
                f'<meta http-equiv="refresh" content="0;url={html.escape(url)}">'
                f"<script>location.replace({json.dumps(url)})</script>\n")
    os.replace(tmp, path)
    return path


def place_rule(workspace: str | None) -> None:
    """Open the window without focus (and on `workspace`), so the game keeps the keys."""
    ws = f', workspace = {lua_str(workspace + " silent")}' if workspace else ""
    lua = (f'hl.window_rule({{ name = "player-two-window", match = {{ class = '
           f'{lua_str(APP_CLASS)} }}, no_initial_focus = true{ws} }})')
    if "ok" in hypr("eval", lua):
        return
    rule = f"noinitialfocus, class:{APP_CLASS}"
    hypr("keyword", "windowrulev2", rule)
    if workspace:
        hypr("keyword", "windowrulev2", f"workspace {workspace} silent, class:{APP_CLASS}")


def profile_owner(profile: Path) -> int | None:
    """The browser process already running this profile (Chromium's lock names it).

    Linux only: Windows Chromium keeps no such lock, and there os.kill(pid, 0) would end
    the process instead of checking it.
    """
    if sys.platform == "win32":
        return None
    try:
        pid = int(os.readlink(profile / "SingletonLock").rsplit("-", 1)[1])
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        return None
    return pid


def stale_window(profile: Path, wait_s: float = 3.0) -> None:
    """End the browser still running `profile`, if any, and wait for it to go."""
    import signal

    pid = profile_owner(profile)
    if pid is None:
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    end = time.monotonic() + wait_s
    while profile_owner(profile) == pid and time.monotonic() < end:
        time.sleep(0.05)
    if profile_owner(profile) == pid:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


class CompanionWindow:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.profile = CACHE_DIR / "window"
        self.tab_opened = False

    def alive(self) -> bool:
        if self.tab_opened:
            return True
        if self.proc is not None and self.proc.poll() is None:
            return True
        return self.pid is not None and profile_owner(self.profile) == self.pid

    def open(self, url: str, game: dict | None = None) -> bool:
        exe = browser()
        if exe is None:
            import webbrowser

            # A tab in the default browser can't be watched or closed: open it once per
            # session, or every "show the window" adds another.
            if self.tab_opened:
                return True
            try:
                self.tab_opened = webbrowser.open(start_page(url).as_uri())
            except OSError:
                return False
            return self.tab_opened
        profile = self.profile
        profile.mkdir(parents=True, exist_ok=True)
        # A window left by a session that crashed has that run's secret and can't
        # reconnect: close it and open a fresh one (asked to open the page again, the
        # running browser would only add a second window).
        stale_window(profile)
        if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
            place_rule(window_workspace(game))
        try:
            page = start_page(url).as_uri()
        except OSError:
            return False
        cmd = [exe, f"--app={page}", f"--user-data-dir={profile}", "--no-first-run",
               "--no-default-browser-check", "--window-size=560,900",
               "--disable-features=Translate", "--autoplay-policy=no-user-gesture-required"]
        flags = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, **flags)
        except OSError:
            return False
        self.pid = self.proc.pid
        return True

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        elif self.pid is not None and profile_owner(self.profile) == self.pid:
            import signal

            os.kill(self.pid, signal.SIGTERM)


# The talk key.

BIND_MARK = "-- Player Two: hold to talk"


def omarchy_bindings() -> Path | None:
    p = Path.home() / ".config/hypr/bindings.lua"
    return p if p.exists() else None


def hyprland_conf() -> Path | None:
    p = Path.home() / ".config/hypr/hyprland.conf"
    return p if p.exists() else None


def player_two_cmd() -> str:
    """How the compositor should run Player Two: the installed command if there is one."""
    exe = shutil.which("player-two")
    return shlex.quote(exe) if exe else f"{shlex.quote(sys.executable)} -m playertwo.cli"


def talk_key_bound(settings=None) -> bool:
    if sys.platform == "win32":
        return True  # Player Two watches the key itself
    if settings is not None and settings.talk_key_set:
        return True
    for p in (omarchy_bindings(), hyprland_conf()):
        if p and "ctl talk-start" in p.read_text(encoding="utf-8", errors="replace"):
            return True
    return gnome() and gnome_talk_key_bound()


# GNOME's shortcuts run a command on press only, so there the key toggles listening.
GNOME_KEYS = "org.gnome.settings-daemon.plugins.media-keys"
GNOME_PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/player-two/"


def gnome() -> bool:
    return "gnome" in os.environ.get("XDG_CURRENT_DESKTOP", "").lower() and bool(
        shutil.which("gsettings"))


def _gsettings(*args: str) -> str:
    try:
        return subprocess.run(["gsettings", *args], capture_output=True, text=True,
                              timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gnome_talk_key_bound() -> bool:
    return GNOME_PATH in _gsettings("get", GNOME_KEYS, "custom-keybindings") and "ctl talk" in (
        _gsettings("get", f"{GNOME_KEYS}.custom-keybinding:{GNOME_PATH}", "command"))


def install_gnome_talk_key() -> str:
    import ast

    raw = _gsettings("get", GNOME_KEYS, "custom-keybindings")
    try:
        paths = ast.literal_eval(raw.removeprefix("@as ")) if raw else []
    except (ValueError, SyntaxError):
        return "I couldn't read your GNOME keyboard shortcuts, so I left them alone."
    if GNOME_PATH not in paths:
        paths.append(GNOME_PATH)
    key = f"{GNOME_KEYS}.custom-keybinding:{GNOME_PATH}"
    _gsettings("set", key, "name", "Player Two: talk")
    _gsettings("set", key, "command", f"{player_two_cmd()} ctl talk")
    _gsettings("set", key, "binding", TALK_KEY)
    _gsettings("set", GNOME_KEYS, "custom-keybindings", str(paths))
    if gnome_talk_key_bound():
        return f"Done. Press {TALK_KEY} to talk, and again when you're done."
    return "That didn't work. Add the shortcut by hand in Settings, Keyboard, Shortcuts."


def bind_lines(cmd: str, lua: bool) -> str:
    if lua:
        return (f"\n{BIND_MARK}\n"
                f'o.bind_hold("{TALK_KEY}", "Player Two: talk", "{cmd} ctl talk-start",\n'
                f'            "Player Two: done talking", "{cmd} ctl talk-stop")\n')
    return (f"\n# Player Two: hold to talk\n"
            f"bind = , {TALK_KEY}, exec, {cmd} ctl talk-start\n"
            f"bindr = , {TALK_KEY}, exec, {cmd} ctl talk-stop\n")


def install_talk_key() -> str:
    """Add the hold-to-talk bind to the compositor config. Returns what happened, said
    plainly. Keeps a backup and never adds it twice."""
    target, lua = omarchy_bindings(), True
    if target is None:
        target, lua = hyprland_conf(), False
    if target is None:
        return ("I couldn't find a Hyprland config. Bind a key to run "
                f"'{player_two_cmd()} ctl talk-start' on press and "
                f"'{player_two_cmd()} ctl talk-stop' on release.")
    text = target.read_text(encoding="utf-8")
    if "ctl talk-start" in text:
        return f"{TALK_KEY} is already your talk key."
    backup = target.with_name(f"{target.name}.bak.player-two-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(target, backup)
    target.write_text(text.rstrip("\n") + "\n" + bind_lines(player_two_cmd(), lua), encoding="utf-8")
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        hypr("reload")
    return f"Done. Hold {TALK_KEY} to talk. Your old config is saved as {backup.name}."


class WindowsTalkKey:
    """F8 on Windows, wherever focus is: calls back on press and release.

    A registered hot key, so the press goes to Player Two and not also to the game. If
    another program already owns F8, it falls back to watching the key, which works but
    lets the game see it too.
    """

    VK: ClassVar[dict[str, int]] = {"F8": 0x77, "F9": 0x78, "F10": 0x79}
    WM_HOTKEY = 0x0312
    MOD_NOREPEAT = 0x4000

    def __init__(self, on_press, on_release, key: str = TALK_KEY):
        self.on_press, self.on_release, self.vk = on_press, on_release, self.VK[key]
        self.running = True
        self.swallowed = False

    def _held(self) -> bool:
        import ctypes

        return bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)

    def run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(None, 1, self.MOD_NOREPEAT, self.vk):
            self._poll()
            return
        self.swallowed = True
        msg = wintypes.MSG()
        try:
            while self.running and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message != self.WM_HOTKEY:
                    continue
                self.on_press()
                while self.running and self._held():
                    time.sleep(0.015)
                self.on_release()
        finally:
            user32.UnregisterHotKey(None, 1)

    def _poll(self) -> None:
        down = False
        while self.running:
            now = self._held()
            if now != down:
                down = now
                (self.on_press if down else self.on_release)()
            time.sleep(0.015)


# The app launcher.

def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))


def icon_svg() -> bytes:
    return (resources.files("playertwo") / "ui" / "icon.svg").read_bytes()


def install_desktop_entry() -> list[Path]:
    """A launcher entry and icon, so Player Two starts like any other app."""
    if sys.platform == "win32":
        return [install_windows_shortcut()]
    icon = data_home() / "icons/hicolor/scalable/apps/player-two.svg"
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(icon_svg())
    entry = data_home() / "applications/player-two.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Player Two\n"
        "GenericName=Game narrator and co-op partner\n"
        "Comment=Tells you what's on screen in your game and plays when you ask\n"
        f"Exec={player_two_cmd()}\n"
        "Icon=player-two\n"
        "Terminal=false\n"
        "Categories=Game;Accessibility;Utility;\n"
        "Keywords=accessibility;blind;screen reader;narrator;game;\n"
        "StartupNotify=false\n")
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(entry.parent)], check=False,
                       capture_output=True)
    return [entry, icon]


def install_windows_shortcut() -> Path:
    """A Start menu shortcut, written with PowerShell's WScript.Shell.

    Its console window starts minimized, so launching Player Two doesn't take focus from
    the game; Player Two talks, so the console isn't needed.
    """
    start = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs"
    link = start / "Player Two.lnk"
    exe = shutil.which("player-two") or sys.executable
    args = "" if exe != sys.executable else "-m playertwo.cli"
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:P2_LINK);"
          "$s.TargetPath=$env:P2_EXE;$s.Arguments=$env:P2_ARGS;$s.WindowStyle=7;"
          "$s.Description='Player Two';$s.Save()")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, timeout=60, check=False,
                       env={**os.environ, "P2_LINK": str(link), "P2_EXE": exe,
                            "P2_ARGS": args})
    if r.returncode != 0 or not link.exists():
        raise OSError(f"PowerShell couldn't make the shortcut: {r.stderr.strip()[:200]}")
    return link
