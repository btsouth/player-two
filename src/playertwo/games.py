"""Finding the game: which window it is and what it's called.

Steam games on Linux get a window class like `steam_app_1113000`; the number is the Steam
app id, and the game's real name is in that app's manifest in one of the Steam libraries.
Window titles are often useless ("Persona4 GOLDEN [REV: 4519] 121fps"), so the manifest
name is what Player Two says and tells the model.
"""

from __future__ import annotations

import os
import re
import sys
from functools import cache
from pathlib import Path

# Windows that are never the game, by the words of their class: terminals, browsers, chat,
# editors, launchers, settings, and Player Two's own window.
NOT_GAME_WORDS = {
    "alacritty", "kitty", "foot", "ghostty", "wezterm", "konsole", "terminal", "xterm",
    "terminator", "tilix", "chrome", "chromium", "firefox", "brave", "vivaldi", "opera",
    "edge", "msedge", "zen", "librewolf", "qutebrowser", "discord", "vesktop", "slack",
    "signal", "telegram", "telegramdesktop", "element", "t3code", "code", "cursor", "zed",
    "neovide", "obsidian", "nautilus", "thunar", "dolphin", "pavucontrol", "wiremix",
    "blueberry", "impala", "walker", "rofi", "wofi", "steam", "steamwebhelper", "lutris",
    "heroic", "bottles", "1password", "bitwarden", "obs", "spotify", "localsend",
    "mpv", "vlc", "celluloid", "totem", "btop", "htop", "thunderbird", "libreoffice",
    "keepassxc", "evince", "zathura", "okular",
    # Windows programs.
    "explorer", "windowsterminal", "openconsole", "conhost", "cmd", "powershell", "pwsh",
    "nvda", "jfw", "narrator", "magnify", "applicationframehost", "textinputhost",
    "searchhost", "shellexperiencehost", "startmenuexperiencehost", "lockapp",
    "systemsettings", "taskmgr", "notepad", "epicgameslauncher", "galaxyclient",
    "eadesktop", "ubisoftconnect", "upc", "playnite", "gamebar", "xboxpcapp",
}
# Emulators and game runners: a window of theirs is a game even when it isn't fullscreen.
GAME_WORDS = {
    "retroarch", "dolphin-emu", "pcsx2", "rpcs3", "ppsspp", "duckstation", "citra", "lime3ds",
    "azahar", "yuzu", "suyu", "ryujinx", "eden", "citron", "cemu", "xemu", "xenia", "melonds",
    "desmume", "mgba", "snes9x", "bsnes", "mame", "mednafen", "flycast", "vita3k", "shadps4",
    "scummvm", "dosbox", "gamescope", "minecraft", "openmw",
}
OWN_TITLE = "Player Two"
STEAM_CLASS = re.compile(r"^steam_app_(\d+)$")


def steam_roots() -> list[Path]:
    home = Path.home()
    roots = [home / ".local/share/Steam", home / ".steam/steam",
             home / ".var/app/com.valvesoftware.Steam/.local/share/Steam"]
    if sys.platform == "win32":
        roots.insert(0, Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
                     / "Steam")
    seen, out = set(), []
    for r in roots:
        try:
            key = r.resolve()
        except OSError:
            continue
        if key not in seen and (r / "steamapps").is_dir():
            seen.add(key)
            out.append(r)
    return out


def _vdf_values(text: str, key: str) -> list[str]:
    """Every value of `key` in a Valve KeyValues file (good enough for paths and names)."""
    return [m.group(1).replace("\\\\", "\\")
            for m in re.finditer(rf'"{key}"\s+"((?:[^"\\]|\\.)*)"', text, re.IGNORECASE)]


@cache
def steam_libraries() -> tuple[Path, ...]:
    libs: list[Path] = []
    for root in steam_roots():
        libs.append(root / "steamapps")
        vdf = root / "steamapps" / "libraryfolders.vdf"
        try:
            text = vdf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for p in _vdf_values(text, "path"):
            lib = Path(p) / "steamapps"
            if lib.is_dir() and lib not in libs:
                libs.append(lib)
    return tuple(libs)


@cache
def steam_game_name(app_id: str) -> str:
    for lib in steam_libraries():
        try:
            text = (lib / f"appmanifest_{app_id}.acf").read_text(encoding="utf-8",
                                                                 errors="replace")
        except OSError:
            continue
        names = _vdf_values(text, "name")
        if names:
            return names[0]
    return ""


@cache
def steam_install_dirs() -> dict[str, str]:
    """Each Steam game's install folder (lower case) and its app id."""
    dirs: dict[str, str] = {}
    for lib in steam_libraries():
        for acf in lib.glob("appmanifest_*.acf"):
            try:
                text = acf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            ids, names = _vdf_values(text, "appid"), _vdf_values(text, "installdir")
            if ids and names:
                dirs[str(lib / "common" / names[0]).lower()] = ids[0]
    return dirs


def steam_app_for_path(path: str) -> str:
    """The Steam app id of the game a program belongs to, from where it's installed."""
    dirs = steam_install_dirs()
    for parent in Path(path).parents:
        app_id = dirs.get(str(parent).lower())
        if app_id:
            return app_id
    return ""


def tidy_title(title: str) -> str:
    """A window title without the build tags, frame counters and bracketed noise."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|\b\d+(\.\d+)?\s*fps\b,?", " ", title, flags=re.IGNORECASE)
    t = re.split(r"\s+[-|–—]\s+(?:vulkan|opengl|dx\d+|direct3d|wine|proton)\b", t,
                 flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", t).strip(" -|,")


def game_name(client: dict) -> str:
    """The name to call the game in this window by."""
    m = STEAM_CLASS.match(client.get("class", "") or client.get("initialClass", ""))
    if m:
        name = steam_game_name(m.group(1))
        if name:
            return name
    return tidy_title(client.get("title", ""))


def is_own(client: dict) -> bool:
    return client.get("title", "") == OWN_TITLE or client.get("title", "").startswith(
        OWN_TITLE + " ")


def looks_like_game(client: dict) -> bool:
    """A Steam game, a Windows game under Wine or Proton, an emulator, or any other app
    that isn't a known non-game and is fullscreen. A game in a small window that's none of
    these is found with `window = "..."` instead."""
    cls = client.get("class", "") or client.get("initialClass", "")
    if STEAM_CLASS.match(cls):
        return True
    if is_own(client) or not cls:
        return False
    low = cls.lower()
    # Emulators first, by whole dotted parts ("org.DolphinEmu.dolphin-emu" is a game,
    # "org.kde.dolphin" the file manager).
    if low.endswith(".exe") or set(re.split(r"[._ ]+", low)) & GAME_WORDS:
        return True
    if set(re.split(r"[.\-_ ]+", low)) & NOT_GAME_WORDS:
        return False
    if client.get("game_dir"):  # Windows: installed in a games folder
        return True
    # Hyprland: 2 is fullscreen, 3 fullscreen and maximized (older builds say True).
    return client.get("fullscreen") in (2, 3, True)


def game_key(client: dict) -> str:
    """What stays the same for one game's window while its title changes."""
    return client.get("initialClass") or client.get("class") or client.get("address", "")


def pick_game(clients: list[dict], want: str = "") -> dict | None:
    """The game window among Hyprland's clients.

    With `want`, the window whose class or title contains it. Otherwise a Steam game
    first, then any other window that doesn't look like a terminal, browser or app,
    most recently focused first. None when nothing looks like a game.
    """
    clients = [c for c in clients if c.get("mapped", True) and not is_own(c)]
    clients.sort(key=lambda c: c.get("focusHistoryID", 99))
    if want:
        w = want.lower()
        return next((c for c in clients if w in c.get("class", "").lower()
                     or w in c.get("title", "").lower()
                     or w in game_name(c).lower()), None)
    steam = [c for c in clients if STEAM_CLASS.match(c.get("class", ""))]
    if steam:
        return steam[0]
    return next((c for c in clients if looks_like_game(c)), None)


def running_steam_game() -> str:
    """The Steam game running right now, on Windows, where Steam records it."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            app_id, _ = winreg.QueryValueEx(k, "RunningAppID")
    except OSError:
        return ""
    return steam_game_name(str(app_id)) if app_id else ""
