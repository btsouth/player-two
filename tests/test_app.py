"""The app around the core loop: finding the game, settings, the talk key, start-up."""

import sys
import tomllib

import pytest

linux_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes, Linux desktops")

from playertwo import config, desktop, games
from playertwo.app import local_action, parse_setting, trouble_words


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    games.steam_libraries.cache_clear()
    games.steam_game_name.cache_clear()
    games.steam_install_dirs.cache_clear()
    yield tmp_path
    games.steam_libraries.cache_clear()
    games.steam_game_name.cache_clear()
    games.steam_install_dirs.cache_clear()


def client(cls, title, focus, **kw):
    return {"class": cls, "title": title, "focusHistoryID": focus, "mapped": True, **kw}


def test_steam_game_name_comes_from_the_manifest_in_any_library(home):
    root = home / ".local/share/Steam/steamapps"
    other = home / "games/SteamLibrary/steamapps"
    root.mkdir(parents=True)
    other.mkdir(parents=True)
    folders = "".join(f' "{i}"\n {{\n  "path"  "{p}"\n }}\n' for i, p in
                      enumerate([home / ".local/share/Steam", home / "games/SteamLibrary"]))
    (root / "libraryfolders.vdf").write_text(f'"libraryfolders"\n{{\n{folders}}}\n')
    (other / "appmanifest_1113000.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"1113000"\n\t"name"\t\t"Persona 4 Golden"\n}\n')
    game = client("steam_app_1113000", "Persona4 GOLDEN [REV: 4519]  121fps, (0.50)", 1)
    assert games.game_name(game) == "Persona 4 Golden"
    assert games.game_name(client("steam_app_999", "Some Game [DX12] 60 fps", 1)) == "Some Game"


def test_pick_game_prefers_steam_and_skips_terminals_browsers_and_itself(home):
    clients = [
        client("Alacritty", "bash", 0),
        client("chrome-127.0.0.1__-Default", "Player Two", 1),
        client("com.t3tools.T3Code", "T3 Code", 2),
        client("retroarch", "RetroArch", 3),
        client("steam_app_1113000", "Persona4 GOLDEN", 4),
        client("steam", "Steam", 5),
    ]
    assert games.pick_game(clients)["class"] == "steam_app_1113000"
    assert games.pick_game(clients[:4])["class"] == "retroarch"
    assert games.pick_game(clients[:3]) is None
    assert games.pick_game(clients, "retro")["class"] == "retroarch"


@linux_only
def test_save_settings_keeps_comments_tables_and_the_file_mode(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('# mine\nverbosity = "normal"  \nbrain = "open"\n\n[keys]\nverbosity = 1\n')
    path.chmod(0o600)
    config.save_settings({"verbosity": "terse", "speech_rate": 1.25, "team_up": True}, path)
    text = path.read_text()
    assert text.startswith("# mine\n")
    data = tomllib.loads(text)
    assert data["verbosity"] == "terse" and data["brain"] == "open"
    assert data["speech_rate"] == 1.25 and data["team_up"] is True
    assert data["keys"] == {"verbosity": 1}
    assert path.stat().st_mode & 0o777 == 0o600


def test_settings_from_the_window_are_checked():
    assert parse_setting("verbosity", "Terse") == "terse"
    assert parse_setting("team_up", "true") is True
    assert parse_setting("speech_rate", "1.3") == 1.3
    assert parse_setting("language", "spanish") == "Spanish"
    for key, raw in [("verbosity", "loud"), ("speech_rate", "9"), ("base_url", "x")]:
        with pytest.raises(ValueError):
            parse_setting(key, raw)


def test_quit_is_only_for_player_two():
    assert local_action("goodbye") == "quit"
    assert local_action("okay, turn yourself off") == "quit"
    assert local_action("quit player two") == "quit"
    assert local_action("quit the game") is None
    assert local_action("quit") is None


@linux_only
def test_talk_key_bind_is_added_once_with_a_backup(home):
    hypr = home / ".config/hypr"
    hypr.mkdir(parents=True)
    (hypr / "bindings.lua").write_text('o.bind("SUPER + R", "Run", "walker")\n')
    assert not desktop.talk_key_bound()
    said = desktop.install_talk_key()
    assert said.startswith("Done.")
    text = (hypr / "bindings.lua").read_text()
    assert text.startswith('o.bind("SUPER + R"')
    assert text.count("ctl talk-start") == 1 and "o.bind_hold(\"F8\"" in text
    assert len(list(hypr.glob("bindings.lua.bak.player-two-*"))) == 1
    assert desktop.talk_key_bound()
    assert "already" in desktop.install_talk_key()
    assert (hypr / "bindings.lua").read_text() == text


@linux_only
def test_talk_key_bind_for_plain_hyprland(home):
    hypr = home / ".config/hypr"
    hypr.mkdir(parents=True)
    (hypr / "hyprland.conf").write_text("monitor = , preferred, auto, 1\n")
    desktop.install_talk_key()
    text = (hypr / "hyprland.conf").read_text()
    assert "bind = , F8, exec," in text and "bindr = , F8, exec," in text


@linux_only
def test_desktop_entry_and_icon(home, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    entry, icon = desktop.install_desktop_entry()
    text = entry.read_text()
    assert "Name=Player Two" in text and "Icon=player-two" in text and "Exec=" in text
    assert icon.read_bytes().startswith(b"<svg")


def test_network_trouble_is_said_plainly():
    import httpx2 as httpx
    import openai

    req = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
    kind, words = trouble_words(openai.APIConnectionError(request=req))
    assert kind == "net" and "internet" in words
    resp = httpx.Response(401, request=req)
    kind, words = trouble_words(openai.AuthenticationError("no", response=resp, body=None))
    assert kind == "key" and "setup" in words


@linux_only
def test_save_settings_keeps_inline_comments_and_writes_through_symlinks(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text('verbosity = "normal"  # how chatty\nlanguage = "a # b"\n')
    real.chmod(0o640)
    link = tmp_path / "config.toml"
    link.symlink_to(real)
    config.save_settings({"verbosity": "quiet", "language": "Spanish"}, link)
    assert link.is_symlink()
    assert real.read_text() == 'verbosity = "quiet"  # how chatty\nlanguage = "Spanish"\n'
    assert real.stat().st_mode & 0o777 == 0o640


def test_only_game_like_windows_count_without_steam(home):
    assert games.looks_like_game(client("hollow_knight.exe", "Hollow Knight", 1))
    assert games.looks_like_game(client("org.DolphinEmu.dolphin-emu", "Dolphin", 1))
    assert games.looks_like_game(client("SomeNativeGame", "Game", 1, fullscreen=2))
    assert not games.looks_like_game(client("SomeNativeGame", "Game", 1))
    assert not games.looks_like_game(client("org.omarchy.btop", "btop", 1, fullscreen=2))
    assert not games.looks_like_game(client("thunderbird", "Inbox", 1))
    assert not games.looks_like_game(client("org.kde.dolphin", "Home", 1, fullscreen=2))


async def test_window_socket_refuses_other_websites():
    import asyncio
    from types import SimpleNamespace

    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    from playertwo.app import Events
    from playertwo.web import TOKEN, serve_ui

    session = SimpleNamespace(events=Events())
    port = 47899
    server = asyncio.create_task(serve_ui(session, port))
    await asyncio.sleep(0.2)
    try:
        with pytest.raises(InvalidStatus):
            async with connect(f"ws://127.0.0.1:{port}/events",
                               origin="https://evil.example"):
                pass
        # Another program on this PC, without the window's secret.
        with pytest.raises(InvalidStatus):
            async with connect(f"ws://127.0.0.1:{port}/events"):
                pass
        with pytest.raises(InvalidStatus):
            async with connect(f"ws://127.0.0.1:{port}/events?k=guess",
                               origin=f"http://127.0.0.1:{port}"):
                pass
        async with connect(f"ws://127.0.0.1:{port}/events?k={TOKEN}",
                           origin=f"http://127.0.0.1:{port}") as ws:
            assert ws.protocol.state.name == "OPEN"
    finally:
        server.cancel()


async def test_the_window_can_only_send_its_own_commands():
    import asyncio
    import json
    from types import SimpleNamespace

    from websockets.asyncio.client import connect

    from playertwo.app import Events
    from playertwo.web import TOKEN, serve_ui

    got: list[str] = []

    async def command(line):
        got.append(line)

    session = SimpleNamespace(events=Events(), command=command)
    port = 47898
    server = asyncio.create_task(serve_ui(session, port))
    await asyncio.sleep(0.2)
    try:
        async with connect(f"ws://127.0.0.1:{port}/events?k={TOKEN}",
                           origin=f"http://127.0.0.1:{port}") as ws:
            for line in ("shot /tmp/x.png", "press a", "do open the menu"):
                await ws.send(json.dumps({"command": line}))
            await asyncio.sleep(0.2)
        assert got == ["do open the menu"]
    finally:
        server.cancel()


def test_spoken_commands_people_actually_say():
    cases = {
        "terse": "terse", "back to normal": "normal", "team up": "team_up",
        "the game reads the menus": "team_up", "watch my health": "hud", "help": "help",
        "what can you do": "help", "speak slower": "slower", "faster": "faster",
        "read that again": "repeat", "what?": "repeat", "stop it": "stop",
        "shut up, player two": "stop", "cancel that": "stop", "what's happening": "describe",
        "show the window": "show",
        # Requests and answers are for the agent.
        "stop attacking and heal": None, "keep going": None, "yes": None,
    }
    for said, action in cases.items():
        assert local_action(said) == action, said


def test_gnome_talk_key_is_added_to_the_existing_shortcuts(monkeypatch):
    store = {("get", desktop.GNOME_KEYS, "custom-keybindings"): "['/custom0/']"}

    def fake(*args):
        if args[0] == "set":
            store[("get", *args[1:3])] = args[3]
            return ""
        return store.get(args, "")

    monkeypatch.setattr(desktop, "_gsettings", fake)
    said = desktop.install_gnome_talk_key()
    assert said.startswith("Done.")
    paths = store[("get", desktop.GNOME_KEYS, "custom-keybindings")]
    assert "/custom0/" in paths and desktop.GNOME_PATH in paths
    key = f"{desktop.GNOME_KEYS}.custom-keybinding:{desktop.GNOME_PATH}"
    assert store[("get", key, "command")].endswith("ctl talk")
    assert store[("get", key, "binding")] == "F8"
    desktop.install_gnome_talk_key()  # twice adds it once
    assert store[("get", desktop.GNOME_KEYS, "custom-keybindings")].count("player-two") == 1


def test_windows_windows_pick_the_game(home):
    lib = home / ".local/share/Steam/steamapps"
    (lib / "common/Hades").mkdir(parents=True)
    (lib / "appmanifest_1145360.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"1145360"\n\t"name"\t\t"Hades"\n'
        '\t"installdir"\t\t"Hades"\n}\n')
    exe = str(lib / "common/Hades/x64/Hades.exe")
    assert games.steam_app_for_path(exe) == "1145360"
    assert games.steam_app_for_path(str(home / "Downloads/Hades.exe")) == ""

    def win(cls, title, z, **kw):
        return client(cls, title, z, **{"fullscreen": 0, "game_dir": False, **kw})

    clients = [
        win("WindowsTerminal", "player-two", -1),
        win("msedge", "News", 0, fullscreen=2),
        win("explorer", "Downloads", 1),
        win("EpicGamesLauncher", "Epic Games Launcher", 2, game_dir=True),
        win("Celeste", "Celeste", 3, game_dir=True),
        win("steam_app_1145360", "Hades", 4),
    ]
    assert games.pick_game(clients)["class"] == "steam_app_1145360"
    assert games.game_name(clients[-1]) == "Hades"
    assert games.pick_game(clients[:5])["class"] == "Celeste"
    assert games.pick_game(clients[:4]) is None
    assert games.pick_game([*clients[:4], win("Indie", "Indie", 5, fullscreen=2)])


def test_hidden_game_is_said_plainly():
    from playertwo.screen import GameHidden

    assert "minimized" in trouble_words(GameHidden())[1]


async def test_scripted_presses_for_a_known_start(tmp_path):
    from PIL import Image

    from playertwo.app import Session
    from playertwo.controls import DryRun
    from playertwo.screen import Frame

    s = Session.__new__(Session)
    s.controls = DryRun()

    async def grab_now(show=False):
        return Frame(Image.new("RGB", (64, 36), (9, 9, 9)), 0)

    s.grab_now = grab_now
    assert await s.command("press start wait0 down*2 a") == "ok"
    assert s.controls.log == ["start", "down", "down", "a"]
    assert (await s.command("press jump")).startswith("unknown button")
    assert (await s.command("press a*x")).startswith("can't read")
    assert s.controls.log == ["start", "down", "down", "a"]  # nothing pressed for a bad line
    assert (await s.command("shot now.png")).startswith("give")
    await s.command(f"shot {tmp_path / 'now.png'}")
    assert Image.open(tmp_path / "now.png").size == (64, 36)


def test_bad_settings_fall_back_and_are_said(tmp_path):
    from playertwo.config import Settings

    p = tmp_path / "config.toml"
    p.write_text('watch_fps = 0\nspeech_rate = "fast"\nverbosity = "chatty"\n'
                 'hands_free = "yes"\nlanguage = "Spanish"\n')
    s = Settings.load(p)
    assert (s.watch_fps, s.speech_rate, s.verbosity, s.hands_free) == (6.0, 1.1, "normal", False)
    assert s.language == "Spanish"
    assert len(s.problems) == 4 and any("watch_fps" in p for p in s.problems)
    p.write_text('verbosity = "terse\n')  # a typo in the file itself
    s = Settings.load(p)
    assert s.verbosity == "normal" and "mistake" in s.problems[0]


def test_errors_are_said_for_what_they_are():
    import subprocess
    from types import SimpleNamespace

    from websockets.exceptions import InvalidStatus

    from playertwo.app import trouble_words
    from playertwo.controls import ControlsError

    assert trouble_words(ControlsError("x"))[0] == "controls"
    assert trouble_words(FileNotFoundError("wtype"))[0] != "net"
    assert trouble_words(InvalidStatus(SimpleNamespace(status_code=401)))[0] == "key"
    assert trouble_words(ConnectionRefusedError())[0] == "net"
    assert trouble_words(subprocess.CalledProcessError(1, "x"))[0] == "other"


def test_whats_wrong_is_answered_out_loud():
    from types import SimpleNamespace

    from playertwo.app import MIC_PROBLEM, Session, local_action
    from playertwo.config import Settings

    assert local_action("Player Two, what's wrong?") == "check"
    assert local_action("are you working") == "check"
    s = Session.__new__(Session)
    s.game, s.capture_failures, s.paused, s.hands = False, 0, False, None
    s.narrator = SimpleNamespace(game="")
    s.controls = SimpleNamespace(kind="dry-run")
    s.problems, s.last_trouble = [MIC_PROBLEM], "The model is having trouble right now."
    s.s = Settings()
    words = s.self_check()
    assert "don't see a game" in words and "can't press anything" in words
    assert "microphone" in words and "last problem" in words


def bare_session():
    """A Session with just what local commands touch."""
    from types import SimpleNamespace

    from playertwo.app import Session
    from playertwo.controls import DryRun

    s = Session.__new__(Session)
    s.controls = DryRun()
    s.events = SimpleNamespace(emit=lambda *a, **k: None, last_said="")
    s._tasks = set()
    s.emit_settings = lambda: None
    s.agent_task = None
    s.answer = None
    s.paused = False
    s.said = []
    cues = []
    s.voice = SimpleNamespace(stop=lambda: None, cue=cues.append)
    s.cues = cues
    s.stop_side = lambda: None

    async def announce(text, **k):
        s.said.append(text)

    s.announce = announce
    return s


async def test_next_presses_a_without_the_model():
    import asyncio

    from playertwo.app import local_action

    assert local_action("next") == "next" and local_action("Player Two, next please") == "next"
    assert local_action("continue") is None  # a title screen option, for the agent
    s = bare_session()
    await s.local("next")
    assert s.controls.log == ["a"]
    # Not in the middle of something the agent is doing.
    s.agent_task = asyncio.create_task(asyncio.sleep(10))
    await s.local("next")
    assert s.controls.log == ["a"] and "stop first" in s.said[-1]
    s.agent_task.cancel()


async def test_stop_ends_the_agent_at_once():
    import asyncio
    from types import SimpleNamespace

    s = bare_session()
    cancelled = []
    s.agent = SimpleNamespace(cancel=lambda: cancelled.append(1))
    s.agent_task = asyncio.create_task(asyncio.sleep(30))  # a model call still running
    await s.local("stop")
    await asyncio.sleep(0)
    assert cancelled and s.agent_task.cancelled()
    assert s.said == ["Okay, stopped."]


async def test_a_command_said_during_a_question_isnt_the_answer():
    import asyncio
    from types import SimpleNamespace

    s = bare_session()
    s.agent = SimpleNamespace(cancelled=False, cancel=lambda: None)
    s.answer = asyncio.get_running_loop().create_future()
    ran = []

    async def local(action):
        ran.append(action)

    s.local = local
    await s.handle("help")
    assert ran == ["help"] and not s.answer.done()  # still waiting for the answer
    await s.handle("next")  # could be the answer ("the next one")
    assert s.answer.result() == "next"


def test_a_hidden_health_bar_is_not_empty():
    import numpy as np
    from PIL import Image

    from playertwo.hud import HudWatch, Meter

    def bar(fill):
        arr = np.full((200, 400, 3), 200, np.uint8)
        arr[20:30, 40:240] = (60, 60, 60)
        arr[20:30, 40:40 + int(200 * fill)] = (170, 40, 40)
        return Image.fromarray(arr)

    m = Meter("health", (0.09, 0.09, 0.61, 0.16), (170, 40, 40), estimate=1.0)
    m.locate(bar(1.0))
    watch = HudWatch([m])
    assert watch.update(bar(1.0), 0) == []
    black = Image.new("RGB", (400, 200))
    assert watch.update(black, 1) == [] and watch.update(black, 6) == []  # a fade
    assert watch.update(bar(0.0), 7) == []  # once: likely covered, not a one-hit loss
    assert [a.level for a in watch.update(bar(0.0), 8)] == ["critical"]  # still so


def test_window_server_serves_only_its_own_files():
    from playertwo.web import FILES, UI, _static

    assert {p.name for p in UI.iterdir() if p.is_file()} == FILES
    assert _static("/?k=x").status_code == 200
    assert _static("/app.js").headers["Content-Type"].startswith("text/javascript")
    for path in ("/C:\\Users\\p\\.config\\player-two\\api-key", "/..\\..\\x", "/../app.js",
                 "/%2e%2e/app.js", "/ui/app.js", "/.env", "/C:/Windows/win.ini"):
        assert _static(path).status_code == 404, path


async def test_the_talk_button_never_waits_behind_a_slow_setting():
    import asyncio
    import json
    from types import SimpleNamespace

    from websockets.asyncio.client import connect

    from playertwo.app import Events
    from playertwo.web import TOKEN, serve_ui

    got: list[str] = []
    slow = asyncio.Event()

    async def command(line):
        got.append(line)
        if line.startswith("set"):
            await slow.wait()

    session = SimpleNamespace(events=Events(), command=command)
    port = 47897
    server = asyncio.create_task(serve_ui(session, port))
    await asyncio.sleep(0.2)
    try:
        async with connect(f"ws://127.0.0.1:{port}/events?k={TOKEN}",
                           origin=f"http://127.0.0.1:{port}") as ws:
            for line in ("set verbosity terse", "pause", "talk-start", "talk-stop"):
                await ws.send(json.dumps({"command": line}))
            await asyncio.sleep(0.2)
            assert sorted(got) == ["set verbosity terse", "talk-start", "talk-stop"]
            slow.set()
            await asyncio.sleep(0.1)
            assert got[-1] == "pause"  # the rest still run, in order
    finally:
        server.cancel()


@linux_only
def test_window_secret_stays_out_of_command_lines(home, monkeypatch, tmp_path):
    from playertwo.web import TOKEN, window_url

    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(run))
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    argv = tmp_path / "argv"
    fake = tmp_path / "chromium"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {argv}\n')
    fake.chmod(0o755)
    monkeypatch.setattr(desktop, "browser", lambda: str(fake))
    monkeypatch.setattr(desktop, "CACHE_DIR", tmp_path / "cache")
    win = desktop.CompanionWindow()
    win.profile = tmp_path / "profile"
    assert win.open(window_url())
    win.proc.wait(5)
    args = argv.read_text()
    assert TOKEN not in args and "--app=file://" in args
    page = run / "player-two" / desktop.PAGE
    assert (run / "player-two").stat().st_mode & 0o777 == 0o700
    assert page.stat().st_mode & 0o777 == 0o600
    assert TOKEN in page.read_text() and page.as_uri() in args
    import re

    assert re.fullmatch(desktop.APP_CLASS, "chrome-127.0.0.1__-Default")
    assert re.fullmatch(desktop.APP_CLASS, f"chrome-__run_user_1000_player-two_{desktop.PAGE}-Default")


@linux_only
def test_a_window_left_by_a_crash_is_replaced(tmp_path):
    import subprocess
    import threading
    import time

    old = subprocess.Popen(["sleep", "30"])
    threading.Thread(target=old.wait, daemon=True).start()  # reap it once it ends
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SingletonLock").symlink_to(f"host-{old.pid}")
    assert desktop.profile_owner(profile) == old.pid
    t = time.monotonic()
    desktop.stale_window(profile)
    assert old.poll() is not None and time.monotonic() - t < 2.5
    assert desktop.profile_owner(profile) is None


def test_orca_lines_starting_with_a_dash_are_spoken_not_options():
    from playertwo.screenreader import ORCA

    cmd = ORCA[0].call(ORCA[0].path, "org.gnome.Orca1.Service.PresentMessage", "-30 gold")
    assert cmd[-2:] == ["--", "-30 gold"]


@linux_only
def test_gdbus_takes_a_dash_line_after_the_separator():
    import shutil
    import subprocess

    if not shutil.which("gdbus"):
        pytest.skip("no gdbus")
    # Option parsing fails before any bus is reached, so no session bus is needed.
    for args, ok in ((["-30 gold"], False), (["--", "-30 gold"], True)):
        r = subprocess.run(["gdbus", "call", "--session", "--dest", "x.y", "--object-path",
                            "/x", "--method", "x.y.Z", *args], capture_output=True,
                           text=True, timeout=10, env={"PATH": "/usr/bin:/bin"}, check=False)
        assert r.stderr.startswith("Usage:") != ok, r.stderr


async def test_a_command_that_fails_is_said_not_silent(monkeypatch):
    import asyncio

    from playertwo import app
    from playertwo.narrator import Narrator

    s = bare_session()
    s.s = config.Settings()
    s.narrator = Narrator.__new__(Narrator)
    troubles = []
    s.trouble = lambda exc, where, answer=False: troubles.append((type(exc), answer))

    def locked(_values, _path=None):
        raise PermissionError("config.toml is read-only")

    monkeypatch.setattr(app, "save_settings", locked)
    await s.handle("terse")  # spoken or typed
    assert troubles == [(PermissionError, True)]
    await s.local("terse")  # from the window or a hotkey
    assert troubles[-1] == (PermissionError, True)

    # Anything else a background task raises is logged, not lost.
    emitted = []
    s.events.emit = lambda kind, **k: emitted.append((kind, k))

    async def boom():
        raise RuntimeError("lost")

    task = s.spawn(boom())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert task.done() and ("error", {"where": "task", "message": "lost"}) in emitted


@linux_only
async def test_sigterm_ends_the_session_the_normal_way():
    import asyncio
    import os
    import signal

    s = bare_session()
    s.main = asyncio.create_task(asyncio.sleep(30))
    s.stop_on_signals()
    loop = asyncio.get_running_loop()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        with pytest.raises(asyncio.CancelledError):
            await s.main  # cancelled like "quit": main_run's finally closes the session
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            loop.remove_signal_handler(sig)
