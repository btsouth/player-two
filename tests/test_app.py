"""The app around the core loop: finding the game, settings, the talk key, start-up."""

import tomllib

import pytest

from playertwo import config, desktop, games
from playertwo.app import local_action, parse_setting, trouble_words


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    games.steam_libraries.cache_clear()
    games.steam_game_name.cache_clear()
    yield tmp_path
    games.steam_libraries.cache_clear()
    games.steam_game_name.cache_clear()


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


def test_talk_key_bind_for_plain_hyprland(home):
    hypr = home / ".config/hypr"
    hypr.mkdir(parents=True)
    (hypr / "hyprland.conf").write_text("monitor = , preferred, auto, 1\n")
    desktop.install_talk_key()
    text = (hypr / "hyprland.conf").read_text()
    assert "bind = , F8, exec," in text and "bindr = , F8, exec," in text


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
    from playertwo.web import serve_ui

    session = SimpleNamespace(events=Events())
    port = 47899
    server = asyncio.create_task(serve_ui(session, port))
    await asyncio.sleep(0.2)
    try:
        with pytest.raises(InvalidStatus):
            async with connect(f"ws://127.0.0.1:{port}/events",
                               origin="https://evil.example"):
                pass
        async with connect(f"ws://127.0.0.1:{port}/events",
                           origin=f"http://127.0.0.1:{port}") as ws:
            assert ws.protocol.state.name == "OPEN"
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
