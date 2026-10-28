"""Settings, loaded from ~/.config/player-two/config.toml with environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "player-two"
CACHE_DIR = Path(
    os.environ.get("PLAYER_TWO_CACHE")
    or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "player-two"
)

STATE_DIR = Path(
    os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "player-two"


def socket_path() -> Path:
    """The running session's control socket (hotkeys and the window talk to it)."""
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return Path(base) / "player-two.sock"


# Windows has no Unix sockets in asyncio; there the control channel is a loopback port,
# which every user on the PC can reach. So the session writes its port and a secret to a
# file in the user's own profile, and a client has to send the secret first. A second
# user's session takes another port instead of finding the first user's.
CONTROL_PORT = 47821


def _control_file() -> Path:
    return STATE_DIR / "control"


async def open_control(limit: int = 1 << 16):
    """Connect to the running session's control channel."""
    import asyncio
    import sys

    if sys.platform == "win32":
        try:
            port, token = _control_file().read_text(encoding="utf-8").split()
        except (OSError, ValueError) as exc:
            raise ConnectionRefusedError("Player Two isn't running") from exc
        reader, writer = await asyncio.open_connection("127.0.0.1", int(port), limit=limit)
        writer.write(f"{token}\n".encode())
        return reader, writer
    return await asyncio.open_unix_connection(str(socket_path()), limit=limit)


async def start_control(client):
    import asyncio
    import contextlib
    import sys

    if sys.platform == "win32":
        import secrets

        token = secrets.token_hex(16)

        async def checked(reader, writer):
            try:
                line = await asyncio.wait_for(reader.readline(), 5)
            except (TimeoutError, OSError, ValueError):
                line = b""
            if not secrets.compare_digest(line.strip(), token.encode()):
                writer.close()
                return
            await client(reader, writer)

        try:
            server = await asyncio.start_server(checked, "127.0.0.1", CONTROL_PORT)
        except OSError:  # taken, by another user's Player Two or anything else
            server = await asyncio.start_server(checked, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        _control_file().write_text(f"{port} {token}\n", encoding="utf-8")
        return server
    path = socket_path()
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return await asyncio.start_unix_server(client, str(path))


@dataclass
class Settings:
    model: str = "muse-spark-1.3"
    # Reasoning effort for live narration. "minimal" keeps answers around two seconds.
    effort: str = "minimal"
    # Agent turns (voice commands) get a little more thought.
    agent_effort: str = "low"
    base_url: str = "https://api.meta.ai/v1"
    # Which Muse Spark does the thinking: "open" (the open-weight Muse Spark Open on a hosted
    # provider), "meta" (Meta Model API), or "auto" (open when its credentials are set up).
    brain: str = "auto"
    open_base_url: str = "https://inference.baseten.co/v1"
    open_model: str = "meta-models/muse-spark-open-1026"
    # Kokoro voice. Empty means the default voice for `language`.
    voice: str = ""
    speech_rate: float = 1.1
    # How chatty narration is: "terse" (bare facts, for fast screen reader users),
    # "quiet" (only dialogue and choices), "normal", "detailed".
    verbosity: str = "normal"
    # Language Player Two speaks. On-screen text in other languages is translated.
    language: str = "English"
    # Where speech goes: "voice" (the built-in narrator) or "screen-reader" (NVDA, Orca).
    output: str = "auto"
    # When the agent checks before acting: "always", "important", or "never".
    confirm: str = "important"
    # The game has its own screen narrator: leave menus to it and cover the rest.
    team_up: bool = False
    # Longest side of the screenshot sent to the model.
    image_max_side: int = 1280
    # Frames per second the screen watcher samples.
    watch_fps: float = 6.0
    # Input backend: "gamepad", "keyboard", or "auto".
    controls: str = "auto"
    capture: str = "auto"
    # The game's window: part of its title or class. Empty picks the last focused window.
    window: str = ""
    game: str = ""
    # The companion window (captions, what Player Two sees and presses): "auto" opens it
    # on Hyprland without taking focus from the game, "on" always, "off" never.
    companion: str = "auto"
    # The talk key starts and stops listening on two presses (GNOME and other desktops
    # without hold binds) instead of being held.
    talk_toggle: bool = False
    # The player bound the talk key by hand where Player Two can't check it.
    talk_key_set: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or CONFIG_DIR / "config.toml"
        data: dict = {}
        if path.exists():
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in data.items() if k in known})
        s.extra = {k: v for k, v in data.items() if k not in known}
        for f in fields(cls):
            env = os.environ.get(f"PLAYER_TWO_{f.name.upper()}")
            if env is None or f.name == "extra":
                continue
            cur = getattr(s, f.name)
            setattr(s, f.name, type(cur)(env) if not isinstance(cur, bool) else env == "1")
        return s


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_settings(changes: dict, path: Path | None = None) -> None:
    """Write changed settings into config.toml, keeping every other line (and comment)."""
    import re

    path = path or CONFIG_DIR / "config.toml"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    todo = dict(changes)
    # Top-level keys end where the first table starts; new ones go just before it.
    end = next((i for i, line in enumerate(lines) if line.lstrip().startswith("[")),
               len(lines))
    line_re = re.compile(r'(\s*)([A-Za-z_]+)(\s*=\s*)("(?:[^"\\]|\\.)*"|[^#]*?)(\s*#.*)?$')
    for i in range(end):
        m = line_re.match(lines[i])
        if m and m.group(2) in todo:
            comment = m.group(5) or ""
            lines[i] = f"{m.group(2)} = {toml_value(todo.pop(m.group(2)))}{comment}"
    lines[end:end] = [f"{k} = {toml_value(v)}" for k, v in todo.items()]
    # Write through a symlink (dotfile managers) to the real file, atomically, and never
    # with looser permissions than it had.
    path = path.resolve() if path.is_symlink() else path
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        if hasattr(os, "fchmod"):  # Windows has no fchmod, and no modes to keep
            os.fchmod(f.fileno(), mode)
        f.write("\n".join(lines) + "\n")
    tmp.replace(path)


def first_run(mark: bool = False) -> bool:
    """True until Player Two has welcomed this player once."""
    p = STATE_DIR / "welcomed"
    seen = p.exists()
    if mark and not seen:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return not seen


def has_api_key() -> bool:
    try:
        api_key()
    except SystemExit:
        return False
    return True


def api_key() -> str:
    """Find the Meta Model API key without ever printing it."""
    for name in ("PLAYER_TWO_API_KEY", "MODEL_API_KEY", "META_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    p = CONFIG_DIR / "api-key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "No Meta Model API key found. Put it in ~/.config/player-two/api-key "
        "or set MODEL_API_KEY."
    )


def _env_file(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not k.lstrip().startswith("#"):
                out[k.strip()] = v.strip()
    return out


def has_open_credentials(base_url: str = "") -> bool:
    try:
        open_credentials(base_url)
    except SystemExit:
        return False
    return True


def open_credentials(base_url: str = "") -> tuple[str, dict[str, str]]:
    """API key and extra headers for the provider serving Muse Spark Open at base_url.

    Modal shared endpoints authenticate with a proxy token (Modal-Key and Modal-Secret
    headers, from MODAL_PROXY_KEY/SECRET or ~/.config/player-two/modal.env); Baseten,
    Fireworks and the rest take a bearer key. A Modal token is only used for a Modal
    address, so switching providers doesn't send it to the wrong one.
    """
    env = {**_env_file(CONFIG_DIR / "modal.env"), **os.environ}
    modal = not base_url or ".modal." in base_url
    if modal and env.get("MODAL_PROXY_KEY") and env.get("MODAL_PROXY_SECRET"):
        return "modal", {"Modal-Key": env["MODAL_PROXY_KEY"],
                         "Modal-Secret": env["MODAL_PROXY_SECRET"]}
    for name in ("PLAYER_TWO_OPEN_API_KEY", "BASETEN_API_KEY", "FIREWORKS_API_KEY"):
        if env.get(name):
            return env[name].strip(), {}
    p = CONFIG_DIR / "open-api-key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip(), {}
    raise SystemExit(
        "brain = open needs provider credentials: a Modal proxy token in "
        "~/.config/player-two/modal.env, or a key in ~/.config/player-two/open-api-key."
    )
