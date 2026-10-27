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


@dataclass
class Settings:
    model: str = "muse-spark-1.3"
    # Reasoning effort for live narration. "minimal" keeps answers around two seconds.
    effort: str = "minimal"
    # Agent turns (voice commands) get a little more thought.
    agent_effort: str = "low"
    base_url: str = "https://api.meta.ai/v1"
    voice: str = "af_heart"
    speech_rate: float = 1.1
    # How chatty narration is: "quiet" (only dialogue and choices), "normal", "detailed".
    verbosity: str = "normal"
    # Longest side of the screenshot sent to the model.
    image_max_side: int = 1280
    # Frames per second the screen watcher samples.
    watch_fps: float = 6.0
    # Input backend: "gamepad", "keyboard", or "auto".
    controls: str = "auto"
    capture: str = "auto"
    game: str = ""
    language: str = "en-us"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or CONFIG_DIR / "config.toml"
        data: dict = {}
        if path.exists():
            data = tomllib.loads(path.read_text())
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


def api_key() -> str:
    """Find the Meta Model API key without ever printing it."""
    for name in ("PLAYER_TWO_API_KEY", "MODEL_API_KEY", "META_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    p = CONFIG_DIR / "api-key"
    if p.exists():
        return p.read_text().strip()
    raise SystemExit(
        "No Meta Model API key found. Put it in ~/.config/player-two/api-key "
        "or set MODEL_API_KEY."
    )
