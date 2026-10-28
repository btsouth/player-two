"""Thin async wrapper over Meta Model API (Chat Completions, OpenAI compatible)."""

from __future__ import annotations

import base64
import io
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from PIL import Image

from .config import Settings, api_key, has_open_credentials, open_credentials


def image_part(img: Image.Image, max_side: int = 1280, quality: int = 82) -> dict:
    """Encode a frame as a JPEG data URL content part, downscaled to max_side."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    seconds: list[float] = field(default_factory=list)

    def add(self, u, dt: float) -> None:
        self.calls += 1
        self.seconds.append(dt)
        if u is None:
            return
        self.prompt_tokens += u.prompt_tokens or 0
        self.completion_tokens += u.completion_tokens or 0
        details = getattr(u, "prompt_tokens_details", None)
        self.cached_tokens += (getattr(details, "cached_tokens", 0) or 0) if details else 0

    def cost_usd(self) -> float:
        # Muse Spark standard mode, per 1M tokens: input 1.25, cached 0.15, output 4.25.
        fresh = self.prompt_tokens - self.cached_tokens
        return (fresh * 1.25 + self.cached_tokens * 0.15 + self.completion_tokens * 4.25) / 1e6


class Muse:
    def __init__(self, settings: Settings):
        self.s = settings
        # SAM 3.1 and anything Meta-only always go through Meta Model API.
        self.client = AsyncOpenAI(base_url=settings.base_url, api_key=api_key(), max_retries=2)
        self.open = settings.brain == "open" or (
            settings.brain == "auto" and has_open_credentials())
        if self.open:
            key, headers = open_credentials()
            self.brain = AsyncOpenAI(base_url=settings.open_base_url, api_key=key,
                                     default_headers=headers, max_retries=2)
            self.model = settings.open_model
        else:
            self.brain = self.client
            self.model = settings.model
        self.usage = Usage()
        self._effort_ok = True

    def _extra(self, effort: str, key: str) -> dict:
        """Parameters only some providers accept."""
        extra: dict = {}
        if self._effort_ok:
            # Muse Spark Open's lowest setting is "low"; "minimal" is Meta API only.
            extra["reasoning_effort"] = "low" if self.open and effort == "minimal" else effort
        if not self.open:
            extra["prompt_cache_key"] = key
        return extra

    async def _create(self, key: str, effort: str, **kw):
        from openai import BadRequestError

        try:
            return await self.brain.chat.completions.create(
                model=self.model, **kw, **self._extra(effort, key))
        except BadRequestError as exc:
            if "reasoning_effort" not in str(exc):
                raise
            self._effort_ok = False  # this provider doesn't take reasoning_effort
            return await self.brain.chat.completions.create(
                model=self.model, **kw, **self._extra(effort, key))

    async def stream(
        self, messages: list[dict], effort: str | None = None, max_tokens: int = 1500
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        t = time.monotonic()
        stream = await self._create(
            "player-two", effort or self.s.effort,
            messages=messages, max_completion_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
        )
        usage = None
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        self.usage.add(usage, time.monotonic() - t)

    async def json(
        self, messages: list[dict], effort: str | None = None, max_tokens: int = 3000
    ) -> dict:
        t = time.monotonic()
        r = await self._create(
            "player-two", effort or self.s.effort,
            messages=messages, max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        self.usage.add(r.usage, time.monotonic() - t)
        return json.loads(r.choices[0].message.content or "{}")

    async def tools(
        self, messages: list[dict], tools: list[dict], effort: str | None = None,
        max_tokens: int = 3000,
    ):
        """One tool-calling turn. Returns the assistant message."""
        t = time.monotonic()
        r = await self._create(
            "player-two-agent", effort or self.s.agent_effort,
            messages=messages, tools=tools, tool_choice="auto",
            max_completion_tokens=max_tokens,
        )
        self.usage.add(r.usage, time.monotonic() - t)
        return r.choices[0].message
