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

from .config import Settings, api_key


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
        self.client = AsyncOpenAI(base_url=settings.base_url, api_key=api_key(), max_retries=2)
        self.usage = Usage()

    async def stream(
        self, messages: list[dict], effort: str | None = None, max_tokens: int = 1500
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        t = time.monotonic()
        stream = await self.client.chat.completions.create(
            model=self.s.model,
            messages=messages,
            reasoning_effort=effort or self.s.effort,
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            prompt_cache_key="player-two",
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
        r = await self.client.chat.completions.create(
            model=self.s.model,
            messages=messages,
            reasoning_effort=effort or self.s.effort,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
            prompt_cache_key="player-two",
        )
        self.usage.add(r.usage, time.monotonic() - t)
        return json.loads(r.choices[0].message.content or "{}")

    async def tools(
        self, messages: list[dict], tools: list[dict], effort: str | None = None,
        max_tokens: int = 3000,
    ):
        """One tool-calling turn. Returns the assistant message."""
        t = time.monotonic()
        r = await self.client.chat.completions.create(
            model=self.s.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            reasoning_effort=effort or self.s.agent_effort,
            max_completion_tokens=max_tokens,
            prompt_cache_key="player-two-agent",
        )
        self.usage.add(r.usage, time.monotonic() - t)
        return r.choices[0].message
