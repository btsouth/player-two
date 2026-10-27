"""The companion window: a local web page fed by session events over a WebSocket.

One port serves both the page and the live event stream. Only 127.0.0.1 is bound.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
from http import HTTPStatus
from importlib import resources
from typing import TYPE_CHECKING

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

if TYPE_CHECKING:
    from .app import Session

PORT = 47820
UI = resources.files("playertwo") / "ui"


def _static(path: str) -> Response | None:
    name = path.split("?", 1)[0].lstrip("/") or "index.html"
    if name == "events":
        return None  # WebSocket upgrade
    if "/" in name or name.startswith(".") or not (UI / name).is_file():
        return Response(HTTPStatus.NOT_FOUND, "Not Found", Headers(), b"not found")
    body = (UI / name).read_bytes()
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    headers = Headers({"Content-Type": ctype, "Cache-Control": "no-cache",
                       "Content-Length": str(len(body))})
    return Response(HTTPStatus.OK, "OK", headers, body)


async def serve_ui(session: Session, port: int = PORT) -> None:
    def process_request(_conn, request):
        return _static(request.path)

    async def handler(ws):
        q: asyncio.Queue = asyncio.Queue()
        session.events.subs.add(q)
        for ev in session.events.snapshot():
            await ws.send(json.dumps(ev))

        async def commands():
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(data, dict) and isinstance(data.get("command"), str):
                    await session.command(data["command"])

        rx = asyncio.create_task(commands())
        try:
            while True:
                await ws.send(json.dumps(await q.get()))
        except ConnectionClosed:
            return
        finally:
            rx.cancel()
            session.events.subs.discard(q)

    async with serve(handler, "127.0.0.1", port, process_request=process_request):
        await asyncio.Future()
