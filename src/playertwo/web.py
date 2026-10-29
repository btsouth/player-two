"""The companion window: a local web page fed by session events over a WebSocket.

One port serves both the page and the live event stream. Only 127.0.0.1 is bound.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import secrets
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
# A secret for this run, in the window's address: other programs on the PC (or other
# users of it) can reach 127.0.0.1 too, and must not get the screen, the mic or the pad.
TOKEN = secrets.token_urlsafe(18)
# What the window itself sends; nothing else is taken from it.
WINDOW_COMMANDS = {"talk-start", "talk-stop", "do", "set", "pause", "resume", "quit", "stop",
                   "repeat", "describe", "check"}


def window_url(port: int = PORT) -> str:
    return f"http://127.0.0.1:{port}/?k={TOKEN}"


def _allowed(path: str) -> bool:
    query = path.split("?", 1)[1] if "?" in path else ""
    keys = [v for k, _, v in (part.partition("=") for part in query.split("&")) if k == "k"]
    return bool(keys) and hmac.compare_digest(keys[0], TOKEN)


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
        if request.path.split("?", 1)[0] == "/events" and not _allowed(request.path):
            return Response(HTTPStatus.FORBIDDEN, "Forbidden", Headers(), b"forbidden")
        return _static(request.path)

    async def handler(ws):
        q: asyncio.Queue = asyncio.Queue()

        async def commands():
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(data, dict) and isinstance(data.get("command"), str):
                    if data["command"].partition(" ")[0] not in WINDOW_COMMANDS:
                        continue
                    try:
                        await session.command(data["command"])
                    except Exception as exc:  # noqa: BLE001  keep the window's commands working
                        session.events.emit("error", where="window", message=str(exc)[:300])

        session.events.subs.add(q)
        rx = asyncio.create_task(commands())
        try:
            for ev in session.events.snapshot():
                await ws.send(json.dumps(ev))
            while True:
                await ws.send(json.dumps(await q.get()))
        except ConnectionClosed:
            return
        finally:
            rx.cancel()
            session.events.subs.discard(q)

    # Only this page may connect: any other site open in the player's browser could
    # otherwise read the screen and the mic and press buttons. No Origin means a
    # non-browser client on this machine.
    origins = [f"http://127.0.0.1:{port}", f"http://localhost:{port}", None]
    async with serve(handler, "127.0.0.1", port, process_request=process_request,
                     origins=origins):
        await asyncio.Future()
