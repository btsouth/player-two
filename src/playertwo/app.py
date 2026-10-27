"""The live session: watch the screen, narrate, listen, act."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import time
from pathlib import Path

from .agent import Agent
from .audio import Mixer
from .config import Settings, socket_path
from .controls import open_controls
from .listen import Mic, Utterance
from .muse import Muse
from .narrator import Narrator
from .screen import Capture, ChangeWatcher, Frame, auto_capture
from .voice import Voice

LOCAL = [
    (re.compile(r"^(stop|quiet|shush|shut up|be quiet|hush)\b", re.IGNORECASE), "stop"),
    (re.compile(r"^(repeat|say that again|what did (you|they|he|she) say|again)\b", re.IGNORECASE),
     "repeat"),
    (re.compile(r"^(describe|what'?s on (the )?screen|where am i|look around)\b", re.IGNORECASE),
     "describe"),
    (re.compile(r"\b(talk less|less detail|quieter mode)\b", re.IGNORECASE), "quiet"),
    (re.compile(r"\b(more detail|talk more|tell me more)\b", re.IGNORECASE), "detailed"),
    (re.compile(r"^(pause|stop narrating)\b", re.IGNORECASE), "pause"),
    (re.compile(r"^(resume|start narrating|keep going)\b", re.IGNORECASE), "resume"),
]


HUD_WORDS = {"low": "{name} low.", "critical": "{name} critical!",
             "recovered": "{name} back up.", "empty": "Out of {name}."}


class Events:
    """Fan-out of session events to any connected viewer (the overlay, logs)."""

    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.log_path: Path | None = None
        self.recent: list[dict] = []
        self.sticky: dict[str, dict] = {}

    def snapshot(self) -> list[dict]:
        """What a window that just opened needs: the latest state, then recent history."""
        return list(self.sticky.values()) + self.recent[-30:]

    def emit(self, kind: str, **data) -> None:
        if isinstance(data.get("text"), str):
            from .voice import tidy

            data["text"] = tidy(data["text"])
        ev = {"t": time.time(), "kind": kind, **data}
        if kind in ("ready", "status", "frame", "settings"):
            self.sticky[kind] = ev
        elif kind in ("say", "heard", "ask", "press", "hold", "error"):
            self.recent = [*self.recent[-59:], ev]
        for q in list(self.subs):
            if q.qsize() < 200:
                q.put_nowait(ev)
        if self.log_path and kind != "frame":
            with self.log_path.open("a") as f:
                f.write(json.dumps(ev) + "\n")


class Session:
    def __init__(self, settings: Settings, capture: Capture | None = None,
                 controls: str | None = None, sink: str = "device"):
        self.s = settings
        self.muse = Muse(settings)
        self.mixer = Mixer(sink)
        self.voice = Voice(settings, self.mixer)
        self.capture = capture or auto_capture(settings.window)
        self.controls = open_controls(controls or settings.controls, focus=self.focus_game)
        self.narrator = Narrator(self.muse, settings.verbosity, settings.game,
                                 settings.language, settings.team_up)
        self.watcher = ChangeWatcher()
        self.events = Events()
        self.latest: Frame | None = None
        self.narrating: asyncio.Task | None = None
        self.agent_task: asyncio.Task | None = None
        self.paused = False
        self.mic: Mic | None = None
        self.utt: Utterance | None = None
        self.keywords: list[str] = []
        self._tasks: set[asyncio.Task] = set()
        self.answer: asyncio.Future[str] | None = None
        self.hud = None
        self.agent = Agent(self.muse, self.controls, lambda: self.grab_now(show=True),
                           self.agent_say,
                           game=settings.game, ask=self.agent_ask, confirm=settings.confirm,
                           language=settings.language, on_action=self.on_action)
        self.status = ""

    def focus_game(self) -> None:
        target = getattr(self.capture, "target", None)
        if target and target.get("address"):
            from .screen import hypr_focus

            hypr_focus(target["address"])

    def spawn(self, coro) -> asyncio.Task:
        """Start a background task and keep a reference so it isn't collected mid-run."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def grab_now(self, show: bool = False) -> Frame:
        frame = await asyncio.to_thread(self.capture.grab)
        self.latest = frame
        if show:
            self.show(frame)
        return frame

    def show(self, frame: Frame) -> None:
        """Send a small copy of the frame to the companion window."""
        import base64
        import io

        img = frame.image.convert("RGB")
        img.thumbnail((720, 405))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        self.events.emit("frame", jpeg=base64.b64encode(buf.getvalue()).decode())

    def on_action(self, name: str, args: dict) -> None:
        if name == "press":
            self.events.emit("press", buttons=args.get("buttons", []))
        elif name == "hold":
            self.events.emit("hold", input=args.get("input", ""),
                             direction=args.get("direction", ""), ms=args.get("ms", 0))

    def current_status(self) -> str:
        if self.utt is not None:
            return "listening"
        if self.agent_task is not None and not self.agent_task.done():
            waiting = self.answer is not None and not self.answer.done()
            if not self.mixer.idle.is_set():
                return "speaking"
            return "listening" if waiting else "acting"
        if not self.mixer.idle.is_set():
            return "speaking"
        if self.narrating is not None and not self.narrating.done():
            return "thinking"
        return "paused" if self.paused else "watching"

    async def track_status(self) -> None:
        while True:
            st = self.current_status()
            if st != self.status:
                self.status = st
                self.events.emit("status", state=st)
            await asyncio.sleep(0.08)

    async def agent_ask(self, question: str, timeout: float = 30.0) -> str:
        """Ask out loud and wait for the next thing the player says."""
        self.events.emit("ask", text=question)
        self.answer = asyncio.get_running_loop().create_future()
        await self.voice.say(question, interrupt=False)
        self.voice.cue("listening")
        try:
            return await asyncio.wait_for(self.answer, timeout)
        except TimeoutError:
            return ""
        finally:
            self.answer = None

    async def agent_say(self, text: str) -> None:
        self.events.emit("say", text=text, source="agent")
        self.narrator.recent.append(text)
        await self.voice.say(text, interrupt=False)

    # Narration

    async def _narrate(self, frame: Frame) -> None:
        t = time.monotonic()

        async def deltas():
            async for d in self.narrator.narrate(frame):
                yield d

        text = await self.voice.say_stream(deltas())
        if text:
            self.events.emit("say", text=text, source="narrator",
                             latency=round(time.monotonic() - t, 2))

    def _start_narration(self, frame: Frame, describe: bool = False) -> None:
        if self.narrating and not self.narrating.done():
            self.narrating.cancel()
        if describe:
            async def run():
                self.voice.stop()
                text = await self.voice.say_stream(self.narrator.describe(frame))
                self.events.emit("say", text=text, source="describe")
            self.narrating = asyncio.create_task(run())
        else:
            self.narrating = asyncio.create_task(self._narrate(frame))

    async def watch(self) -> None:
        period = 1.0 / self.s.watch_fps
        while True:
            t = time.monotonic()
            try:
                frame = await self.grab_now()
            except Exception as exc:  # noqa: BLE001  keep watching through hiccups
                self.events.emit("error", where="capture", message=str(exc))
                await asyncio.sleep(1.0)
                continue
            if self.hud is not None:
                self.on_hud(self.hud.update(frame.image, frame.t))
            busy = self.agent_task is not None and not self.agent_task.done()
            if self.watcher.update(frame) and not self.paused and not busy:
                self.events.emit("settled", label=frame.label)
                self.show(frame)
                self._start_narration(frame)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t)))

    # Talking

    def talk_start(self) -> None:
        if self.utt is not None:
            return
        self.voice.stop()
        self.voice.cue("listening")
        waiting = self.answer is not None and not self.answer.done()
        if self.agent_task and not self.agent_task.done() and not waiting:
            self.agent.cancelled = True
        self.utt = Utterance(self.keywords,
                             on_partial=lambda p: self.events.emit("hearing", text=p))
        self.utt.start()
        self.mic = Mic()
        self.mic.start()
        self.spawn(self._pump(self.mic, self.utt))

    async def _pump(self, mic: Mic, utt: Utterance) -> None:
        while self.utt is utt:
            try:
                frame = await asyncio.wait_for(mic.queue.get(), 0.5)
            except TimeoutError:
                continue
            utt.feed(frame)

    async def talk_stop(self) -> None:
        utt, mic = self.utt, self.mic
        if utt is None:
            return
        self.utt, self.mic = None, None
        await asyncio.sleep(0.15)  # keep the tail of the last word
        if mic:
            while not mic.queue.empty():
                utt.feed(mic.queue.get_nowait())
            mic.stop()
        utt.finish()
        try:
            text = await utt.text()
        except Exception as exc:  # noqa: BLE001
            self.voice.cue("error")
            self.events.emit("error", where="listen", message=str(exc))
            return
        self.voice.cue("heard")
        self.events.emit("heard", text=text)
        if text:
            await self.handle(text)

    async def handle(self, text: str) -> None:
        if self.answer is not None and not self.answer.done():
            self.answer.set_result(text)
            return
        for pattern, action in LOCAL:
            if pattern.search(text.strip()):
                await self.local(action)
                return
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
        self.agent_task = asyncio.create_task(self._agent(text))

    async def _agent(self, text: str) -> None:
        if self.narrating and not self.narrating.done():
            self.narrating.cancel()
        self.voice.cue("working")
        run = await self.agent.run(text, self.latest)
        self.events.emit("agent", request=text, outcome=run.outcome,
                         steps=[{"tool": s.tool, "args": s.args} for s in run.steps],
                         seconds=round(run.seconds, 2))
        # The agent already said what happened. Count the result as seen.
        frame = await self.grab_now()
        self.watcher.update(frame)
        self.watcher.base = None
        self.watcher.prev = None

    def on_hud(self, alerts) -> None:
        for a in alerts:
            cue = {"low": "low", "critical": "critical", "empty": "ping"}.get(a.level)
            if cue:
                self.voice.cue(cue)
            self.events.emit("hud", meter=a.meter, level=a.level, value=round(a.value, 2))
            if a.level in ("low", "critical", "recovered") or "stamina" not in a.meter:
                words = HUD_WORDS[a.level].format(name=a.meter.capitalize())
                self.spawn(self.voice.say(words, interrupt=a.level == "critical"))

    async def start_hud(self) -> None:
        from .hud import HudWatch, calibrate

        frame = await self.grab_now()
        meters = await calibrate(self.muse, frame.image)
        if not meters:
            await self.voice.say("I can't find any health or stamina bars on this screen.")
            return
        self.hud = HudWatch(meters)
        self.s.watch_fps = max(self.s.watch_fps, 12.0)
        names = ", ".join(m.name for m in meters)
        await self.voice.say(f"Okay, I'm watching your {names}. I'll warn you when it "
                             "gets low.")
        self.events.emit("hud_on", meters=[m.name for m in meters])

    def emit_settings(self) -> None:
        self.events.emit("settings", verbosity=self.narrator.verbosity,
                         team_up=self.narrator.team_up, paused=self.paused,
                         language=self.narrator.language)

    async def local(self, action: str) -> None:
        self.events.emit("local", action=action)
        self.spawn(self._after_local())
        if action == "stop":
            self.voice.stop()
            if self.agent_task and not self.agent_task.done():
                self.agent.cancelled = True
        elif action == "repeat":
            last = self.narrator.recent[-1] if self.narrator.recent else "Nothing yet."
            await self.voice.say(last)
        elif action == "describe":
            frame = await self.grab_now()
            self._start_narration(frame, describe=True)
        elif action in ("terse", "quiet", "normal", "detailed"):
            self.narrator.verbosity = action
            await self.voice.say({"terse": "Terse mode.",
                                  "quiet": "Okay, I'll keep it short.",
                                  "detailed": "Sure, more detail from now on.",
                                  "normal": "Back to normal."}[action])
        elif action == "team_up":
            self.narrator.team_up = True
            await self.voice.say("Got it. I'll leave the menus to the game's narrator "
                                 "and cover everything else.")
        elif action == "solo":
            self.narrator.team_up = False
            await self.voice.say("Okay, I'll read the menus too.")
        elif action == "hud":
            await self.start_hud()
        elif action == "hud_off":
            self.hud = None
            await self.voice.say("Okay, no more health warnings.")
        elif action == "pause":
            self.paused = True
            await self.voice.say("Okay, I'll stay quiet until you say resume.")
        elif action == "resume":
            self.paused = False
            self.watcher.force()
            await self.voice.say("I'm back.")

    # Control socket: hotkeys and the overlay talk to the session through it.

    async def serve(self) -> None:
        path = socket_path()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

        async def client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                line = (await reader.readline()).decode().strip()
                if line == "events":
                    q: asyncio.Queue = asyncio.Queue()
                    self.events.subs.add(q)
                    try:
                        while True:
                            writer.write((json.dumps(await q.get()) + "\n").encode())
                            await writer.drain()
                    finally:
                        self.events.subs.discard(q)
                reply = await self.command(line)
                writer.write((reply + "\n").encode())
                await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_unix_server(client, str(path))
        async with server:
            await server.serve_forever()

    async def command(self, line: str) -> str:
        cmd, _, arg = line.partition(" ")
        if cmd == "talk-start":
            self.talk_start()
        elif cmd == "talk-stop":
            self.spawn(self.talk_stop())
        elif cmd == "talk":
            if self.utt is None:
                self.talk_start()
            else:
                self.spawn(self.talk_stop())
        elif cmd == "do" and arg:
            self.events.emit("heard", text=arg, typed=True)
            self.spawn(self.handle(arg))
        elif cmd in ("stop", "repeat", "describe", "terse", "quiet", "normal", "detailed",
                     "team_up", "solo", "pause", "resume", "hud", "hud_off"):
            self.spawn(self.local(cmd))
        elif cmd == "status":
            u = self.muse.usage
            return json.dumps({"calls": u.calls, "cost_usd": round(u.cost_usd(), 4),
                               "paused": self.paused, "verbosity": self.narrator.verbosity})
        else:
            return f"unknown command: {cmd}"
        return "ok"

    async def _after_local(self) -> None:
        await asyncio.sleep(0)
        self.emit_settings()

    async def run(self, ui: bool = True) -> None:
        from .web import PORT, serve_ui

        await asyncio.to_thread(self.voice.warm_up)
        self.events.emit("ready", voice=self.voice.voice_name, device=self.voice.device,
                         game=self.s.game or "Ready when you are", talk_key="F8")
        self.emit_settings()
        tasks = [self.watch(), self.serve(), self.track_status()]
        if ui:
            tasks.append(serve_ui(self))
            print(f"Player Two window: http://127.0.0.1:{PORT}/", file=sys.stderr)
        await self.voice.say("Player Two is here. Hold the talk key and tell me what you need.")
        await asyncio.gather(*tasks)

    def close(self) -> None:
        self.controls.close()
        self.mixer.close()


async def send(line: str) -> str:
    reader, writer = await asyncio.open_unix_connection(str(socket_path()))
    writer.write((line + "\n").encode())
    await writer.drain()
    reply = (await reader.readline()).decode().strip()
    writer.close()
    return reply


def main_run(settings: Settings, controls: str | None = None, log: str | None = None) -> None:
    session = Session(settings, controls=controls)
    if log:
        session.events.log_path = Path(log)
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        print(f"\n{session.muse.usage.calls} model calls, ${session.muse.usage.cost_usd():.3f}",
              file=sys.stderr)
