"""The live session: watch the screen, narrate, listen, act."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

from .agent import Agent
from .audio import Mixer
from .config import Settings, first_run, open_control, save_settings, socket_path, start_control
from .controls import BUTTONS, ControlsError, open_controls
from .listen import HandsFree, Mic, Utterance, addressed
from .muse import Muse
from .narrator import Narrator
from .screen import (
    Capture,
    CaptureError,
    ChangeWatcher,
    Frame,
    GameHidden,
    NoGame,
    auto_capture,
)
from .screenreader import open_voice

# Short commands handled without the model. They must be the whole utterance: "keep going,
# but don't leave" or "stop attacking and heal" are requests for the agent, not commands.
LOCAL = [
    ((r"stop|stop it|quiet|shush|shut up|be quiet|hush|stop talking|cancel|cancel that"
      r"|never ?mind|forget it|stop that"), "stop"),
    ((r"repeat|repeat that|say that again|say it again|read that again|what did (you|they|he"
      r"|she) say|again|come again|what|what was that|sorry|pardon|huh"), "repeat"),
    ((r"describe|describe (the|this) screen|what'?s on (the )?screen|where am i|look around"
      r"|what do you see|what'?s (going on|happening)"), "describe"),
    (r"help|what can you do|what can i say|commands|list (the )?commands", "help"),
    # Moving dialogue along without a model turn: a press of a, then the narrator reads
    # the new line. Not "continue": that's a title screen option.
    (r"next|next line|next page|press a|hit a", "next"),
    ((r"what'?s wrong|what is wrong|(are you|is everything) (ok|okay|working)|check yourself"
      r"|self check|status( report)?|what'?s (broken|not working)"), "check"),
    (r"(be )?terse|terse mode|just the facts", "terse"),
    (r"talk less|less detail|quieter|be quieter|quiet mode", "quiet"),
    (r"(back to )?normal|normal mode|regular mode", "normal"),
    (r"more detail|more details|talk more|detailed( mode)?", "detailed"),
    (r"(talk|speak|read) (slower|more slowly)|slow down|slower", "slower"),
    (r"(talk|speak|read) faster|speed up|faster", "faster"),
    ((r"team up|the game (reads|narrates) (the |its own )?menus|leave (the )?menus to the"
      r" game"), "team_up"),
    (r"(you )?read the menus( too)?|solo( mode)?", "solo"),
    ((r"watch my (health|hp|stamina)|health warnings( on)?|warn me (about|when) (my )?"
      r"(health|hp) (is )?(gets )?low"), "hud"),
    (r"stop watching my (health|hp|stamina)|(no|stop) (the )?health warnings", "hud_off"),
    # Bare "pause" and "unpause" are about the game, so they go to the agent.
    (r"stop narrating|pause (the )?narration|pause narrating|stop describing", "pause"),
    (r"start narrating|keep narrating|resume (the )?narration|resume narrating", "resume"),
    (r"(show|open) (the |your )?(player two )?window", "show"),
    ((r"good ?bye|bye( bye)?|(quit|exit|close|turn off|shut down)"
      r" (player two|yourself)|turn yourself off|that'?s all for (now|today)"), "quit"),
]
LOCAL_RE = [(re.compile(p), action) for p, action in LOCAL]


def local_action(text: str) -> str | None:
    """The local command this utterance is, if the whole thing is one."""
    t = re.sub(r"[^\w' ]+", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^((hey|ok|okay|please|player two|can you|could you) )+", "", t)
    # "Quit player two" needs its name; "shut up, player two" doesn't.
    for form in (t, re.sub(r"( (please|now|for me|player two|thanks|thank you))+$", "", t)):
        for pattern, action in LOCAL_RE:
            if pattern.fullmatch(form):
                return action
    return None


HELP = ("{talk} and ask me anything about the game, or tell me what to do, like: "
        "open the menu, or: use a potion on Yosuke. You can also say: next, to press A and "
        "hear the next line, describe, repeat, stop, help, what's wrong, terse, more "
        "detail, slower, faster, pause narration, resume narration, watch my health, or "
        "goodbye.")

MIC_PROBLEM = ("I can't find a microphone, so I can't hear you. I opened the Player Two "
               "window, where you can type to me instead.")
SOUND_PROBLEM = ("I can't find a speaker or headset. I'll start talking as soon as one is "
                 "connected.")
# "Player Two, stop" (and its other ways to say stop), anywhere in what's heard, as a whole
# command: followed by the end of a sentence or its name again, so "Player Two, stop
# narrating" and "quiet mode" stay commands of their own. In words still coming in, the
# end of the text isn't the end of the sentence: STOP_SO_FAR needs punctuation.
_STOP = (r"(?:player|layer|play a)s?[\s,-]*(?:(?:two|2|ii)\b|(?:to|too)(?=\s*[,.!?]))"
         r"[\s,.!?:-]*(?:stop|stop it|stop that|stop talking|be quiet|quiet|shush|shut up"
         r"|cancel|cancel that|hush|never ?mind)")
STOP_CALL = re.compile(_STOP + r"(?=\s*(?:[,.!?;]|$)|\s+(?:hey |ok |okay )?(?:player|layer))",
                       re.IGNORECASE)
STOP_SO_FAR = re.compile(_STOP + r"(?=\s*[,.!?;]|\s+(?:hey |ok |okay )?(?:player|layer))",
                         re.IGNORECASE)
HANDS_FREE_PROBLEM = ("I couldn't start hands-free listening, so for now hold the talk key, "
                      "or use the talk button in the Player Two window. Run player-two setup "
                      "to try again.")
TALK_KEY_PROBLEM = ("Your talk key isn't set up yet. Run player-two setup in a terminal to "
                    "set it up. Until then, I opened the Player Two window: type to me "
                    "there, or use its talk button.")

# Settings the companion window can change while playing. Each is saved to config.toml.
SETTABLE: dict[str, tuple] = {
    "verbosity": ("terse", "quiet", "normal", "detailed"),
    "confirm": ("always", "important", "never"),
    "language": ("English", "British English", "Spanish", "French", "Italian",
                 "Portuguese", "Hindi"),
    "output": ("auto", "voice", "screen-reader"),
    "team_up": (True, False),
    "companion": ("auto", "on", "off"),
    "hands_free": (True, False),
}
SPEECH_RATES = (0.8, 2.0)


def parse_setting(key: str, raw: str):
    """The value to store for `key`, or ValueError when it isn't one of the choices."""
    if key == "speech_rate":
        v = round(float(raw), 2)
        if not SPEECH_RATES[0] <= v <= SPEECH_RATES[1]:
            raise ValueError(raw)
        return v
    if key not in SETTABLE:
        raise ValueError(key)
    for choice in SETTABLE[key]:
        if str(choice).lower() == raw.strip().lower():
            return choice
    raise ValueError(raw)


def trouble_words(exc: BaseException) -> tuple[str, str]:
    """A kind and a spoken line for a model or network failure."""
    import openai

    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return "key", ("The model turned down my API key. Run player-two setup in a "
                       "terminal to check it.")
    if isinstance(exc, openai.APITimeoutError):
        return "slow", "The model is taking too long right now. I'll keep trying."
    if isinstance(exc, TimeoutError):
        return "timeout", "That took too long. Try again?"
    if isinstance(exc, GameHidden):
        return "nogame", "The game is minimized or hidden. Bring it back and I'll look again."
    if isinstance(exc, NoGame):
        return "nogame", "I don't see a game yet. Start one and I'll find it."
    if isinstance(exc, openai.RateLimitError):
        return "rate", "I'm being rate limited, so I'll slow down for a bit."
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if type(exc).__name__ == "InvalidStatus" and status in (401, 403):
        return "key", ("Speech recognition turned down my API key. Run player-two setup in "
                       "a terminal to check it.")
    if isinstance(exc, ControlsError):
        return "controls", ("I can't press buttons in the game right now. Run player-two setup "
                            "to check the controller.")
    if isinstance(exc, PermissionError | FileNotFoundError):
        return "other", f"Something is missing or locked on this computer: {exc}."
    if isinstance(exc, openai.APIConnectionError | openai.APITimeoutError | OSError):
        return "net", ("I can't reach the model right now. Check the internet connection. "
                       "I'll keep trying.")
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return "server", "The model is having trouble right now. I'll keep trying."
    return "other", "Something went wrong on my end. I'll keep trying."


def game_keywords(name: str) -> list[str]:
    """Words from the game's name that help speech recognition ("Persona", "Golden")."""
    return [w for w in re.findall(r"[A-Za-z][\w']+", name) if len(w) > 2][:8]


HUD_WORDS = {"low": "{name} low.", "critical": "{name} critical!",
             "recovered": "{name} back up.", "empty": "Out of {name}."}


class Events:
    """Fan-out of session events to any connected viewer (the overlay, logs)."""

    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.log_path: Path | None = None
        self.recent: list[dict] = []
        self.sticky: dict[str, dict] = {}
        self.last_said = ""

    def snapshot(self) -> list[dict]:
        """What a window that just opened needs: the latest state, then recent history."""
        return list(self.sticky.values()) + self.recent[-30:]

    def emit(self, kind: str, **data) -> None:
        if isinstance(data.get("text"), str):
            from .voice import tidy

            data["text"] = tidy(data["text"])
        ev = {"t": time.time(), "kind": kind, **data}
        if kind in ("say", "ask") and data.get("text") and not data.get("status"):
            self.last_said = data["text"]
        if kind in ("ready", "status", "frame", "settings", "game"):
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
        self.voice = open_voice(settings, self.mixer)
        self.capture = capture or auto_capture(settings.window, settings.capture)
        self.controls = open_controls(controls or settings.controls, focus=self.focus_game)
        # Things the player should hear about at start-up (no controller, no mic).
        self.problems: list[str] = [self.controls.problem] if self.controls.problem else []
        self.problems += settings.problems
        if self.mixer.lost:
            self.problems.append(SOUND_PROBLEM)
        self.narrator = Narrator(self.muse, settings.verbosity, settings.game,
                                 settings.language, settings.team_up)
        self.watcher = ChangeWatcher()
        self.events = Events()
        self.latest: Frame | None = None
        self.narrating: asyncio.Task | None = None
        # Newest settled screen not yet narrated, and a back-off while screens keep
        # settling with nothing new to say (animated menus, fog, idle cutscenes).
        self.pending: tuple[Frame, float] | None = None
        self.silent_streak = 0
        self.quiet_until = 0.0
        # After an error, no narration until then, scene cut or not.
        self.error_until = 0.0
        # Until then, settled screens don't count as scene cuts (after Player Two forced a
        # fresh look itself), so they don't talk over what it just said.
        self.soft_until = 0.0
        self.talk_started = 0.0
        self.capture_failures = 0
        self.asked = ""
        self._last_show = 0.0
        self.agent_task: asyncio.Task | None = None
        self.paused = False
        self.mic: Mic | None = None
        self.utt: Utterance | None = None
        # Hands-free listening: the listener and its task, whether the player has said
        # "Player Two" in what's being heard now (and when), until when a follow-up
        # needs no name (after "Player Two" on its own), and when Player Two last spoke
        # (what the microphone heard then may be its own voice).
        self.hands: HandsFree | None = None
        self.hands_task: asyncio.Task | None = None
        self.hands_active = False
        self.woke_at = 0.0
        self.woke_for = 0.0  # when the speech that said "Player Two" began
        self.follow_until = 0.0
        self.spoke_at = 0.0
        self.narrating_text = ""
        # When Player Two was speaking lately, as [start, end] times.
        self.spans: deque[list[float]] = deque(maxlen=64)
        # The speech (by when it began) and how many "Player Two, stop"s in it were acted on.
        self.stop_seen: tuple[float, int] = (0.0, 0)
        self.hands_lock = asyncio.Lock()
        self.keywords: list[str] = []
        self._tasks: set[asyncio.Task] = set()
        self.side_task: asyncio.Task | None = None  # a "where is" or health-bar search
        self.unnamed_answer = False  # the answer came hands-free, without "Player Two"
        self.last_trouble = ""  # the last problem said, for "what's wrong?"
        self.fps_before_hud: float | None = None
        self.answer: asyncio.Future[str] | None = None
        self.hud = None
        self.agent = Agent(self.muse, self.controls, lambda: self.grab_now(show=True),
                           self.agent_say,
                           game=settings.game, ask=self.agent_ask, confirm=settings.confirm,
                           language=settings.language, on_action=self.on_action)
        self.status = ""
        self.transcribing = False
        # The game in view: None until the first look, then its name ("" when it has
        # none) or False while no game window is open.
        self.game: str | bool | None = None
        self.game_id: str | None = None
        self.no_game_since: float | None = None
        self.troubled: dict[str, float] = {}
        self.window = None
        self.ui = True  # the window's web server runs (not with --no-ui)
        self.unheard = ""  # the greeting, when there was no sound device to say it on
        self.main: asyncio.Task | None = None

    def focus_game(self) -> None:
        self.capture.focus()

    def fresh_look(self) -> None:
        """Narrate the screen again soon, without cutting off what's being said."""
        self.watcher.force()
        self.soft_until = time.monotonic() + 3.0

    def listening(self) -> bool:
        """The player is talking, or what they said is being worked out."""
        return self.utt is not None or self.transcribing or self.hands_active

    def announce(self, text: str, interrupt: bool = False, status: bool = False,
                 wait_turn: bool = False) -> asyncio.Task:
        """Say a line of Player Two's own (not narration) and caption it in the window.

        A `status` line (an error, "didn't catch that") isn't what "repeat" replays.
        With `wait_turn` (news nobody asked for), it waits until the player has finished
        talking instead of speaking over them.
        """
        self.events.emit("say", text=text, source="p2", status=status)

        async def run():
            while wait_turn and self.listening():
                await asyncio.sleep(0.1)
            await self.voice.say(text, interrupt=interrupt)

        return self.spawn(run())

    def update_game(self, announce: bool = True) -> None:
        """Check which game the capture is looking at (cheap: called on every frame)."""
        key = self.capture.game_key()
        if key == self.game_id and self.game not in (None, False):
            self.no_game_since = None
            return  # same window; its title may change, the game doesn't
        self.set_game(self.capture.game(), announce=announce, key=key)

    def set_game(self, found: str | bool, announce: bool = True, key: str | None = None) -> None:
        """Track which game is open; say so when one opens or closes mid-session."""
        if found is False:
            # A game window can blink out while it changes resolution; wait before
            # calling it closed.
            now = time.monotonic()
            self.no_game_since = self.no_game_since or now
            if self.game is not None and now - self.no_game_since < 5.0:
                return
            if self.game is False:
                return
        else:
            self.no_game_since = None
            if key is not None and key == self.game_id and self.game is not None:
                return
        self.game_id = key
        before, self.game = self.game, found
        name = self.s.game or (found if isinstance(found, str) else "")
        self.narrator.game = self.agent.game = name
        self.keywords = game_keywords(name)
        self.events.emit("game", name=name, found=found is not False)
        if not announce or before is None:
            return
        if found is False:
            self.announce("The game closed. I'll wait for the next one.", wait_turn=True)
            return
        # A different game: start fresh, as if Player Two had just sat down.
        self.narrator.recent.clear()
        self.fresh_look()
        self.announce(f"I can see {name}." if name else "I can see your game.", wait_turn=True)

    def trouble(self, exc: BaseException, where: str, answer: bool = False) -> None:
        """Say what went wrong, once a minute per kind of problem, and log it.

        With `answer` (the player just asked for something), there's always a reply:
        the full explanation once a minute, a short one in between.
        """
        kind, words = trouble_words(exc)
        self.events.emit("error", where=where, message=str(exc)[:300] or kind)
        self.last_trouble = words
        now = time.monotonic()
        if now - self.troubled.get(kind, -1e9) < 60:
            if answer:
                self.voice.cue("error")
                self.announce("Sorry, that didn't work.", interrupt=True, status=True)
            return
        if kind == "key" and kind in self.troubled and not answer:
            return  # a rejected key won't fix itself: said once until they ask again
        self.troubled[kind] = now
        self.voice.cue("error")
        self.announce(words, status=True, wait_turn=not answer)

    def recovered(self) -> None:
        if self.troubled.pop("net", None) is not None or self.troubled.pop("server", None):
            self.troubled.clear()
            self.announce("I'm back online.", status=True, wait_turn=True)

    def stop_side(self) -> None:
        """Cancel a "where is it" or "watch my health" still on its way."""
        if self.side_task is not None and not self.side_task.done():
            self.side_task.cancel()
        self.side_task = None

    def spawn(self, coro) -> asyncio.Task:
        """Start a background task and keep a reference so it isn't collected mid-run."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._task_failed)
        return task

    def _task_failed(self, task: asyncio.Task) -> None:
        """A background task that raised: log it (what the player asked for says it)."""
        if task.cancelled() or task.exception() is None:
            return
        exc = task.exception()
        name = getattr(task.get_coro(), "__qualname__", "a task")
        print(f"Player Two: {name} failed: {exc!r}", file=sys.stderr)
        self.events.emit("error", where="task", message=str(exc)[:300] or type(exc).__name__)

    async def grab_now(self, show: bool = False) -> Frame:
        frame = await asyncio.to_thread(self.capture.grab)
        self.latest = frame
        if show:
            self.show(frame)
        return frame

    def show(self, frame: Frame, force: bool = False) -> None:
        """Send a small copy of the frame to the companion window (a few a second)."""
        import base64
        import io

        now = time.monotonic()
        # Settled screens always go out (a window that opens later shows the latest one);
        # the agent's many looks while it settles only a few times a second, and only
        # when a window is watching.
        if not force and (not self.events.subs or now - self._last_show < 0.25):
            return
        self._last_show = now

        img = frame.image.convert("RGB")
        img.thumbnail((720, 405))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        self.events.emit("frame", jpeg=base64.b64encode(buf.getvalue()).decode())

    def on_action(self, name: str, args: dict) -> None:
        # Most games ignore controllers and keys while another window has focus.
        with contextlib.suppress(Exception):
            self.focus_game()
        if name == "press":
            self.events.emit("press", buttons=args.get("buttons", []))
        elif name == "hold":
            self.events.emit("hold", input=args.get("input", ""),
                             direction=args.get("direction", ""), ms=args.get("ms", 0))
        elif name == "walk_to":
            self.events.emit("walk", target=args.get("target", ""))
        elif name == "walk_to_map":
            self.events.emit("walk", target=args.get("label", "a spot on the map"))
        elif name == "keep_walking":
            way = str(args.get("direction", "")).replace("_", " and ")
            self.events.emit("walk", target="", direction=way)
        elif name == "walk_note":
            self.events.emit("walk_note", text=args.get("text", ""))

    def current_status(self) -> str:
        if self.utt is not None or self.hands_active:
            return "listening"
        if self.transcribing:
            return "thinking"
        if self.agent_task is not None and not self.agent_task.done():
            waiting = self.answer is not None and not self.answer.done()
            if not self.voice.idle():
                return "speaking"
            return "listening" if waiting else "acting"
        if not self.voice.idle():
            return "speaking"
        if self.narrating is not None and not self.narrating.done():
            return "thinking"
        return "paused" if self.paused else "watching"

    async def track_status(self) -> None:
        checked = time.monotonic()
        while True:
            # Finding a speaker again restarts the sound system, which closes the
            # hands-free mic for a moment: less often then, and never mid-sentence.
            every = 2.0 if self.hands is None else 10.0
            hearing = self.hands is not None and (self.hands.utt is not None
                                                  or self.hands_active)
            if time.monotonic() - checked > every and self.utt is None and not hearing:
                checked = time.monotonic()
                if await asyncio.to_thread(self.mixer.check):
                    back = f"My sound is back. {self.unheard}".strip()
                    self.unheard = ""
                    self.announce(back, status=True, wait_turn=True)
            now = time.monotonic()
            if not self.voice.idle():
                self.spoke_at = now
                if self.spans and now - self.spans[-1][1] < 0.25:
                    self.spans[-1][1] = now
                else:
                    self.spans.append([now, now])
            if self.hands_active and now - self.woke_at > 20:
                # What they were saying never came back (the mic went away mid-word).
                self.hands_active = False
                if self.utt is None:
                    self.agent.resume()
            st = self.current_status()
            if st != self.status:
                self.status = st
                self.events.emit("status", state=st)
            await asyncio.sleep(0.08)

    async def agent_ask(self, question: str, timeout: float = 30.0) -> str:
        """Ask out loud and wait for the next thing the player says."""
        self.events.emit("ask", text=question)
        self.answer = asyncio.get_running_loop().create_future()
        self.unnamed_answer = False
        self.asked = question
        await self.voice.say(question, interrupt=False)
        await self.voice.wait_idle(15)
        self.voice.cue("listening")
        try:
            asked = time.monotonic()
            end = asked + timeout
            while True:
                try:
                    got = await asyncio.wait_for(asyncio.shield(self.answer),
                                                 max(0.1, end - time.monotonic()))
                    if got and self.unnamed_answer:
                        got += (' (heard hands-free without "Player Two": if it doesn\'t '
                                "sound like an answer, it may be the game talking; for "
                                "anything hard to undo, ask again)")
                    return got
                except TimeoutError:
                    # Still talking or being transcribed: give them time to finish
                    # (hands-free, up to a point: the game's own voices count as talking).
                    hearing = self.hands is not None and self.hands.utt is not None and (
                        time.monotonic() - asked < 3 * timeout)
                    if self.utt is not None or self.transcribing or hearing:
                        end = time.monotonic() + 10
                        continue
                    return ""
        finally:
            self.answer = None

    async def agent_say(self, text: str) -> None:
        self.events.emit("say", text=text, source="agent")
        self.narrator.recent.append(text)
        await self.voice.say(text, interrupt=False)

    # Narration

    async def _narrate(self, frame: Frame, interrupt: bool = False) -> None:
        t = time.monotonic()

        self.narrating_text = ""

        async def deltas():
            async for d in self.narrator.narrate(frame):
                self.narrating_text += d
                yield d

        try:
            text = await self.voice.say_stream(deltas(), interrupt=interrupt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  say it, back off, keep watching
            self.trouble(exc, "narrate")
            self.error_until = time.monotonic() + 10.0
            self.pending = None
            self.watcher.force()  # look again once the back-off is over
            return
        self.recovered()
        if text:
            self.silent_streak = 0
            self.quiet_until = 0.0
            self.events.emit("say", text=text, source="narrator",
                             latency=round(time.monotonic() - t, 2))
        else:
            self.silent_streak += 1
            self.quiet_until = time.monotonic() + min(16.0, 2.0 ** self.silent_streak)

    # A settled screen this different from the last one is a new scene, back-off or not.
    SCENE_CUT = 0.3

    def _maybe_narrate(self) -> None:
        """Narrate the newest settled screen once the last narration has finished.

        Never cancels a narration in flight: on busy screens that would restart the
        model call every second, say nothing and pay for all of it.
        """
        if self.pending is None or (self.narrating and not self.narrating.done()):
            return
        if self.listening():
            return  # never over the player; the newest screen waits until they're done
        if time.monotonic() < self.error_until:
            return
        frame, change = self.pending
        cut = change >= self.SCENE_CUT and time.monotonic() >= self.soft_until
        if not cut and not self.voice.idle():
            return  # let the last line finish; we'll narrate the newest screen after
        if not cut and time.monotonic() < self.quiet_until:
            return
        self.pending = None
        # A whole new scene (a battle starting, a menu opening) cuts off old talk about
        # the last one, like a friend would.
        self.narrating = asyncio.create_task(self._narrate(frame, interrupt=cut))

    def _start_narration(self, frame: Frame, describe: bool = False) -> None:
        if self.narrating and not self.narrating.done():
            self.narrating.cancel()
        if describe:
            async def run():
                self.voice.stop()
                try:
                    text = await self.voice.say_stream(self.narrator.describe(frame))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.trouble(exc, "describe", answer=True)
                    return
                self.events.emit("say", text=text, source="describe")
            self.narrating = asyncio.create_task(run())
        else:
            self.narrating = asyncio.create_task(self._narrate(frame))

    async def watch(self) -> None:
        while True:
            period = 1.0 / self.s.watch_fps
            t = time.monotonic()
            try:
                frame = await self.grab_now()
            except GameHidden:
                self.update_game()  # still open: no "the game closed"
                await asyncio.sleep(1.0)
                continue
            except NoGame:
                self.set_game(False)
                await asyncio.sleep(1.0)
                continue
            except Exception as exc:  # noqa: BLE001  keep watching through hiccups
                self.capture_failures += 1
                if self.capture_failures == 5:  # a few seconds of it: say so, once
                    self.events.emit("error", where="capture", message=str(exc)[:300])
                    self.voice.cue("error")
                    # A desktop capture can't work on says how to fix it.
                    say = (str(exc) if isinstance(exc, CaptureError)
                           else "I can't see the screen right now. I'll keep trying.")
                    self.announce(say, status=True, wait_turn=True)
                await asyncio.sleep(1.0)
                continue
            if self.capture_failures >= 5:
                self.announce("I can see the screen again.", status=True, wait_turn=True)
            self.capture_failures = 0
            try:
                self._look(frame)
            except Exception as exc:  # noqa: BLE001  one bad frame must not end the session
                self.events.emit("error", where="watch", message=str(exc)[:300])
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t)))

    def _look(self, frame: Frame) -> None:
        notice, self.capture.notice = self.capture.notice, ""
        if notice:
            self.events.emit("error", where="capture", message=notice)
            self.announce(notice, status=True, wait_turn=True)
        self.update_game()
        if self.hud is not None:
            self.on_hud(self.hud.update(frame.image, frame.t))
        busy = self.agent_task is not None and not self.agent_task.done()
        if self.watcher.update(frame) and not self.paused and not busy:
            self.events.emit("settled", label=frame.label,
                             change=round(self.watcher.last_change, 3))
            self.show(frame, force=True)
            self.pending = (frame, self.watcher.last_change)
        if not self.paused and not busy:
            self._maybe_narrate()

    # Talking

    def talk_start(self) -> None:
        if self.utt is not None:
            return
        self.hands_active = False  # the talk key takes over from hands-free listening
        self.voice.stop()
        self.voice.cue("listening")
        waiting = self.answer is not None and not self.answer.done()
        if self.agent_task and not self.agent_task.done() and not waiting:
            self.agent.hold()  # until it's clear what they want; see talk_stop
        mic = Mic()
        try:
            mic.start()
        except Exception as exc:  # noqa: BLE001  no microphone: say so, don't crash
            self.agent.resume()
            self.voice.cue("error")
            self.events.emit("error", where="mic", message=str(exc)[:300])
            self.announce(MIC_PROBLEM, interrupt=True, status=True)
            self.open_window(refocus=False)
            return
        self.mic = mic
        self.talk_started = time.monotonic()
        self.utt = Utterance(self.keywords,
                             on_partial=lambda p: self.events.emit("hearing", text=p))
        self.utt.start()
        self.spawn(self._pump(self.mic, self.utt))

    async def _pump(self, mic: Mic, utt: Utterance) -> None:
        while self.utt is utt:
            if utt.result is not None and utt.result.done():
                # The speech session ended (an error). Don't wait for the key release,
                # which may never come; talk_stop says what went wrong.
                self.spawn(self.talk_stop())
                break
            if time.monotonic() - self.talk_started > 60:
                # A lost key release would leave the mic open for good.
                self.spawn(self.talk_stop())
                break
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
        self.transcribing = True
        try:
            text = await utt.text()
        except Exception as exc:  # noqa: BLE001
            self.agent.resume()
            self.trouble(exc, "listen", answer=True)
            return
        finally:
            self.transcribing = False
        self.voice.cue("heard")
        self.events.emit("heard", text=text or "")
        try:
            if text:
                await self.handle(text)
            else:
                await self.announce("Sorry, I didn't catch that.", interrupt=True, status=True)
        finally:
            # Whatever they said that wasn't stop or a new request (a repeat, a question,
            # nothing), the action they asked for earlier carries on.
            self.agent.resume()

    # Hands-free

    def open_ear(self, at: float) -> bool:
        """Whether speech that began at `at` counts without "Player Two" first: an answer
        to a question, or what comes right after "Player Two" on its own."""
        return (self.answer is not None and not self.answer.done()) or at < self.follow_until

    def maybe_echo(self, start: float, end: float) -> bool:
        """Whether speech heard from `start` to `end` could be Player Two's own voice: it
        was talking for most of that time. (Talking over it, the player's "Player Two"
        stops its voice, so their sentence is mostly quiet on its side.)"""
        spoke = sum(max(0.0, min(b, end) - max(a, start)) for a, b in self.spans)
        return spoke >= 0.6 * max(0.1, end - start)

    def echo(self, text: str, named: bool = False) -> bool:
        """Whether `text` reads like Player Two's own recent words picked up by the mic.

        With `named`, only its lines that say its own name count ("Hi, I'm Player Two"),
        and what was heard after the name is compared with all of such a line. One word
        after the name ("Player Two, stop") always counts as the player's.
        """
        def words(t: str | None) -> set[str]:
            return set(re.findall(r"[a-z0-9']+", (t or "").lower()))

        said = self.said_lately()
        heard = words(addressed(text) if named else text)
        if named:
            said = [line for line in said if addressed(line) is not None]
            if not heard:
                return bool(said)  # just the name, while it said its own name lately
            if len(heard) == 1:
                return False
        elif not heard:
            return True
        for line in said:
            w = words(line)
            if w and len(heard & w) / len(heard) >= 0.6:
                return True
        return False

    def said_lately(self) -> list[str]:
        """Player Two's own lines from the last 30 s, and the narration being said now."""
        now = time.time()
        said = [e["text"] for e in self.events.recent
                if e["kind"] in ("say", "ask") and now - e["t"] < 30 and e.get("text")]
        return [*said, self.narrating_text]

    def new_stops(self, text: str, started: float, final: bool = False) -> bool:
        """Whether `text` (the speech that began at `started`, so far, or all of it when
        `final`) has a "Player Two, stop" not acted on yet. It counts even mixed into
        Player Two's own voice (speakers, no headset), unless it's the very words Player
        Two said."""
        def norm(t: str) -> str:
            return " ".join(re.findall(r"[a-z0-9']+", t.lower()))

        said = [norm(line) for line in self.said_lately()]
        found = [m for m in (STOP_CALL if final else STOP_SO_FAR).finditer(text)
                 if not any(norm(m.group(0)) in line for line in said)]
        seen = self.stop_seen[1] if self.stop_seen[0] == started else 0
        if len(found) <= seen:
            return False
        self.stop_seen = (started, len(found))
        self.voice.cue("heard")
        self.events.emit("heard", text="stop")
        return True

    def hands_partial(self, text: str, started: float) -> None:
        """The words so far of speech that began at `started`."""
        hands = self.hands
        if self.utt is not None or hands is None or hands.utt is None or (
                hands.started != started):
            return  # the talk key is in use, or a dropped stretch's last words
        if self.new_stops(text, started):
            self.spawn(self.local("stop"))
            return
        if self.stop_seen[0] == started and self.stop_seen[1]:
            # Stopped already in this breath: only its name again starts something new
            # (and cutting the voice off now would cut off "Okay, stopped").
            stops = list(STOP_SO_FAR.finditer(text))
            if not stops or not addressed(text[stops[-1].end():]):
                return
            text = text[stops[-1].end():]
        said = addressed(text)
        if said is None:
            if self.open_ear(started) or self.hands_active:
                self.events.emit("hearing", text=text)
            return
        # While it's saying its own name ("Hi, I'm Player Two"), the words so far can't
        # tell its voice from the player's: the whole sentence decides, in hands_heard.
        own_name = time.monotonic() - self.spoke_at < 1.0 and self.echo("Player Two",
                                                                         named=True)
        if not self.hands_active and not own_name and not self.echo(text, named=True):
            # "Player Two": stop talking and wait to hear what they want.
            self.hands_active, self.woke_at = True, time.monotonic()
            self.woke_for = started
            self.voice.stop()
            self.voice.cue("listening")
            waiting = self.answer is not None and not self.answer.done()
            if self.agent_task and not self.agent_task.done() and not waiting:
                self.agent.hold()
        if self.hands_active and said:
            self.events.emit("hearing", text=said)

    def hands_dropped(self) -> None:
        """Speech being heard was cut off before its words came back (the talk key went
        down, the mic went away): let go of what "Player Two" held."""
        if self.hands_active:
            self.hands_active = False
            if self.utt is None:
                self.agent.resume()

    async def hands_heard(self, text: str, error: BaseException | None,
                          started: float, ended: float) -> None:
        """Speech heard hands-free, from `started` to `ended`: act on it if it was said to
        Player Two."""
        # Only the speech that said "Player Two" lets go of what that held: another
        # sentence may already have started.
        woke = self.hands_active and self.woke_for == started
        if woke:
            self.hands_active = False
        try:
            if self.utt is not None:
                return  # the talk key is in use now; that's what counts
            if error is not None:
                if woke:
                    self.trouble(error, "listen", answer=True)
                else:
                    self.events.emit("error", where="listen", message=str(error)[:300])
                return
            if self.new_stops(text, started, final=True):
                await self.local("stop")
            stops = list(STOP_CALL.finditer(text))
            if self.stop_seen[0] == started and self.stop_seen[1] and stops:
                # Stopped in this breath: only what came after counts ("Player Two, stop.
                # Player Two, open the map"), and only if it's said to Player Two too.
                text = text[stops[-1].end():]
                if not addressed(text):
                    return
            echo_possible = self.maybe_echo(started, ended)
            said = addressed(text)
            if said is not None and echo_possible and self.echo(text, named=True):
                said = None
            elif said is None and text and not (echo_possible and self.echo(text)):
                busy = (self.agent_task is not None and not self.agent_task.done()) or (
                    not self.voice.idle())
                if self.open_ear(started) or (busy and local_action(text) == "stop"):
                    said = text  # an answer, a follow-up, or "stop" while it's busy
                    # Heard without its name: over speakers that could be the game.
                    self.unnamed_answer = self.answer is not None and not self.answer.done()
            if said is None:
                if text:
                    self.events.emit("overheard", text=text)
                return
            if not said:
                # Just "Player Two": the next thing they say is for it, name or not.
                self.follow_until = time.monotonic() + 8.0
                self.voice.cue("listening")
                return
            self.follow_until = 0.0
            self.voice.cue("heard")
            self.events.emit("heard", text=said)
            await self.handle(said)
        finally:
            if woke and self.utt is None:
                self.agent.resume()

    async def start_hands_free(self) -> bool:
        """Start listening hands-free. False (and the reason logged) when it can't."""
        from .listen import SpeechDetector, ensure_vad

        async with self.hands_lock:
            if self.hands_task is not None and not self.hands_task.done():
                return True
            try:
                detector = await asyncio.to_thread(lambda: SpeechDetector(ensure_vad()))
            except Exception as exc:  # noqa: BLE001  no download, no onnxruntime
                self.events.emit("error", where="hands_free", message=str(exc)[:300])
                return False
            if not self.s.hands_free:
                return True  # turned off again while the detector loaded
            self.hands = HandsFree(
                self.hands_partial, self.hands_heard, self.hands_dropped,
                lambda: ["Player Two", *self.keywords],
                lambda: self.utt is not None or self.transcribing,
                detector=detector, refresh=self.mixer.refresh)
            self.hands_task = self.spawn(self.hands.run())
            return True

    def stop_hands_free(self) -> None:
        if self.hands_task is not None:
            self.hands_task.cancel()
        self.hands, self.hands_task = None, None
        self.hands_dropped()

    async def handle(self, text: str) -> None:
        """Act on what the player said or typed; if that fails, they hear why."""
        try:
            await self._handle(text)
        except Exception as exc:  # noqa: BLE001  never fail silently
            self.trouble(exc, "command", answer=True)

    async def _handle(self, text: str) -> None:
        action = local_action(text)
        if action is None and self.paused and local_action(f"{text} narration") == "resume":
            action = "resume"  # "resume" right after pausing narration means narration
        if self.answer is not None and not self.answer.done():
            if action == "repeat":
                await self.voice.say(self.asked or "Sorry, say that again?")
                self.voice.cue("listening")
                return
            if action == "stop":
                self.agent.cancel()
                self.answer.set_result("")
                await self.announce("Okay, I won't.", interrupt=True)
                return
            if action and action != "next" and not self.agent.cancelled:
                # "Help", "describe", "slower": a command of its own, not the answer. The
                # question still waits.
                await self.local(action)
                self.voice.cue("listening")
                return
            if not self.agent.cancelled:
                self.answer.set_result(text)
                return
            # The player pressed talk to start something new, not to answer.
            self.answer.set_result("")
        if action:
            await self.local(action)
            return
        from .locate import where_request

        phrase = where_request(text)
        self.stop_side()
        if phrase:
            self.side_task = self.spawn(self.where(phrase))
            return
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
        self.agent_task = asyncio.create_task(self._agent(text))

    async def _agent(self, text: str) -> None:
        if self.narrating and not self.narrating.done():
            self.narrating.cancel()
        self.pending = None
        self.voice.cue("working")
        ticks = self.spawn(self._working_ticks())
        try:
            run = await self.agent.run(text, self.latest if self.game is not False else None)
        except asyncio.CancelledError:
            self.controls.steer(None)  # never leave the d-pad held
            self.events.emit("agent", request=text, outcome="cancelled", steps=[], turns=[],
                             seconds=0)
            raise
        except Exception as exc:  # noqa: BLE001  say it out loud instead of going silent
            self.trouble(exc, "agent", answer=True)
            return
        finally:
            ticks.cancel()
            self.pending = None
        self.events.emit("agent", request=text, outcome=run.outcome,
                         steps=[{"tool": s.tool, "args": s.args, "t": round(s.t, 2),
                                 "result": s.result[:200]}
                                for s in run.steps],
                         turns=run.turns, seconds=round(run.seconds, 2))
        # The agent already said what happened: count the screen it left as seen, so the
        # narrator doesn't talk over its last line about the same thing.
        with contextlib.suppress(Exception):
            self.watcher.accept(await self.grab_now())

    async def _working_ticks(self) -> None:
        """A soft tick every few seconds while acting, so a long silence isn't a mystery."""
        while True:
            await asyncio.sleep(4.0)
            if self.voice.idle() and self.answer is None:
                self.voice.cue("working")

    def on_hud(self, alerts) -> None:
        for a in alerts:
            cue = {"low": "low", "critical": "critical", "empty": "ping"}.get(a.level)
            if cue:
                self.voice.cue(cue)
            self.events.emit("hud", meter=a.meter, level=a.level, value=round(a.value, 2))
            if self.listening():
                continue  # the tone says it; words would land in what they're saying
            if a.level in ("low", "critical", "recovered") or "stamina" not in a.meter:
                words = HUD_WORDS[a.level].format(name=a.meter.capitalize())
                self.spawn(self.voice.say(words, interrupt=a.level == "critical"))

    async def where(self, phrase: str) -> None:
        """Find things with SAM 3.1, say where they are, and ping each one from its side."""
        try:
            await self._where(phrase)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.trouble(exc, "where", answer=True)

    async def _where(self, phrase: str) -> None:
        from .locate import describe, locate, point

        if self.narrating and not self.narrating.done():
            self.narrating.cancel()
        frame = await self.grab_now(show=True)
        matches, secs = await locate(self.muse, frame.image, phrase)
        how = "sam-3.1"
        if not matches:
            matches, how = await point(self.muse, frame.image, phrase), "muse-spark"
        text = describe(phrase, matches)
        self.events.emit("where", phrase=phrase, via=how, seconds=round(secs, 2),
                         points=[[round(m.cx, 3), round(m.cy, 3)] for m in matches[:12]])
        self.events.emit("say", text=text, source="agent")
        self.narrator.recent.append(text)
        await self.voice.say(text)
        await self.voice.wait_idle(10)
        for m in sorted(matches, key=lambda m: m.cx)[:8]:
            self.voice.cue("ping", pan=m.pan)
            await asyncio.sleep(0.28)

    async def start_hud(self) -> None:
        from .hud import HudWatch, calibrate

        try:
            frame = await self.grab_now()
            meters = await calibrate(self.muse, frame.image)
        except Exception as exc:  # noqa: BLE001
            self.trouble(exc, "hud", answer=True)
            return
        if not meters:
            await self.announce("I can't find any health or stamina bars on this screen.")
            return
        self.hud = HudWatch(meters)
        if self.fps_before_hud is None:
            self.fps_before_hud = self.s.watch_fps
        self.s.watch_fps = max(self.s.watch_fps, 12.0)
        names = ", ".join(m.name for m in meters)
        await self.announce(f"Okay, I'm watching your {names}. I'll warn you when it "
                             "gets low.")
        self.events.emit("hud_on", meters=[m.name for m in meters])

    def self_check(self) -> str:
        """Everything that could be wrong, said plainly: a way to find out without
        seeing the screen or a terminal."""
        from .desktop import talk_key_bound

        lines = []
        if self.game is False:
            lines.append("I don't see a game window.")
        elif self.capture_failures:
            lines.append("I'm having trouble seeing the screen.")
        else:
            lines.append(f"I can see {self.narrator.game or 'the game'}.")
        lines.append({"gamepad": "My virtual controller is plugged in.",
                      "keyboard": "I'm pressing keyboard keys, not a controller.",
                      "dry-run": "I can't press anything in the game right now."
                      }.get(self.controls.kind, ""))
        if MIC_PROBLEM in self.problems:
            lines.append("I can't find a microphone.")
        elif self.hands is not None:
            lines.append("I'm listening hands-free for my name.")
        elif talk_key_bound(self.s):
            lines.append("The talk key is set up.")
        else:
            lines.append("No talk key is set up; type in my window, or run player-two "
                         "setup.")
        if self.paused:
            lines.append("Narration is paused; say resume narration.")
        lines += [p for p in self.problems if p not in (MIC_PROBLEM,)]
        if self.last_trouble:
            lines.append(f"The last problem was: {self.last_trouble}")
        if len(lines) <= 3 and not self.last_trouble and not self.problems:
            lines.append("Everything looks fine.")
        return " ".join(line for line in lines if line)

    def emit_settings(self) -> None:
        self.events.emit("settings", verbosity=self.narrator.verbosity,
                         team_up=self.narrator.team_up, paused=self.paused,
                         language=self.narrator.language, confirm=self.agent.confirm,
                         speech_rate=self.s.speech_rate, output=self.s.output,
                         companion=self.s.companion, hands_free=self.hands is not None,
                         choices={k: list(v) for k, v in SETTABLE.items()},
                         speech_rates=list(SPEECH_RATES))

    async def set_setting(self, key: str, raw: str) -> str:
        """Change a setting now and keep it for next time."""
        try:
            value = parse_setting(key, raw)
        except ValueError:
            return f"bad setting: {key} {raw}"
        if key == "output" and value != self.s.output:
            # Loading a voice can take seconds (or a download): keep the session running.
            old = self.s.output
            self.s.output = value
            try:
                voice = await asyncio.to_thread(open_voice, self.s, self.mixer)
                await asyncio.to_thread(voice.warm_up)
            except Exception as exc:  # noqa: BLE001
                self.s.output = old
                self.events.emit("error", where="settings", message=str(exc)[:300])
                self.emit_settings()
                await self.announce("I couldn't switch how I talk. I'll keep this voice.")
                return "failed"
            self.voice.stop()
            self.voice = voice
        setattr(self.s, key, value)
        if key in ("verbosity", "team_up", "language"):
            setattr(self.narrator, key, value)
        if key in ("confirm", "language"):
            setattr(self.agent, key, value)
        if key == "language" and hasattr(self.voice, "lang"):
            from .voice import voice_for

            self.voice.lang, self.voice.voice_name = voice_for(value, self.s.voice)
        if key == "companion" and value == "on":
            self.open_window()
        if key == "hands_free":
            if value and not await self.start_hands_free():
                self.s.hands_free = False
                self.emit_settings()
                await self.announce(HANDS_FREE_PROBLEM, status=True)
                return "failed"
            if self.s.hands_free != value:
                return "ok"  # switched back while the detector loaded; that one counts
            if not value:
                self.stop_hands_free()
        save_settings({key: value})
        self.emit_settings()
        if key == "hands_free":
            await self.announce("Okay, I'm listening for my name. Say Player Two, then "
                                "what you need." if value else "Okay, hands-free is off.")
        if key in ("output", "language", "speech_rate"):
            await self.announce("Like this?" if key == "speech_rate" else "Okay.")
        return "ok"

    async def local(self, action: str) -> None:
        try:
            await self._local(action)
        except Exception as exc:  # noqa: BLE001  never fail silently
            self.trouble(exc, "command", answer=True)

    async def _local(self, action: str) -> None:
        self.events.emit("local", action=action)
        self.spawn(self._after_local())
        if action == "stop":
            self.stop_side()
            self.voice.stop()
            if self.answer is not None and not self.answer.done():
                self.answer.set_result("")
            if self.agent_task and not self.agent_task.done():
                # Said now, not when the model's reply comes back: no more presses or
                # words come from the run after this, and narration is back at once.
                self.agent.cancel()
                self.agent_task.cancel()
                await self.announce("Okay, stopped.", interrupt=True)
        elif action == "repeat":
            await self.voice.say(self.events.last_said or "I haven't said anything yet.")
        elif action == "next":
            if self.agent_task and not self.agent_task.done():
                await self.announce("I'm in the middle of something. Say stop first.")
                return
            try:
                await asyncio.to_thread(self.controls.press, "a")
            except Exception as exc:  # noqa: BLE001
                self.trouble(exc, "next", answer=True)
                return
            if self.paused:
                # Narration is off, so read the new screen once, after the text types out.
                await asyncio.sleep(0.8)
                with contextlib.suppress(Exception):
                    self._start_narration(await self.grab_now(), describe=True)
        elif action == "describe":
            try:
                frame = await self.grab_now()
            except Exception as exc:  # noqa: BLE001
                self.trouble(exc, "describe", answer=True)
                return
            self._start_narration(frame, describe=True)
        elif action == "check":
            await self.announce(self.self_check())
        elif action == "help":
            from .desktop import talk_key_bound

            talk = ("Start with my name, Player Two," if self.hands is not None
                    else "Press the talk key, speak, and press it again,"
                    if self.s.talk_toggle and talk_key_bound(self.s)
                    else "Hold the talk key while you speak," if talk_key_bound(self.s)
                    else "Type in the Player Two window, or use its talk button,")
            await self.announce(HELP.format(talk=talk), interrupt=True)
        elif action in ("slower", "faster"):
            if not hasattr(self.voice, "lang"):
                await self.announce("Your screen reader sets my speed. Change it there.")
                return
            step = 0.15 if action == "faster" else -0.15
            rate = round(min(2.0, max(0.8, self.s.speech_rate + step)), 2)
            await self.set_setting("speech_rate", str(rate))
        elif action == "show":
            self.open_window(refocus=False)
            await self.announce("Okay, the Player Two window is open.")
        elif action in ("terse", "quiet", "normal", "detailed"):
            self.narrator.verbosity = self.s.verbosity = action
            save_settings({"verbosity": action})
            self.emit_settings()
            await self.announce({"terse": "Terse mode.",
                                  "quiet": "Okay, I'll keep it short.",
                                  "detailed": "Sure, more detail from now on.",
                                  "normal": "Back to normal."}[action])
        elif action == "team_up":
            self.narrator.team_up = self.s.team_up = True
            save_settings({"team_up": True})
            self.emit_settings()
            await self.announce("Got it. I'll leave the menus to the game's narrator "
                                 "and cover everything else.")
        elif action == "solo":
            self.narrator.team_up = self.s.team_up = False
            save_settings({"team_up": False})
            self.emit_settings()
            await self.announce("Okay, I'll read the menus too.")
        elif action == "hud":
            self.side_task = self.spawn(self.start_hud())  # "stop" can cancel it
        elif action == "hud_off":
            self.hud = None
            if self.fps_before_hud is not None:
                self.s.watch_fps, self.fps_before_hud = self.fps_before_hud, None
            await self.announce("Okay, no more health warnings.")
        elif action == "pause":
            self.paused = True
            await self.announce("Okay, no narration until you say resume narration. "
                                "Questions still work.")
        elif action == "resume":
            self.paused = False
            self.fresh_look()
            await self.announce("I'm back.")
        elif action == "quit":
            await self.quit()

    async def quit(self) -> None:
        self.voice.stop()
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
        self.events.emit("bye")  # the window closes now: not a lost connection
        await self.announce("Bye for now. Have fun.")
        await self.voice.wait_idle(5)
        if self.main is not None:
            self.main.cancel()

    # Control socket: hotkeys and the overlay talk to the session through it.

    async def serve(self) -> None:
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

        server = await start_control(client)
        self.serving.set()
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
                     "team_up", "solo", "pause", "resume", "hud", "hud_off", "quit", "help",
                     "check",
                     "slower", "faster"):
            self.spawn(self.local(cmd))
        elif cmd == "press" and arg:
            # For scripted setups (tests, a known start for a task): no model involved.
            # "start wait2 down*2 a": buttons, pauses in seconds, repeats.
            plan: list[tuple[str, float]] = []
            for token in arg.split():
                try:
                    if token.startswith("wait"):
                        plan.append(("", min(float(token[4:] or 1), 30)))
                        continue
                    button, _, times = token.partition("*")
                    n = int(times or 1)
                except ValueError:
                    return f"can't read {token!r}: use a button, button*N or waitSECONDS"
                if button not in BUTTONS or not 1 <= n <= 50:
                    return f"unknown button or count: {token}"
                plan += [(button, 0.25)] * n
            for button, pause in plan:
                if button:
                    await asyncio.to_thread(self.controls.press, button)
                await asyncio.sleep(pause)
        elif cmd == "shot" and arg:
            if not os.path.isabs(arg):
                return "give the screenshot a full path"
            (await self.grab_now()).image.save(arg)
        elif cmd == "set":
            key, _, value = arg.partition(" ")
            return await self.set_setting(key, value)
        elif cmd == "show":
            if arg != "again" or self.want_window():
                self.open_window(refocus=arg == "again")
            if arg == "again":
                self.announce("I'm already running.")
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

    def greeting(self) -> str:
        """The first thing Player Two says: hello, which game, anything wrong, the key."""
        from .desktop import TALK_KEY, talk_key_bound

        new = first_run()
        terse = self.narrator.verbosity == "terse"
        name = self.narrator.game
        parts = ["Hi, I'm Player Two. I'll tell you what's happening in your game, and I "
                 "can play it for you." if new else "Player Two here."]
        if self.game is False:
            parts.append("I don't see a game yet. Start one and I'll find it.")
        elif name:
            parts.append(f"I can see {name}.")
        parts += self.problems
        how = (f"Press {TALK_KEY} to talk, and again when you're done" if self.s.talk_toggle
               else f"Hold {TALK_KEY} to talk")
        if self.hands is not None:
            if new:
                parts.append("I'm listening hands-free. Start with my name, like: Player "
                             "Two, open the menu. Ask for help to hear what else you can "
                             "say, and say stop any time to cut me off.")
            elif not terse:
                parts.append("Say Player Two, then what you need.")
        elif not talk_key_bound(self.s):
            parts.append(TALK_KEY_PROBLEM)
        elif new:
            parts.append(f"{how}, and tell me what you need. Say help to hear what else "
                         "you can say, and stop any time to cut me off.")
        elif not terse:
            parts.append(f"{how}.")
        if terse and not new:
            parts = [p for p in parts if p != "Player Two here."] or ["Ready."]
            parts.insert(0, "Player Two.")
        return " ".join(parts)

    def want_window(self) -> bool:
        import os

        if self.s.companion == "on":
            return True
        return self.s.companion == "auto" and bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"))

    async def _serve_ui(self) -> None:
        """The window's server; if it can't start (its port taken), go on without it."""
        from .web import PORT, serve_ui

        try:
            await serve_ui(self)
        except OSError as exc:
            self.ui = False
            self.events.emit("error", where="window", message=str(exc)[:300])
            self.announce(f"My window can't open: port {PORT} is taken by another program. "
                          "I'll keep going without it.", status=True, wait_turn=True)

    def open_window(self, refocus: bool = True) -> None:
        """Open the companion window. With `refocus`, hand focus back to the game after,
        in case the window took it; not when the player asked for the window or typing
        there is their only way in."""
        from .desktop import CompanionWindow
        from .web import window_url

        if not self.ui or (self.window is not None and self.window.alive()):
            return
        self.window = CompanionWindow()
        self.window.open(window_url(), getattr(self.capture, "target", None))
        if refocus:
            self.spawn(self._refocus_game())

    async def _refocus_game(self) -> None:
        await asyncio.sleep(1.5)
        with contextlib.suppress(Exception):
            self.focus_game()

    def windows_talk_key(self) -> None:
        import threading

        from .desktop import WindowsTalkKey

        loop = asyncio.get_running_loop()
        key = WindowsTalkKey(lambda: loop.call_soon_threadsafe(self.talk_start),
                             lambda: loop.call_soon_threadsafe(self.spawn, self.talk_stop()))
        threading.Thread(target=key.run, daemon=True).start()

    async def run(self, ui: bool = True) -> None:
        self.ui = ui
        from .listen import mic_ok
        from .web import window_url

        self.main = asyncio.current_task()
        # Open the control socket first, so a talk key pressed during start-up still lands.
        self.serving = asyncio.Event()
        tasks = [asyncio.create_task(self.serve()), asyncio.create_task(self.track_status())]
        if ui:
            tasks.append(asyncio.create_task(self._serve_ui()))
            # The address holds this run's secret: show where to find it, not the secret.
            from .desktop import start_page

            print(f"Player Two window: {start_page(window_url()).as_uri()}", file=sys.stderr)
        # Wait for the control channel, or for it to fail (port taken, no permission).
        ready = asyncio.create_task(self.serving.wait())
        await asyncio.wait({ready, tasks[0]}, return_when=asyncio.FIRST_COMPLETED)
        if tasks[0].done():
            ready.cancel()
            tasks[0].result()
        if sys.platform == "win32":
            self.windows_talk_key()
        if not mic_ok():
            self.problems.append(MIC_PROBLEM)
        # Even without a mic yet: hands-free opens one as soon as it's plugged in.
        if self.s.hands_free and not await self.start_hands_free():
            self.problems.append(HANDS_FREE_PROBLEM)
        await asyncio.to_thread(self.voice.warm_up)
        # Find the game before saying hello, so the greeting can name it.
        try:
            frame = await self.grab_now(show=True)
            self.update_game(announce=False)
            self.watcher.update(frame)
        except GameHidden:
            self.update_game(announce=False)
        except NoGame:
            self.set_game(False, announce=False)
        except Exception as exc:  # noqa: BLE001  the watch loop will report it
            self.events.emit("error", where="capture", message=str(exc)[:300])
        # Without a mic or a talk key, typing in the window is the only way to reach me.
        from .desktop import talk_key_bound as _bound

        typing_only = MIC_PROBLEM in self.problems or (
            not _bound(self.s) and self.hands is None)
        if ui and (self.want_window() or typing_only or SOUND_PROBLEM in self.problems):
            self.open_window(refocus=not typing_only)
        from .desktop import TALK_KEY, talk_key_bound

        self.events.emit("ready", voice=self.voice.voice_name, device=self.voice.device,
                         game=self.narrator.game, found=self.game is not False,
                         talk_key=TALK_KEY, talk_key_bound=talk_key_bound(self.s),
                         talk_toggle=self.s.talk_toggle, hands_free=self.hands is not None,
                         brain=self.muse.model, first_run=first_run(),
                         problems=self.problems)
        self.emit_settings()
        greeting = self.greeting()
        self.events.emit("say", text=greeting, source="hello")
        if self.mixer.lost and hasattr(self.voice, "lang"):
            # Nobody heard it: say it once sound is back.
            self.unheard = greeting.replace(" " + SOUND_PROBLEM, "")
        await self.voice.say(greeting)
        first_run(mark=True)
        # Let the hello finish before the first look at the game starts talking.
        await self.voice.wait_idle(20)
        self.watcher.force()
        await asyncio.gather(self.watch(), *tasks)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.window is not None:
            self.window.close()
        self.controls.close()
        self.mixer.close()
        with contextlib.suppress(OSError):
            socket_path().unlink()
        if sys.platform == "win32":
            from .config import STATE_DIR

            with contextlib.suppress(OSError):
                (STATE_DIR / "control").unlink()


async def send(line: str) -> str:
    reader, writer = await asyncio.wait_for(open_control(), 2)
    writer.write((line + "\n").encode())
    await writer.drain()
    # A session that's stuck must not keep a new start waiting forever.
    reply = (await asyncio.wait_for(reader.readline(), 2)).decode().strip()
    writer.close()
    return reply


def already_running() -> bool:
    try:
        return asyncio.run(send("show again")) == "ok"
    except (OSError, ConnectionError, TimeoutError):
        return False


def _plain(exc: BaseException) -> str:
    """An error in a few words a person can hear, not a traceback."""
    text = " ".join(str(exc).split())[:120]
    return text or type(exc).__name__


def say_and_exit(settings: Settings, message: str) -> None:
    """Start-up failed before the session could talk: say why anyway, then stop."""
    print(message, file=sys.stderr)
    try:
        voice = open_voice(settings, Mixer("device"))

        async def speak():
            await asyncio.to_thread(voice.warm_up)
            await voice.say(message)
            await voice.wait_idle(20)

        asyncio.run(speak())
    except Exception:  # noqa: BLE001, S110  no way to speak: the printed line is all we have
        pass
    sys.exit(1)


def main_run(settings: Settings, controls: str | None = None, log: str | None = None) -> None:
    from .config import has_api_key

    if already_running():
        print("Player Two is already running.", file=sys.stderr)
        return
    if not has_api_key():
        say_and_exit(settings, "Hi, I'm Player Two. I need a Meta API key before I can "
                     "help. Open a terminal and run player-two setup, and I'll walk you "
                     "through it.")
    try:
        session = Session(settings, controls=controls)
    except SystemExit as exc:
        say_and_exit(settings, str(exc))
        return
    except Exception as exc:  # noqa: BLE001  whatever it is, the player must hear it
        say_and_exit(settings, f"Sorry, I couldn't start: {_plain(exc)}. Try starting me "
                     "again; if it keeps happening, run player-two setup.")
        return
    if log:
        session.events.log_path = Path(log)
    try:
        asyncio.run(session.run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:  # noqa: BLE001
        session.close()
        say_and_exit(settings, f"Sorry, I've stopped working: {_plain(exc)}. Start me "
                     "again to keep playing.")
    finally:
        session.close()
        print(f"\n{session.muse.usage.calls} model calls, ${session.muse.usage.cost_usd():.3f}",
              file=sys.stderr)
