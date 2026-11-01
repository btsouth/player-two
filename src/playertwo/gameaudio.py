"""Hearing the game: when its own voices speak, so Player Two doesn't talk over them.

A blind player isn't deaf. In a voiced cutscene or dialogue the game's actors say the
lines, so Player Two must neither read their subtitles aloud nor talk over them; like an
audio description track, it fits a few words about what's seen into the gaps.

Player Two listens to the game's own sound stream on PipeWire or PulseAudio (the stream the
game plays into, found by the game window's process), never to what the speakers play, so
its own voice is never in what it hears. The same speech detector as hands-free listening
(Silero VAD, on the CPU) runs on it, and `SpeechTrack` turns its per-frame scores into
stretches of speech: when the game started and stopped talking. Music and sound effects
score low; a voice over music scores high.

Linux with `pactl` and `parec` (PipeWire's pulse server or PulseAudio) only; elsewhere
nothing is heard and narration is as before.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

RATE = 24_000
FRAME_S = 0.08
FRAME_BYTES = int(RATE * FRAME_S) * 2  # 80 ms of 16-bit mono, what SpeechDetector takes
# Quieter than this (RMS of full scale) is silence: no detector run.
SILENCE_RMS = 10 ** (-55 / 20)


class SpeechTrack:
    """Stretches of speech from per-frame speech probabilities.

    Speech starts when `start_frames` frames in a row score at least `on` (the stretch
    starts at the first of them), and goes on while frames score at least `off`; it ends
    `hang_s` after the last such frame, and its end is that frame's time. Times are the
    end of each frame, on the caller's clock (time.monotonic() in a session).
    Tuned on XCOM 2's cinematics (voices over music and effects, timed by their subtitle
    files) and on Chrono Trigger's music: see scripts/game_speech_bench.py.
    """

    def __init__(self, on: float = 0.5, off: float = 0.3, start_frames: int = 3,
                 hang_s: float = 0.5, frame_s: float = FRAME_S, keep_s: float = 120.0,
                 on_start: Callable[[float], None] | None = None):
        self.on, self.off = on, off
        self.start_frames = start_frames
        self.hang_s = hang_s
        self.frame_s = frame_s
        self.keep_s = keep_s
        self.on_start = on_start
        self.run = 0  # frames in a row at or above `on`, while not speaking
        self.lock = threading.Lock()  # pushed from a listening thread, read from the loop
        self.active = False
        self.last_voiced = -1e9  # end of the last frame that kept speech going
        # [start, end]; end is None while it goes on.
        self.segments: deque[list] = deque()
        self.t = -1e9  # the last frame's time

    def push(self, t: float, p: float) -> None:
        with self.lock:
            self._push(t, p)

    def _push(self, t: float, p: float) -> None:
        self.t = t
        if not self.active:
            self.run = self.run + 1 if p >= self.on else 0
            if self.run >= self.start_frames:
                self.active = True
                self.last_voiced = t
                start = t - self.start_frames * self.frame_s
                self.segments.append([start, None])
                self.run = 0
                if self.on_start is not None:
                    self.on_start(start)
        elif p >= self.off:
            self.last_voiced = t
        elif t - self.last_voiced >= self.hang_s:
            self.active = False
            self.segments[-1][1] = self.last_voiced
        while self.segments and self.segments[0][1] is not None \
                and t - self.segments[0][1] > self.keep_s:
            self.segments.popleft()

    def close(self) -> None:
        """The stream ended: speech going on ends at its last voiced frame."""
        with self.lock:
            if self.active:
                self.active = False
                self.segments[-1][1] = self.last_voiced

    def speaking(self, now: float | None = None) -> bool:
        """Speaking now. With `now`, only if frames are still coming in: a stream that
        went quiet without ending (paused, its listener gone) isn't left speaking."""
        with self.lock:
            if now is not None and now - self.t > self.hang_s + 0.5:
                return False
            return self.active

    def quiet_for(self, now: float) -> float:
        """Seconds since the game last spoke (0 while it speaks; very long if never)."""
        with self.lock:
            if self.active:
                return 0.0
            if not self.segments:
                return 1e9
            return max(0.0, now - self.segments[-1][1])

    def heard_between(self, a: float, b: float) -> bool:
        """Whether there was speech at any moment from a to b. Speech still going counts
        up to the last frame heard."""
        with self.lock:
            return any(start <= b and (self.t if end is None else end) >= a
                       for start, end in self.segments)

    def ever(self) -> bool:
        with self.lock:
            return bool(self.segments)


# Finding the game's sound stream

def sink_inputs() -> list[dict]:
    """The sound streams playing now (pactl's JSON), or [] when that can't be had."""
    try:
        out = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"], capture_output=True,
                             text=True, timeout=3, check=False).stdout
        data = json.loads(out or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return data if isinstance(data, list) else []


def stream_pid(stream: dict) -> int | None:
    """The process playing a stream: the server's own record of the client first (it's
    the real one, even from a container), then what the client said."""
    props = stream.get("properties") or {}
    for key in ("pipewire.sec.pid", "application.process.id"):
        with contextlib.suppress(KeyError, TypeError, ValueError):
            return int(props[key])
    return None


def _environ(pid: int) -> dict[str, str]:
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()  # noqa: SIM115
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        k, _, v = item.partition(b"=")
        if k in (b"SteamAppId", b"SteamGameId"):
            out[k.decode()] = v.decode(errors="replace")
    return out


def _parent(pid: int) -> int | None:
    try:
        stat = open(f"/proc/{pid}/stat").read()  # noqa: SIM115
    except OSError:
        return None
    # The process name is in parentheses and may hold spaces or ")".
    fields = stat[stat.rfind(")") + 2:].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def _comm(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return ""


# Program names too common to tie a stream to a game by name alone.
GENERIC = {"python", "python3", "wine", "wine64", "wine-preloader", "wine64-preloader",
           "java", "mono", "steam", "sh", "bash", "pw-play", "paplay", "aplay", "ffplay",
           "chromium", "chrome", "firefox", "brave", "vivaldi", "opera", "msedge",
           "steamwebhelper", "electron"}


def _ancestors(pid: int, parent=_parent, most: int = 32) -> list[int]:
    out = []
    while pid and pid > 1 and len(out) < most:
        pid = parent(pid)
        if not pid:
            break
        out.append(pid)
    return out


def game_streams(streams: list[dict], game_pid: int | None, app_id: str = "",
                 own_pid: int | None = None, environ=_environ, parent=_parent,
                 comm=_comm) -> list[int]:
    """The indexes of the streams the game plays: from the game window's own process or
    one it started, from a program of the same name (a game in a container reports its
    own process ids), or (a Steam game) any process Steam started for that game. Never
    Player Two's own stream, nor anything not tied to the game."""
    own = os.getpid() if own_pid is None else own_pid
    game_name = (comm(game_pid) if game_pid else "").lower()
    out = []
    for s in streams:
        pid = stream_pid(s)
        if pid is None or pid == own or "index" not in s:
            continue
        binary = str((s.get("properties") or {}).get("application.process.binary") or "")
        mine = bool(game_pid) and (pid == game_pid or game_pid in _ancestors(pid, parent))
        mine = mine or bool(game_name and game_name not in GENERIC
                            and binary.lower()[:15] == game_name[:15])
        if mine or (app_id and app_id in (environ(pid).get("SteamAppId"),
                                         environ(pid).get("SteamGameId"))):
            out.append(int(s["index"]))
    return out


STEAM_CLASS = re.compile(r"^steam_app_(\d+)$")


def window_identity(client: dict | None) -> tuple[int | None, str]:
    """(process id, Steam app id) of a Hyprland window, as far as they're known."""
    if not client:
        return None, ""
    pid = client.get("pid")
    m = STEAM_CLASS.match(client.get("class", "") or client.get("initialClass", "") or "")
    return (pid if isinstance(pid, int) and pid > 0 else None), (m.group(1) if m else "")


# Listening

def record_command(index: int) -> list[str]:
    """Records one stream's own sound (what the game plays into it, before the mix)."""
    return ["parec", f"--monitor-stream={index}", "--format=s16le", f"--rate={RATE}",
            "--channels=1", "--latency-msec=40", "--client-name=Player Two (game ears)",
            "--stream-name=game speech"]


class _Stream:
    """One game stream, recorded with parec and scored frame by frame."""

    def __init__(self, index: int, detector, track: SpeechTrack, clock=time.monotonic):
        self.index = index
        self.detector = detector
        self.track = track
        self.clock = clock
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.started = 0.0

    def start(self) -> None:
        self.started = self.clock()
        self.proc = subprocess.Popen(record_command(self.index), stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._run, name=f"game-ears-{self.index}",
                                       daemon=True)
        self.thread.start()

    def _run(self) -> None:
        out = self.proc.stdout
        while True:
            data = out.read(FRAME_BYTES)
            if not data or len(data) < FRAME_BYTES:
                break
            self.feed(data, self.clock())
        self.track.close()  # the stream ended: whatever was being said ended with it

    def feed(self, data: bytes, t: float) -> None:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if float(np.sqrt(np.mean(x * x))) < SILENCE_RMS:
            p = 0.0
            self.detector.reset()
        else:
            p = self.detector(data)
        self.track.push(t, p)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=1)
        self.track.close()


def can_listen() -> bool:
    return sys.platform.startswith("linux") and bool(shutil.which("pactl")) \
        and bool(shutil.which("parec"))


class GameEars:
    """Listens to the game's own sound streams and says when its voices are speaking.

    `target()` gives the game window's (process id, Steam app id); every couple of
    seconds the game's streams are looked up again, since games open and close them (a
    movie, a new scene). `on_speech(start)` is called (from a listening thread) when the
    game starts to speak.
    """

    RESCAN_S = 2.0

    def __init__(self, target: Callable[[], tuple[int | None, str]],
                 on_speech: Callable[[float], None] | None = None,
                 detector_factory: Callable[[], object] | None = None,
                 clock=time.monotonic):
        self.target = target
        self.on_speech = on_speech
        self.detector_factory = detector_factory
        self.clock = clock
        self.streams: dict[int, _Stream] = {}
        self.failed: dict[int, float] = {}  # streams parec couldn't record, until when
        self.tracks: list[SpeechTrack] = []  # every stream's, kept a while after it ends
        self.lock = threading.Lock()  # streams and tracks, read from the event loop
        self.scan_lock = threading.Lock()  # one rescan (or stop) at a time
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.hearing = False  # a game stream is being listened to

    def start(self) -> bool:
        if not can_listen():
            return False
        if self.detector_factory is None:
            from .listen import SpeechDetector, ensure_vad

            path = ensure_vad()
            self.detector_factory = lambda: SpeechDetector(path)
        self.thread = threading.Thread(target=self._run, name="game-ears", daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.stopping.set()
        with self.scan_lock:  # after a rescan in progress, so nothing it starts is left
            with self.lock:
                streams = list(self.streams.values())
                self.streams.clear()
            for s in streams:
                s.stop()

    def _run(self) -> None:
        while not self.stopping.is_set():
            with contextlib.suppress(Exception):
                self._rescan()
            self.stopping.wait(self.RESCAN_S)

    def _rescan(self) -> None:
        with self.scan_lock:
            if self.stopping.is_set():
                return
            pid, app = self.target()
            want = set(game_streams(sink_inputs(), pid, app)) if (pid or app) else set()
            now = self.clock()
            with self.lock:
                gone = [self.streams.pop(i) for i in list(self.streams)
                        if i not in want or not self.streams[i].alive()]
                new = [i for i in want - set(self.streams) if self.failed.get(i, 0.0) <= now]
            # Starting and stopping recorders (and loading a detector) happen outside the
            # lock the event loop reads through, so it never waits on them.
            for s in gone:
                if not s.alive() and now - s.started < 3.0:
                    self.failed[s.index] = now + 30.0  # can't record it: not every scan
                s.stop()
            started = []
            for idx in new:
                s = _Stream(idx, self.detector_factory(), SpeechTrack(on_start=self._started),
                            self.clock)
                s.start()
                started.append(s)
            now = self.clock()
            with self.lock:
                for s in started:
                    self.streams[s.index] = s
                    self.tracks.append(s.track)
                live = {id(s.track) for s in self.streams.values()}
                self.tracks = [t for t in self.tracks if id(t) in live or now - t.t < 120.0]
                self.hearing = bool(self.streams)

    def _started(self, start: float) -> None:
        if self.on_speech is not None:
            with contextlib.suppress(Exception):
                self.on_speech(start)

    # What was heard (any of the game's streams)

    def speaking(self) -> bool:
        now = self.clock()
        with self.lock:
            return any(t.speaking(now) for t in self.tracks)

    def quiet_for(self) -> float:
        now = self.clock()
        with self.lock:
            return min((t.quiet_for(now) for t in self.tracks), default=1e9)

    def heard_between(self, a: float, b: float) -> bool:
        with self.lock:
            return any(t.heard_between(a, b) for t in self.tracks)

    def ever(self) -> bool:
        with self.lock:
            return any(t.ever() for t in self.tracks)


# Whether a game's lines are spoken

VOICED_HINT = re.compile(r"(?im)^\s*[-*]?\s*voiced\s*:\s*(all|some|none|yes|no)\b")


# Games known to have no voice acting at all, when no pack says otherwise: their text is
# always read, and nothing is listened to.
UNVOICED = ("chrono trigger",)


def known_voicing(game: str) -> str:
    name = re.sub(r"\s+", " ", (game or "").lower()).strip()
    return "none" if any(name.startswith(g) for g in UNVOICED) else ""


def voiced_hint(pack: str) -> str:
    """A game pack's word on its voice acting: "all" (every dialogue line and subtitle is
    spoken), "none" (nothing is), or "" (some lines, or it doesn't say: listen)."""
    m = VOICED_HINT.search(pack or "")
    if not m:
        return ""
    word = m.group(1).lower()
    return {"yes": "all", "no": "none", "some": ""}.get(word, word)
