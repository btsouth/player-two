"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from .config import Settings


def _settings(args) -> Settings:
    s = Settings.load()
    for name in ("game", "verbosity", "voice", "model", "effort", "window", "language",
                 "confirm", "brain", "output", "companion"):
        v = getattr(args, name, None)
        if v:
            setattr(s, name, v)
    return s


async def _narrate(args) -> None:
    from .muse import Muse
    from .narrator import Narrator
    from .screen import FileCapture

    s = _settings(args)
    muse = Muse(s)
    nar = Narrator(muse, s.verbosity, s.game, s.language, s.team_up)
    voice = None
    if args.speak or args.wav:
        from .audio import Mixer
        from .voice import Voice

        voice = Voice(s, Mixer("file" if args.wav else "device"))
    cap = FileCapture([Path(p) for p in args.images])
    for _ in args.images:
        frame = cap.grab()
        t = time.monotonic()
        first = None
        out = ""

        async def deltas():
            nonlocal first, out
            async for d in nar.narrate(frame):
                if first is None:
                    first = time.monotonic() - t
                out += d
                yield d

        if voice:
            await voice.say_stream(deltas())
            await voice.wait_idle()
        else:
            async for _ in deltas():
                pass
        total = time.monotonic() - t
        first_s = f"{first:.2f}s" if first is not None else "silent"
        print(f"[{frame.label}] first text {first_s}, done {total:.2f}s")
        print(f"  {out.strip() or '<silent>'}")
    if voice and args.wav:
        import numpy as np
        import soundfile as sf

        from .audio import RATE

        sf.write(args.wav, np.concatenate(voice.mixer.recorded), RATE)
        print(f"wrote {args.wav}")
    u = muse.usage
    print(f"{u.calls} calls, {u.prompt_tokens} in ({u.cached_tokens} cached), "
          f"{u.completion_tokens} out, ${u.cost_usd():.4f}")


async def _say(args) -> None:
    from .audio import Mixer
    from .screenreader import open_voice

    s = _settings(args)
    voice = open_voice(s, Mixer("file" if args.wav else "device"))
    await voice.say(" ".join(args.text))
    await voice.wait_idle()
    if not voice.first_audio_ms:
        print(f"sent to {voice.voice_name}")
        return
    if args.wav:
        import numpy as np
        import soundfile as sf

        from .audio import RATE

        sf.write(args.wav, np.concatenate(voice.mixer.recorded), RATE)
    print(f"first audio {voice.first_audio_ms[0]:.0f} ms")


async def _ask(args) -> None:
    from .agent import Agent
    from .controls import open_controls
    from .muse import Muse
    from .screen import FileCapture

    s = _settings(args)
    muse = Muse(s)
    controls = open_controls("dry-run" if args.dry_run else s.controls)
    cap = FileCapture([Path(p) for p in args.images])

    async def grab():
        return cap.grab()

    async def speak(text: str) -> None:
        print(f"  says: {text}")

    agent = Agent(muse, controls, grab, speak, game=s.game, confirm=s.confirm,
                  language=s.language)
    run = await agent.run(" ".join(args.request))
    for st in run.steps:
        print(f"  {st.t:5.2f}s {st.tool} {st.args}")
    print(f"outcome: {run.outcome} ({run.seconds:.2f}s, {muse.usage.calls} calls, "
          f"${muse.usage.cost_usd():.4f})")


def _run(args) -> None:
    from .app import main_run

    main_run(_settings(args), controls=args.controls, log=args.log)


async def _ctl(args) -> None:
    # Kept light on purpose: a talk key press should reach the session instantly.
    from .config import open_control

    try:
        # Frame events carry a JPEG, far past asyncio's default 64 KB line limit.
        reader, writer = await open_control(limit=1 << 24)
    except OSError:
        print("Player Two isn't running. Start it with: player-two", file=sys.stderr)
        sys.exit(1)
    if args.command == ["events"]:
        writer.write(b"events\n")
        await writer.drain()
        while line := await reader.readline():
            print(line.decode().rstrip(), flush=True)
        return
    writer.write((" ".join(args.command) + "\n").encode())
    await writer.drain()
    print((await reader.readline()).decode().strip())
    writer.close()


COMMANDS = ("run", "setup", "install-desktop", "narrate", "say", "ask", "ctl")


def main(argv: list[str] | None = None) -> None:
    import truststore

    truststore.inject_into_ssl()
    argv = sys.argv[1:] if argv is None else list(argv)
    # Plain "player-two" (or with run's flags) starts a session.
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["run", *argv]
    p = argparse.ArgumentParser(
        prog="player-two",
        description="Player Two: an AI co-op partner that tells you what's on screen and "
        "plays when you ask. Run with no command to start.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--game")
        sp.add_argument("--verbosity", choices=["terse", "quiet", "normal", "detailed"])
        sp.add_argument("--voice")
        sp.add_argument("--model")
        sp.add_argument("--effort")
        sp.add_argument("--window", help="part of the game window's title or class")
        sp.add_argument("--language", help="language to speak, e.g. Spanish")
        sp.add_argument("--confirm", choices=["always", "important", "never"])
        sp.add_argument("--output", choices=["auto", "voice", "screen-reader"],
                        help="speak with Player Two's voice or through your screen reader")
        sp.add_argument("--brain", choices=["auto", "meta", "open"],
                        help="open: Muse Spark Open; meta: Muse Spark on Meta Model API; "
                        "auto: open when its credentials are set up")

    n = sub.add_parser("narrate", help="narrate screenshots, one after another")
    common(n)
    n.add_argument("images", nargs="+")
    n.add_argument("--speak", action="store_true", help="speak through the speakers")
    n.add_argument("--wav", help="write the spoken narration to a file instead")

    sy = sub.add_parser("say", help="speak a line in the narrator voice")
    common(sy)
    sy.add_argument("text", nargs="+")
    sy.add_argument("--wav")

    a = sub.add_parser("ask", help="run one voice command against screenshots")
    common(a)
    a.add_argument("request", nargs="+")
    a.add_argument("--images", nargs="+", required=True,
                   help="screens the agent sees, in order (the last one repeats)")
    a.add_argument("--dry-run", action="store_true", help="log button presses only")

    sub.add_parser("setup", help="set up your key, voice, microphone, controller and "
                   "talk key, one question at a time")
    sub.add_parser("install-desktop", help="add Player Two to the app launcher")

    r = sub.add_parser("run", help="start a live session (the default)")
    common(r)
    r.add_argument("--controls", choices=["auto", "gamepad", "keyboard", "dry-run"])
    r.add_argument("--log", help="append session events as JSON lines to this file")
    r.add_argument("--companion", choices=["auto", "on", "off"],
                   help="open the Player Two window (auto: on Hyprland, without focus)")

    c = sub.add_parser("ctl", help="send a command to the running session "
                       "(talk-start, talk-stop, talk, repeat, describe, stop, status, events, "
                       "do TEXT)")
    c.add_argument("command", nargs="+")

    args = p.parse_args(argv)
    if args.cmd == "run":
        _run(args)
        return
    if args.cmd == "setup":
        from .onboarding import run_setup

        run_setup()
        return
    if args.cmd == "install-desktop":
        from .desktop import install_desktop_entry

        for path in install_desktop_entry():
            print(f"added {path}")
        return
    try:
        asyncio.run({"narrate": _narrate, "say": _say, "ask": _ask, "ctl": _ctl}[args.cmd](args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
