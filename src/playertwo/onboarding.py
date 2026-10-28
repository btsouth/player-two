"""player-two setup: a plain question-and-answer setup that reads well in a screen reader.

Every question is one line with the default in brackets; Enter keeps it. Choices are
numbered, and the number or the word both work. Nothing here draws on the screen.
"""

from __future__ import annotations

import getpass
import os
import sys

from .config import CONFIG_DIR, Settings, has_api_key, has_open_credentials, save_settings

UINPUT_RULE = 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"'
UINPUT_RULE_FILE = "/etc/udev/rules.d/60-player-two-uinput.rules"
VIGEM_URL = "https://github.com/nefarius/ViGEmBus/releases"

VERBOSITY_HELP = {
    "terse": "bare facts, for fast screen reader users",
    "quiet": "only dialogue and choices",
    "normal": "what's new and matters, in a few words",
    "detailed": "a little more about each scene",
}
CONFIRM_HELP = {
    "always": "check with you before every press that changes the game",
    "important": "check before anything that matters and is hard to undo",
    "never": "just do it (still asks before deleting saves or quitting)",
}


def say(text: str = "") -> None:
    print(text, flush=True)


def ask(question: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{hint} ").strip()
    except EOFError:
        answer = ""
    return answer or default


def ask_yes(question: str, default: bool = True) -> bool:
    while True:
        a = ask(question, "yes" if default else "no").lower()
        if a in ("y", "yes", "yeah", "sure", "ok"):
            return True
        if a in ("n", "no", "nope"):
            return False
        say("Please answer yes or no.")


def ask_choice(question: str, options: list[str], default: str,
               notes: dict[str, str] | None = None) -> str:
    say(question)
    for i, o in enumerate(options, 1):
        note = f": {notes[o]}" if notes and o in notes else ""
        say(f"  {i}. {o}{note}")
    while True:
        a = ask("Your choice:", default)
        if a.isdigit() and 1 <= int(a) <= len(options):
            return options[int(a) - 1]
        match = [o for o in options if o.lower() == a.lower()]
        if match:
            return match[0]
        say(f"Type a number from 1 to {len(options)}.")


def write_secret(name: str, value: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        if hasattr(os, "fchmod"):
            os.fchmod(f.fileno(), 0o600)  # an older file may have been readable
        f.write(value.strip() + "\n")


def check_key(s: Settings) -> str | None:
    """None when the saved Meta key works, else what went wrong."""
    import asyncio

    import openai

    from .config import api_key

    async def probe():
        client = openai.AsyncOpenAI(base_url=s.base_url, api_key=api_key(), max_retries=1,
                                    timeout=20)
        await client.chat.completions.create(
            model=s.model, max_completion_tokens=8,
            messages=[{"role": "user", "content": "Say OK."}])

    try:
        asyncio.run(probe())
    except openai.AuthenticationError:
        return "Meta turned that key down."
    except openai.APIConnectionError:
        return "I couldn't reach Meta Model API. Check the internet connection."
    except openai.APIStatusError as exc:
        return f"Meta Model API answered with an error ({exc.status_code})."
    return None


def step_key(s: Settings) -> None:
    say("\nStep 1 of 6: your Meta Model API key.")
    if has_api_key():
        if not ask_yes("A key is already saved. Keep it?"):
            _enter_key()
    else:
        say("Player Two uses Meta's models to see the screen and hear you. Paste your key "
            "from the Meta Model API dashboard. It won't be shown as you type.")
        _enter_key()
    if not has_api_key():
        say("No key saved. Run player-two setup again when you have one.")
        return
    env = next((n for n in ("PLAYER_TWO_API_KEY", "MODEL_API_KEY", "META_API_KEY")
                if os.environ.get(n)), None)
    if env:
        say(f"Note: {env} is set in your environment, and it wins over the saved key. "
            "That's the one I'll check.")
    for _ in range(3):
        say("Checking the key...")
        problem = check_key(s)
        if problem is None:
            say("The key works.")
            break
        say(problem)
        if env or "turned that key down" not in problem:
            say("You can keep going with setup and check the key later.")
            break
        say("Paste the key again, or press Enter to keep going without changing it.")
        if not _enter_key():
            break
    if s.brain != "meta" and has_open_credentials(s.open_base_url):
        say("Your Muse Spark Open provider is set up too, so I'll think with the open model.")


def _enter_key() -> bool:
    # Piped in, read it plainly: Windows' hidden prompt reads only the keyboard.
    key = (getpass.getpass("Key: ") if sys.stdin.isatty() else sys.stdin.readline()).strip()
    if key:
        write_secret("api-key", key)
        say("Saved.")
    return bool(key)


def step_speech(s: Settings, changes: dict) -> None:
    from .screenreader import find_backend

    say("\nStep 2 of 6: how I talk.")
    reader = find_backend()
    if reader is not None and reader.name == "speech-dispatcher":
        reader = None  # installed isn't the same as a screen reader in use
    output = s.output
    if reader is not None:
        use = ask_yes(f"{reader.name} is running. Should I speak through {reader.name}, at "
                      "your own voice and speed, instead of my voice? (Whenever it isn't "
                      "running, I'll use my voice.)", s.output != "voice")
        output = "auto" if use else "voice"
    elif s.output == "screen-reader":
        say("I don't see a screen reader running, so I'll use my own voice for now.")
        output = "auto"
    changes["output"] = output
    s.output = output
    languages = ["English", "British English", "Spanish", "French", "Italian",
                 "Portuguese", "Hindi"]
    lang = ask_choice("Which language should I speak? I translate on-screen text into it.",
                      languages, s.language if s.language in languages else "English")
    changes["language"] = s.language = lang
    verbosity = ask_choice("How much should I say?", list(VERBOSITY_HELP), s.verbosity,
                           VERBOSITY_HELP)
    changes["verbosity"] = s.verbosity = verbosity
    if reader is None or output == "voice":
        from .voice import MODEL_FILES, model_dir

        if not all((model_dir() / f).exists() for f in MODEL_FILES):
            say("Getting my voice ready (a one-time download of about 350 MB)...")
    test_voice(s, changes)


def test_voice(s: Settings, changes: dict) -> None:
    import asyncio

    from .audio import Mixer
    from .screenreader import open_voice

    try:
        voice = open_voice(s, Mixer("device"))

        async def speak():
            await asyncio.to_thread(voice.warm_up)
            await voice.say("Hi, I'm Player Two. This is how I sound.")
            await voice.wait_idle(15)

        asyncio.run(speak())
    except Exception as exc:  # noqa: BLE001
        say(f"I couldn't play sound: {exc}. Check that a speaker or headset is connected "
            "and set as the default output.")
        return
    if not ask_yes("Did you hear me?"):
        say("Check that your speakers or headset are the default sound output, then run "
            "player-two say hello to try again.")
    elif voice.device != "screen reader":  # the screen reader keeps its own speed
        rate = ask_choice("Speaking speed?", ["slower", "same", "faster"], "same")
        if rate != "same":
            s.speech_rate = round(s.speech_rate + (0.15 if rate == "faster" else -0.15), 2)
            changes["speech_rate"] = s.speech_rate


def step_mic() -> None:
    say("\nStep 3 of 6: your microphone.")
    try:
        import numpy as np
        import sounddevice as sd

        from .audio import tone
        from .listen import RATE

        ask("Press Enter, then say a few words after the beep. I'll listen for three "
            "seconds, until the second beep.", "")
        sd.play(tone(880, 120), RATE, blocking=True)
        rec = sd.rec(int(3 * RATE), samplerate=RATE, channels=1, dtype="int16")
        sd.wait()
        sd.play(tone(660, 160), RATE, blocking=True)
    except Exception as exc:  # noqa: BLE001
        say(f"I couldn't open a microphone ({exc}). You can still type to me in the Player "
            "Two window.")
        return
    peak = int(np.abs(rec).max())
    if peak < 300:
        say("That was silent. Check that your microphone is the default input and isn't "
            "muted.")
    elif peak < 1500:
        say("I heard you, but quietly. Speak up or move the microphone closer.")
    else:
        say("I heard you clearly.")


def step_controller() -> None:
    say("\nStep 4 of 6: the virtual controller I play with.")
    from .controls import open_controls

    try:
        pad = open_controls("gamepad")
        pad.close()
        say("The virtual controller works.")
        return
    except Exception as exc:  # noqa: BLE001
        problem = exc
    if sys.platform == "win32":
        say("The virtual controller needs the ViGEmBus driver. Run the Player Two "
            "installer again to add it, or get it from "
            f"{VIGEM_URL}. Until then I'll press keyboard keys.")
        return
    say(f"I can't create a virtual controller yet ({problem}). A udev rule gives you "
        "access to /dev/uinput, the same one Steam uses for its controller support:")
    say(f"  {UINPUT_RULE}")
    if ask_yes(f"Add it to {UINPUT_RULE_FILE} now with sudo? You'll be asked for your "
               "password.", True):
        import subprocess

        cmd = (f"echo '{UINPUT_RULE}' > {UINPUT_RULE_FILE} && udevadm control --reload "
               "&& udevadm trigger --name-match=uinput && modprobe uinput")
        r = subprocess.run(["sudo", "sh", "-c", cmd], check=False)
        if r.returncode == 0:
            say("Done. You may need to log out and back in once for it to apply.")
        else:
            say("That didn't work. Until it's fixed I'll press keyboard keys instead.")


def step_talk_key(changes: dict) -> None:
    from .desktop import (
        TALK_KEY,
        gnome,
        install_gnome_talk_key,
        install_talk_key,
        player_two_cmd,
        talk_key_bound,
    )

    say("\nStep 5 of 6: the talk key.")
    if sys.platform == "win32":
        say(f"Hold {TALK_KEY} to talk to me, in any window, while I'm running.")
        return
    if talk_key_bound():
        say(f"{TALK_KEY} is already your talk key.")
        return
    hyprland = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or os.path.exists(
        os.path.expanduser("~/.config/hypr"))
    if hyprland:
        if ask_yes(f"Make {TALK_KEY} your hold-to-talk key? I'll add it to your Hyprland "
                   "config and keep a backup."):
            say(install_talk_key())
        return
    if gnome():
        if ask_yes(f"Make {TALK_KEY} your talk key? Press it once to talk and again when "
                   "you're done. I'll add it to your GNOME keyboard shortcuts."):
            say(install_gnome_talk_key())
            changes["talk_toggle"] = True
        return
    say(f"In your desktop's keyboard shortcut settings, bind {TALK_KEY} to run "
        f"'{player_two_cmd()} ctl talk'. Press it once to talk and again when you're "
        "done.")
    if ask_yes("Is that set up now?", False):
        changes["talk_toggle"] = True
        changes["talk_key_set"] = True
    else:
        say("Okay. Until then, type to me in the Player Two window. Say 'show the "
            "window', or run player-two ctl show.")


def step_app(s: Settings, changes: dict) -> None:
    from .desktop import install_desktop_entry

    say("\nStep 6 of 6: the app.")
    # Only Hyprland can open the window without taking focus from the game; elsewhere
    # it stays closed unless asked for.
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        show = ask_yes("Open the Player Two window when I start? It shows big captions, "
                       "what I see, and what I press, without taking focus from the game.",
                       s.companion != "off")
        changes["companion"] = "auto" if show else "off"
    else:
        say("There's also a Player Two window with big captions and settings. Say 'show "
            "the window' any time to open it.")
    if ask_yes("Add Player Two to your app launcher?"):
        try:
            for p in install_desktop_entry():
                say(f"Added {p}")
        except Exception as exc:  # noqa: BLE001  setup still finishes
            say(f"I couldn't add it ({exc}). You can start me with: player-two")


def run_setup() -> None:
    s = Settings.load()
    changes: dict = {}
    say("Player Two setup. Press Enter to keep the answer in brackets. Ctrl+C stops at any "
        "time.")
    steps = [lambda: step_key(s), lambda: step_speech(s, changes), step_mic,
             step_controller, lambda: step_talk_key(changes), lambda: step_app(s, changes)]
    try:
        for step in steps:
            step()
            if changes:  # keep each answer even if a later step fails
                save_settings(changes)
    except KeyboardInterrupt:
        say("\nStopped. The answers you gave are saved.")
        if changes:
            save_settings(changes)
        sys.exit(130)
    how = "Press F8 to talk, and again when you're done" if (
        changes.get("talk_toggle") or s.talk_toggle) else "Hold F8 and talk"
    say("\nAll set. Start your game, then run player-two, or pick Player Two in your app "
        f"launcher. {how}. Say help to hear what else you can say.")
