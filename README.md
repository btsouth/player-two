# Player Two

**An AI co-op partner for gamers who can't see the screen, or can't easily hold a controller.**

A lot of blind and low-vision players only get to play with a sighted friend on the couch, reading
the screen and pressing buttons for them. Player Two is that friend, there whenever you want to play.

It watches the game, tells you what matters in a warm, natural voice, and when you ask, it plays
for you through a virtual controller plugged in as player two. No mods, and nothing the game
developer has to build.

> Built for the Meta AI Developer Hackathon (Oct 27 to Nov 4, 2026). Everything here was written
> during the event.

## What it does

- **Narrates new screens on its own.** Dialogue read word for word with the speaker's name, what's
  highlighted in a menu, whose turn it is, when your health is low. It stays quiet when nothing
  changed.
- **Answers questions.** "What's on screen?" "Who's here?" "Which enemy is hurt?"
- **Plays when you ask.** "Skip to the battle and hit the left one with Zio." It looks, moves the
  cursor, checks the highlight, confirms, then looks again to make sure it worked.
- **Checks before anything that matters.** If the request is vague or it can't read the screen, it
  asks instead of guessing: "Garu on Shadow A, the left one. Go?"
- **Warns you in real-time games.** It finds your health and stamina bars once, then reads them on
  every frame with plain pixel math and plays a sound when they get low.
- **Speaks through your screen reader if you like.** With `output = "screen-reader"` its words go
  to Orca (also shown on a braille display) or speech-dispatcher on Linux, or to NVDA or JAWS on
  Windows, at the voice and speed you already use. Sound cues still come from Player Two.
- **Fits how you play.** Terse mode for fast screen reader users, narration in English, Spanish,
  French, Italian, Portuguese or Hindi (it translates on-screen text, so a Japanese-only game can
  be played in English), and a team-up mode for games that already narrate their own menus.

It works best in turn-based and menu-driven games: RPGs, JRPGs, visual novels, card games,
strategy, point and click. It does not play fast action games for you; there it gives warnings.

## How it uses Meta's models

| Job | Model |
|---|---|
| Seeing the screen, deciding what to say, planning button presses, finding HUD bars | **Muse Spark 1.3** (vision, tool calling, structured output, streaming) |
| Understanding the player's voice, live, with game names as keywords | **Muse Voice Transcribe** (realtime WebSocket) |
| "Where is it?": exact positions of things on screen, spoken and played as directional sound | **SAM 3.1** |
| The same narrator and agent on open weights (optional) | **Muse Spark Open** (hosted endpoint) |

**Open weights, if you want them.** By default the narrator and agent use Muse Spark on Meta Model
API, which caches the parts of each request that repeat. They also run on **Muse Spark Open**,
Meta's open-weight model, through a hosted provider (we tested a Modal shared endpoint; Baseten and
Fireworks work the same way). Once provider credentials are set up, `brain = "auto"` uses it, and
`brain = "open"` or `"meta"` picks one. SAM 3.1 and speech recognition stay on Meta Model API. On
our scored battle trials both brains get 8 of 8 right. Hosted open-model endpoints bill every
token, so they cost more for the same play.

```toml
brain = "open"
open_base_url = "https://YOUR-WORKSPACE--ep-NAME-server.us-west.modal.direct/v1"
open_model = "meta-models/Muse-Spark-Open-816B-A42B"
# Modal proxy token in ~/.config/player-two/modal.env:
#   MODAL_PROXY_KEY=wk-...
#   MODAL_PROXY_SECRET=ws-...
```

Meta's API has no text-to-speech model, so the narrator voice is
[Kokoro](https://github.com/hexgrad/kokoro), a free open-weight voice that runs on your own PC
(about 0.12 s per sentence on an RTX 4070, and it falls back to the CPU).

## How it works

```
 game window ──► screen watcher ──► "a new screen settled" ──► Muse Spark ──► voice (Kokoro)
                      │                                                         ▲
                      └─► HUD watcher (pixels, every frame) ──► warning sounds ─┤
 talk key ──► mic ──► Muse Voice Transcribe ──► agent (Muse Spark + tools) ─────┘
                                                   │
                                                   └─► virtual Xbox controller ──► game
```

- **Screen watcher** (`screen.py`): samples the game window a few times a second, learns which parts
  are always moving (a flickering TV, rain), and fires once when a new screen has stopped moving.
- **Narrator** (`narrator.py`): streams Muse Spark's answer into the voice sentence by sentence, so
  speech starts before the model has finished.
- **Agent** (`agent.py`): a tool loop with `press`, `hold`, `say`, `ask_player` and `finish`. A rule
  in code stops any press sequence just before a confirm that follows cursor movement, so it always
  looks at the highlight before pressing A.
- **Controller** (`controls.py`): a uinput Xbox 360 pad on Linux, ViGEm on Windows, or keyboard
  keys for PC-only games.
- **Companion window** (`ui/`): what Player Two sees, says, hears and presses, with big captions
  for low vision, a box to type to it, and settings. It opens in its own app window without taking
  focus from the game (on another monitor if you have one). `?captions` shows a single caption bar
  for streaming.
- **Finding the game** (`games.py`): the Steam game window first, named from its Steam library;
  otherwise the most recent window that isn't a terminal, browser or app. `window = "..."` picks
  one by name.

## Quick start

On Windows, paste this into PowerShell. It installs everything, including the virtual controller
driver (Windows asks for permission once), then starts setup:

```powershell
irm https://raw.githubusercontent.com/btsouth/player-two/main/install.ps1 | iex
```

On Linux, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/btsouth/player-two && cd player-two
uv tool install . --excludes uv-excludes.txt   # keeps the GPU build of the voice model
player-two setup         # key, voice, microphone, controller, talk key, launcher: one question at a time
```

Then start your game and run `player-two`, or pick **Player Two** in your app launcher. It finds
the game window by itself (a Steam game is named from its Steam library), says hello, and starts
narrating. Hold **F8** and talk. Say "help" any time to hear what you can say.

Setup is plain questions and answers, so it reads well in a screen reader. It:

- saves your Meta Model API key (mode 600) and checks that it works
- offers your screen reader (Orca, NVDA, JAWS) for speech, and lets you pick language, how much
  Player Two says, and its speaking speed
- checks your microphone and the virtual controller, and offers the one-line udev rule if
  `/dev/uinput` needs it
- adds F8 as a hold-to-talk key to your Omarchy or Hyprland config (with a backup). On GNOME it
  adds F8 as a shortcut you press once to start talking and again to stop. On Windows, Player Two
  takes F8 itself, so the game doesn't also get it. Elsewhere, bind `player-two ctl talk-start` to
  a key press and `player-two ctl talk-stop` to its release.
- adds Player Two to your app launcher

When something is wrong (no key, no microphone, no controller, no network), Player Two says so out
loud and says how to fix it, instead of printing it where you can't see it.

## Things you can say

| Say | What happens |
|---|---|
| anything about the game | Player Two looks and answers, or does it |
| "describe" / "what's on screen" | a full spoken tour of the screen |
| "repeat" | says the last line again |
| "stop" / "quiet" / "cancel that" | stops talking (and cancels a running command) |
| "help" | lists what you can say |
| "slower" / "faster" | changes its speaking speed |
| "terse", "less detail", "more detail", "back to normal" | how much it says |
| "the game reads the menus" / "team up" | leaves menus to the game's own narrator |
| "watch my health" | starts health and stamina warnings |
| "pause narration" / "resume narration" | stops and restarts narration ("pause" alone pauses the game) |
| "show the window" | opens the companion window, where you can also type |
| "goodbye" | Player Two says bye and quits |

## Settings

The settings button in the companion window changes the common ones while you play and keeps
them. All of them live in `~/.config/player-two/config.toml`, or as flags on `player-two`:

```toml
voice = "af_heart"        # Kokoro voice
speech_rate = 1.1
verbosity = "normal"      # terse, quiet, normal, detailed
language = "English"      # or British English, Spanish, French, Italian, Portuguese, Hindi
output = "auto"           # your screen reader when one is running, else the voice; or "voice", "screen-reader"
confirm = "important"     # always, important, never
team_up = false
window = ""               # part of the game window's title; empty means find the game
controls = "auto"         # gamepad, keyboard
companion = "auto"        # the window: auto (on Hyprland), on, off
```

## Safety and limits

- Single-player only. Virtual controllers can trip anti-cheat in online games.
- It never deletes or overwrites saves, quits, or makes story choices without asking.
- It reads the screen, not game memory, so it can misread tiny or blurry text. It says so when it
  can't read something instead of guessing.
- On Windows, the virtual controller needs the ViGEmBus driver. Its maintainer has retired it; it
  still installs and works, and Player Two falls back to keyboard input without it.
- Screen reader output is tested with Orca 51 and speech-dispatcher on Linux and NVDA 2026.3 on
  Windows (NVDA's own controller client, downloaded once and checksum verified). JAWS goes through
  accessible_output2 and is not tested yet.
- Each described screen costs about a third of a cent in Muse Spark usage.

## Development

```bash
uv sync --group dev --extra cuda
uv run pytest                     # unit tests
uv run player-two narrate shot1.png shot2.png --wav out.wav
uv run player-two ask --dry-run --images shot.png -- "who is talking?"
scripts/arena_trials.sh < tests/arena/trials.txt   # scored runs against tests/arena
```

`tests/arena/index.html` is a small JRPG-style scene (dialogue, menus, a turn-based battle) used to
test the full loop end to end in an isolated desktop.

## License

MIT. Kokoro's model weights are Apache 2.0.
