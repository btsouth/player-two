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
- **Walks you there in 2D games.** "Take me to the man at the stall." Muse Spark points out your
  character and the stall once, then code on your PC steers there about a dozen times a second,
  goes around what's in the way, and stops next to it, facing it, so the next "talk to him" works.
  In Chrono Trigger most walks to something on screen took 2 to 8 seconds in our sessions. For
  places not on screen yet, it walks on in a direction while watching and stops at walls and
  doorways. Everything that scrolls past is stitched into a map of the area, which it can look at
  to find exits and gaps and walk back to.
- **Works hands-free.** For players who can't easily press a key, it can listen all the time and
  act on what starts with its name: "Player Two, open the menu." A small speech detector runs on
  your PC, so only speech is sent to be transcribed, and talk in the room that isn't for Player
  Two is ignored.
- **Checks before anything that matters.** If the request is vague or it can't read the screen, it
  asks instead of guessing: "Garu on Shadow A, the left one. Go?"
- **Warns you in real-time games.** It finds your health and stamina bars once, then reads them on
  every frame with plain pixel math and plays a sound when they get low.
- **Speaks through your screen reader if you like.** With `output = "screen-reader"` its words go
  to Orca (also shown on a braille display) or speech-dispatcher on Linux, or to NVDA or JAWS on
  Windows, at the voice and speed you already use. Sound cues still come from Player Two.
- **Fits how you play.** Terse mode for fast screen reader users, narration in English, Spanish,
  French, Italian, Portuguese or Hindi (it can translate on-screen text into your language), and a
  team-up mode for games that already narrate their own menus.

It works best in turn-based and menu-driven games: RPGs, JRPGs, visual novels, card games,
strategy, point and click. It walks you around 2D games. Walking in 3D games, like a Persona
dungeon, isn't good enough for a blind player yet: it gets lost. It does not play fast action
games for you; there it gives warnings.

## What you need

- Windows 10 or 11, or Linux (tested on Arch with Hyprland, and GNOME).
- A Meta Model API key, from [Meta's developer site](https://dev.meta.ai). Setup asks for it.
- Speakers or a headset. A headset is best for hands-free.
- A graphics card is optional: the voice runs on the processor too.

## Quick start

On Windows, paste this into PowerShell. It installs everything, including the virtual controller
driver (Windows asks for permission once), then starts setup:

```powershell
irm https://raw.githubusercontent.com/btsouth/player-two/main/install.ps1 | iex
```

On Linux, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/btsouth/player-two && cd player-two
uv tool install . --excludes uv-excludes.txt
player-two setup
```

The install keeps the graphics card build of the voice model. Setup asks one question at a time:
your key, voice, microphone, controller, talk key and app launcher.

Then start your game and run `player-two`, or pick **Player Two** in your app launcher. It finds
the game window by itself (a Steam game is named from its Steam library), says hello, and starts
narrating. Hold **F8** and talk, or with hands-free on, just say "Player Two" and what you need.
Say "help" any time to hear what you can say.

Setup is plain questions and answers, so it reads well in a screen reader. It:

- saves your Meta Model API key (mode 600) and checks that it works
- offers your screen reader (Orca, NVDA, JAWS) for speech, and lets you pick language, how much
  Player Two says, and its speaking speed
- checks your microphone and the virtual controller, and offers the one-line udev rule if
  `/dev/uinput` needs it
- offers hands-free listening, for anyone who can't easily press a key
- adds F8 as a hold-to-talk key to your Omarchy or Hyprland config (with a backup). On GNOME it
  adds F8 as a shortcut you press once to start talking and again to stop. On Windows, Player Two
  takes F8 itself, so the game doesn't also get it. Elsewhere, bind `player-two ctl talk-start` to
  a key press and `player-two ctl talk-stop` to its release.
- adds Player Two to your app launcher

**Hands-free works best with a headset or earbuds.** Over speakers, the microphone also hears
Player Two and the game. It still ignores its own voice, and you can talk to it whenever it's
quiet, but words said over its longer lines can get lost. Each stretch of speech, including the
game's, is sent to be transcribed, so speakers also cost a little more.

When something is wrong (no key, no microphone, no controller, no network), Player Two says so out
loud and says how to fix it, instead of printing it where you can't see it.

## How it uses Meta's models

| Job | Model |
|---|---|
| Seeing the screen, deciding what to say, planning button presses, finding HUD bars | **Muse Spark 1.3** (vision, tool calling, structured output, streaming) |
| Understanding the player's voice, live, with game names as keywords | **Muse Voice Transcribe** (realtime WebSocket) |
| "Where is it?": exact positions of things on screen, spoken and played as directional sound | **SAM 3.1** |
| The same narrator and agent on open weights (optional) | **Muse Spark Open** (hosted endpoint) |

**Open weights, if you want them.** By default the narrator and agent use Muse Spark on Meta Model
API, which caches the parts of each request that repeat. They also run on **Muse Spark Open**,
Meta's open-weight model, through a hosted provider (we tested Baseten and a Modal shared endpoint;
other OpenAI-compatible hosts should work the same way). Once provider credentials are set up, `brain = "auto"` uses it, and
`brain = "open"` or `"meta"` picks one. SAM 3.1 and speech recognition stay on Meta Model API.

On our own test scene (`tests/arena`), both brains got 8 of 8 battle commands right, and 20 of 22
on our Persona 4 Golden narration checks (our screenshots, not in the repo). The difference is speed: on Baseten a narration line takes a median 5.3 seconds,
against 1.4 on Meta Model API. That's why Meta is the default for live play.

```toml
brain = "open"
open_base_url = "https://inference.baseten.co/v1"
open_model = "meta-models/muse-spark-open-1026"
# Baseten API key in ~/.config/player-two/open-api-key (or BASETEN_API_KEY).
# For a Modal shared endpoint, use its .modal.direct address and put the proxy token
# in ~/.config/player-two/modal.env as MODAL_PROXY_KEY=wk-... and MODAL_PROXY_SECRET=ws-...
```

Meta's API has no text-to-speech model, so the narrator voice is
[Kokoro](https://github.com/hexgrad/kokoro), a free open-weight voice that runs on your own PC
(about 0.12 s per sentence on an RTX 4070, and it falls back to the CPU).

## How it works

1. The screen watcher sees that a new screen has settled.
2. Muse Spark decides what's worth saying, and Kokoro (or your screen reader) says it.
3. Meanwhile the HUD watcher reads health bars on every frame and plays a sound when they get low.
4. You talk (with the talk key, or by saying "Player Two"). Muse Voice Transcribe turns it into
   text.
5. The agent (Muse Spark with tools) looks at the screen and presses buttons on a virtual Xbox
   controller, looking again after each step, until it's done.

- **Screen watcher** (`screen.py`): samples the game window a few times a second, learns which parts
  are always moving (a flickering TV, rain), and fires once when a new screen has stopped moving.
- **Narrator** (`narrator.py`): streams Muse Spark's answer into the voice sentence by sentence, so
  speech starts before the model has finished.
- **Agent** (`agent.py`): a tool loop with `press`, `hold`, `walk_to`, `talk_to`, `keep_walking`,
  `area_map`, `walk_to_map`, `say`, `ask_player`, `remember` and `finish`. A rule in code stops any press sequence just before a confirm that follows cursor
  movement, so it always looks at the highlight before pressing A.
- **Walking** (`walk.py`): first, in each new area, it taps the d-pad one way and back and
  watches what starts and stops moving with it: that's your character, not someone else with the
  same red hair, and not a party member catching up. Then one Muse Spark call finds the target
  and a way around anything in the way. Then plain image math takes over: phase correlation follows the scenery as
  it scrolls, your character is pinned down by what moves differently from the scenery and then
  followed by its colours (which hold when the sprite turns), and the stick and d-pad steer it in
  eight directions. It asks Muse Spark again only when it's stuck or loses track, and stops when
  the screen fades to a new place.
- **Exploring** (`explore.py`): `keep_walking` holds a direction and watches every frame: how far
  the view scrolled, when the character stopped (a wall), and when the picture changed to a new
  place or a text box came up. It reports that in steps ("blocked after 8 steps up"). The agent
  keeps a rough map of the current area from where each walk ended, and sees it with every
  request, so it tries new exits instead of the same ones. Once it has walked in a game, a long
  d-pad hold is watched the same way.
- **Area map** (`areamap.py`): every frame a walk looks at is laid where the camera was, and each
  spot keeps the colour seen there most often, so people walking about don't smear it. After a
  gap (the model thinking, a button pressed by hand) it finds the screen on the map again, and it
  starts a new map in a new place. The model sees it with what hasn't been seen yet in dark grey,
  points at a spot, and `walk_to_map` walks there even if it's off screen.
- **Game notes** (`notes.py`): what the agent learns about a game (which button opens the menu,
  how battles go, where a place is) is kept in a plain text file a game in
  `~/.config/player-two/games/`, and read at the start of every session. A newer note replaces
  a similar older one, so a correction doesn't leave both.
- **Controller** (`controls.py`): a uinput Xbox 360 pad on Linux, ViGEm on Windows, or keyboard
  keys for PC-only games.
- **Hands-free listening** (`listen.py`): Silero VAD (a 2 MB model on the CPU, well under a
  millisecond per 80 ms of sound) finds stretches of speech, which stream to Muse Voice
  Transcribe as they're spoken. Only what starts with "Player Two" counts, plus an answer to a
  question it just asked. What it hears of its own voice is recognized and ignored, and
  "Player Two, stop" works the moment it's heard.
- **Companion window** (`ui/`): what Player Two sees, says, hears and presses, with big captions
  for low vision, a box to type to it, and settings. It opens in its own app window without taking
  focus from the game (on another monitor if you have one). `?captions` shows a single caption bar
  for streaming.
- **Finding the game** (`games.py`): the Steam game window first, named from its Steam library;
  otherwise the most recent window that isn't a terminal, browser or app. `window = "..."` picks
  one by name.

## Things you can say

| Say | What happens |
|---|---|
| anything about the game | Player Two looks and answers, or does it |
| "Player Two" first (hands-free) | the same, with no key; "Player Two" alone, then the request, works too |
| "take me to the stairs" / "go talk to her" | walks your character there (2D games) |
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
hands_free = false        # listen all the time for "Player Two, ..."; no talk key needed
```

## Safety and limits

- Single-player only. Virtual controllers can trip anti-cheat in online games.
- It's told to ask before saving, loading, quitting or making a story choice, and never to delete
  or overwrite a save without asking. That's an instruction to the model, not a hard block in
  code, so back up saves you care about.
- Walking goes to what it can see, or to a spot on the map of what it has seen. Finding a place
  it hasn't seen yet, in a town or forest, is still hit and miss: it can miss a way in (a gap in
  a fence, a chain to climb), so tell it which way when you know.
- In 3D games it can't find its way around a dungeon yet. Menus, battles and dialogue work.
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
