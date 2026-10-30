# Player Two

**An AI co-op partner for gamers who can't see the screen, or can't easily hold a controller.**

A lot of blind and low-vision players only get to play with a sighted friend on the couch, reading
the screen and pressing buttons for them. Player Two is that friend, there whenever you want to play.

It watches the game, tells you what matters in a warm, natural voice, and when you ask, it plays
for you through a virtual controller plugged in as player two. No mods, and nothing the game
developer has to build.

> Built for the Meta AI Developer Hackathon (Oct 27 to Nov 4, 2026). Everything here was written
> during the event.

## What you need

- Windows 10 or 11, or Linux.
  - Windows: run the game windowed or borderless. Exclusive fullscreen can capture as a black
    screen.
  - Linux: Hyprland (Omarchy), where it captures just the game window (it uses `grim`), or an X11
    session, where it captures the whole first screen, so keep the game on it. GNOME and KDE on
    Wayland can't be captured yet; Player Two says so if the picture comes back black.
  - Linux also needs `git`, [uv](https://docs.astral.sh/uv/), a C compiler with kernel headers
    (for the virtual controller), PortAudio (`libportaudio2` on Debian and Ubuntu, `portaudio`
    on Arch and Fedora), and for screen reader output, `gdbus`.
- A Meta Model API key, from [Meta's developer site](https://dev.meta.ai), with billing set up:
  you pay Meta for what you use. Each described screen costs about a third of a cent. Setup asks
  for the key.
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

With an NVIDIA card, install `".[cuda]"` in place of `.` to run the voice on it (about 2 GB
more): each line starts about half a second sooner than on the processor.

Setup asks one question at a time: your key, screen reader, language, how much it says,
microphone, controller, hands-free, talk key and app launcher.

Then start your game and run `player-two`, or pick **Player Two** in your app launcher. It finds
the game window by itself (a Steam game is named from its Steam library), says hello, and starts
narrating. Hold **F8** and talk, or with hands-free on, just say "Player Two" and what you need.
Say "help" any time to hear what you can say, and "what's wrong?" if something doesn't work:
it says what it can and can't reach (the game, controller, microphone, talk key) and the last
problem it hit.

Setup is plain questions and answers, so it reads well in a screen reader. It:

- saves your Meta Model API key (mode 600) and checks that it works
- offers your screen reader (Orca, NVDA, JAWS) for speech, and lets you pick language, how much
  Player Two says, and its speaking speed
- checks your microphone and the virtual controller, and offers the one-line udev rule if
  `/dev/uinput` needs it
- offers hands-free listening, for anyone who can't easily press a key
- adds F8 as a hold-to-talk key to your Omarchy or Hyprland config (with a backup). On GNOME it
  adds F8 as a shortcut you press once to start talking and again to stop. On Windows, Player Two
  takes F8 itself, so the game doesn't also get it. Elsewhere, bind `player-two ctl talk` to a
  key in your desktop's shortcut settings (press once to talk, again to stop), or
  `player-two ctl talk-start` and `player-two ctl talk-stop` to a key's press and release. If
  that's hard to set up alone, turn on hands-free instead.
- adds Player Two to your app launcher

**Hands-free works best with a headset or earbuds.** Over speakers, the microphone also hears
Player Two and the game. It still ignores its own voice, and you can talk to it whenever it's
quiet, but words said over its longer lines can get lost. Each stretch of speech, including the
game's, is sent to be transcribed, so speakers also cost a little more.

When something is wrong (no key, no microphone, no controller, no network), Player Two says so out
loud and says how to fix it, instead of printing it where you can't see it.

## What it does

- **Narrates new screens on its own.** Dialogue read word for word with the speaker's name, what's
  highlighted in a menu, whose turn it is, when your health is low. It stays quiet when nothing
  changed.
- **Answers questions.** "What's on screen?" "Who's here?" "Which enemy is hurt?"
- **Plays when you ask.** "Skip to the battle and hit the left one with Zio." It looks, moves the
  cursor, checks the highlight, confirms, then looks again to make sure it worked.
- **Walks you there in 2D games.** "Take me to the man at the stall." Muse Spark points out your
  character and the stall once, SAM 3.1 pins the stall keeper down to the pixel, then code on your
  PC steers there about ten times a second, goes around what's in the way, and stops next to it,
  facing it, so the next "talk to him" works. It only says it got there when it can see your
  character there. In Chrono Trigger, walks to something already on screen took 2 to 8 seconds in
  my sessions; longer trips (downstairs, outside, across the map) took 20 seconds to a minute. For
  places not on screen yet, it walks on in a direction while watching and stops at walls and
  doorways. Everything that scrolls past is stitched into a map of the area, which it can look at
  to find exits and gaps and walk back to.
- **Works hands-free.** For players who can't easily press a key, it can listen all the time and
  act on what starts with its name: "Player Two, open the menu." A small speech detector runs on
  your PC, so only stretches of speech are sent to Meta to be transcribed; what isn't for Player
  Two is then ignored. Saying "next" presses A and reads the next line, with no wait for the
  model.
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

It works best in turn-based and menu-driven games played with a controller or keyboard: RPGs,
JRPGs, visual novels, card games, strategy. It can't use a mouse or touch screen yet. It walks
you around 2D games. Walking in 3D games, like a Persona
dungeon, isn't good enough for a blind player yet: it gets lost. It does not play fast action
games for you; there it gives warnings.

## How it uses Meta's models

| Job | Model |
|---|---|
| Seeing the screen, deciding what to say, planning button presses, finding HUD bars | **Muse Spark 1.3** (vision, tool calling, structured output, streaming) |
| Understanding the player's voice, live, with game names as keywords | **Muse Voice Transcribe** (realtime WebSocket) |
| "Where is it?": exact positions of things on screen, spoken and played as directional sound; the exact spot of whoever or whatever a walk is going to | **SAM 3.1** |
| The same narrator and agent on open weights (optional) | **Muse Spark Open** (hosted endpoint) |

**Open weights, if you want them.** By default the narrator and agent use Muse Spark on Meta Model
API, which caches the parts of each request that repeat. They also run on **Muse Spark Open**,
Meta's open-weight model, through a hosted provider (I tested Baseten and a Modal shared endpoint;
other OpenAI-compatible hosts should work the same way). Once provider credentials are set up, `brain = "auto"` uses it, and
`brain = "open"` or `"meta"` picks one. SAM 3.1 and speech recognition stay on Meta Model API.

In my runs, both brains got 8 of 8 battle commands right on a small test scene (`tests/arena`),
and 20 of 22 Persona 4 Golden narration checks (on my own screenshots, which aren't in the repo,
so that one can't be rerun from it). The difference is speed: on Baseten a narration line took a
median 5.3 seconds, against 1.4 on Meta Model API. That's why Meta is the default for live play.

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

How each part works, including walking and the area map: [docs/how-it-works.md](docs/how-it-works.md).


## Things you can say

| Say | What happens |
|---|---|
| anything about the game | Player Two looks and answers, or does it |
| "Player Two" first (hands-free) | the same, with no key; "Player Two" alone, then the request, works too |
| "take me to the stairs" / "go talk to her" | walks your character there (2D games) |
| "describe" / "look around" | a full spoken tour: who and what is around you, and the ways out |
| "next" | presses A and reads the next line of dialogue, without waiting for the model |
| "repeat" | says the last line again |
| "stop" / "quiet" / "cancel that" | stops talking (and cancels a running command) |
| "help" | lists what you can say |
| "what's wrong?" | says what it can and can't do right now: the game, controller, microphone, talk key, and the last problem |
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
uv run --extra cuda pytest        # unit tests (plain uv run drops the CUDA extra)
uv run player-two narrate shot1.png shot2.png --wav out.wav
uv run player-two ask --dry-run --images shot.png -- "who is talking?"
scripts/arena_trials.sh < tests/arena/trials.txt   # scored runs against tests/arena
```

`arena_trials.sh` runs the game scene in an isolated desktop with my own tool for that (omabox),
so it won't run as is elsewhere.

`tests/arena/index.html` is a small JRPG-style scene (dialogue, menus, a turn-based battle) used to
test the full loop end to end in an isolated desktop.

## License

MIT. Kokoro's model weights are Apache 2.0.
