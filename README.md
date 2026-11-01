# Player Two

**An AI co-op partner for gamers who can't see the screen, or can't easily hold a controller.**

A lot of blind and low-vision players only get to play with a sighted friend on the couch, reading
the screen and pressing buttons for them. Player Two is that friend, there whenever you want to play.

It watches the game, tells you what matters in a warm, natural voice, and when you ask, it plays
for you through a virtual controller plugged in as player two. No mods, and nothing the game
developer has to build.

> Built for the Meta AI Developer Hackathon (Oct 27 to Nov 4, 2026). Everything here was written
> during the event.

## Where it stands

- **In any game** it reads the screen, answers questions, presses buttons when you ask, and walks
  your character to things it can see in 2D games. It does that by looking with Muse Spark, which
  takes a second or two per look.
- **In Chrono Trigger on Steam** it goes much further, and that's the game it's measured on. It
  reads the game's own text, menus and HP from the pixels in about a tenth of a second, plays
  battles, shops and menus with checks on every press, and knows the maps, so "take me to the
  fair" walks straight there. Some of that needs map data you build yourself (see
  [Game data](#game-data)).
- Nobody blind has tried it yet. It has only been tested by me.

Live results on Steam, nights of Oct 31 and Nov 1 (each request sent to the running app as text):

| Request | Passed | Time |
|---|---|---|
| "Where am I?" | 8/8 | 4 to 10 s |
| "Take me to the fair", from the map | 11/11 | 3 to 10 s, once 38 s |
| "Buy three potions" | 3/3 | 28 to 48 s |
| "What are we fighting?" | 9/9 | 3 to 6 s |
| "Attack the one on the left" | 10/11 | 7 to 10 s |
| "Win this fight for me" | 14/14 | 10 to 155 s |
| "Talk to ..." (someone by name or description) | 11/16 | 8 to 107 s |

Player Two starts speaking within 0.1 to 2 seconds of a request; the times are for the whole job.

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
  you pay Meta for what you use. A described screen costs between a quarter of a cent and a cent
  in my tests, depending on the game and the screen size. Setup asks for the key.
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

## Things you can say

In any game:

| Say | What happens |
|---|---|
| anything about the game | Player Two looks and answers, or does it |
| "Player Two" first (hands-free) | the same, with no key; "Player Two" alone, then the request, works too |
| "what's on the screen?" / "describe" / "look around" | a spoken tour of the screen: who and what is around you, and the ways out |
| "where is the door?" | says where it is from your character, with a short sound from that side |
| "take me to the stairs" / "go talk to her" | walks your character there (2D games) |
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

Every request gets an instant spoken reply, in your own words when it's something to do ("Okay,
I'll buy three potions") or "Let me look" for a question, so you know at once if it heard you
wrong.

In Chrono Trigger on Steam, these were measured live (Battle Mode set to Wait):

| Say | What happens | Live |
|---|---|---|
| "where am I?" | the era and the nearest places, from the map: "on the world map in 1000 A.D., just north of the Truce houses, by Leene Square" | 8/8 |
| "take me to the fair" (or any place on the map) | walks there and goes in, through other places when it has to | 11/11 from the map, 4/4 from inside a building |
| "take me back outside" | walks out of the building or town | 13/14 |
| "buy three potions" | walks to the seller, buys, and reads the gold before and after | 3/3 |
| "rest at the inn" | walks to the inn, pays, and reads everyone's HP after | 4/4 since the last fixes |
| "what are we fighting?" | names the enemies | 9/9 |
| "attack the one on the left" | the next fighter attacks that enemy | 10/11 |
| "Crono, use Cyclone" | that fighter uses that tech; says so if they aren't ready yet | 7/7 when ready |
| "heal Marle" (in a fight) | a heal on her, and what happened | 10/12 |
| "win this fight for me" | plays the whole fight, saying each decision | 14/14 |
| "run away" | flees the fight | 6/6 |
| "heal everyone up" (outside a fight) | heal techs until everyone is full | 4/4 |
| "set the battle mode to wait" | changes the game's setting | 8/8 |

## What it does

- **Narrates new screens on its own.** Dialogue read word for word with the speaker's name, what's
  highlighted in a menu or a Yes/No choice, whose turn it is, when your health is low. It stays
  quiet when nothing changed, and silent while it's carrying out what you asked.
- **Answers questions.** "What's on screen?" "Who's here?" "Which enemy is hurt?"
- **Plays when you ask.** "Skip to the battle and hit the left one with Zio." It looks, moves the
  cursor, checks the highlight, confirms, then looks again to make sure it worked. In Chrono
  Trigger it says what happened, including the bad news ("Cyclone went off, but Marle dropped to
  1 HP").
- **Walks you there in 2D games.** "Take me to the man at the stall." Muse Spark points out your
  character and the stall once, SAM 3.1 pins the stall keeper down to the pixel, then code on your
  PC steers there about ten times a second, finds its own way around what's in the way (it
  remembers where it bumped and plans again), and stops next to it, facing it, so the next "talk
  to him" works. It only says it got there when it can see your character there. With a game's
  maps (see [Game data](#game-data)) it knows where every place is and walks straight there.
- **Works hands-free.** For players who can't easily press a key, it can listen all the time and
  act on what starts with its name: "Player Two, open the menu." A small speech detector runs on
  your PC, so only stretches of speech are sent to Meta to be transcribed; what isn't for Player
  Two is then ignored.
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
JRPGs, visual novels, card games, strategy. It can't use a mouse or touch screen yet. Walking in
3D games, like a Persona dungeon, isn't good enough for a blind player yet: it gets lost. It does
not play fast action games for you; there it gives warnings.

## How it uses Meta's models

| Job | Model |
|---|---|
| Understanding your request and planning it with tools, describing the screen, deciding a fight's next move when something surprising happens, finding HUD bars | **Muse Spark 1.3** (vision, tool calling, structured output, streaming) |
| Building a game's data once: reading the place names on world-map images, naming the people and exits in surveyed rooms, turning a walkthrough into a list of steps | **Muse Spark 1.3** |
| Understanding your voice, live, with the game's names as keywords | **Muse Voice Transcribe** (realtime WebSocket) |
| "Where is it?": exact positions of things on screen, spoken and played as directional sound; the exact spot of whoever or whatever a walk is going to | **SAM 3.1** |
| The same narrator and agent on open weights (optional) | **Muse Spark Open** (hosted endpoint) |

**Open weights, if you want them.** By default the narrator and agent use Muse Spark on Meta Model
API, which caches the parts of each request that repeat. They also run on **Muse Spark Open**,
Meta's open-weight model, through a hosted provider (I tested Baseten and a Modal shared endpoint;
other OpenAI-compatible hosts should work the same way). Once provider credentials are set up,
`brain = "auto"` uses it, and `brain = "open"` or `"meta"` picks one. SAM 3.1 and speech
recognition stay on Meta Model API.

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

## What runs on your PC, and why

Someone who can't see the screen can't check what the AI did. A misread HP number or a press on
the wrong menu option is worse than silence, and waiting a couple of seconds for a model on every
button press makes a game unplayable. So the model decides, and code on your PC does the parts
that have to be exact and fast:

- **Reading a game's own text.** For Chrono Trigger on Steam, a reader learned from real frames
  reads its font, cursor, HP and menus straight from the pixels, in about a tenth of a second. On
  163 labelled Steam frames, reading first and asking Muse Spark only about what it couldn't read
  raised the share of what had to be said from 92% to 100% (averaged per frame), cut the first
  word from a median 1.45 s to 0.30 s, and cut the cost about four times.
- **Carrying out commands.** Battles, shops, inns and field menus in Chrono Trigger are played by
  local executors. Muse Spark says what to do once ("Crono: Cyclone on the left one"); the executor
  presses, checks every press against the reader, and reads the result (HP before and after, gold
  before and after). In a fight the game is paused while the model thinks, so the enemies don't
  act while it decides.
- **Walking.** One look by Muse Spark, then image tracking steers several times a second. With a
  game's maps, it finds you on the map by image matching and plans the route itself.
- **The instant echo**, the voice ([Kokoro](https://github.com/hexgrad/kokoro), about 0.12 s a
  sentence on an RTX 4070, and it falls back to the CPU; Meta's API has no text-to-speech model),
  the speech detector for hands-free (Silero VAD) and the health bar watcher.

## Game data

Player Two keeps what it knows about a game in `~/.config/player-two/`. None of it is in this
repo:

- `games/<game>.txt`: notes it writes itself with the "remember" tool (which button opens the
  menu, where a place is). Read at the start of every session.
- `games/<game>.pack.txt`: a game pack, plain text on how the game works (its menus, its tech
  and item names). Written once per game.
- `guides/<game>.txt`: a walkthrough you supply, for "what do I do next?" and "how do I get to
  ...?". Blind players already play with text guides; this reads one so you don't have to.
- `atlas/<game>/`: the game's maps. World maps come from fan-made map images, labelled with Muse
  Spark (`scripts/label_atlas.py`) and checked by a person. Where you can walk on each map, and
  every room and town, is surveyed by walking it in an emulator with the game's memory as ground
  truth (`scripts/survey.py`, `scripts/survey_rooms.py`). The maps contain game art, so they can't
  be shared here.

Without these, Player Two still narrates, answers and plays, and walks to what it can see.
Finding a place it hasn't seen yet is then slow (it tries the ways out one by one), so it helps to
tell it which way when you know ("Gato is west of the bell").

## How it works

1. The screen watcher sees that a new screen has settled.
2. The reader (where the game has one) or Muse Spark decides what's worth saying, and Kokoro (or
   your screen reader) says it.
3. Meanwhile the HUD watcher reads health bars on every frame and plays a sound when they get low.
4. You talk (with the talk key, or by saying "Player Two"). Muse Voice Transcribe turns it into
   text.
5. The agent (Muse Spark with tools) looks at the screen and acts: it presses buttons on a virtual
   Xbox controller and looks again after each step, or hands the job to a walker or an executor
   that checks every step itself, until it's done.

How each part works, including walking, the maps and the executors:
[docs/how-it-works.md](docs/how-it-works.md).

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
- Only tested by me. I'm very nearsighted, not blind, and no blind player has used it yet.
- The Chrono Trigger work was tested on Linux (Hyprland) with the Steam version only. It needs
  Battle Mode set to Wait: in Active mode its commands land too late. Start the game on the
  monitor it will stay on; a game started on one monitor and moved to another can draw at a
  slightly different scale, and then map positions come out wrong.
- In Chrono Trigger, these still go wrong: long trips through forests (2 to 4 minutes, with
  fights on the way), "talk to" someone described by position or someone walking around, walking
  to a save point, and boss fights (it lost all 3 tries against one).
- Finding a place it hasn't seen yet, without maps, is slow, and it can miss a way in (a gap in a
  fence, a chain to climb).
- In 3D games it can't find its way around a dungeon yet. Menus, battles and dialogue work.
- Without a game's reader, it reads the screen through Muse Spark, so it can misread tiny or
  blurry text. It says so when it can't read something instead of guessing.
- On Windows, the virtual controller needs the ViGEmBus driver. Its maintainer has retired it; it
  still installs and works, and Player Two falls back to keyboard input without it.
- Screen reader output is tested with Orca 51 and speech-dispatcher on Linux and NVDA 2026.3 on
  Windows (NVDA's own controller client, downloaded once and checksum verified). JAWS goes through
  accessible_output2 and is not tested yet.
- Hands-free listening is tested only with generated voice clips in an isolated desktop, not with a
  real voice in a real room.

## Development

```bash
uv sync --group dev --extra cuda
uv run --extra cuda pytest        # unit tests (plain uv run drops the CUDA extra)
uv run player-two narrate shot1.png shot2.png --wav out.wav
uv run player-two ask --dry-run --images shot.png -- "who is talking?"
scripts/arena_trials.sh < tests/arena/trials.txt   # scored runs against tests/arena
```

`arena_trials.sh` runs the game scene in an isolated desktop with my own tool for that (omabox),
so it won't run as is elsewhere. The same goes for the emulator benches (`scripts/emu_tasks.py`,
`scripts/rooms_bench.py`), which run the SNES version of Chrono Trigger in RetroArch and judge
each task by reading the game's memory, and for `scripts/play_tasks.py`, which runs tasks on a
real Steam copy.

`tests/arena/index.html` is a small JRPG-style scene (dialogue, menus, a turn-based battle) used to
test the full loop end to end in an isolated desktop.

`player-two ctl do "take me to the fair"` sends a request to the running session as if you'd said
it, and `player-two run --log events.jsonl` keeps everything it heard and said. For demo videos,
`scripts/demo_record.sh` records a monitor with sound and `scripts/demo_captions.py` turns the log
into subtitles.

## License

MIT. Kokoro's model weights are Apache 2.0.
