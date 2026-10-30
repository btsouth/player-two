# How Player Two works

The parts, in the order a moment of play goes through them. File names are under
`src/playertwo/`.

- **Screen watcher** (`screen.py`): samples the game window a few times a second, learns which parts
  are always moving (a flickering TV, rain), and fires once when a new screen has stopped moving.
- **Narrator** (`narrator.py`): streams Muse Spark's answer into the voice sentence by sentence, so
  speech starts before the model has finished.
- **Agent** (`agent.py`): a tool loop with `press`, `hold`, `walk_to`, `talk_to`, `keep_walking`,
  `exits`, `area_map`, `walk_to_map`, `say`, `ask_player`, `remember` and `finish`. A rule in code stops
  any press sequence just before an A that follows any other press, so it always looks at the
  highlight before confirming, and it can't press A twice and skip a line you didn't hear.
- **Walking** (`walk.py`): first, in each new area, it taps the d-pad one way and back and
  watches what starts and stops moving with it: that's your character, not someone else with the
  same red hair, and not a party member catching up, and that went the way the d-pad was held.
  Then one Muse Spark call finds the target and a way around anything in the way. At the same
  time SAM 3.1 looks for the target by name (and as "person", for someone to talk to): Muse
  Spark's boxes are often a sprite off, and the walk stops and faces by them, so when SAM finds
  the same thing its exact box is used. SAM gets no longer than the model took plus a moment, so
  it adds no wait, and without a match the model's box stands. Then plain image math takes over: phase correlation follows the
  scenery as it scrolls, your character is pinned down by what moves differently from the scenery and then
  followed by its colours (which hold when the sprite turns), and the stick and d-pad steer it in
  eight directions. When the character stops moving against something, it marks that spot on
  a grid of character-sized cells fixed to the world (it moves with the scroll), plans a path
  around what it knows is blocked (A* over the four d-pad directions, unknown cells counted as
  open), and follows it, planning again at every bump (`nav.py`). It asks Muse Spark again only
  when no path is left, after many bumps, or when it loses track, and never says it arrived
  while your character is hidden behind something. When the screen changes, at
  once or over a fade through a door, it lets go, waits for the picture to settle, and says
  whether it's a new place.
- **Exploring** (`explore.py`): `keep_walking` holds a direction and watches every frame: how far
  the view scrolled, when the character stopped (a wall), and when the picture changed to a new
  place or a text box came up. It reports that in steps ("blocked after 8 steps up"). The agent
  keeps a rough map of the current area from where each walk ended, and sees it with every
  request, so it tries new exits instead of the same ones. Once it has walked in a game, a long
  d-pad hold is watched the same way. To look for something off screen, `exits` asks Muse Spark
  for the ways out of the place on screen and lists them nearest first, from your character,
  with the way it came in (opposite the way it was walking when the place changed) marked, and
  the agent tries them one at a time.
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
  focus from the game (on another monitor if you have one). Adding `&captions` to its address
  shows a single caption bar for streaming.
- **Finding the game** (`games.py`): the Steam game window first, named from its Steam library;
  otherwise the most recent window that isn't a terminal, browser or app. `window = "..."` picks
  one by name.

## Tests on real game frames

`tests/test_real_frames.py` checks walking, fades and the area map on Chrono Trigger frames
recorded while holding the d-pad, at full and slow capture rates. The recordings are game footage,
so they aren't in the repo; the tests skip without them (`P2_RECORDINGS` points at them).
