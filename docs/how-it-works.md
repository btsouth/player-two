# How Player Two works

The parts, in the order a moment of play goes through them. File names are under
`src/playertwo/`.

Runtime Steam play reads screen pixels and sends input. Map/room preparation is a separate,
emulator-memory-assisted process: RAM supplies coordinates, camera position, collision and
exit ground truth during surveys and benchmarks. Muse labels the resulting pictures; a human
checks those labels. That preparation is not inferred from live pixels alone. The reference
pictures, room/player templates and atlas configuration used in the demo are not distributed,
so the demonstrated configuration cannot currently be reproduced from the repo alone.
See [game data](game-data.md).

- **Screen watcher** (`screen.py`): samples the game window a few times a second, learns which parts
  are always moving (a flickering TV, rain), and fires once when a new screen has stopped moving.
- **Reader** (`glyphread.py`, with a game profile in `ctsteam.py` and `ctscreen.py`): reads a
  game's own pixel font, cursor and HP bars with no model, in about a tenth of a second. Every
  frame is scaled to one canvas, text pixels are found as bright cores inside the font's dark
  outline, cut into glyphs and named by their nearest match in a library learned from real frames
  whose text was labelled by hand (`scripts/ct_glyph_learn.py`; a line is only learned when its
  glyph count matches the label). The cursor is found by template matching. A run of text is only
  trusted when it reads as words, to reduce false reads from busy backgrounds. `ctscreen`
  turns frames into what a screen reader would say: whose turn it is, what the cursor is on, a
  Yes/No choice with the highlighted option, dialogue once it has stopped typing out, who is in
  danger, each said once when it changes. Chrono Trigger on Steam is the only profile so far.
- **Narrator** (`narrator.py`): says what the reader read. Screens the reader can't make sense of,
  and every screen in games without a profile, go to Muse Spark, whose answer streams into the
  voice sentence by sentence, so speech starts before the model has finished. It suppresses automatic narration
  while the agent is acting.
- **Hearing the game** (`gameaudio.py`): a blind player can still hear, so Player Two aims not to
  talk over the game's own voices. It listens to the game's own sound stream on PipeWire or
  PulseAudio (found from the game window's process, so Player Two's own voice is never in
  it), and the speech detector used for hands-free listening (Silero VAD) tells voices from
  music and effects, frame by frame. Narration waits for a gap in the game's speech, and a
  sentence already playing stops within about a quarter second of a voice starting and
  starts over once it ends (or is dropped after a few seconds, the moment gone). Subtitles of
  lines the game is saying aloud aren't read; in a voiced cutscene the narrator adds only
  what's seen, in a short sentence, like an audio description track. Text no one speaks is
  read as before. A game pack's `Voiced: all` or `Voiced: none` line settles it for games
  that always or never speak; otherwise each screen is judged by what's heard. Measured with
  `scripts/game_speech_bench.py` on game audio (XCOM 2's cinematic music and effects,
  Chrono Trigger's music, and lines laid over them at known times). Linux only for now;
  `hear_game = false` turns it off.
- **Places** (`narrator.py`, `rooms.py`): "what does it look like?" gets a few sentences of
  setting, light, colours and mood from Muse Spark, after the place's name from the room survey.
  The first time you walk into a surveyed place in a session, its name comes with one line on
  what it's like, drafted once from the survey's picture of it (`scripts/atmos_rooms.py`).
  Never in terse or quiet mode, and never over a text box, a menu or a fight.
- **Voice** (`voice.py`, `audio.py`, `screenreader.py`): Kokoro speaks each sentence as it
  arrives (about 0.1 s a sentence on a GPU, and it runs on the CPU too). With
  `output = "screen-reader"` the words go to Orca, speech-dispatcher, NVDA or JAWS instead.
  Sound cues (low health, the "where is it?" pings) always come from Player Two.
- **Instant echo** (`ack.py`): says your request back in your own words the moment it arrives,
  with no model call ("Okay, Crono will use Cyclone"), so you can notice a misheard request. It
  states what it will try, never a result.
- **Fast path** (`fast.py`): supported Chrono Trigger commands such as buying, attacking and
  taking a fight are parsed locally and dispatched to the same executors without a planning
  model call. Unmatched requests continue to the agent. Unit coverage does not establish live timing.
- **Agent** (`agent.py`): Muse Spark in a tool loop. The tools are `press`, `hold`, `walk_to`,
  `talk_to`, `keep_walking`, `travel`, `exits`, `area_map`, `walk_to_map`, `wait`, `say`,
  `ask_player`, `remember` and `finish`, plus `where_am_i`, `go_to_place`, `go_to_exit` and
  `go_to_era`, which use a game's maps, and `guide` and `next_goal`, which use a walkthrough.
  Those are always offered; without maps or a walkthrough they say so, and the agent falls
  back on the general tools. In Chrono Trigger on Steam it also gets the `shop` and
  `field_menu` executors below, and in a fight only `battle`, `fight`, `say`, `ask_player`,
  `remember`, `wait` and `finish`. A rule in code stops any press sequence just
  before an A that follows any other press, so it always looks at the highlight before
  confirming, reducing the risk of confirming an unseen choice.
- **Executors** (`battle.py`, `shop.py`, `menu.py`, Chrono Trigger on Steam): the model says what
  to do once, and local code does it. `battle` takes one command ("Crono, Cyclone on the one on the
  left"): it unpauses, picks the fighter's command window, the command, the list entry and the
  target, checks each press against the reader, confirms, watches it play out and pauses again,
  then reports what changed (HP and MP before and after, the message bar). Anything unexpected
  (the hand gone, the entry not in the list, the fighter knocked out) stops it, paused, with the
  reason in words. `fight` with "win" loops over the same steps with local attack/heal/revive
  rules. `battle.Tuning.model_decides` defaults to False; `agent.py` passes no model decision
  callback in that configuration. The optional model-decision path is not the default demo.
  Clear supported commands can also bypass planning through `fast.py`. "Run" holds the
  two shoulder buttons. `shop` walks to a seller, talks, buys N of an item, and reads the gold
  before and after; it also rests at an inn and reads everyone's HP after. `field_menu` uses a
  tech or an item on someone outside a fight, or changes a battle setting.
- **Walking** (`walk.py`): first, in each new area, it taps the d-pad one way and back and
  watches what starts and stops moving with it: that's your character, not someone else with the
  same red hair, and not a party member catching up, and that went the way the d-pad was held.
  Then one Muse Spark call finds the target and a way around anything in the way. At the same
  time SAM 3.1 looks for the target by name (and as "person", for someone to talk to): Muse
  Spark's boxes are often a sprite off, and the walk stops and faces by them, so when SAM finds
  the same thing its box is used. SAM gets no longer than the model took plus a moment, so
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
- **People** (`people.py`): in a surveyed place, "talk to the guy selling swords" is matched
  with no model call against everything known about each person there: the survey's name and
  other names they go by, what they said when talked to, what they are ("guy" is a man,
  "soldier" a guard; a word that contradicts someone counts against them), and where they
  stand ("on the left", "by the bell"). The best fit wins when it's clearly ahead; when two or
  three fit about as well, it asks which one. "Who's here?" is answered the same way, nearest
  first, in steps and d-pad directions.
- **Where is it?** (`locate.py`, `app.py`): SAM 3.1 finds every match for the thing asked
  about and returns segmentation masks/boxes; these can be empty or wrong. At the same time Muse Spark looks for it too (SAM doesn't know many
  game things, like a pixel-art cat) and for your character. The answer is from your character
  when it's known ("4 steps up and 2 to your left", nearest first), and each match gets a short
  sound panned to its side.
- **Health warnings** (`hud.py`): for real-time games, Muse Spark finds the health and stamina
  bars once; then a few lines of pixel math read them on every frame and play a sound when they
  get low.
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
- **World maps** (`atlas.py`): telling world-map icons apart and knowing a game's geography is
  where a vision model fails (it took a village for the castle, and put the fair west of town
  when it's north). A sighted friend would look at a map, so this does too. An atlas is a game's
  map images, with each place's spot on them, labelled once with Muse Spark
  (`scripts/label_atlas.py`) and checked by a person, plus where you can walk, surveyed by walking
  the map in an emulator with the game's memory saying where the character is
  (`scripts/survey.py`). The screen is found on the map by image matching, the route is planned on
  the walkable ground, and the walk checks where it is several times a second. `where_am_i`
  answers from it, relative to the nearest places.
- **Rooms and towns** (`rooms.py`): the same idea indoors. `scripts/survey_rooms.py` walks every
  place in the emulator, notes which tiles can be walked and where each way out goes, and
  stitches the place into one picture; Muse Spark names the people and exits on it. In play, the
  screen is found on that picture, the character on it by its look, and `go_to_exit` walks to a
  named way out or person, through other places when it's further. The Steam version's redrawn
  screens are found on the SNES pictures.
- **The Epoch** (`epoch.py`): flying Chrono Trigger's time machine across the sea and to another
  era with `go_to_era`. Tested on the SNES version only.
- **Guides** (`guide.py`, `quests.py`): a walkthrough you put in `~/.config/player-two/guides/` is
  searched for "how do I get to ...?", and turned once into a list of story steps (one Muse Spark
  call per chapter, `scripts/build_quests.py`) for "what do I do next?".
- **Game notes** (`notes.py`): what the agent learns about a game (which button opens the menu,
  how battles go, where a place is) is kept in a plain text file per game in
  `~/.config/player-two/games/`, and read at the start of every session. A newer note replaces
  a similar older one, so a correction doesn't leave both. A game pack (`<game>.pack.txt` in the
  same folder) is read the same way: how the game works, written once per game.
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

## Tests and benches

`tests/test_real_frames.py` checks walking, fades and the area map on Chrono Trigger frames
recorded while holding the d-pad, at full and slow capture rates. The recordings are game footage,
so they aren't in the repo; the tests skip without them (`P2_RECORDINGS` points at them). Tests
that need the maps or real Steam frames skip the same way.

The benches need an isolated desktop (the external omabox tool) and a copy of the game,
so they don't run as is elsewhere:

- `scripts/emu_tasks.py` and `scripts/rooms_bench.py` run the SNES version in RetroArch, give
  Player Two (or just the walker) a task, and judge it by reading the game's memory: where the
  character ended up, whether a fight is still on. `scripts/atlas_walk.py` does the same for the
  world-map walker alone. These emulator benches are distinct from the Steam results in the README.
- `scripts/narrator_bench.py` narrates labelled Steam frames and scores what had to be said, what
  was made up, and how soon the first word came.
- `scripts/play_tasks.py` runs tasks on a real Steam copy, with Muse Spark judging the last screen.
