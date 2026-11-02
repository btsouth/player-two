# Build the Player Two demo

Encode on devbox. Python 3, Pillow and ffmpeg with libass are required. The renderer opens no
game or desktop. Inter 4.1 and its OFL license are bundled in `assets/video`; libass loads that
folder directly, so no system font install is needed.

```sh
python3 scripts/make_video.py takes --out round2-sample.mp4 --sample --sheet --jobs 2
python3 scripts/make_video.py takes --out draft3d-proxy.mp4 --proxy --sheet --jobs 2
python3 scripts/make_video.py takes --out draft3d.mp4 --sheet --jobs 2
python3 scripts/make_video.py takes --out final.mp4 --final --sheet --jobs 2
```

Full: 1920x1080 at 60 fps, full-bleed Lanczos scaling, 42px request chips, 34px measured badges,
48px speech captions (80px on black). Proxies: 960x540 at 30 fps. No FPS-counter masking.

## Takes

Use `take-A.mp4`, `take-B.mp4`, `game2.mp4`, `game3.mp4`, optional `sam.mp4` and `goodbye.mp4`.
Each has `<name>.mp4.start` (Unix recording start) and `<name>.events.jsonl` (original log).
For other filenames, `takes.json` maps names to relative `video`, `start` and `events` paths.
All referenced files must be included when syncing the takes folder. Symlinks may be dereferenced.
A manifest entry with `"reject": "reason"` excludes a take. Replace all source files before
removing it. `--sample` alone permits rejected footage for an explicitly marked style test;
`--sample --final` is forbidden. Never submit private notifications or rejected gameplay.

A and B requests are selected from their actual logs. `--pick A4=1` selects the first matching
attempt (default last). `--clip game2=12-35` or `--clip sam=2-10` selects a continuous excerpt.
The cutter cannot judge gameplay, privacy, spoken correctness or whether an action succeeded.

## Cold opening and ending

The accepted B sequence is fight narration, “How are we doing?”, “Crono, use Cyclone.”, then
“Win this fight for me.” The picture is black from before the logged first “Fight.” through the
choice. Every spoken line stays in order. It fades up on the actual spin, which must be measured
from source frames, not guessed from the completed action event. The reveal and spin stay normal
speed. Manifest settings can accelerate silent gaps at 4x and shorten only their silent middles;
all original spoken spans remain normal speed and chronological. Every accelerated piece has a
speed label; shortened silence also says “Silent gap shortened”. Typed requests
stay visibly typed. Missing opening speech or unmeasured reveal is refused by final mode.

Add measured settings to the take-B manifest entry, for example:

```json
{
  "take-A": {"edit": {"recording_delay": 0.4}},
  "take-B": {
    "edit": {"recording_delay": 0.32, "reveal_at": 33.35,
             "reveal_duration": 0.25, "normal_until": 35.35,
             "gap_speed": 4, "gap_max_output": 0.3,
             "min_gap": 0.35, "speech_pad": 0.06}
  },
  "edit": {"drop": ["A2"]}
}
```

`recording_delay` corrects the measured delay between the shell timestamp and actual recorder
start, verified against the voice stem. It adjusts log-to-video mapping without changing any
original timestamps. Reveal values remain absolute SOURCE VIDEO seconds, independent of that
epoch correction. Those values apply to the reviewed run, not arbitrary replacement footage. Review every new
reveal against its frames. Use the original full session log when a per-check log omits speech
from before the first request. Preserve original timestamps and never synthesize a missing line.

The same run's final real “Won the fight…” spoken record is reserved once for the black-screen
ending, labelled “The same fight. The result.” It is neither repeated in the opening nor dubbed.
A missing result remains a placeholder. `goodbye.mp4` supplies the separate real closing exchange:
logged “Goodbye.” followed by actual spoken “Bye for now.” and “Have fun.”. Missing or out-of-order
records are refused. Until captured, that slot is explicitly labelled as an ending design.

The film follows with a two-second product card and four-second RetroArch excerpt containing
“Lavos is still ahead. Here’s how far we’ve got.” `vo-ahead.wav` optionally adds a spoken ahead card. No glasses story.
Then map proof, actual walking/shop, optional SAM when usable footage exists, other games, how,
honest limit card, real win and goodbye, then the end card. The complete Square Enix proprietary
notice is copied verbatim from its Chrono Trigger policy into the end card, alongside the general
publisher attribution and no-endorsement line. These notices are not a claim of contest clearance. `A3b` (“What does it look like?”) retains whole descriptive sentences
and marks its shortened tail. No rates wall. `vo-how.wav` voices the how card; `vo-why` is retired.

The earlier `--sample` remains a 45-second style sample based on a continuous Win excerpt; it is
not the new full choice/delegation story. The full render is the default command.

## Model evidence and SAM

Only completed, logged measurements get badges. `model_s` is measured model time; local action
seconds include walking/animation. Local classification requires empty model turns AND a recognized
local step. Empty turns alone do not prove no model work. Scene narration is separately labelled.
Typed requests stay visibly typed. Never claim a transcriber latency without a measured event.
Every badge retains its source event timestamp in the evidence JSON.

SAM requires `where.via == sam-3.1`; the rectangle comes only from that event's normalized `boxes`
array, never from a centre point or Spark fallback. The app logs returned SAM boxes and queried-frame
time. Capture a stationary camera/target around the response so a returned box remains valid on the
recorded frame. The renderer displays it for at most 1.5 seconds after completion. Keep the actual
stereo ping. A SAM take without real logged boxes is refused by final mode.

`atlas-proof.json` supplies one real `raw_place` from saved label_atlas output, the same-name
`checked_place` with x/y, and `raw_sha256` / `checked_sha256`. The five-second sequence shows saved
model JSON, the checked destination, then local-navigation data. It is clearly a saved response,
not a fabricated live model call; those stages are not represented as latency measurements.
Do not include fan-made map art or source dumps in the public repo.

## Audio and checks

Optional `<take>.game.wav` and `<take>.voice.wav` replace the mixed recording audio. Both must
start at video time zero and cover its duration. Voice contains microphone + Player Two; game is
separate. The game ducks under voice (15ms attack, 220ms release, 8:1). An incomplete pair is an
error. Premixed takes cannot be independently ducked. No added music. Accelerated gaps are silent;
all speech stays at normal speed. Cut ramps prevent clicks. Two-pass normalization targets -14 LUFS
and -2.5 dBTP; measure decoded AAC with ebur128 again.

Cut/evidence JSON, caption warnings and optional sheets accompany the output. `--cut` replays a
hand-edited cut; rebuild without it to discover new source/VO files. Cache keys include renderer,
font, source size/mtime, captions and settings. `--final` refuses placeholders, missing how VO,
incomplete choice/delegation opening, missing geometry/timing/captions, unlabelled speed-ups, stale health facts,
sped-up speech and duration above 3:00, including after encoding. `--accept-captions` cannot bypass
stale facts or sped-up speech. Human frame and full-speed listening checks remain essential.

Completed action windows hold at least 1.2 seconds after the logged result so measured completion
badges remain readable. Speed labels stay physically clear of request chips; shop speed labels use
the lower-right open area, away from its quantity/item menus. Recheck placement on new games.

A take can also supply `edit.shop_menu: [83.5, 87.6]` (measured absolute source-video seconds).
While that menu is open, the request chip uses its empty lower item panel; controls, quantity,
prices, gold and HP remain visible. Only timed overlays change, never source video/audio boundaries.

## Red Dead: description, then restraint

Use `rdr-look.mp4` and `rdr-dialogue.mp4` with start, events and both stems. Per-take `edit.clip`
selects the exact continuous source interval. The description is normal-speed and any omitted
last sentence is labelled. Choose a whole spoken sentence boundary from the actual voice stem.
The dialogue segment refuses logged Player Two speech in its interval. It keeps BOTH stems,
with sidechain compression disabled, so real game voices stay audible and unexpected Player Two
speech cannot be hidden by muting its stem. Verify digital silence in that source voice window.

`rdr-look.edit.notes_only_verified: true` enables “No code written for this game. One line of notes.”
only after checking current runtime code and the actual notes file. Do not carry that flag forward
without rechecking it. Recorded typed questions remain explicitly typed. Source timing and old/new
build limits belong in the private QC, not a false claim that the latest code recorded an older take.

`take-A.edit.speeds` may override a single action's silent-gap speed, for example `{"A5": 4}`.
`duck: false` in resolved cut JSON preserves both stems without ducking. Retake source selection,
clock offsets, reveal points and menu positions remain in the manifest rather than hand edits to code.

## A single Monster Hunter run

`game2.edit.story` selects ordered `normal` source windows and a `gap_speed`. The entire source
interval between the first and last window remains continuous: intervening footage is accelerated,
labelled, silent and has captions disabled. This permits a scored-out false utterance to remain in
the source chronology without putting its claim on the soundtrack or in captions. A guard refuses
the known false technical-counter line if it intersects any audible window. Verify the rendered
interval itself for silence and absence of caption pixels; a plan alone is insufficient.

The first window requires a real completed `mh_muse` wheel event and displays its measured seconds.
The chapter identifies custom combat code explicitly. Its story path never inherits the generic
no-code overlay. `unducked_windows` preserves both audio stems without sidechain compression where
an NPC response needs to remain audible. Caption placement leaves the battle menus and party HP
clear. Source offsets and all windows belong in the take manifest, not the public repository.
