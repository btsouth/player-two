# Build the demo video

Run encodes on devbox. Python 3, Pillow, ffmpeg with libass and Liberation Sans are required.
No game or desktop session is opened by this tool.

Put `take-A.mp4`, `take-B.mp4`, `game2.mp4`, `game3.mp4` in one directory. Each has
`<name>.mp4.start` (recording start, Unix seconds) and `<name>.events.jsonl` (original log).
Missing games produce obvious placeholders. Never use footage containing private notifications.

```sh
python3 scripts/make_video.py takes --out proxy.mp4 --proxy --sheet --jobs 2
python3 scripts/make_video.py takes --out draft.mp4 --sheet --jobs 2
python3 scripts/make_video.py takes --out final.mp4 --rates rates.txt --final --sheet --jobs 2
```

Drop `vo-why.wav` and `vo-how.wav` into `takes` to replace the marked voice-over stand-ins.
Other supported formats include FLAC, M4A and MP3. No code edits are needed. With no voice-over,
the why text is shown silently over blurred footage. The final render requires both recordings.

For source paths with other names, use `takes/takes.json`, with paths relative to that folder:

```json
{
  "take-A": {
    "video": "chrono/walk.mp4",
    "start": "chrono/walk.mp4.start",
    "events": "chrono/events.jsonl"
  }
}
```

Keep one run per request. `--pick A4=1` selects the first matching attempt; default is last.
Use `--clip game2=12-35` to choose one continuous secondary-game excerpt. Inspect every chosen
excerpt: the automatic cutter does not judge gameplay, truth, privacy, or whether an action succeeded.
Do not label a game "no code written" unless that is true; Monster Hunter has game-specific code.

Optional `take-A.game.wav` and `take-A.voice.wav` (similarly for other takes) replace the mixed
recording audio. Both must start at exactly the video start and cover its duration. Voice contains
the microphone and Player Two; game contains only the game. The game is sidechain-ducked under
voice. A premixed recording cannot be independently ducked; fix that at recording time. Voice-over
background audio is reduced separately. There is no added music. Accelerated gaps are silent,
all speech remains normal speed, and cut boundaries have short ramps to prevent clicks.
The final mix uses two-pass -14 LUFS / -1.5 dBTP normalization; verify decoded AAC with ebur128.

The output includes a cut JSON, evidence JSON and caption warning report. `--cut` can replay a
hand-edited cut, but automatic source selection/voice-over replacement requires rebuilding without
`--cut`. Encoded pieces are cached by settings, source size/mtime, subtitles and renderer revision.
Proxies use 960x540 at 30 fps; full renders use 1920x1080 at 60 fps. All text is laid out at 1080p.

Evidence badges appear only after the logged completion time. `model_s` sums measured model
turns; action seconds include walking/animation. These are labelled separately. Spark badges
require a logged Muse brain identity. Scene narration is labelled separately from local actions.
Explicit `model_stage` events can supply `model` (`Muse Voice Transcribe` or `SAM 3.1`) and
`seconds`; absent instrumentation produces no invented stage or latency badge. Every badge's
source timestamp is written to the evidence JSON. Request captions distinguish typed from heard.

`--final` refuses placeholders, missing voice-over, draft rates, missing timing/captions, unlabelled
speed-ups, stale health facts, sped-up speech and a duration above three minutes (including after
encoding). The deprecated `--accept-captions` cannot bypass stale facts or sped-up speech.
A final render still needs human viewing and listening. Check the contact sheet, every caption
state, every model badge and final loudness; a passing script cannot certify the video.

A manifest entry with `"reject": "reason"` explicitly excludes that take. A missing/rejected
fight becomes a visible placeholder, never a silent omission. Remove `reject` only when the video,
log and start timestamp have all been replaced with an accepted run.
