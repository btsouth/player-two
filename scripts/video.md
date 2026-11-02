# Build the Player Two demo

Run heavy encoding on devbox. Python 3, Pillow and ffmpeg with libass are required. Inter4.1
and its OFL license are bundled in assets/video. No game or desktop is opened by the renderer.

```sh
python3 scripts/make_video.py TAKES --out draft.mp4 --sheet --jobs 2
python3 scripts/make_video.py TAKES --out proxy.mp4 --proxy --sheet --jobs 2
python3 scripts/make_video.py TAKES --out final.mp4 --final --sheet --jobs 2
```

Full output is1920x1080 at60fps; proxies are960x540 at30fps. Gameplay fills the frame, with Lanczos
scaling and no FPS-counter masking. Speech is plain white52px (80px on black); the person's
request has a separate chip with a real typed/spoken label. Measured model badges are at least34px.

## Portable takes and swaps

Each take has video, a recording-start Unix timestamp, original events.jsonl, and optional paired
.game.wav/.voice.wav stems. The stems start at video zero and must cover the video. takes.json
maps names to relative video/start/events paths and reviewed edit settings. Symlinks are fine
when rsync dereferences them. Never change source logs or reuse timing values for another run.

The full story uses take-A, take-B, nav, game2, rdr-sam and goodbye. Missing footage remains a
marked placeholder and blocks final mode. A reject reason in a take manifest excludes it.

```sh
python3 scripts/select_video_take.py TAKES take-A REVIEWED_RECORDER_FOLDER \
  --delay MEASURED_SECONDS --shop-menu SOURCE_START SOURCE_END
python3 scripts/select_video_take.py TAKES nav REVIEWED_RECORDER_FOLDER \
  --delay MEASURED_SECONDS --clip SOURCE_START SOURCE_END
```

The selector requires run.mp4, run.mp4.start, events.jsonl and both stems. It preserves originals,
creates a portable source alias, and changes only the manifest. An optional --session-log prepends
the last original ready record before recording into a derived sidecar if the take omitted model
identity. It preserves original event bytes. Never infer a backend from a request's wording.

recording_delay corrects measured recorder launch delay without rewriting original timestamps.
Measure it against the actual voice waveform. Clip, menu, reveal and ping times are SOURCE VIDEO
seconds, independent of that offset. Inspect each replacement's own frames and resolved plan.

## Opening, audience, quote and ending

One B run supplies damage, How are we doing, Cyclone, delegation, healing and the real win. All
shown speech stays normal speed, in order. Black begins before the actual opening_line caption;
reveal_at/reveal_duration/normal_until must match the recorded Cyclone spin. Only quiet gaps may
be shortened or accelerated. The win now pays off in the opening, before the audience title.
There is no later replay of that result. The real goodbye is a separate short field excerpt:
logged Goodbye, actual Bye for now and Have fun, then the five-second end card.

The audience title holds one real field frame (identity_at), with a founder VO slot underneath.
The complete RetroArch quote, source date, explanation of Lavos and honest still-ahead line stay
over moving map footage. The quote is attributed to the December2019 RetroArch1.8.2 release notes.
It is outside evidence of a gap, not a claim this prototype has beaten Lavos.

The full publisher notices stay intact on the end card, alongside the general ownership and
no-endorsement line. Notices are attribution, not a claim of additional publication clearance.
The tagline is “You choose. It reads and presses.” Input-method wording comes from the actual
shown requests, followed by “Waits are cut and fast parts are marked.”

## Navigation and model evidence

The fair entry is a door press and is omitted as navigation proof. The nav slot reserves about12s
for a reviewed real world-map walk at1x. It requires a logged local go_to_place result and the
saved atlas-proof.json name/check/hash evidence. Only the single line “Muse labelled this map.
I checked it. Local code walks it.” is displayed; coordinates and raw model output are not shown.
The seller walk can use take-A.edit.speeds.A4=1. shop_menu changes chip placement without altering
video/audio boundaries, keeping quantity, prices, gold and HP clear.

Badges require completed logged measurements. Local action seconds include movement/animation;
model_s is inference time; scene latency is labelled generation. Empty model turns alone do not
prove local execution: a recognized local step is also required. Source timestamps are retained
in evidence JSON. The opening keeps request chips uncluttered; local measurements remain in the
sidecar rather than being repeated over the paused fight. pause_windows needs frame/code proof
that those exact intervals are held for the next player request, not model deliberation.

## Persistent timing disclosures

Cold-open and Monster Hunter summaries stay visible throughout their montages. They are computed
from the actual first/last source boundary and the summed output durations. The cold line states
“Waits between lines cut: N s of play in M s”. MH states the source fight span, output duration,
and that narration is muted in fast parts. No unreadable six-frame disclosure is relied on.

Final mode checks the summary's arithmetic, common source, ordered nonoverlapping windows and
full coverage of every piece. Removing a label, shortening it or changing its values fails. Other
accelerated clips still require their full-duration numerical label. These are checks, not an
exception that permits unlabelled compression.

## Monster Hunter

The story manifest supplies ordered normal windows, gap_speed and optional request_intro:
[source_start, first_normal_start, speed]. The intro must contain a real logged request; it is
explicitly accelerated/muted so a partial earlier answer is not presented as the current answer.
The first completed mh_muse weak-point read supplies its actual duration. The visible language
explains weak points instead of unexplained “wheel” jargon. Code written for this game takes
turns after the read; this path never inherits a no-code claim. A Speed-versus-Power gloss explains
the attack triangle when that advice is retained.

Intervening footage stays in source order, accelerated, silent and uncaptioned. The known false
technical-counter line is refused if it overlaps an audible window. Verify the rendered muted
piece itself. unducked_windows retains both stems for a game character's reply. Replace video,
start, log, stems and all reviewed windows together for a new take.

## Red Dead and its benchmark

rdr-sam uses one source and its original events. Edit windows are view, horse, wagon, dialogue and
how. Horse and wagon require actual where events with via=sam-3.1, normalized boxes and measured
seconds. Boxes appear only once the result arrives and remain through the real ping. A Spark
fallback or a centre point is not SAM proof. The movement chip explicitly says the person walks.

ping_boosts supplies measured [source_start, source_end, gain] windows. Multiply both channels
identically to preserve pan. Never move, extend or synthesize the cue. horse_wait_cut may remove
only checked silence and carries a steady measured wait-cut disclosure. Verify actual cue timing
and direction in the finished master, not just its isolated clip.

The short actor shot retains BOTH stems with ducking disabled and refuses logged Player Two
speech. Its caption is a separate recorded-clip benchmark, not a claim that silence in this shot
proves the fix. Private rdr-bench contains words.json plus before/after.events.jsonl and t0.json
files from the same replayed source. The renderer recomputes overlap over170s, merging word gaps
under0.35s, and only displays the12.7s/1.1s comparison when it matches. Missing or contradictory proof drops the shot. Raw benchmark video and private source dumps do not belong in the repo.

## Monday voice slots and audio

Put vo-intro.wav, vo-how.wav and vo-limit.wav beside takes.json, then rebuild without --cut.
Current slot ceilings: intro8.50s, How7.00s, limit6.00s. Longer takes are refused. The exact prompts
are stored in the cut's vo_prompt fields. Intro and How use quiet game-only illustrative footage;
that is never used as evidence of the app's silence. The creator-testing limit remains a clean
card, with the first-person limit line spoken over it. Missing VO still blocks final mode.

The game ducks under real voice stems (15ms attack,220ms release,8:1). No added music. Accelerated
opening gaps are silent. Speech-free navigation fight gaps retain pitch-preserving game audio;
any logged speech overlapping such a gap is refused. Clean t13 excerpts keep both stems at1x;
cuts remove intervening waits and turns. Short cut ramps avoid clicks. Bounded padding flushes finite-stem sidechain
buffers, then trims to the real cut so a last word or ping cannot disappear at EOF.

Audio is decoded and trimmed per piece to its actual video frame count before joining. Video
uses those same durations. Do not restore packet-copy AAC joins: per-clip priming accumulates.
The master has one continuous audio clock. Two-pass loudness targets-14LUFS with headroom;
a192kHz oversampled limiter precedes48kHz/AAC. Always measure decoded ebur128 loudness/true peak.

--cut replays a resolved cut. Cache keys include renderer/font/source metadata/captions/settings.
--final refuses placeholders, missing required VO, incomplete opening, unsupported timing/SAM,
stale spoken facts, sped-up speech, missing time disclosures and duration above3:00, including
after encoding. --accept-captions cannot bypass stale facts or sped-up speech. Inspect contact
sheets, individual overlay states, source facts and full-speed sound before submission.

Quote source: [Archived release notes](https://web.archive.org/web/2020/https://www.libretro.com/index.php/category/accessibility/).

## Castle-story and clean-combat cut

The navigation manifest can provide edit.story windows with role/a/b and optional speed. It
requires the original interrupted castle route, the model's fight decision and its multi-turn
exit plan. Spoken spans always remain1x. Only verified speech-free gaps use faster video/audio;
the game track is time-stretched without changing pitch. The source/output disclosure stays
visible, and the join to the fair's other recording says it is another run of the same save.

Clean Monster Hunter footage uses story.cuts_only=true. Every selected clip keeps both original
stems at normal speed. There is no narration muting or accelerated speech; the steady line says
cuts occur between requests and turns. carry_requests can recall an actual typed request on its
answer; microphone requests must remain in the selected media. Status and weak-point badges
retain their distinct real call durations. All source windows need fresh review on replacement.
