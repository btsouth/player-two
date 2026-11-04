# Validation boundaries

## Fresh application installs, Nov 2

Both trials began with a fresh public clone of main at `259ca29` and followed its README
before consulting implementation notes. Times are wall-clock measurements, not estimates.

| Machine | Measured result | What was checked |
|---|---|---|
| Disposable Ubuntu VM | 112 s for prerequisites, uv, clone, tool installation and help; 14 s for setup | Blank-key setup, packaged reader loading, blank-frame reading and A* wall detour passed. |
| Windows 11 Pro laptop | 72.2 s for the public installer and setup; source-clone offline check 11.7 s | uv and Player Two were initially absent. Python, Windows extras, ViGEmBus and voice downloads completed; help and offline checks passed. This was an existing OS, not a clean Windows installation. |

Setup reported a silent microphone on both machines. Voice assets downloaded and setup reported
controller access, but no person verified audible output, microphone recognition or a live game.
The laptop already had a Visual C++ runtime. Its installation on a clean Windows OS and Windows
ARM remain unverified. Offline checks do not open audio, a controller or a desktop.

## A real call without a game

The original static fixture `tests/arena/sample.png` adds a headless vision check using the
existing `player-two narrate` command. A Meta key is required. On Nov 2, both machines returned
correct descriptions of the practice screen and its selected Attack option. This validates
one image narration, not tool use, navigation, speech playback or gameplay. Provider cost is
estimated by the CLI, not measured from an invoice.

The Linux VM's normal dual-stack calls returned HTTP 404 for the configured Muse model.
Restricting the same installed client to IPv4 succeeded; a controlled comparison also reproduced
the failure in the source environment without the IPv4 restriction. After disabling IPv6 on
that disposable VM, the unmodified installed command succeeded. This is a network workaround,
not proof that all Linux networks work or a change to the application's transport. The Windows
call needed no network workaround. The tested workaround on the disposable Ubuntu VM was:

```bash
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
```

This changes that VM's networking; it is not a Player Two setting. Temporary API-key files were
restricted to the test user and removed after the requests; setup stored no API key in these trials.

## Public checks versus live play

CI runs lint, the offline smoke script and the test suite on Linux; Windows runs the offline
script, suite and a window-enumeration import check. Frame/map tests skip when their private
game data is absent. A green run does not establish real microphone quality, accessible
independent play, game capture, input delivery or reliable live-game narration.

## Notes on the live results

Action times end at logged task completion; the final spoken result can continue afterwards.
Local-answer times include the spoken answer. First words can include an acknowledgement. Completion
counts are not narration-quality pass rates: all six purchase runs finished but missed the narration
timing checks. One more fight run didn't count because the launch script failed before the fight
started. Just before these runs, on a slightly older build, a purchase wandered for 65 s by the
seller and said nothing, so the walker was changed to give up sooner. Earlier builds lost 3/3 boss
attempts. The 63 spoken-line check covers transcription and routing, not completed actions or a real
microphone.

## Notes on the film

The film shows chosen takes from developer sessions. Eight recorded Somnacanth takes won eight fights;
frame review found false next-attack narration in three earlier takes. In the recorded Red Dead SAM
session, two of six requests returned no boxes. In the recorded cover-question session on Red Dead,
4 of 8 attempts were true against the frame SAM saw. The Linux game-speech hold reduced overlap on one
recorded clip from 12.7 s to 1.1 s, and one separate 87 s live cutscene had no Player Two line. There
is no claim of beating Lavos, playing arbitrary games or replacing a game's own accessibility features.

## Hosted open-weight check

A Nov 1 saved-frame narration through Baseten's Muse Spark Open endpoint returned the visible
result text in 6.0 s, with 2,401 input and 294 output tokens, estimated at $0.0043 at the published
rates. A shorter JSON probe exhausted its completion budget without a usable description.
This is historical narration evidence, not a fresh install check, a live agent benchmark or
local inference. SAM and transcription continue to use Meta's API in open-weight mode.
