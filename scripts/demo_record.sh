#!/bin/sh
# Record a demo take: the game monitor at 60 fps. Stops on Ctrl+C or TERM. Three sound tracks:
# everything together, the game alone, and Player Two's voice with the microphone, so the edit
# can duck the game under speech. Writes out.game.wav and out.voice.wav beside the video.
#   scripts/demo_record.sh [monitor] [out.mp4]
# DEMO_MIC=0 leaves the microphone out (requests typed with `player-two ctl do`).
# Then: python scripts/demo_captions.py <events.jsonl> --video out.mp4 --burn out-captioned.mp4
set -e
MON=${1:-DP-2}
OUT=${2:-$HOME/Videos/player-two/take-$(date +%Y%m%d-%H%M%S).mp4}
mkdir -p "$(dirname "$OUT")"
P2_APP=$(gpu-screen-recorder --list-application-audio 2>/dev/null | grep -m1 -i -E 'python|player' || true)
MIX="default_output"
GAME="default_output"
VOICE="default_input"
if [ -n "$P2_APP" ]; then
  GAME="app-inverse:$P2_APP"
  VOICE="app:$P2_APP"
  [ "${DEMO_MIC:-1}" = 0 ] || VOICE="$VOICE|default_input"
elif [ "${DEMO_MIC:-1}" = 0 ]; then
  echo "Player Two isn't making sound yet, so its voice can't get its own track" >&2
  VOICE=""
fi
[ "${DEMO_MIC:-1}" = 0 ] || MIX="default_output|default_input"
# The captions line up with the video from this moment (the recorder starts within ~0.3 s).
date +%s.%N > "$OUT.start"
echo "Recording $MON to $OUT (Ctrl+C to stop)"
gpu-screen-recorder -w "$MON" -f 60 -k h264 -q very_high -cursor no \
  -a "$MIX" -a "$GAME" ${VOICE:+-a "$VOICE"} -ac aac -o "$OUT" &
GSR=$!
trap 'kill -INT $GSR 2>/dev/null' INT TERM
wait $GSR || wait $GSR || true
if [ -n "$VOICE" ] && [ -s "$OUT" ]; then
  B=${OUT%.*}
  ffmpeg -v error -y -i "$OUT" -map 0:a:1 -c:a pcm_s16le "$B.game.wav" -map 0:a:2 -c:a pcm_s16le "$B.voice.wav"
fi
