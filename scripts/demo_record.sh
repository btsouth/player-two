#!/bin/sh
# Record a demo take: the game monitor at 60 fps with the desktop's sound (the game and
# Player Two's voice) and the microphone (the player). Stops on Ctrl+C.
#   scripts/demo_record.sh [monitor] [out.mp4]
# DEMO_MIC=0 leaves the microphone out (requests typed with `player-two ctl do`).
# Then: python scripts/demo_captions.py <events.jsonl> --video out.mp4 --burn out-captioned.mp4
set -e
MON=${1:-DP-2}
OUT=${2:-$HOME/Videos/player-two/take-$(date +%Y%m%d-%H%M%S).mp4}
mkdir -p "$(dirname "$OUT")"
# The captions line up with the video from this moment (the recorder starts within ~0.3 s).
date +%s.%N > "$OUT.start"
echo "Recording $MON to $OUT (Ctrl+C to stop)"
AUDIO="default_output|default_input"
[ "${DEMO_MIC:-1}" = 0 ] && AUDIO="default_output"
exec gpu-screen-recorder -w "$MON" -f 60 -k h264 -q very_high -cursor no \
  -a "$AUDIO" -ac aac -o "$OUT"
