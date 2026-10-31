#!/bin/sh
# Record a demo take: the game monitor at 60 fps with the desktop's sound (the game and
# Player Two's voice) and the microphone (the player). Stops on Ctrl+C.
#   scripts/demo_record.sh [monitor] [out.mp4]
# Then: python scripts/demo_captions.py <events.jsonl> --video out.mp4 --burn out-captioned.mp4
set -e
MON=${1:-DP-2}
OUT=${2:-$HOME/Videos/player-two/take-$(date +%Y%m%d-%H%M%S).mp4}
mkdir -p "$(dirname "$OUT")"
# The captions line up with the video from this moment (the recorder starts within ~0.3 s).
date +%s.%N > "$OUT.start"
echo "Recording $MON to $OUT (Ctrl+C to stop)"
exec gpu-screen-recorder -w "$MON" -f 60 -k h264 -q very_high \
  -a "default_output|default_input" -ac aac -o "$OUT"
