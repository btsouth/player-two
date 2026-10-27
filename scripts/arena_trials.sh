#!/usr/bin/env bash
# Run voice commands against the test arena in an omabox box and check the outcome.
# Each line of TRIALS: command | expected title fragment
set -u
BIN=/home/bts/Projects/player-two/.venv/bin/player-two
URL=file:///home/bts/Projects/player-two/tests/arena/index.html
pass=0; total=0
# Forward the open-weight brain settings into the box when they're set.
EXTRA=()
for v in PLAYER_TWO_BRAIN PLAYER_TWO_OPEN_BASE_URL PLAYER_TWO_OPEN_MODEL MODAL_PROXY_KEY MODAL_PROXY_SECRET; do
  [ -n "${!v:-}" ] && EXTRA+=(--pass "$v")
done
while IFS="|" read -r cmd want <&3; do
  [ -z "$cmd" ] && continue
  total=$((total+1))
  omabox run -- sh -c 'pkill -f "[b]in/player-two run"; pkill -f "[t]ests/arena"; rm -f /tmp/trial.jsonl; true' >/dev/null 2>&1
  omabox run -d --wait -- chromium --kiosk --no-first-run --user-data-dir=/tmp/arena-trial "$URL" >/dev/null 2>&1
  omabox run -d --pass MODEL_API_KEY "${EXTRA[@]}" -- "$BIN" run --controls keyboard --window Arena \
    --game "a small JRPG test scene" --log /tmp/trial.jsonl >/dev/null 2>&1
  omabox wait until --timeout 40 -- sh -c 'grep -q "\"kind\": \"ready\"" /tmp/trial.jsonl' >/dev/null 2>&1
  start=$(date +%s.%N)
  omabox run -- "$BIN" ctl do "$cmd" >/dev/null 2>&1
  omabox wait until --timeout 120 -- sh -c 'grep -q "\"kind\": \"agent\"\|\"kind\": \"ask\"" /tmp/trial.jsonl' >/dev/null 2>&1
  if omabox run -- sh -c 'grep -q "\"kind\": \"ask\"" /tmp/trial.jsonl' >/dev/null 2>&1; then
    q=$(omabox run -- sh -c 'grep "\"kind\": \"ask\"" /tmp/trial.jsonl | tail -1' | python3 -c 'import json,sys;print(json.load(sys.stdin)["text"])')
    omabox run -- "$BIN" ctl do "yes" >/dev/null 2>&1
    omabox wait until --timeout 120 -- sh -c 'grep -q "\"kind\": \"agent\"" /tmp/trial.jsonl' >/dev/null 2>&1
  else q=""; fi
  secs=$(python3 -c "import time;print(f'{time.time()-$start:.1f}')")
  title=$(omabox run -- hyprctl -j clients | python3 -c 'import json,sys;print(next((c["title"] for c in json.load(sys.stdin) if c["title"].startswith("Arena")),"?"))')
  if [[ "$title" == *"$want"* ]]; then ok=PASS; pass=$((pass+1)); else ok=FAIL; fi
  echo "$ok ${secs}s | $cmd | want '$want' got '$title'${q:+ | asked: $q}"
done 3<&0 </dev/null
echo "$pass/$total passed"
