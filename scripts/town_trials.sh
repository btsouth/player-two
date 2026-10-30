#!/usr/bin/env bash
# Walking and searching trials in the offline test town (tests/town/index.html), run in an
# omabox box and scored by the game's window title.
#   OMABOX_BOX=box-3 RUNS=2 ONLY=gato scripts/town_trials.sh [trials file]
# Each line of the trials file: start area | request | expected | limit seconds.
# "expected" is key=value from the title: area= must match, talked= and items= must be in
# the list. Logs and last screenshots go to $OUT (~/.cache/player-two/town/<time>).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/.venv/bin/player-two
URL=file://$ROOT/tests/town/index.html
TRIALS=${1:-$ROOT/tests/town/trials.txt}
RUNS=${RUNS:-1}
ONLY=${ONLY:-}
OUT=${OUT:-$HOME/.cache/player-two/town/$(date +%Y%m%d-%H%M%S)}
LOG=/tmp/town-trial.jsonl
mkdir -p "$OUT"
B=()
[ -n "${OMABOX_BOX:-}" ] && B=(-b "$OMABOX_BOX")
ob() { omabox "${B[@]}" "$@"; }
# Forward the open-weight brain settings into the box when they're set.
EXTRA=()
for v in PLAYER_TWO_AGENT_EFFORT PLAYER_TWO_BRAIN PLAYER_TWO_OPEN_BASE_URL PLAYER_TWO_OPEN_MODEL MODAL_PROXY_KEY MODAL_PROXY_SECRET; do
  [ -n "${!v:-}" ] && EXTRA+=(--pass "$v")
done
count() { ob run -- sh -c "grep -c '\"kind\": \"$1\"' $LOG || true" 2>/dev/null | tail -1; }

pass=0; total=0; cost=0
for run in $(seq 1 "$RUNS"); do
  n=0
  while IFS="|" read -r start req want limit <&3; do
    [ -z "$start" ] || [[ "$start" == \#* ]] && continue
    n=$((n+1))
    [ -n "$ONLY" ] && ! [[ "$start|$req|$want" =~ $ONLY ]] && continue
    total=$((total+1))
    id="r$run-t$n"
    ob run -- sh -c "pkill -f '[b]in/player-two run'; pkill -f '[t]ests/town'; pkill wtype; rm -f $LOG; true" >/dev/null 2>&1
    ob run -d --wait -- chromium --kiosk --no-first-run --user-data-dir=/tmp/town-chrome "$URL?start=$start" >/dev/null 2>&1
    ob wait window Town --timeout 20 >/dev/null 2>&1
    up=""
    for try in 1 2; do  # a session still closing can hold the control socket: once more
      ob run -d --pass MODEL_API_KEY "${EXTRA[@]}" -- "$BIN" run --controls keyboard --window Town \
        --companion off --game "a small top-down SNES-style RPG" --log $LOG >/dev/null 2>&1
      if ob wait until --timeout 60 -- sh -c "grep -q '\"kind\": \"ready\"' $LOG" >/dev/null 2>&1; then up=1; break; fi
      ob run -- sh -c "pkill -f '[b]in/player-two run'; rm -f $LOG; true" >/dev/null 2>&1
    done
    if [ -z "$up" ]; then
      echo "FAIL  -     | $start | $req | Player Two didn't start"; continue
    fi
    t0=$(date +%s.%N)
    ob run -- "$BIN" ctl do "$req" >/dev/null 2>&1
    answered=0; timedout=""
    while :; do
      left=$(python3 -c "import time;print(max(1,int($limit-(time.time()-$t0))))")
      if [ "$left" -le 1 ]; then timedout=1; break; fi
      ob wait until --timeout "$left" -- sh -c "grep -q '\"kind\": \"agent\"' $LOG || [ \$(grep -c '\"kind\": \"ask\"' $LOG) -gt $answered ]" >/dev/null 2>&1
      if [ "$(count agent)" -gt 0 ]; then break; fi
      if [ "$(count ask)" -gt "$answered" ]; then
        answered=$((answered+1))
        ob run -- "$BIN" ctl do yes >/dev/null 2>&1
        continue
      fi
      timedout=1; break
    done
    secs=$(python3 -c "import time;print(f'{time.time()-$t0:.0f}')")
    title=$(ob run -- hyprctl -j clients | python3 -c 'import json,sys;print(next((c["title"] for c in json.load(sys.stdin) if c["title"].startswith("Town")),"?").removesuffix(" - Chromium"))')
    if [ -n "$timedout" ]; then
      ob run -- "$BIN" ctl stop >/dev/null 2>&1
      ob wait until --timeout 30 -- sh -c "grep -q '\"kind\": \"agent\"' $LOG" >/dev/null 2>&1
    fi
    c=$(ob run -- "$BIN" ctl status 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("cost_usd", 0))
except Exception: print(0)')
    cost=$(python3 -c "print(round($cost+${c:-0},4))")
    ob run -- cat $LOG > "$OUT/$id.jsonl" 2>/dev/null
    shot=$(ob shot --window Town 2>/dev/null | tail -1)
    [ -f "$shot" ] && cp "$shot" "$OUT/$id.png"
    ob run -- "$BIN" ctl quit >/dev/null 2>&1
    ok=$(python3 - "$title" "$want" <<'EOF'
import sys
title, want = sys.argv[1], sys.argv[2].strip()
fields = dict(p.split("=", 1) for p in title.split()[1:] if "=" in p)
k, _, v = want.partition("=")
have = fields.get(k, "")
print("PASS" if (have == v if k in ("area", "x", "y") else v in have.split(",")) else "FAIL")
EOF
)
    [ "$ok" = PASS ] && pass=$((pass+1))
    why=$(python3 - "$OUT/$id.jsonl" <<'EOF'
import json, sys
agent, said, walks = None, "", []
for line in open(sys.argv[1], errors="replace"):
    try: e = json.loads(line)
    except ValueError: continue
    if e.get("kind") == "agent": agent = e
    if e.get("kind") == "say" and e.get("text") and e.get("source") == "agent": said = e["text"]
    if e.get("kind") == "walk": walks.append(e.get("target", ""))
out = " ".join(str((agent or {}).get("outcome", "no agent result")).split())
if not (agent or {}).get("turns"):  # stopped at the limit: no turns recorded
    out = f"{out}; {len(walks)} walks, last to '{walks[-1] if walks else ''}'; said '{said[:60]}'"
print(f"{len((agent or {}).get('turns', []))} turns | {out[:170]}")
EOF
)
    printf '%s %4ss | %-10s | %s | want %s | %s%s | %s\n' "$ok" "$secs" "$start" "$req" "$want" \
      "$(sed 's/^Town //' <<<"$title")" "${timedout:+ | TIMED OUT}" "$why"
  done 3<"$TRIALS"
done
ob run -- sh -c "pkill -f '[b]in/player-two run'; pkill -f '[t]ests/town'; pkill wtype; true" >/dev/null 2>&1
echo "$pass/$total passed, model cost \$$cost, logs in $OUT"
