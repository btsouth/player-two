"""Summarize where the last voice command's time went, from a session event log on stdin."""

import json
import sys

agent = None
for line in sys.stdin:
    if '"kind": "agent"' in line:
        agent = json.loads(line)
turns = (agent or {}).get("turns", [])
total = {k: sum(t.get(k, 0) for t in turns) for k in ("model_s", "act_s", "look_s")}
print(f"{len(turns)} turns, model {total['model_s']:.1f}s, act {total['act_s']:.1f}s, "
      f"look {total['look_s']:.1f}s")
