# Validation boundaries

Checked Nov 1, 2026. A passing unit suite is not a live game or accessibility acceptance result.

## Installation

A disposable Ubuntu VM installed the README's Linux prerequisites and uv, cloned main at
813f519 and completed `uv tool install --python 3.12 . --excludes uv-excludes.txt`.
`player-two --help` succeeded. Setup with no API key completed and wrote configuration;
it reported a silent microphone. No human listened to its audio and no live capture was tested.
The VM exposed uinput and setup reported controller creation success; no game was controlled.

The same VM then installed d3/repo at 7ba9fe5 and passed the offline smoke script: four packaged
cursor templates loaded, the pixel reader processed a blank frame, and A* found a seven-step
route around a wall. That second installation is upgrade evidence, not another clean machine.
Later documentation/CI-only commits do not change the installed runtime or smoke script.
The VM was deleted after testing.

Windows: install.ps1 was checked against the README. It downloads main, selects Python 3.12,
installs Windows extras and conditionally installs VC runtime/ViGEmBus, then calls setup.
This pass did not run a fresh Windows installer. Earlier Windows VM/input checks are historical
and do not establish current Chrono Trigger gameplay on Windows.

## Models

With the configured Baseten endpoint and `--brain open`, saved-frame narration returned the
visible result/MVP text correctly in 6.0 seconds. Tokens: 2,401 input, 294 output. Estimated
cost $0.0043 at the provider's published rates. One frame is a connectivity/narration check,
not an agent benchmark. A prior 400-completion-token JSON probe returned no usable description.
No Modal run was made in this pass. Default live configuration remains Muse Spark 1.3.

## Automated and live evidence

The latest historical main CI at 813f519 passed: Linux 406 passed/78 skipped; Windows
397 passed/87 skipped. The preceding Windows failure at 3b98a99 was a walker timeout:
396 passed/87 skipped/1 failed. The later main run passed the test.

On devbox, dev 661b176 plus this branch's lint fix finished with **600 passed, 23 skipped,
1 failed in 369.60 seconds**. `test_walk_to_talk_across_a_counter` expected one confirm press
and observed three before arriving. A focused room-suite reproduction passed 12/12 in 9.54
seconds. The failure is timing-sensitive and unresolved; it is not a green full-suite result.
The synthetic simulation integrates wall-clock time and tests exact tile alignment; failed
presses occurred just over a tile boundary. No runtime change or assertion relaxation was made.
Ruff and the offline smoke script passed on devbox. This branch changes docs, CI, a standalone
smoke script and one lint annotation, not live gameplay behavior. The old main counts do not
describe this newer dev build. CI now runs on dev and d3 branches, including the offline smoke check.
Frame tests skip when their private recordings are absent. No tests or timeouts were weakened
in this repo pass. The lint fix adds explicit check=False for an intentionally crashing subprocess
whose signal return code is still asserted.

The README's live gameplay table comes from the Nov 1 afternoon developer run ending at
3b98a99, on selected saves with typed requests and private atlas data. Final spoken results
are TODO-FINAL. No blind player validation or general success rate is claimed.
