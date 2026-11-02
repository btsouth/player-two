"""The recorder must include both voices, or refuse an incomplete voice stem."""
import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/demo_record.sh"


def run_record(tmp_path, apps, mic=None):
    gsr = tmp_path / "gpu-screen-recorder"
    gsr.write_text("#!/usr/bin/env python3\nimport os,sys,json\n"
                   "if '--list-application-audio' in sys.argv: print(os.environ['APPS'])\n"
                   "else: open(os.environ['ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n")
    gsr.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", APPS=apps,
               ARGS=str(tmp_path / "args.json"))
    env.pop("DEMO_MIC", None)
    if mic is not None:
        env["DEMO_MIC"] = mic
    result = subprocess.run([str(SCRIPT), "TEST", str(tmp_path / "take.mp4")],
                            env=env, capture_output=True, text=True, check=False)
    args = json.loads((tmp_path / "args.json").read_text()) if (tmp_path / "args.json").exists() else []
    return result, args


def tracks(args):
    return [args[i + 1] for i, word in enumerate(args) if word == "-a"]


def test_default_mic_goes_to_full_mix_and_voice_stem(tmp_path):
    result, args = run_record(tmp_path, "PipeWire ALSA [python3.14]")
    assert result.returncode == 0
    assert tracks(args) == ["default_output|default_input",
                           "app-inverse:PipeWire ALSA [python3.14]",
                           "app:PipeWire ALSA [python3.14]|default_input"]


def test_typed_rehearsal_excludes_mic_from_both_tracks(tmp_path):
    result, args = run_record(tmp_path, "PipeWire ALSA [python3.14]", "0")
    assert result.returncode == 0
    assert tracks(args) == ["default_output", "app-inverse:PipeWire ALSA [python3.14]",
                           "app:PipeWire ALSA [python3.14]"]


def test_no_player_stream_refuses_mic_only_voice_stem(tmp_path):
    result, args = run_record(tmp_path, "")
    assert result.returncode == 1 and not args
    assert "let it speak" in result.stderr
    assert not (tmp_path / "take.mp4.start").exists()
