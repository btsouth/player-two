"""Drop-in staging stays isolated and refuses unreviewed source clocks."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dropin", Path(__file__).parents[1] / "scripts/video_dropin.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (template / "takes.json").write_text('{"take-B":{"video":"old.mp4"}}')
    (template / "pacing.json").write_text(
        json.dumps({"shots": [{"id": "hook", "take": "take-B", "a": 1, "b": 4}], "exceptions": {}})
    )
    for name in ("new.mp4", "new.mp4.start", "events.jsonl", "new.game.wav", "new.voice.wav"):
        (source / name).write_text("data")
    return source, {
        "template": str(template),
        "takes": {
            "take-B": {
                "video": "new.mp4",
                "start": "new.mp4.start",
                "events": "events.jsonl",
                "edit": {"recording_delay": 0},
            }
        },
    }


def test_new_take_needs_every_window_and_reviewed_overlays(tmp_path):
    source, config = fixture(tmp_path)
    with pytest.raises(ValueError, match="reviewed a/b/overlays"):
        m.stage(source, config, tmp_path / "stage")


def test_staging_keeps_source_and_old_cut_untouched(tmp_path):
    source, config = fixture(tmp_path)
    config["shots"] = {"hook": {"a": 2, "b": 5, "overlays": []}}
    dest = m.stage(source, config, tmp_path / "stage")
    assert json.loads((dest / "pacing.json").read_text())["shots"][0]["a"] == 2
    assert json.loads((Path(config["template"]) / "pacing.json").read_text())["shots"][0]["a"] == 1
    assert (source / "new.mp4").read_text() == "data"


def test_stand_in_cannot_be_staged_under_a_real_cut_name(tmp_path):
    source, config = fixture(tmp_path)
    (source / "STAND-IN.json").write_text("{}")
    with pytest.raises(ValueError, match="TEMP-NOT-FOR-SUBMISSION"):
        m.stage(source, config, tmp_path / "real-cut")


def test_changed_take_requires_its_own_measured_recorder_offset(tmp_path):
    source, config = fixture(tmp_path)
    config["takes"]["take-B"].pop("edit")
    with pytest.raises(ValueError, match="measured recording_delay"):
        m.stage(source, config, tmp_path / "stage")


def test_fallback_does_not_require_an_unrecorded_walk_or_its_limit_audio(tmp_path):
    source, config=fixture(tmp_path)
    config['guide_mode']='fallback'
    config['takes']['guide']={'video':'not-recorded.mp4','start':'missing.start','events':'missing.events'}
    config['voiceovers']={'limit':'missing-walk-limit.wav'}
    config['shots']={'hook':{'a':2,'b':5,'overlays':[]}}
    result=m.stage(source,config,tmp_path/'stage')
    assert json.loads((result/'pacing.json').read_text())['guide_mode']=='fallback'
    assert not (result/'vo-limit.wav').exists()


def test_tonight_checklist_null_windows_fail_with_a_useful_error(tmp_path):
    source,config=fixture(tmp_path)
    config['shots']={'hook':{'a':None,'b':None,'overlays':[]}}
    with pytest.raises(ValueError,match='measured numeric'):
        m.stage(source,config,tmp_path/'stage')


def test_replacing_a_pending_take_clears_only_that_shot(tmp_path):
    source,config=fixture(tmp_path)
    t=Path(config['template'])/'pacing.json';p=json.loads(t.read_text());p['shots'][0]['stand_in']=True
    p['shots'].append({'id':'other','section':'game2','kind':'card','secs':5,'stand_in':True})
    t.write_text(json.dumps(p));config['shots']={'hook':{'a':2,'b':5,'overlays':[]}}
    result=m.stage(source,config,tmp_path/'stage');shots=json.loads((result/'pacing.json').read_text())['shots']
    assert not shots[0]['stand_in'] and shots[1]['stand_in']


def test_byte_identical_dry_run_does_not_clear_a_pending_spoken_take(tmp_path):
    source,config=fixture(tmp_path);template=Path(config['template'])
    (template/'old.mp4').write_text('data')
    p=json.loads((template/'pacing.json').read_text());p['shots'][0]['stand_in']=True
    (template/'pacing.json').write_text(json.dumps(p));config['unchanged_windows']=['take-B']
    result=m.stage(source,config,tmp_path/'stage')
    assert json.loads((result/'pacing.json').read_text())['shots'][0]['stand_in']
