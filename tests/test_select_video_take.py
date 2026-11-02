"""A reviewed-source swap keeps originals intact and the manifest portable for rsync."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="takes are selected with a symlink swap")


def test_selection_preserves_originals_and_other_takes(tmp_path):
    source=tmp_path/'reviewed take'
    source.mkdir()
    names=['run.mp4','run.mp4.start','events.jsonl','run.game.wav','run.voice.wav']
    for n in names:
        (source/n).write_text('original '+n)
    takes=tmp_path/'takes'
    takes.mkdir()
    (takes/'takes.json').write_text(json.dumps({'game2':{'edit':{'reviewed':True}}}))
    script=Path(__file__).parents[1]/'scripts/select_video_take.py'
    subprocess.run([sys.executable,str(script),str(takes),'take-A',str(source),
                    '--delay','0.44','--shop-menu','20','24'],check=True)
    data=json.loads((takes/'takes.json').read_text())
    assert data['game2']=={'edit':{'reviewed':True}}
    assert data['take-A']['edit']=={'recording_delay':.44,'shop_menu':[20,24]}
    assert (takes/data['take-A']['video']).resolve()==source/'run.mp4'
    assert all((source/n).read_text()=='original '+n for n in names)

    (source/'run.mp4.start').write_text('1000')
    session=tmp_path/'session.jsonl'
    session.write_text(json.dumps({'kind':'ready','t':999,'brain':'muse-spark-1.3'})+'\n')
    before={n:(source/n).read_bytes() for n in names}
    subprocess.run([sys.executable,str(script),str(takes),'take-A',str(source),
                    '--delay','0.42','--shop-menu','21','25','--session-log',str(session)],check=True)
    data=json.loads((takes/'takes.json').read_text())
    derived=(takes/data['take-A']['events']).read_bytes()
    assert json.loads(derived.splitlines()[0])['brain']=='muse-spark-1.3'
    assert derived.split(b'\n',1)[1]==before['events.jsonl']
    assert all((source/n).read_bytes()==before[n] for n in names)


def test_nav_selection_requires_and_saves_the_reviewed_interval(tmp_path):
    source=tmp_path/'walk';source.mkdir();takes=tmp_path/'takes';takes.mkdir()
    for name in ['run.mp4','run.mp4.start','events.jsonl','run.game.wav','run.voice.wav']:
        (source/name).write_text('unchanged')
    script=Path(__file__).parents[1]/'scripts/select_video_take.py'
    command=[sys.executable,str(script),str(takes),'nav',str(source),'--delay','0.4']
    assert subprocess.run(command,check=False,capture_output=True).returncode!=0
    subprocess.run([*command,'--clip','10','22'],check=True)
    entry=json.loads((takes/'takes.json').read_text())['nav']
    assert entry['edit']=={'recording_delay':.4,'clip':[10,22]}
    assert (takes/entry['video']).resolve()==source/'run.mp4'
