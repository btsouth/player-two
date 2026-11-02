"""A reviewed-source swap keeps originals intact and the manifest portable for rsync."""
import json
import subprocess
import sys
from pathlib import Path


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
