#!/usr/bin/env python3
"""Select a reviewed recorder folder without copying or rewriting its original files."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('takes',type=Path)
    p.add_argument('name',choices=['take-A','take-B','goodbye','game2','nav'])
    p.add_argument('source',type=Path)
    p.add_argument('--delay',type=float,required=True,help='Measured recording launch delay in seconds')
    p.add_argument('--shop-menu',nargs=2,type=float,help='Measured source-video menu interval for A')
    p.add_argument('--session-log',type=Path,help='Original session log supplying missing ready metadata')
    p.add_argument('--clip',nargs=2,type=float,help='Reviewed source-video interval for the walking slot')
    a=p.parse_args()
    source=a.source.resolve()
    for name in ['run.mp4','run.mp4.start','events.jsonl','run.game.wav','run.voice.wav']:
        if not (source/name).is_file():
            p.error(f'Missing source file: {source/name}')
    if a.name=='take-A' and (not a.shop_menu or a.shop_menu[0]>=a.shop_menu[1]):
        p.error('A selection requires a measured --shop-menu START END')
    if a.name=='nav' and (not a.clip or not 0<=a.clip[0]<a.clip[1]):
        p.error('Navigation selection requires a reviewed --clip START END')
    prefix=b''
    if a.session_log:
        epoch=float((source/'run.mp4.start').read_text())
        ready=[line for line in a.session_log.read_text().splitlines()
               if (e:=json.loads(line)).get('kind')=='ready' and e.get('t',0)<=epoch]
        if not ready:
            p.error('No original ready record before this recording')
        prefix=(ready[-1]+'\n').encode()
    manifest=a.takes/'takes.json'
    data=json.loads(manifest.read_text()) if manifest.exists() else {}
    selected='selected-'+a.name
    alias=a.takes/selected
    if alias.exists() and not alias.is_symlink():
        p.error(f'Preserve existing real folder: {alias}')
    temp=a.takes/(selected+'.next')
    if temp.exists() or temp.is_symlink():
        p.error(f'Previous selection still present: {temp}')
    temp.symlink_to(source,target_is_directory=True)
    os.replace(temp,alias)
    config=data.setdefault(a.name,{})
    config.update(video=selected+'/run.mp4',start=selected+'/run.mp4.start',events=selected+'/events.jsonl')
    if prefix:
        derived=selected+'.events.jsonl'
        (a.takes/derived).write_bytes(prefix+(source/'events.jsonl').read_bytes())
        config['events']=derived
    edit=config.setdefault('edit',{})
    edit['recording_delay']=a.delay
    if a.shop_menu:
        edit['shop_menu']=a.shop_menu
    if a.clip:
        edit['clip']=a.clip
    config.pop('reject',None)
    staging=manifest.with_suffix('.json.next')
    staging.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(staging,manifest)
    print(f'{a.name}: {source}; originals unchanged. Rebuild after inspecting the plan.')


if __name__=='__main__':
    main()
