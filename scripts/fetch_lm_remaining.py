#!/usr/bin/env python3
"""One finite asset group; independent failures do not suppress other assets."""
import argparse
from pathlib import Path
from tidegraph.lm.assets import fetch, ready
from tidegraph.experiment.records import write_json, now

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--group',choices=('dense-moe-small','qwen-moe'),required=True)
args=p.parse_args()
(args.root/'recovery').mkdir(exist_ok=True)
keys=('olmoe','qwen7') if args.group=='dense-moe-small' else ('qwenmoe',)
results=[]
for key in keys:
    if ready(key,args.root/'models'):
        results.append(dict(model=key,state='already_complete'));continue
    try:
        fetch(key,args.root/'models',args.root,project_gib=160,account_gib=300,chunk_mib=32,retries=6)
        results.append(dict(model=key,state='complete'))
    except Exception as error:
        results.append(dict(model=key,state='failed',error=str(error)[:1000]))
        print(f'{key}: failed: {error}',flush=True)
    write_json(args.root/'recovery'/('assets-'+args.group+'.json'),dict(finished=now(),results=results))
raise SystemExit(1 if any(x['state']=='failed' for x in results) else 0)
