#!/usr/bin/env python3
"""One finite asset group; independent failures do not suppress other assets."""
import argparse
from pathlib import Path
import time
from tidegraph.lm.assets import fetch, ready
from tidegraph.experiment.records import write_json, read_json, now

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--group',choices=('dense-moe-small','qwen-moe'),required=True)
p.add_argument('--chunk-mib',type=int,default=32)
p.add_argument('--retries',type=int,default=6)
p.add_argument('--wait-for-job',help='serialize model downloads after this durable job is terminal')
p.add_argument('--attempt',default='',help='distinct summary identity; asset history is also retained')
args=p.parse_args()
if not 1<=args.chunk_mib<=64 or not 0<=args.retries<=10:p.error('chunk-mib 1..64 and retries 0..10 required')
(args.root/'recovery').mkdir(exist_ok=True)
summary=args.root/'recovery'/('assets-'+args.group+('-'+args.attempt if args.attempt else '')+'.json')
if args.wait_for_job:
    started=time.monotonic()
    write_json(summary,dict(state='waiting_dependency',job=args.wait_for_job,started=now()))
    while time.monotonic()-started<30*3600:
        dependency=args.root/'jobs'/args.wait_for_job/'status.json'
        if dependency.exists() and read_json(dependency).get('state') in {'passed','failed','cancelled','timeout','timed_out'}:break
        time.sleep(30)
    else:raise TimeoutError('30h download dependency limit')
keys=('olmoe','qwen7') if args.group=='dense-moe-small' else ('qwenmoe',)
results=[]
for key in keys:
    if ready(key,args.root/'models'):
        results.append(dict(model=key,state='already_complete'));continue
    try:
        fetch(key,args.root/'models',args.root,project_gib=160,account_gib=300,
              chunk_mib=args.chunk_mib,retries=args.retries)
        results.append(dict(model=key,state='complete'))
    except Exception as error:
        results.append(dict(model=key,state='failed',error=str(error)[:1000]))
        print(f'{key}: failed: {error}',flush=True)
    write_json(summary,dict(finished=now(),results=results))
write_json(summary,dict(finished=now(),results=results))
raise SystemExit(1 if any(x['state']=='failed' for x in results) else 0)
