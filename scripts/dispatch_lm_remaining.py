#!/usr/bin/env python3
"""Wait for finite asset groups, then submit independent immutable compute queues."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from tidegraph.lm.assets import ready
from tidegraph.lm.recovery import manifest
from tidegraph.experiment.records import write_json, read_json, fingerprint, now
from tidegraph.lm.selection import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--attempt',required=True)
    p.add_argument('--small-job',required=True);p.add_argument('--moe-job',required=True)
    p.add_argument('--wait-hours',type=float,default=30)
    p.add_argument('--manifest',type=Path,help='explicit finite selection instead of the old recovery matrix')
    p.add_argument('--npu-helper',type=Path,help='reviewed site scheduler or explicit sharing adapter')
    args=p.parse_args()
    if not 0<args.wait_hours<=30:p.error('dependency wait hours must be in (0,30]')
    root=args.root.resolve();out=root/'recovery'/args.attempt;out.mkdir(exist_ok=False)
    selection=validate(args.manifest) if args.manifest else None
    plans={key:manifest(key) for key in ('ready','small','moe')}
    if selection:
        for key in plans:
            cases=[x for x in selection['cases'] if
                   ('small' if x['config']['model']=='qwen7' else 'moe' if x['config']['model']=='qwenmoe' else 'ready')==key]
            plans[key]={**selection,'cases':cases,'identity':fingerprint(cases)}
    groups={key:(sorted({x['config']['model'] for x in plans[key]['cases']}),
                 args.small_job if key=='small' else args.moe_job if key=='moe' else None)
            for key in plans if plans[key]['cases']}
    state=dict(schema='tide-lm-dispatch-v1',started=now(),state='waiting',submitted=[],pending=list(groups))
    def save():write_json(out/'dispatch.json',state)
    save();start=time.monotonic()
    try:
        while state['pending'] and time.monotonic()-start<args.wait_hours*3600:
            for group in list(state['pending']):
                keys,job=groups[group]
                complete=all(ready(key,root/'models') for key in keys)
                status_path=root/'jobs'/job/'status.json' if job else None
                status=read_json(status_path) if status_path and status_path.exists() else {}
                terminal=status.get('state') in {'passed','failed','cancelled','timed_out','timeout'}
                if not complete and not terminal:continue
                value=plans[group];path=out/(group+'.json');write_json(path,value)
                for backend in ('cpu','npu'):
                    if not any(x['config']['execution']['backend']==backend for x in value['cases']):continue
                    label=f'{backend}-{group}-{args.attempt}'
                    cmd=[sys.executable,'-B','scripts/submit_lm_job.py','--unit','gef-lm-'+label,
                         '--output',str(root/'jobs'/label),'--seconds','86400','--memory-gib','384','--',
                         '/bin/bash','scripts/run_lm_local.sh',str(root),'queue','--root',str(root),
                         '--backend',backend,'--manifest',str(path),'--attempt',group+'-'+args.attempt,'--hours','23.5']
                    if backend=='npu':cmd+=['--npu-helper',str(args.npu_helper or root/'site/run_on_free_npu.py')]
                    subprocess.run(cmd,check=True)
                    state['submitted'].append(dict(group=group,backend=backend,unit='gef-lm-'+label+'.service',
                                                   job=str(root/'jobs'/label),time=now()))
                    save()
                state['pending'].remove(group);save()
            if state['pending']:time.sleep(30)
        state.update(state='submitted' if not state['pending'] else 'dependency_deadline',finished=now());save()
    except BaseException as error:
        state.update(state='failed',error=str(error)[:2000],finished=now());save();raise
    return 0 if not state['pending'] else 1


if __name__=='__main__':raise SystemExit(main())
