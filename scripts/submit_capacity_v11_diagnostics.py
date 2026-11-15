#!/usr/bin/env python3
"""Submit finite diagnostics and fresh data preparation as durable services."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import verify_source

def main(a):
    root=a.root.resolve();source=a.source.resolve();sid=verify_source(source)
    base=source/'configs/studies/language/capacity-v11';table=base/'diagnostics.json'
    budget=json.loads((base/'budget.json').read_text());deadline=budget['deadline_epoch']
    control=root/'runs/v11-diagnostic-control';control.mkdir(parents=True,exist_ok=False)
    state=dict(state='submitting',source_commit=sid['commit'],jobs=[],started_at=now());atomic_json(control/'state.json',state)
    def submit(job,device,argv,inputs,threads=4,cap=7200):
        cap=min(cap,int(deadline-time.time()-60));dest=control/job;dest.mkdir()
        baseargv=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
            '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
            '--env','TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'),'--device',device,'--threads',str(threads),
            '--memory-gib','64','--timeout',str(cap)]
        if device=='npu':
            policy=json.loads((base/'resource.json').read_text());policy['minimum_free_hbm_mib']=48000
            rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
            baseargv+=['--env','TIDE_LAB_RESOURCE_POLICY='+str(rp),'--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(cap-300)]
            inputs=[*inputs,rp]
        for p in inputs:baseargv+=['--input',str(p)]
        baseargv+=['--','python',*argv]
        row=dict(job=job,state='submitting',argv=baseargv);state['jobs'].append(row);atomic_json(control/'state.json',state)
        r=subprocess.run(baseargv,capture_output=True,text=True);atomic_json(dest/'submission.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
        row.update(state='submitted',at=now());atomic_json(control/'state.json',state);print(job,flush=True)
    submit('v11-confirmation-data','cpu',['scripts/prepare_capacity_v11_confirmation.py','--root',str(root),
        '--output',str(root/'runs/v11-confirmation-data'),'--deadline-epoch',str(deadline)],
        [root/p for p in ('runs/v9-data-r3/data-lock.json','runs/v6-data-final/data-lock.json','runs/lm-data-v1/tokens/tokenizer-a.json','runs/lm-data-v1/documents/documents.jsonl')],cap=1800)
    for i,c in enumerate(json.loads(table.read_text())['cases']):
        submit('v11-diag-'+c['id'],'npu',['scripts/diagnose_capacity_v11.py','--root',str(root),'--table',str(table),
            '--index',str(i),'--data',str(root/'runs/v9-data-r3'),'--output',str(root/'runs'/('v11-diag-'+c['id'])),
            '--lens-cli',str(a.lens_cli.resolve()),'--deadline-epoch',str(deadline)],
            [table,root/c['config'],root/c['checkpoint'],root/'runs/v9-data-r3/data-lock.json',*[root/p for p in c['history'].values()]],threads=4,cap=14400)
    state.update(state='submitted',finished_at=now());atomic_json(control/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
