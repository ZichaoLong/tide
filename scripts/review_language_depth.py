#!/usr/bin/env python3
"""Read-only workload inspection; publish only human-review evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import digest,now
from tide_lab.supervisor import verify_source


def review(root):
    config_path=next(root/'configs/studies/language'/name for name in
        ['budget-guard-diagnosis-v5.json','budget-guard-parallel-v5.json','budget-guard-v5.json']
        if (root/'configs/studies/language'/name).exists())
    config=json.loads(config_path.read_text())
    out=dict(schema='tide-depth-handoff-v1',at=now(),project='tide-learning-lab',jobs={},sources={},lens=[],
        axes='execution, strict numerics, discrete semantics, recovery, learning, performance remain separate',
        data_lock_sha256=digest(root/'runs/lm-depth-data-v5/tokens/data-lock.json'),
        resource_decision='docs/evidence/language-depth-resource-decision.json',
        artifact_retirement='docs/evidence/artifact-retirement-v5.json',lens_retirement='docs/evidence/lens-retirement-v5.json')
    inputs={}
    for name in [*config['jobs'],'lm-depth-budget-v5','lm-depth-budget-parallel-v5','lm-depth-budget-diagnosis-v5','lm-a-language-v4']:
        record=root/'artifacts/jobs'/name/'job.json'
        if not record.exists():continue
        job=json.loads(record.read_text());life=inspect(record);source=job['source']['path']
        if source not in out['sources']:
            s=verify_source(source);out['sources'][source]=dict(commit=s['commit'],tree_sha256=s['content']['tree_sha256'])
        for item in [job['environment_script'],*job['inputs']]:inputs[item['path']]=item['sha256']
        if job.get('queue_helper'):inputs[job['queue_helper']['path']]=job['queue_helper']['sha256']
        row=dict(lifecycle=life,source_commit=job['source']['commit'],device=job['device'],resources=job['resources'])
        base=root/'runs'/name
        if (base/'stage.json').exists():
            chain=json.loads((base/'stage.json').read_text());row.update(chain=chain,status=chain['state'])
            for step in chain['steps']:
                folder=Path(step['output']);manifest=folder/'manifest.json';projection=folder/'projection.json'
                if projection.exists():
                    pr=json.loads(projection.read_text());m=json.loads(manifest.read_text())
                    out['lens'].append(dict(job_id=name,study_id=m['study_id'],run_id=m['run_id'],stage=m['stage'],
                        projection=str(projection.relative_to(root)),**{k:pr.get(k) for k in ['destination_name','destination_run_id','status','event_count','readback_matches']}))
                if step['state']=='running' and (folder/'metrics.jsonl').exists():
                    text=(folder/'metrics.jsonl').read_text();lines=text.splitlines()
                    if not text.endswith('\n'):lines=lines[:-1]
                    row['latest_metrics']=json.loads(lines[-1]) if lines else None
        else:
            wait=root/'runs'/(name+'-wait')/'state.json'
            row['dependency_wait']=json.loads(wait.read_text()) if wait.exists() else None
            waiting=(row['dependency_wait'] or {}).get('state')=='waiting_dependency' or (life.get('queue') or {}).get('status','').startswith('waiting')
            row['status']='queued' if waiting else life['state']['state']
        out['jobs'][name]=row
    mismatches=[p for p,h in inputs.items() if digest(p)!=h]
    if mismatches:raise ValueError('active/frozen input changed: '+str(mismatches))
    out['fixed_inputs']=dict(unique_files=len(inputs),all_hashes_match=True)
    lock=json.loads((root/'dependencies/installation.lock.json').read_text())['installed_identity']['package']
    bad=[name for name,h in lock['files_sha256'].items() if digest(Path(lock['path'])/name)!=h]
    if bad:raise ValueError('installed foundation changed: '+str(bad))
    out['installed_foundation']=dict(files_checked=len(lock['files_sha256']),all_hashes_match=True)
    guard=next((root/'runs'/name/'state.json' for name in
        ['lm-depth-budget-diagnosis-v5','lm-depth-budget-parallel-v5','lm-depth-budget-v5']
        if (root/'runs'/name/'state.json').exists()),root/'runs/lm-depth-budget-v5/state.json')
    out['budget']=json.loads(guard.read_text()) if guard.exists() else config
    p=root/'runs/lm-depth-data-v5/projection.json';pr=json.loads(p.read_text())
    out['data_projection']={k:pr.get(k) for k in ['source_run_id','project','destination_name','destination_run_id','status','readback_matches']}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,default=Path('docs/evidence/language-depth-handoff.json'))
    a=p.parse_args();r=review(a.root);write_review_json(a.output,r,root=a.root)
    print('observed',r['at'],'fixed',r['fixed_inputs'],'installed',r['installed_foundation'])
    for name,row in r['jobs'].items():print(name,row['status'])
