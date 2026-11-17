#!/usr/bin/env python3
"""Read-only live Mem study view; never launches or changes run records."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import now

def read(p):return json.loads(p.read_text()) if p.is_file() else {}

def review(root):
 base=root/'configs/studies/language/hb-mem-v13';t=read(base/'table.json');rows=[]
 for case in t['cases']:
  job='v13-main-'+case['id'];out=root/'runs'/job;record=root/'artifacts/jobs'/job/'job.json'
  ob=inspect(record) if record.is_file() else {};w=read(out.with_name(job+'-worker')/'state.json')
  stages=w.get('cases',[{}])[-1].get('stages',[]) if w.get('cases') else [];stage=stages[-1]['name'] if stages else None
  progress=read(out/(stage or 'train')/'progress.json');result=read(out/(stage or 'train')/'result.json')
  projection=read(out/(stage or 'train')/'projection.json');cfg=read(root/case['config'])
  rows.append(dict(id=case['id'],group=case['group'],mem=case['mem'],seed=case['seed'],job=job,config=case['config'],
   source=read(record).get('source',{}).get('commit'),execution=ob.get('state',{}).get('state','not_submitted'),
   terminated=ob.get('termination_verified_now'),device=(ob.get('queue') or {}).get('physical_id'),stage=stage,
   stage_state=result.get('state'),tokens=progress.get('tokens',result.get('progress',{}).get('tokens')),
   error=result.get('error',w.get('error')),lens=projection.get('status'),lens_name=projection.get('destination_name'),
   physical_batch=cfg['training']['batch'],global_batch=cfg['training']['global_batch'],parameters=cfg['planned_inventory']['lm_trainable'],
   recovery=read(out/'trained-compare/recovery.json').get('state'),confirmation=read(out/'confirmation/result.json').get('state')))
 reused=[]
 for case in t.get('qry_reuse',[]):
  job='v13-reuse-'+case['id'];p=root/'artifacts/jobs'/job/'job.json';ob=inspect(p) if p.is_file() else {};r=read(root/'runs'/job/'result.json')
  reused.append(dict(id=case['id'],job=job,execution=ob.get('state',{}).get('state','not_submitted'),
   terminated=ob.get('termination_verified_now'),device=(ob.get('queue') or {}).get('physical_id'),
   state=r.get('state'),nll=r.get('nll'),error=r.get('error')))
 return dict(schema='tide-hb-mem-v13-review-v1',observed_at=now(),budget=read(base/'budget.json'),
  counts=dict(Counter(r['execution'] for r in rows)),new_training=rows,reused_qry_confirmation=reused,
  scientific_state='awaiting terminal paired review; execution is not strict numerical or quality acceptance')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path)
 a=p.parse_args();r=review(a.root.resolve())
 if a.output:write_review_json(a.output,r)
 else:print(json.dumps(r,indent=2,ensure_ascii=False))
