#!/usr/bin/env python3
"""Read-only endpoint audit; failed post-training diagnostics stay failed."""
import argparse,json,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_moe_results import endpoint
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.records import digest,now

def read(p):return json.loads(p.read_text()) if p.is_file() else {}

def audit(root):
 rows=[];jobs=[];signatures={};paths={};train_sha=digest(root/'runs/v9-data-r3/data-lock.json');conf_sha=digest(root/'runs/v12-data/data-lock.json')
 for p in sorted((root/'artifacts/jobs').glob('v12-*/job.json')):
  ob=inspect(p);jobs.append(dict(id=p.parent.name,state=ob['state']['state'],terminated=ob['termination_verified_now']))
 for p in sorted((root/'runs').glob('v12-*/confirmation/result.json')):
  cr=read(p);out=p.parent.parent;train=out/'train';tr=read(train/'result.json');m=read(train/'manifest.json');c=m['config'];cid=c['id']
  cfg=root/'configs/studies/language/hb-v12/configs'/(cid+'.json');identity=tr['identity'];cp=train/'latest.pt';ck=read(train/'checkpoint.json')
  if not (tr['state']=='completed' and tr['progress']['tokens']==128*2**20 and tr['inference']['state']=='completed'
    and c==read(cfg) and digest(cfg)==identity['config_sha256'] and identity['data_lock_sha256']==train_sha
    and m['identity']==identity and cr['training_identity']==identity and cr['exposure']==128*2**20
    and cr['state']=='completed' and cr['confirmation_lock_sha256']==conf_sha and not cr['test_access']):
   raise ValueError('endpoint/config/data mismatch: '+cid)
  sha=digest(cp)
  if sha!=ck['sha256'] or sha!=cr['checkpoint_sha256']:raise ValueError('checkpoint hash drift: '+cid)
  values={}
  for split,f,n in [('validation',train/'validation-final.json',131072),('confirmation',p.with_name('confirmation.json'),262144)]:
   val,sig=endpoint(f,n)
   if split in signatures and sig!=signatures[split]:raise ValueError('evaluation blocks mismatch')
   signatures[split]=sig;values[split]=dict(nll=val['nll'],record=str(f.relative_to(root)),sha256=digest(f),work=val['work'])
  recovery=[]
  for f in (out/'recovery.json',out/'trained-compare/recovery.json',out/'recovery-compare/recovery.json'):
   r=read(f)
   if r:recovery.append(dict(record=str(f.relative_to(root)),state=r['state'],scope=r.get('scope'),identity=r.get('identity')))
  if not recovery or any(r['state']!='passed' for r in recovery):raise ValueError('endpoint recovery missing: '+cid)
  lens={}
  for stage in ('train','confirmation'):
   r=read(out/stage/'projection.json')
   if r.get('status')!='verified' or not r.get('readback_matches'):raise ValueError('Lens not verified: '+cid+'/'+stage)
   lens[stage]={k:r.get(k) for k in ('status','readback_matches','destination_name','destination_run_id','source_run_id')}
  w=values['confirmation']['work'];dg=read(out/'diagnostic/result.json');checks=read(out/'check/result.json').get('checks',{})
  shape=cid.rsplit('-s',1)[0];paths[(shape,c['seed'])]=p.with_name('confirmation.json')
  rows.append(dict(id=cid,shape=shape,seed=c['seed'],job=out.name,config=str(cfg.relative_to(root)),
   checkpoint=str(cp.relative_to(root)),checkpoint_sha256=sha,bytes=cp.stat().st_size,identity=identity,
   parameters=tr['inventory']['trainable_parameters'],exposure=tr['progress']['tokens'],inference=tr['inference']['state'],
   validation=values['validation'],confirmation=values['confirmation'],recovery=recovery,lens=lens,
   full_all=w['work/full']/w['work/node_slots'],receipt_all=w['work/candidates']/w['work/node_slots'],
   active_parameters=w['work/active_parameter_uses']/w['work/prediction_positions'],
   diagnostic=dict(state=dg.get('state','missing'),error=dg.get('error'),record=str((out/'diagnostic/result.json').relative_to(root))),
   strict_forward=checks.get('forward'),strict_vjp_failed=sum(v.get('state')=='failed' for v in checks.get('parameter_vjp',{}).values())))
 if len(rows)!=38 or len({r['id'] for r in rows})!=38:raise ValueError('38 distinct128Mi endpoints required')
 means=[]
 for shape in sorted({r['shape'] for r in rows}):
  rs=[r for r in rows if r['shape']==shape]
  if {r['seed'] for r in rs}!={11,22}:raise ValueError('paired seeds missing')
  means.append(dict(shape=shape,parameters=rs[0]['parameters'],seed_nll={str(r['seed']):r['confirmation']['nll'] for r in rs},
   mean_nll=statistics.mean(r['confirmation']['nll'] for r in rs),full_all=statistics.mean(r['full_all'] for r in rs),
   active_parameters=statistics.mean(r['active_parameters'] for r in rs)))
 comparisons=[]
 pairs=[('share8-q3','base-q3'),('share0-q3','base-q3'),('fullshare8-q3','base-q3'),('share8-d128-q3','base128-q3'),
  ('share8-d192-q3','share8-d128-q3'),('readcontent-q3','base-q3'),('base128-q3','base-q3')]
 for left,right in pairs:
  for seed in (11,22):
   r=paired(paths[(left,seed)],paths[(right,seed)]);r.update(left=str(paths[(left,seed)].relative_to(root)),right=str(paths[(right,seed)].relative_to(root)),
    scope='fresh v12 confirmation, document-bootstrap conditional on fixed seed/checkpoints; not selection/seed uncertainty',left_shape=left,right_shape=right,seed=seed);comparisons.append(r)
 controls=[]
 for p in sorted((root/'runs').glob('v12-control-*/result.json')):
  r=read(p)
  if r['state']!='completed':
   controls.append(dict(id=p.parent.name.removeprefix('v12-control-'),state=r['state'],error=r.get('error'),record=str(p.relative_to(root))))
   continue
  v,sig=endpoint(p.with_name('confirmation.json'),262144)
  if r['state']!='completed' or r['confirmation_lock_sha256']!=conf_sha or sig!=signatures['confirmation']:raise ValueError('historical control not matched')
  controls.append(dict(id=p.parent.name.removeprefix('v12-control-'),state='completed',nll=v['nll'],exposure=r['exposure'],checkpoint=r['checkpoint'],
   checkpoint_sha256=r['checkpoint_sha256'],training_identity=r['training_identity'],record=str(p.relative_to(root))))
 return dict(schema='tide-hb-v12-terminal-review-v1',state='reviewed_with_limits',observed_at=now(),
  training_lock_sha256=train_sha,confirmation_lock_sha256=conf_sha,rows=rows,means=means,comparisons=comparisons,controls=controls,
  jobs=jobs,all_services_terminated=all(r['terminated'] for r in jobs),diagnostic_counts=dict(Counter(r['diagnostic']['state'] for r in rows)),
  limits=['38 completed128Mi training/confirmation endpoints are not38 successful original services.',
   'Original numerical failures and administrative pauses remain;8 historical diagnostic directory-read failures are not repaired.',
   'Direct implementation only, dense physical Full, scoped language findings; numerical equivalence and general TIDE scaling not established.',
   'Fresh confirmation is now seen evidence. Test remains sealed. Small paired bootstrap omits seed/model-selection uncertainty.'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();r=audit(a.root.resolve());write_review_json(a.output,r);print(json.dumps({k:r[k] for k in ('state','all_services_terminated','diagnostic_counts','means')},indent=2))
