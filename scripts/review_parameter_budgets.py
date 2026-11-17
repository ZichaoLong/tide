#!/usr/bin/env python3
"""Read existing evaluations only; separate splits and logical parameter budgets."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from scripts.review_moe_results import endpoint
from tide_lab.records import now,digest

def read(p):return json.loads(p.read_text())

def review(root):
 source=root/'docs/evidence/hb-v12-results-20261117.json';old=read(source);rows=[];signature=None
 for r in old['rows']:
  path=root/r['confirmation']['record'];v,sig=endpoint(path,262144)
  if signature is not None and sig!=signature:raise ValueError('different confirmation blocks')
  signature=sig;w=v['work'];pos=w['work/prediction_positions']
  parts={k.removeprefix('work/'):w[k]/pos for k in ('work/full_parameter_uses','work/receive_parameter_uses','work/shared_parameter_uses')}
  active=w['work/active_parameter_uses']/pos
  if abs(sum(parts.values())-active)>1e-5:raise ValueError('parameter partition mismatch')
  rows.append(dict(model=r['shape'],family='HB',seed=r['seed'],parameters=r['parameters'],active_parameters=active,
   parts=parts,nll=v['nll'],confirmation=str(path.relative_to(root)),full_all=r['full_all'],source_commit=r['identity']['source_commit']))
 for c in old['controls']:
  if c['state']!='completed':continue
  path=(root/c['record']).parent;manifest=read(path/'manifest.json');cfg=manifest['config'];v,sig=endpoint(path/'confirmation.json',262144)
  if sig!=signature:raise ValueError('control confirmation blocks differ')
  p=cfg['planned_inventory']['lm_trainable'];w=v['work']
  active=w['work/active_parameter_uses']/w['work/prediction_positions'] if w else p
  rows.append(dict(model=c['id'].rsplit('-s',1)[0],family='MoE' if cfg['kind']=='moe' else 'Transformer' if cfg['kind']=='transformer' else 'HB',
   seed=cfg['seed'],parameters=p,active_parameters=active,nll=v['nll'],confirmation=str((path/'confirmation.json').relative_to(root)),
   source_commit=c['training_identity']['source_commit']))
 means=[]
 for name in sorted({r['model'] for r in rows}):
  rs=[r for r in rows if r['model']==name]
  means.append(dict(model=name,family=rs[0]['family'],parameters=rs[0]['parameters'],active_parameters=statistics.mean(r['active_parameters'] for r in rs),
   nll=statistics.mean(r['nll'] for r in rs),seeds=sorted(r['seed'] for r in rs),
   seed_active_parameters={str(r['seed']):r['active_parameters'] for r in rs},seed_nll={str(r['seed']):r['nll'] for r in rs}))
 by={r['model']:r for r in means};hb=by['share8-d192-q3'];tf=by['parent128-tf-d192-l6']
 small=['parent128-tf-d192-l6','parent128-small-e20-lr0.003-b0.01','balance128-small-e64-lr0.003-b0.01',
  'parent128-hb-d64-r4-dense-match','base-q10','base-q3']
 large=['parent128-large-dense-lr0.001','parent128-large-e20-lr0.001-b0.01','parent128-large-e64-lr0.001-b0.01','base128-q3']
 for name in small+large:
  if by[name]['seeds']!=[11,22]:raise ValueError('paired seeds missing')
 v10=read(root/'docs/evidence/direct-v10-results.json')
 matched_total=[r for r in v10['means'] if r['exposure']==128*2**20 and r['shape'] in
  ('hb-d128-r28-pool','hb-d128-r28-q10','hb-d128-r64-pool','hb-d128-r64-q3')]
 return dict(schema='tide-parameter-budget-review-v1',observed_at=now(),source_record=str(source.relative_to(root)),source_sha256=digest(source),
  scope='Read-only re-tabulation of existing128Mi endpoints. No new model execution, test access, resume, or active-input modification.',
  definition='Average logical parameter participation per model position, including padded positions. HB Full/private and shared-presence, receiving/Read/router, other shared and tied vocabulary parameters. Shared regional matrices counted once when used. Not FLOPs, gradient count, or physical executed work.',
  v12_confirmation_lock_sha256=old['confirmation_lock_sha256'],v12_rows=rows,v12_means=means,
  strongest_low_activity_hb_vs_named_transformer=dict(hb=hb,transformer=tf,active_ratio=hb['active_parameters']/tf['active_parameters'],
   total_ratio=hb['parameters']/tf['parameters'],nll_delta=hb['nll']-tf['nll']),
  near_active_small=[by[n] for n in small],near_active_large=[by[n] for n in large],
  near_total_cross_family=[by['share8-d128-q3'],by['balance128-small-e64-lr0.003-b0.01']],
  same_total_same_graph_v10=matched_total,
  limits=['v12 and v10 confirmation sets differ: compare NLL only within each table.',
   'Near active parameters are not exact matches or equal FLOPs; HB remains dense physical Full with masks.',
   'Historical controls and new v12 HB retain distinct numerical/repair ancestry.',
   'R3/R4/R28/R64 comparisons also change local region size, selection and sometimes FFN recipe.',
   'No matched100M-total or13.2M-active Transformer at this128Mi endpoint; FC evidence is older16Mi validation only.',
   'Two seeds and finite calibrated recipes do not establish architecture-wide scaling laws.'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();r=review(a.root.resolve());write_review_json(a.output,r)
 print(json.dumps(r['strongest_low_activity_hb_vs_named_transformer'],indent=2))
