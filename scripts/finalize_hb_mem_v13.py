#!/usr/bin/env python3
"""Resolve effective batches and explicit historical versus fresh qry controls."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest
from scripts.prepare_direct_v10 import finalize

def main():
 root=Path.cwd();base=root/'configs/studies/language/hb-mem-v13';table=json.loads((base/'table.json').read_text());extra=[];reuse=[]
 for row in table['cases']:
  group=row['group'];fresh=group in ('q10-l','pool-l');cp=root/'runs'/('v13-cal-'+('r2-' if fresh else '')+group)/'result.json'
  cal=json.loads(cp.read_text());projection=json.loads(cp.with_name('projection.json').read_text())
  if cal['state'] not in ('completed','failed') or projection.get('status')!='verified' or not projection.get('readback_matches'):raise ValueError('calibration not terminal/Lens verified')
  if cal['state']=='failed' and not str(cal.get('error','')).startswith('OutOfMemoryError'):raise ValueError('unhandled calibration failure')
  c=json.loads((root/row['config']).read_text())
  for mode in (('sum','qry') if fresh else ('sum',)):
   good=[v for v in cal['cases'] if v['state']=='completed' and v['peak_mib']<=34000
    and ((v['implementation']=='qry')==(mode=='qry'))]
   if not good:raise ValueError('no completed finite capacity-safe observation: '+group+mode)
   fastest=max(v['targets_per_second'] for v in good)
   # Smallest batch within5% of best avoids spending HBM for negligible gain.
   chosen=min((v for v in good if v['targets_per_second']>=.95*fastest),key=lambda v:(v['micro'],-v['targets_per_second']))
   cfg=copy.deepcopy(c);cfg['training'].update(batch=chosen['micro'],eval_batch=min(16,chosen['micro']))
   if mode=='sum':cfg['implementation']['sum_implementation']=chosen['implementation']
   else:
    cfg.update(id=f'{group}-qry-s{row["seed"]}');cfg['mem_comparison']['mem']='qry'
    for mem in cfg['direct']['plan']['memories']:mem['memory_read']='qry'
    cfg['implementation'].pop('sum_implementation',None);cfg.pop('planned_inventory',None);finalize(cfg)
   cfg['resource_admission']={k:chosen[k] for k in ('micro','global_batch','targets_per_second','peak_mib')}
   cfg['resource_admission'].update(source=str(cp.relative_to(root)),sha256=digest(cp),calibration_state=cal['state'],
    bounded_choice='completed finite rows only, peak<=34000MiB; smallest batch within5% of measured best; failed larger batch remains failed',
    strict_numerical='see separate qualification; finite exploration')
   path=base/'configs'/(cfg['id']+'.json');atomic_json(path,cfg)
   if mode=='qry':extra.append(dict(row,id=cfg['id'],mem='qry',config=str(path.relative_to(root)),
    reason='conditional v10 control: old native quotient/allocator or mixed repaired ancestry not used as same-source single-Mem comparison'))
  if not fresh:
   qry=json.loads((root/row['qry_config']).read_text());matches=[]
   for p in (root/'runs').glob('v12-*/confirmation/result.json'):
    r=json.loads(p.read_text())
    if r.get('state')=='completed' and r.get('config_sha256')==digest(root/row['qry_config']):matches.append((p,r))
   if len(matches)!=1:raise ValueError('qry endpoint ambiguous')
   p,r=matches[0];checkpoint=Path(r['checkpoint'])
   if digest(checkpoint)!=r['checkpoint_sha256']:raise ValueError('qry checkpoint drift')
   reuse.append(dict(id=f'{group}-qry-s{row["seed"]}',group=group,seed=row['seed'],mem='qry',config=row['qry_config'],
    checkpoint=str(checkpoint.relative_to(root)),sha256=r['checkpoint_sha256'],training_identity=r['training_identity'],
    original_confirmation=str(p.relative_to(root)),original_job=p.parent.parent.name,
    scope='same original qry parameter/function/recipe; retain historical numerical source/repair provenance; fresh evaluator source recorded, not exact trajectory equivalence'))
 table['cases']+=extra;table.update(calibration_resolved=True,qry_reuse=reuse,new_sum_chains=12,new_qry_chains=4,
  preserved_calibration_failures=[str(p.relative_to(root)) for p in (root/'runs').glob('v13-cal-*/result.json') if json.loads(p.read_text()).get('state')=='failed'])
 if len(table['cases'])!=16 or len(reuse)!=8:raise ValueError('bounded table changed')
 atomic_json(base/'table.json',table)

if __name__=='__main__':main()
