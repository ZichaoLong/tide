#!/usr/bin/env python3
"""Resolve the authorized six Mem pairs; no language-quality-based selection."""
import copy,datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest

GROUPS={
 'q10-s':('hb-v12/configs/base-q10-s{seed}.json','quad'),
 'q3-s':('hb-v12/configs/base-q3-s{seed}.json','quad'),
 'q10-l':('direct-v10/table/main/hb-d128-r28-q10-s{seed}.json','quad'),
 'q3-l':('hb-v12/configs/base128-q3-s{seed}.json','quad'),
 'pool-l':('direct-v10/table/main/hb-d128-r64-pool-s{seed}.json','pool'),
 'content-s':('hb-v12/configs/readcontent-q3-s{seed}.json','quad'),
}

def main():
 from scripts.prepare_direct_v10 import finalize
 root=Path.cwd();base=root/'configs/studies/language/hb-mem-v13';base.mkdir(exist_ok=False)
 (base/'configs').mkdir();rows=[]
 start=datetime.datetime.fromisoformat('2026-11-17T17:44:00+08:00');deadline=start+datetime.timedelta(hours=48)
 budget=dict(schema='tide-hb-mem-v13-budget-v1',authorized='2026-11-17 user approval of hb-sum-qry-proposal',
  start=start.isoformat(),deadline=deadline.isoformat(),deadline_epoch=deadline.timestamp(),
  max_sum_chains=12,max_conditional_qry_chains=4,target_tokens=128*2**20,max_parameters=200000000,
  max_new_concurrent_npus=16,aggregate_compute_cap=None,account_hard_limit_gib=700,
  planned_new_disk_gib=80,filesystem_low_water_gib=32,account_checkpoint_reserve_gib=32,
  no_automatic_retry=True,original_feasibility_threshold=2e-5)
 atomic_json(base/'budget.json',budget)
 policy=json.loads((root/'configs/studies/language/hb-v12/resource.json').read_text())
 policy.update(deadline_epoch=deadline.timestamp(),max_new_concurrent_npus=16,
  latest_authorization='2026-11-17 six Mem comparisons; all healthy shared cards with observed headroom',minimum_free_hbm_mib=56000)
 atomic_json(base/'resource.json',policy)
 for group,(source,route) in GROUPS.items():
  for seed in (11,22):
   parent=root/'configs/studies/language'/source.format(seed=seed);original=json.loads(parent.read_text());c=copy.deepcopy(original)
   c.update(id=f'{group}-sum-s{seed}',campaign='hb-mem-v13',study_id='TL-HB-MEM-SUMQRY',
    comparison_scope='Mem.sum versus matched historical qry; all receiving candidates update memory; Full attention unchanged; different model functions, not equivalent speedup',
    mem_comparison=dict(group=group,mem='sum',qry_config=str(parent.relative_to(root)),qry_config_sha256=digest(parent),route=route))
   c.pop('planned_inventory',None);c.pop('resource_admission',None)
   for mem in c['direct']['plan']['memories']:mem['memory_read']='sum'
   c['implementation']['sum_implementation']='matrix'
   if group in ('q10-l','pool-l'):
    c['implementation']['aggregate_vjp']='ordered-v1'
    c['mem_comparison']['conditional_qry']='new matched qry because historical native quotient/old allocator or mixed repaired ancestry is not a same-source single-Mem control'
   c['training'].update(target_tokens=128*2**20,milestones=[8*2**20,32*2**20,128*2**20],full_probe_batch=4,full_probe_length=32)
   finalize(c);cfg=base/'configs'/(c['id']+'.json');atomic_json(cfg,c)
   rows.append(dict(id=c['id'],group=group,seed=seed,mem='sum',route=route,config=str(cfg.relative_to(root)),
    qry_config=str(parent.relative_to(root)),sum_parameters=c['planned_inventory']['lm_trainable'],qry_parameters=original['planned_inventory']['lm_trainable']))
 atomic_json(base/'table.json',dict(schema='tide-hb-mem-v13-table-v1',cases=rows,calibration_resolved=False,
  selection='implementation and physical microbatch by bounded synthetic cost only; no quality pruning',
  confirmation='fresh262K excluding old locks; old qry endpoints re-evaluated; test sealed'))

if __name__=='__main__':main()
