#!/usr/bin/env python3
"""Audit completed MoE endpoints and fixed comparisons, without running models."""
import argparse
from collections import defaultdict,Counter
import datetime
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_moe_controls import review,read
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.records import digest,now
from tide_lab.quota import account_quota

M=2**20


def endpoint(path,tokens):
    v=read(path)
    if not (v.get('state')=='completed' and v.get('tokens')==tokens and not v.get('test_access',True)
            and math.isfinite(v['nll']) and sum(r['tokens'] for r in v['records'])==tokens):
        raise ValueError('incomplete evaluation: '+str(path))
    if not math.isclose(sum(r['nll_sum'] for r in v['records'])/tokens,v['nll'],rel_tol=0,abs_tol=1e-12):
        raise ValueError('evaluation aggregate mismatch')
    return v,sorted((r['block_id'],r['document_id'],r['tokens']) for r in v['records'])


def audit(root):
    obs=review(root);controller=read(root/'runs/moe-controller-r2/state.json')
    if obs['controller']!='completed' or len(obs['cases'])!=66 or any(not r['terminal'] or r['application_state']!='completed' for r in obs['cases']):
        raise ValueError('MoE campaign is not fully terminal')
    if any(r['lens_status']!='verified' or not r['lens_readback_matches'] for r in obs['cases']):
        raise ValueError('missing Lens readback')
    co=inspect(root/'artifacts/jobs/moe-controller-r2/job.json')
    if not co['termination_verified_now']:raise ValueError('controller still alive')
    rel=lambda p:str(Path(p).resolve().relative_to(root))
    data=digest(root/'runs/v9-data-r3/data-lock.json');rows=[];signatures={};paths={};external=[]
    confirmations={r['id']:r for r in controller['phases']['confirm']['cases']}
    for case in controller['phases']['main128']['cases']:
        cid=case['id'];out=Path(case['output']);train=out/'train';c=read(Path(case['config']))
        result=read(train/'result.json');manifest=read(train/'manifest.json');identity=result['identity']
        check=read(out/'check/result.json');recovery=read(out/'recovery.json')
        if (identity['config_sha256']!=digest(case['config']) or identity['data_lock_sha256']!=data
                or manifest['identity']!=identity or manifest['config']!=c
                or result['progress']['tokens']!=128*M or result['inference']['state']!='completed'
                or not check.get('admitted_for_finite_exploration') or recovery['state']!='passed'):
            raise ValueError('endpoint identity/mechanism/recovery mismatch: '+cid)
        identity_without_mode=lambda x:{k:v for k,v in x.items() if k!='mode'}
        if identity_without_mode(recovery['identity'])!=identity_without_mode(identity):raise ValueError('recovery identity mismatch')
        conf=Path(confirmations[cid]['output']);cr=read(conf/'result.json')
        checkpoint=read(train/'checkpoint.json');cp=train/checkpoint['file']
        cp_hash=digest(cp)
        if cp_hash!=checkpoint['sha256'] or cp_hash!=cr['checkpoint_sha256'] or cr['training_identity']!=identity:
            raise ValueError('confirmation/checkpoint identity mismatch: '+cid)
        values={}
        for split,p,n in [('validation',train/'validation-final.json',131072),('confirmation',conf/'confirmation.json',262144)]:
            v,signature=endpoint(p,n)
            if split not in signatures:signatures[split]=signature
            if signature!=signatures[split]:raise ValueError('evaluation blocks differ: '+cid)
            values[split]=dict(nll=v['nll'],tokens=n,record=rel(p),sha256=digest(p),work=v['work'])
            paths[(c['selection_key'],c['seed'],split)]=p
        checks=check['checks'];work=values['validation']['work'];metrics=[json.loads(s)['metrics'] for s in (train/'metrics.jsonl').read_text().splitlines()]
        training=[m for m in metrics if 'train/loss' in m];tail=training[-64:]
        # Summaries are conditional on recorded batches/layers, not global expert death.
        cold=[v for m in tail for k,v in m.items() if k.endswith('/cold_experts')]
        loads=[v for m in tail for k,v in m.items() if k.endswith('/max_load_fraction')]
        curves={}
        for t in (8*M,32*M,128*M):
            v,_=endpoint(train/f'validation-{t}.json',131072);curves[str(t)]=v['nll']
        rows.append(dict(id=cid,shape=c['selection_key'],seed=c['seed'],kind=c['kind'],config=rel(case['config']),
            identity=identity,parameters=result['inventory']['trainable_parameters'],active_parameters=c['planned_inventory']['active_per_token'],
            lr=c['training']['lr'],balance=c['training']['balance_coefficient'],training=c['training'],
            tokens=result['progress']['tokens'],steps=result['progress']['step'],dropped_tail_tokens=result['progress']['dropped_tail_tokens'],
            validation=values['validation'],confirmation=values['confirmation'],validation_curve=curves,
            training_seconds=result['training_seconds'],elapsed_seconds=result['elapsed_seconds'],
            targets_per_second=128*M/result['training_seconds'],
            max_logged_gradient_norm=max(m['train/gradient_norm'] for m in training),
            peak_npu_mib=max(m.get('resources/npu_peak_allocated_mib',0) for m in training),
            physical_expert_fraction=work['work/physical_expert_rows']/work['work/expert_slots'] if 'work/expert_slots' in work else None,
            padding_factor=work['work/physical_expert_rows']/work['work/expert_assignments'] if 'work/expert_assignments' in work else None,
            late_batch_cold_experts_mean=statistics.mean(cold) if cold else None,
            late_batch_max_load_fraction=max(loads) if loads else None,
            recovery=dict(state=recovery['state'],scope=recovery['scope'],record=rel(out/'recovery.json')),
            strict_forward=checks['forward'],strict_vjp_failed=sum(v['state']=='failed' for v in checks['parameter_vjp'].values()),
            strict_causal=checks['causal'],causal_mechanism=checks.get('moe_causal_diagnosis',{}).get('mechanism_state'),
            checkpoint=dict(record=rel(cp),sha256=cp_hash,verified_bytes=cp.stat().st_size),
            lens=dict(train=read(train/'projection.json'),confirmation=read(conf/'projection.json'))))
    for seed in (11,22):
        for name,cid in [('old-small-tf',f'tf-d192-l6-s{seed}'),('old-large-tf',f'tf-d256-l12-s{seed}'),
                         ('hb-small-q10',f'hb-d64-r28-q10-s{seed}'),('hb-small-q3',f'hb-d64-r64-q3-s{seed}'),
                         ('hb-large-q10',f'hb-d128-r28-q10-s{seed}'),('hb-large-q3',f'hb-d128-r64-q3-s{seed}')]:
            out=root/'runs'/('v10-main-'+cid)/'train';conf=root/'runs'/('v10-confirm-main-'+cid)
            vr,vs=endpoint(out/'validation-final.json',131072);cr,cs=endpoint(conf/'confirmation.json',262144)
            if vs!=signatures['validation'] or cs!=signatures['confirmation']:raise ValueError('external block mismatch')
            r=read(out/'result.json');c=read(out/'manifest.json')['config']
            if r['identity']['data_lock_sha256']!=data or r['progress']['tokens']!=128*M:raise ValueError('external exposure/data mismatch')
            paths[(name,seed,'confirmation')]=conf/'confirmation.json'
            external.append(dict(shape=name,id=cid,seed=seed,parameters=r['inventory']['trainable_parameters'],
                lr=c['training']['lr'],validation=vr['nll'],confirmation=cr['nll'],source=r['identity']['source_commit'],
                active_parameters=vr['work'].get('work/active_parameter_uses',0)/vr['work']['work/prediction_positions'] if vr['work'].get('work/prediction_positions') else r['inventory']['trainable_parameters']))
    groups=defaultdict(list)
    for r in rows:groups[r['shape']].append(r)
    means=[]
    for shape,items in sorted(groups.items()):
        if sorted(r['seed'] for r in items)!=[11,22]:raise ValueError('not two independent formal seeds')
        means.append(dict(shape=shape,parameters=items[0]['parameters'],active_parameters=items[0]['active_parameters'],
            lr=items[0]['lr'],balance=items[0]['balance'],
            validation=statistics.mean(r['validation']['nll'] for r in items),
            confirmation=statistics.mean(r['confirmation']['nll'] for r in items),
            seed_confirmation=[r['confirmation']['nll'] for r in items],
            seed_validation=[r['validation']['nll'] for r in items],
            targets_per_second=statistics.mean(r['targets_per_second'] for r in items),
            training_hours=statistics.mean(r['training_seconds'] for r in items)/3600,
            max_peak_npu_gib=max(r['peak_npu_mib'] for r in items)/1024,
            physical_expert_fraction=statistics.mean(r['physical_expert_fraction'] for r in items) if items[0]['kind']=='moe' else None,
            validation_curve={t:statistics.mean(r['validation_curve'][t] for r in items) for t in items[0]['validation_curve']}))
    comparisons=[]
    for seed in (11,22):
        for left,right in [('small-e20','small-dense'),('small-e64','small-dense'),('large-e20','large-dense'),('large-e64','large-dense'),
                           ('small-e64','small-e20'),('large-e64','large-e20'),('small-e20','old-small-tf'),('small-e64','old-small-tf'),
                           ('large-e20','old-large-tf'),('large-e64','old-large-tf'),('small-e20','hb-small-q10'),
                           ('small-e64','hb-small-q3'),('large-e20','hb-large-q10'),('large-e64','hb-large-q3')]:
            value=paired(paths[(left,seed,'confirmation')],paths[(right,seed,'confirmation')])
            if not value['complete_same_blocks'] or value['tokens']!=262144:raise ValueError('incomplete paired confirmation')
            value.update(left=rel(value['left']),right=rel(value['right']),left_shape=left,right_shape=right,seed=seed,
                scope='independent confirmation; 2000 paired document bootstrap resamples, conditional on fixed seed/checkpoints; excludes seed/selection uncertainty')
            comparisons.append(value)
    budget=read(root/'configs/studies/language/moe-controls/budget.json')
    ended=datetime.datetime.fromisoformat(controller['finished_at']).timestamp()
    retirement=read(root/'runs/moe-controller-r2/calibration-optimizer-retirement.json')
    return dict(schema='tide-moe-terminal-review-v1',observed_at=now(),state='terminal_reviewed',source=controller['source'],
        finished_at=controller['finished_at'],wall_hours_from_budget_start=(ended-budget['started_epoch'])/3600,
        phase_counts=dict(Counter(r['phase'] for r in obs['cases'])),all_services_terminated=True,lens_verified=66,
        controller_terminated=True,original_attempts=obs['original_attempts'],new_training_failures=controller['retained_failures'],
        selection32=controller['selection32'],rows=rows,means=means,external128=external,comparisons=comparisons,
        confirmation_targets=262144,confirmation_blocks=len(signatures['confirmation']),test_access=False,
        quota=account_quota(),resources=controller['resources'],
        calibration_optimizer_retirement=dict(state=retirement['state'],files=len(retirement['files']),bytes=sum(x['bytes'] for x in retirement['files'])),
        scope='Reviewed finite next-token/capacity comparison; no full TIDE execution equivalence, global locality, universal scaling or strict numerical pass claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--write',type=Path)
    a=p.parse_args();v=audit(a.root.resolve())
    if a.write:write_review_json(a.write,v)
    print(json.dumps({k:v[k] for k in ('finished_at','phase_counts','wall_hours_from_budget_start','means','new_training_failures')},ensure_ascii=False,indent=2))
