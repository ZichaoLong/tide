#!/usr/bin/env python3
"""Audit available v11 endpoints and scoped in-flight comparisons without training."""
import argparse,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_moe_results import endpoint
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.records import digest,now


def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {}


def audit(root,verify_new_weights=True):
    spec=read(root/'configs/studies/language/capacity-v11/confirmation-table.json');rows=[];signature=None;paths={}
    data=digest(root/'runs/v9-data-r3/data-lock.json');fresh=digest(root/'runs/v11-confirmation-data/data-lock.json')
    for c in spec['cases']:
        out=root/'runs'/('v11-fresh-'+c['id']);r=read(out/'result.json')
        if r.get('state')!='completed':continue
        val,sig=endpoint(out/'confirmation.json',262144)
        if signature is None:signature=sig
        if signature!=sig:raise ValueError('fresh evaluation blocks differ')
        train=read(root/c['training_result']);identity=train['identity'];cp=root/c['checkpoint'];cfg=read(root/c['config'])
        if (train['state']!='completed' or train['progress']['tokens']!=c['tokens'] or r['exposure']!=c['tokens'] or
            r['training_identity']!=identity or identity['data_lock_sha256']!=data or identity['config_sha256']!=digest(root/c['config']) or
            r['confirmation_lock_sha256']!=fresh or r['training_lock_sha256']!=data):raise ValueError('identity/endpoint mismatch: '+c['id'])
        if not inspect(root/'artifacts/jobs'/out.name/'job.json')['termination_verified_now']:raise ValueError('confirmation still has descendants')
        if c.get('producer') and not inspect(root/c['producer']/'job.json')['termination_verified_now']:raise ValueError('training still has descendants')
        new=not c['id'].startswith('parent128-')
        if new and verify_new_weights:
            if digest(cp)!=r['checkpoint_sha256']:raise ValueError('new endpoint weight bytes changed')
        ck=read(cp.parent/'checkpoint.json')
        if ck['file']=='latest.pt' and ck['sha256']!=r['checkpoint_sha256']:raise ValueError('saved checkpoint receipt differs')
        projection=read(out/'projection.json')
        if projection.get('status')!='verified' or not projection.get('readback_matches'):raise ValueError('fresh Lens readback missing')
        row=dict(id=c['id'],config=c['config'],seed=cfg['seed'],tokens=c['tokens'],nll=val['nll'],validation=train['final_validation_nll'],
            confirmation=str((out/'confirmation.json').relative_to(root)),confirmation_sha256=digest(out/'confirmation.json'),
            checkpoint=dict(path=str(cp.relative_to(root)),sha256=r['checkpoint_sha256'],new_full_hash_verified=new and verify_new_weights),
            training_identity=identity,lens=projection,termination_verified=True,parameters=cfg['planned_inventory']['lm_trainable'])
        if new:
            owner=cp.parent.parent
            recovery=read(owner/'recovery-compare/recovery.json') if 'extend256' in c['id'] else read(owner/'recovery.json')
            if recovery['state']!='passed':raise ValueError('new endpoint recovery missing')
            row['recovery']=dict(state=recovery['state'],scope=recovery['scope'])
            check=read(owner/'check/result.json');row['strict_forward']=check['checks']['forward']
            row['strict_vjp_failed']=sum(v['state']=='failed' for v in check['checks']['parameter_vjp'].values())
        if cfg['kind']=='moe':
            metrics=[json.loads(s)['metrics'] for s in (cp.parent/'metrics.jsonl').read_text().splitlines()]
            tail=[r for r in metrics if 'train/loss' in r][-64:]
            cold=[v for r in tail for k,v in r.items() if k.endswith('/cold_experts')]
            load=[v for r in tail for k,v in r.items() if k.endswith('/max_load_fraction')]
            row['late_logged_batch_cold_experts_mean']=statistics.mean(cold)
            row['late_logged_batch_max_load_fraction']=max(load)
            row['load_scope']='last64 logged global batches, per-layer; not lifetime use'
        rows.append(row);paths[c['id']]=out/'confirmation.json'
    groups=defaultdict(list)
    for r in rows:groups[r['id'].rsplit('-s',1)[0]].append(r)
    means={k:dict(seeds=[r['seed'] for r in rs],nll=statistics.mean(r['nll'] for r in rs),seed_nll=[r['nll'] for r in rs]) for k,rs in groups.items()}
    comparisons=[]
    pairs=[('extend256-large-e64-lr0.001-b0.01','extend256-large-e20-lr0.001-b0.01'),
        ('parent128-large-e64-lr0.001-b0.01','parent128-large-e20-lr0.001-b0.01'),
        ('balance128-small-e64-lr0.003-b0.01','parent128-small-e64-lr0.003-b0.001'),
        ('balance128-small-e64-lr0.003-b0.01','parent128-small-e20-lr0.003-b0.01')]
    for shape in ['large-dense-lr0.001','large-e20-lr0.001-b0.01','large-e64-lr0.001-b0.01']:
        pairs.append(('extend256-'+shape,'parent128-'+shape))
    for left,right in pairs:
        for seed in [11,22]:
            l=left+'-s'+str(seed);r=right+'-s'+str(seed)
            if l in paths and r in paths:
                pair=paired(paths[l],paths[r]);pair.update(left=str(paths[l].relative_to(root)),right=str(paths[r].relative_to(root)),seed=seed,
                    scope='Fresh262K confirmation, paired document bootstrap2000; conditional on fixed checkpoints/seed, excludes seed and recipe-selection uncertainty')
                comparisons.append(pair)
    hb=[]
    for c in read(root/'configs/studies/language/capacity-v11/hb-table.json')['cases']:
        job='v11-hb-'+c['id'];out=root/'runs'/job;w=read(root/'runs'/(job+'-worker')/'state.json');r=read(out/'train/result.json');p=read(out/'train/progress.json')
        config=read(root/'.local/source-capacity-v11-hb'/c['config']);seed=config['seed'];baseline=root/f'runs/v10-main-hb-d64-r64-q3-s{seed}/train'
        pairs=[]
        # Existing64Mi validation boundary is identical for the same seed/global cursor.
        for label in ['tokens-33554432','tokens-67116646' if seed==11 else 'tokens-67112640']:
            tokens=int(label.split('-')[1]);v=out/'train'/f'validation-{tokens}.json';b=baseline/v.name
            if v.exists() and b.exists():
                vr,vs=endpoint(v,131072);br,bs=endpoint(b,131072)
                if vs!=bs or read(v)['progress']['tokens']!=read(b)['progress']['tokens']:raise ValueError('HB matched evaluation differs')
                pairs.append(dict(tokens=tokens,variant_nll=vr['nll'],baseline_nll=br['nll'],delta=vr['nll']-br['nll'],
                    variant_record=str(v.relative_to(root)),baseline_record=str(b.relative_to(root))))
        live=inspect(root/'artifacts/jobs'/job/'job.json')
        hb.append(dict(id=c['id'],state=w['state'],tokens=p.get('tokens'),error=r.get('error'),
            physical_device=w.get('physical_visibility'),updated_at=p.get('updated_at'),failure_at=r.get('finished_at') if r.get('state')=='failed' else None,
            termination_verified=live['termination_verified_now'],comparisons=pairs,recovery=read(out/'trained-compare/recovery.json')['state'],
            pilot_gate=read(out/'pilot-gate.json')['state'],last_update={k:v for k,v in p.get('last_update',{}).items() if k.startswith(('train/','activity/','resources/'))},
            warning='Nested axes can retain running after an exception; top-level result/worker/error govern lifecycle. Last completed progress is not the failing update or necessarily the latest saved checkpoint.'))
    controller=read(root/'runs/v11-fresh-controller/state.json')
    return dict(schema='tide-capacity-v11-progress-review-v1',observed_at=now(),state='partial_batch_review',rows=rows,means=means,comparisons=comparisons,hb=hb,
        fresh_lock_sha256=fresh,confirmation_tokens=262144,confirmation_blocks=len(signature),test_access=False,
        confirmation_controller=controller['state'],confirmation_states=dict(Counter(c['state'] for c in controller['cases'])),
        limits=['MoE8 new endpoints completed; HB incomplete with two failures','all reported new final means use the same fresh holdout','Mi means2**20 target tokens,256Mi repeats the128Mi train pool',
            'strict finite mismatches remain failed; allocation failures are not waived','HB midcourse matches one seed at common exposure; no final HB benefit claim','no new training, repair or model evaluation was launched by this review'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',type=Path);p.add_argument('--skip-new-weight-hash',action='store_true');a=p.parse_args()
    value=audit(Path(__file__).resolve().parents[1],not a.skip_new_weight_hash)
    if a.write:write_review_json(a.write,value)
    print(json.dumps({k:value[k] for k in ('observed_at','means','confirmation_states','hb')},ensure_ascii=False,indent=2))
