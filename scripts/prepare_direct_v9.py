#!/usr/bin/env python3
"""Resolve the accepted finite v9 table from public metadata and local variants."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tidegraph.planning.recipes import prepare
from tide_lab.campaign_config import local_candidates
from tide_lab.records import atomic_json
from scripts.resolve_direct_v8 import recipe
from scripts.prepare_direct_v8 import common

M=2**20


def inlet_tree(plan):
    """Append IDs, prepend ranks: retain every body parameter role and class."""
    p=copy.deepcopy(plan);g=p['graph'];n0=g['nodes'];r0=g['regions'];entry=list(g['inputs'])
    if len(entry)%16:raise ValueError('tree banks need a multiple of16 entry width')
    old_stride=p['stride'];new_stride=old_stride+3
    p.update(body_regions=r0,body_nodes=n0,light_regions=[r0,r0+1],stride=new_stride,
        inlet=dict(kind='copy1-copy4-gate4x4-v1',body_entry_width=len(entry),gate_strength_budget=1,
                   body_initialization='unchanged role names, shapes and shared source classes'))
    for m in p['memories']:
        m['window']=round(m['window']*new_stride/old_stride)
        m['beta_short']*=old_stride/new_stride
        m['position_scale']*=new_stride/old_stride
        m['position_origin']+=3
    g['ranks']=[r+3 for r in g['ranks']]+[1,2,3,3,3,3]
    g['node_regions'] += [r0]+[r0+1]*4+[r0+2+i//4 for i in range(16)]
    g['nodes']+=21;g['regions']+=6;g['inputs']=[n0]
    old_class=p['classes']['inputs'][0];p['classes']['inputs']=[old_class]
    def edge(src,dst):
        g['edges'].append([src,dst,1]);p['classes']['edges'].append(old_class)
    for i in range(4):
        edge(n0,n0+1+i)
        for j in range(4):
            gate=n0+5+4*i+j;edge(n0+1+i,gate)
            for dst in entry[(4*i+j)*len(entry)//16:(4*i+j+1)*len(entry)//16]:edge(gate,dst)
    m=copy.deepcopy(p['memories'][0]);m['position_origin']=1
    p['memories'] += [copy.deepcopy(m) for _ in range(21)]
    s=copy.deepcopy(p['regions'][0]);s.pop('activity_schedule',None)
    s.update(k=1,region_control_mode='fixed',total_fraction=.5)
    s.pop('content_independent_rotation',None)
    p['regions'] += [copy.deepcopy(s) for _ in range(6)]
    g['counts']=dict(body_nodes=n0,boundary_nodes=21,body_regions=r0,reachable_nodes=g['nodes'],
        output_reachable_nodes=g['nodes'],body_edges=len(plan['graph']['edges']),total_edges=len(g['edges']))
    return p


def direct(name,topology,depth,width,candidate,*,region_width=4,k=None,tree=False,ffn=None,seed=11):
    rows={r['id']:r for r in local_candidates()}
    spec=recipe(rows[candidate],topology,width);spec.update(depth=depth,region_width=region_width)
    if ffn:spec['memory_overrides']['ffn_width']=ffn
    plan,_=prepare(spec,samples=1,positions=1,seed=seed)
    keys=('graph','memories','regions','classes','source_binding','source_hidden_width','alpha_agg','stride','width','upstream','sha256','recipe')
    plan={q:plan[q] for q in keys};plan['aggregation_reduction']='matmul'
    if k is not None:
        for s in plan['regions']:
            s.update(k=k,region_control_mode='fixed',total_fraction=.5)
            if s['regularizer']=='quad':
                s['activity_schedule']=dict(start_k=region_width/2,start_temperature=1.,warmup_tokens=2*M,end_tokens=8*M)
    if 'rotate' in name:
        for s in plan['regions']:s.update(content_independent_rotation=True,rotation_tokens=16384)
    if tree:plan=inlet_tree(plan)
    # Parent public metadata digest is preserved separately; local graph has its own identity.
    plan['parent_public_plan_sha256']=plan.pop('sha256')
    plan['sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    c=base(name,seed,'direct_semantic');c['direct']=dict(plan=plan)
    c['candidate']=dict(id=candidate,topology=topology,depth=depth,width=width,region_width=region_width,tree=tree)
    c['initialization']='role-keyed-fan-in-v1'
    c['comparison_scope']='Independent direct local formulas; dense physical Full with masks, experiment-owned inlet. Not foundation continuation certification.'
    c['numerical_policy']=dict(strict_atol=1e-6,strict_rtol=1e-5,allocation_fp32_tolerance=2e-5,
        positive_floor='normal-tiny on eligible positive functions; original finite/discrete failures retained')
    return c


def base(name,seed,kind):
    c=common(name+'-s'+str(seed),kind,seed);c.update(campaign='direct-v9',study_id='TL-SCALE-v9')
    c['training'].update(target_tokens=64*M,milestones=[4*M,16*M,64*M],length=256,batch=8,global_batch=64,
        eval_batch=8,eval_tokens=131072,initial_eval_tokens=32768,warmup_tokens=1*M,
        checkpoint_seconds=1800,validation_every_tokens=4*M,timeout_seconds=46*3600,log_every=16,
        keep_previous=False,milestone_weights_only=True)
    c['resources']['max_rss_gib']=48
    return c


def tf(name,width,layers,seed=11):
    c=base(name,seed,'transformer');c['transformer']=dict(width=width,layers=layers,heads=4,ffn=4*width,
        position_scale=math.sqrt(2)*.02,initialization='fan_in_depth');return c


def build():
    main=[]
    for depth,n,(w,l) in zip((2,3,4),(40,88,184),((384,6),(400,12),(512,16))):
        for seed in (11,22):
            for kind,cand in (('pool','send-pooled_attention'),('quad','temperature1')):
                c=direct(f'hb{n}-{kind}','hb',depth,128,cand,seed=seed)
                c['selection_key']=f'hb{n}-{kind}';main.append(c)
            c=tf(f'tf-hb{n}',w,l,seed);c['selection_key']=f'tf-hb{n}';main.append(c)
    for w,ff in ((192,None),(256,1280)):
        for seed in (11,22):main.append(direct(f'hb40-wide{w}-pool','hb',2,w,'send-pooled_attention',ffn=ff,seed=seed))
    for depth,n,(w,l) in zip((4,8,16),(16,32,64),((240,6),(336,6),(416,8))):
        for seed in ((11,22) if n==64 else (11,)):
            c=direct(f'fc{n}-direct','fc',depth,128,'send-direct_attention',seed=seed)
            t=tf(f'tf-fc{n}',w,l,seed);t['selection_key']=f'tf-fc{n}'
            for x in (c,t):x['training'].update(target_tokens=16*M,milestones=[4*M,16*M]);main.append(x)
    activity=[]
    for r,h,kind in [(32,2,k) for k in ('bern','pool','q10')]+[(64,2,k) for k in ('bern','pool','q10','q3','rotate3')]+[(32,3,k) for k in ('bern','q10')]:
        for seed in (11,22):
            candidate='send-regional_total-bern' if kind=='bern' else 'send-pooled_attention' if kind=='pool' else 'temperature1'
            k=4 if r==32 else 10 if kind=='q10' else 2
            name=f'r{r}-h{h}-{kind}'
            c=direct(name,'hb',h,64,candidate,region_width=r,k=None if kind=='pool' else k,seed=seed)
            c['study_id']='TL-ACTIVITY-v9';c['training'].update(target_tokens=16*M,milestones=[4*M,16*M])
            if kind.startswith('q'):c['activity_selection_key']=name
            activity.append(c)
    main+=activity
    for r,kind in ((32,'q10'),(64,'q3')):
        for seed in (11,22):
            for mode in (kind,'bern'):
                c=direct(f'tree-r{r}-h2-{mode}','hb',2,64,'temperature1' if mode!='bern' else 'send-regional_total-bern',
                         region_width=r,k=4 if r==32 else 2,tree=True,seed=seed)
                c['study_id']='TL-INLET-v9';c['training'].update(target_tokens=16*M,milestones=[4*M,16*M])
                if mode!='bern':c['activity_selection_key']=f'tree-r{r}-h2-{mode}'
                main.append(c)
    for c in main:
        c['stage_until_tokens']=c['training']['target_tokens']
        c['training']['lr_decay_tokens']=c['training']['target_tokens']
        if c.get('selection_key') in ('hb88-pool','hb88-quad','tf-hb88'):
            c['training']['target_tokens']=256*M;c['training']['milestones'].append(256*M)
        elif c['study_id']=='TL-ACTIVITY-v9':
            c['training']['target_tokens']=64*M;c['training']['milestones'].append(64*M)
    screens=[]
    for c in main:
        if c['seed']!=11:continue
        key=c.get('selection_key')
        if key:
            for init in (('gpt2','fan_in_depth') if c['kind']=='transformer' else (None,)):
                for lr in ((.0003,.001,.003) if c['kind']=='transformer' else (.0003,.001)):
                    x=copy.deepcopy(c);x['id']=f'cal-{key}-{init or "fan"}-lr{lr:g}-s11'
                    if init:x['transformer']['initialization']=init
                    x['training'].update(lr=lr,target_tokens=c['training']['target_tokens'],milestones=[4*M],
                        eval_tokens=65536,discard_optimizer_on_completion=True)
                    x['stage_until_tokens']=4*M;screens.append(x)
        key=c.get('activity_selection_key')
        if key:
            for tau in (1.,.5,.25,.1):
                x=copy.deepcopy(c);x['id']=f'cal-{key}-tau{tau:g}-s11'
                for s in x['direct']['plan']['regions']:
                    s.pop('activity_schedule',None);s['allocation_temperature']=tau
                x['training'].update(target_tokens=262144,lr_decay_tokens=262144,milestones=[262144],warmup_tokens=32768,
                    eval_tokens=32768,discard_optimizer_on_completion=True,validation_every_tokens=262144)
                x['stage_until_tokens']=262144;screens.append(x)
    for c in screens:c['study_id']='TL-CALIBRATION-v9'
    return main,screens


def main(a):
    import torch
    from tide_lab.campaign_runtime import make_model
    main,screens=build();a.output.mkdir(parents=True,exist_ok=False)
    for folder,rows in (('main',main),('screen',screens)):
        dest=a.output/folder;dest.mkdir()
        for c in rows:
            with torch.device('meta'):model=make_model(c,torch.device('meta'))
            c['planned_inventory']=dict(lm_trainable=sum(p.numel() for p in model.parameters()))
            if c.get('direct'):
                plan=c['direct']['plan'];plan.pop('sha256',None)
                plan['sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
            atomic_json(dest/(c['id']+'.json'),c)
        atomic_json(dest/'catalog.json',dict(cases=[c['id'] for c in rows]))
    print(json.dumps(dict(main=len(main),screens=len(screens))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
