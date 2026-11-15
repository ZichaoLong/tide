"""Finite approved MoE/dense grid, independent of the old v10 inputs."""
import math
from .moe import parameter_inventory
from scripts.prepare_direct_v8 import common

M=2**20
SHAPES={'small-e20':(192,6,384,20),'small-e64':(192,6,384,64),
        'large-e20':(256,12,480,20),'large-e64':(256,12,480,64),
        'small-dense':(192,6,768,0),'large-dense':(256,12,960,0)}


def config(shape,lr,balance=.001,seed=33,micro=64):
    d,l,h,e=SHAPES[shape]
    cid=f'{shape}-lr{lr:g}'+(f'-b{balance:g}' if e else '')+f'-s{seed}'
    c=common(cid,'moe' if e else 'transformer',seed)
    c.update(schema='tide-moe-control-config-v1',campaign='moe-controls',selection_key=shape,
             study_id='TL-MOE-CAL' if seed==33 else 'TL-MOE-CONTROL',threads=4)
    c['npu_transformer_attention']='npu_fusion'
    c['training'].update(target_tokens=128*M,lr_decay_tokens=256*M,milestones=[8*M,32*M,128*M],
        length=256,batch=micro,global_batch=64,eval_batch=micro,lr=lr,betas=[.9,.95],eps=1e-8,
        optimizer_foreach=False,validation_every_tokens=8*M,eval_tokens=131072,initial_eval_tokens=32768,
        warmup_tokens=M,checkpoint_seconds=1800,timeout_seconds=48*3600,log_every=16,
        keep_previous=False,milestone_weights_only=True,balance_coefficient=balance if e else 0.,
        z_coefficient=.001 if e else 0.,decay_policy='matrix_only',full_probe_length=32,full_probe_batch=4)
    c['resources']['max_rss_gib']=64
    c['numerical_policy']=dict(strict_atol=1e-6,strict_rtol=1e-5,routing='FP32 top2 selected-logit softmax; dropless',
        auxiliary='global64 valid-token frequencies stop-gradient; mean over layers; no-grad prepass if accumulated',
        finite_exploration='bounded mechanism checks; retain finite strict mismatches, block nonfinite/discrete/recovery failures')
    if e:
        c['moe']=dict(width=d,layers=l,hidden=h,experts=e,heads=4,position_scale=math.sqrt(2)*.02,
            expert_implementation='bucketed_bmm')
        p=parameter_inventory(d,l,h,e);c['planned_inventory']=dict(lm_trainable=p['total'],active_per_token=p['active'])
        c['comparison_scope']='ordinary Transformer-MoE; dense attention, top2 FFN; expert fraction is not HB Full/all'
    else:
        c['transformer']=dict(width=d,layers=l,ffn=h,heads=4,position_scale=math.sqrt(2)*.02,initialization='gpt2')
        p=4096*d+l*(4*d*d+8*d+2*d*h+h+d)+2*d+4096
        c['planned_inventory']=dict(lm_trainable=p,active_per_token=p)
        c['comparison_scope']='matched top2 total FFN width; common exposure, not strict FLOPs or walltime matching'
    if c['planned_inventory']['lm_trainable']>200000000:raise ValueError('200M parameter cap')
    return c


def screening_configs(micros=None):
    rows=[]
    for name,(_,_,_,e) in SHAPES.items():
        for lr in ((.0003,.001,.003) if name.startswith('small') else (.0001,.0003,.001)):
            for balance in ((.001,.01) if e else (0.,)):
                rows.append(config(name,lr,balance,micro=(micros or {}).get(name,64)))
    return rows
