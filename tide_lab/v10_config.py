"""Finite v10 configuration helpers; planning metadata only."""
import math
from scripts.prepare_direct_v9 import direct,tf

M=2**20


def common(c,group):
    c.update(campaign='direct-v10',study_id=group)
    c['training'].update(target_tokens=256*M,lr_decay_tokens=256*M,milestones=[8*M,32*M,128*M,256*M],
        warmup_tokens=M,validation_every_tokens=8*M,eval_tokens=131072,initial_eval_tokens=32768,
        checkpoint_seconds=1800,timeout_seconds=46*3600,log_every=16,keep_previous=False,milestone_weights_only=True)
    c['resources']['max_rss_gib']=64
    return c


def hb(name,d,r,target=None,seed=11,*,micro=8,temperature=.25,controller=True,ffn=None):
    candidate='temperature1' if target is not None else 'send-pooled_attention'
    c=common(direct(name,'hb',2,d,candidate,region_width=r,k=2 if target is not None else None,seed=seed,ffn=ffn),
             'TL-CAPACITY-v10' if target else 'TL-ACTIVE-v10')
    c['kind']='direct_specialized';c['training'].update(batch=micro,eval_batch=min(micro,16),optimizer_foreach=False)
    c['implementation']=dict(cpu_attention='math',npu_attention='npu_fusion_augmented',aggregation='neighbors',ffn_checkpoint=False)
    if target is not None:
        if controller:c['implementation']['activity_controller']=dict(target=target,start_tokens=2*M,interval_updates=8,ema_decay=.9)
        for s in c['direct']['plan']['regions']:
            s['allocation_temperature']=temperature
            s['activity_schedule']=dict(start_k=r/2,start_temperature=1.,warmup_tokens=M//2,end_tokens=2*M)
    c['numerical_policy'].update(attention='unfloored real-arithmetic weighted softmax; log-key bias augmentation; original normal-tiny reference retained',
        state_updates='all receiving candidates; dense physical Full; no sparse speedup claim')
    return c


def transformer(name,width,layers,seed=11,*,initialization='fan_in_depth',lr=.001):
    c=common(tf(name,width,layers,seed),'TL-CONTROL-v10');c['npu_transformer_attention']='npu_fusion'
    c['transformer']['initialization']=initialization;c['training'].update(batch=64,eval_batch=64,lr=lr)
    return c
