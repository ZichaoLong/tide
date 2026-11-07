#!/usr/bin/env python3
"""One bounded CPU-dispatch profile of a selected NPU workload; no kernel-time claim."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    import torch
    from tide_lab.train import runtime,make_model,synchronize
    from tide_lab.tasks import RetrievalTask
    c=json.loads(Path(a.config).read_text());d=runtime(c['device'],c['threads'],c['seed'])
    task=RetrievalTask(c['task']);model=make_model(c,task,None).to(d)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['training']['lr'],
        eps=c['training']['eps'],weight_decay=c['training']['weight_decay'],foreach=False)
    def step(i):
        x,y=task.batch('train',c['training']['batch_size'],i,c['training']['data_seed'],d)
        opt.zero_grad(set_to_none=True);out=model(x);loss=torch.nn.functional.cross_entropy(out,y)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite profile loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
        opt.step();synchronize(d)
    step(0)
    started=time.monotonic()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
        step(1)
    elapsed=time.monotonic()-started
    rows=sorted(prof.key_averages(group_by_input_shape=True),key=lambda e:e.self_cpu_time_total,reverse=True)
    events=[dict(operator=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
                 total_cpu_ms=e.cpu_time_total/1000,input_shapes=e.input_shapes) for e in rows[:40]]
    atomic_json(a.output,dict(schema='tide-lab-operator-observation-v1',state='completed',
        independent_sample_batch=c['training']['batch_size'],positions=task.sequence_length,
        accumulation=1,device=str(d),execution='public eager mixed-c packed, host scheduler workers1',
        synchronized_profiled_step_seconds=elapsed,work=model.last_metrics,events=events,
        scope='CPU dispatch/operator input shapes including host synchronization; profiler overhead included; not NPU kernel durations or a speedup benchmark'))


if __name__=='__main__': main()
