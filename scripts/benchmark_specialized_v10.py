#!/usr/bin/env python3
"""Finite representative FP32 complete-step and peak-HBM calibration."""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens


def main(a):
    import torch
    from scripts.prepare_direct_v9 import direct
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model,optimizer_for,update,memory,synthetic_batch
    dev=runtime('npu:0',2,11);a.output.mkdir(parents=True,exist_ok=False)
    run_id='tl-v10-benchmark-d'+str(a.width)
    report=dict(schema='tide-v10-performance-v1',state='running',source=source_identity(),installation=installed_identity(),
        started_at=now(),cases=[],scope='same-weight FP32 synthetic L256 complete update; finite numerical policy difference retained; not learning evidence')
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=run_id,name=run_id,
        group='TL-PERF-v10',study_id='TL-PERF-v10',created_at=now()))
    metrics=Metrics(a.output,run_id);step=0
    for r in (20,64):
        base=direct(f'v10-r{r}-d{a.width}','hb',2,a.width,'temperature1',region_width=r,k=2)
        base['training'].update(lr_decay_tokens=256*2**20,lr=.001)
        for s in base['direct']['plan']['regions']:
            s.pop('activity_schedule',None);s['allocation_temperature']=.25
        # Baseline is held at B4. Optimized candidates use increasing batches;
        # checkpointing is only tried on the largest shape, not a broad matrix.
        for variant in ('reference','specialized','specialized-foreach','specialized-recompute'):
            if variant=='specialized-recompute' and not (r==64 and a.width==128):continue
            batches=(4,) if variant in ('reference','specialized-foreach') else (4,8,16,32,64)
            previous=None
            for batch in batches:
                if time.time()>a.deadline_epoch-60:break
                if previous and previous['peak_mib']*1.9>46000:
                    report['cases'].append(dict(region_width=r,width=a.width,variant=variant,batch=batch,state='not_attempted',reason='measured conservative growth exceeds46000MiB'))
                    break
                c=copy.deepcopy(base);c['kind']='direct_semantic' if variant=='reference' else 'direct_specialized'
                if c['kind']=='direct_specialized':c['implementation']=dict(npu_attention='npu_fusion_augmented',cpu_attention='math',ffn_checkpoint=variant=='specialized-recompute',aggregation='neighbors')
                c['training']['optimizer_foreach']=variant=='specialized-foreach';c['training']['batch']=batch
                config_path=a.output/(f'config-r{r}-{variant}-b{batch}.json');atomic_json(config_path,c)
                torch.npu.empty_cache();model=make_model(c,dev);opt=optimizer_for(model,c);torch.npu.reset_peak_memory_stats()
                rows=[];item=dict(region_width=r,width=a.width,variant=variant,batch=batch,config=str(config_path),parameters=sum(p.numel() for p in model.parameters()))
                try:
                    for i in range(5):
                        b=synthetic_batch(batch,256,771+i,dev)
                        rows.append(update(model,opt,b,c,9*2**20,dev))
                    seconds=statistics.median(x['seconds'] for x in rows[2:]);mem=memory(dev)
                    item.update(state='completed',median_seconds=seconds,positions_per_second=batch*256/seconds,
                        peak_mib=mem['resources/npu_peak_allocated_mib'],resources=mem,updates=rows)
                    previous=item
                    metrics.log(step,{'timing/positions_per_second':item['positions_per_second'],'resources/peak_mib':item['peak_mib'],
                        'batch/physical':batch,'shape/width':a.width,'shape/region_nodes':r})
                except Exception as error:item.update(state='failed',error=repr(error),updates=rows)
                report['cases'].append(item);atomic_json(a.output/'result.json',report);step+=1
                print(json.dumps({k:v for k,v in item.items() if k not in ('updates','resources')}),flush=True)
                del model,opt
                if 'b' in locals():del b
                gc.collect();torch.npu.empty_cache()
                if item['state']=='failed':break
                if previous['peak_mib']>46000:break
    report.update(state='completed',finished_at=now(),meaning='finite benchmark matrix terminal; failed/not-attempted cases retained')
    atomic_json(a.output/'result.json',report);metrics.close();atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']))
    project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,choices=(64,128),required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True)
    main(p.parse_args())
