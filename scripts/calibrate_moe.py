#!/usr/bin/env python3
"""Finite full-update batch calibration and Lens admission for one MoE tier."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.train import runtime,source_identity,installed_identity
from tide_lab.moe_config import config,M
from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,synthetic_batch,memory
from tide_lab.moe_checks import mechanism_checks
from tide_lab.records import atomic_json,now,Metrics,project_to_lens


def main(a):
    dev=runtime('npu:0',4,33);a.output.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',started_at=now(),source=source_identity(),installation=installed_identity(),shape=a.shape,cases=[])
    run_id='tl-'+a.output.name;metrics=Metrics(a.output,run_id)
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',name=run_id,run_id=run_id,
        group='TL-MOE-CAL',study_id='TL-MOE-CAL',stage='batch-calibration',created_at=now(),config=dict(shape=a.shape)))
    try:
        if not a.shape.endswith('dense'):report['mechanisms']=mechanism_checks(dev)
        for micro in a.batches:
            if micro not in (16,32,64):raise ValueError('finite approved batch choices only')
            c=config(a.shape,.001,micro=micro);atomic_json(a.output/f'config-b{micro}.json',c)
            m=make_model(c,dev);opt=optimizer_for(m,c);torch.npu.reset_peak_memory_stats()
            rows=[]
            for step in range(5):
                b=synthetic_batch(64,256,501+step,dev)
                batches=[{k:v[i:i+micro] for k,v in b.items()} for i in range(0,64,micro)]
                rows.append(logical_update(m,opt,batches,c,8*M,dev))
            seconds=statistics.median(r['seconds'] for r in rows[2:]);mem=memory(dev)
            item=dict(state='completed',microbatch=micro,global_batch=64,accumulation=64//micro,
                median_seconds=seconds,positions_per_second=64*256/seconds,parameters=sum(p.numel() for p in m.parameters()),
                peak_mib=mem['resources/npu_peak_allocated_mib'],updates=rows,
                timing_scope='full model, auxiliary prepass if needed, task/aux VJP, clip, AdamW and synchronization; input already on device')
            report['cases'].append(item);atomic_json(a.output/'result.json',report)
            metrics.log(micro,{'timing/positions_per_second':item['positions_per_second'],'resources/peak_mib':item['peak_mib'],
                'batch/physical':micro,'batch/global':64,'model/parameters':item['parameters'],
                'train/loss':rows[-1]['loss'],**rows[-1]['work']})
            del m,opt,b,batches;gc.collect();torch.npu.empty_cache()
        report['state']='completed'
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lens-cli',required=True);p.add_argument('--batches',nargs='+',type=int,default=[16,32,64]);main(p.parse_args())
