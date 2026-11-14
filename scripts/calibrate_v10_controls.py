#!/usr/bin/env python3
"""Finite complete-update memory and batch checks for the selected controls."""
import argparse
import gc
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens


def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model,optimizer_for,update,memory,synthetic_batch
    from tide_lab.v10_config import hb,M
    dev=runtime('npu:0',2,11);a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tide-v10-controls-calibration-v1',state='running',source=source_identity(),
        installation=installed_identity(),started_at=now(),cases=[],scope='synthetic complete microbatch updates; no learning claim')
    run_id='tl-v10-cal-controls';metrics=Metrics(a.output,run_id)
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=run_id,
        name=run_id,group='TL-PERF-v10',study_id='TL-PERF-v10',created_at=now()))
    # The two dense endpoints match the received/Full/shared inventory from
    # calibration. Two same-shape pools check the worst receipt occupancy.
    shapes=[(64,4,288,[32,64]),(128,3,576,[32,64]),(128,28,512,[16]),(128,64,512,[8])]
    try:
        for d,r,ffn,batches in shapes:
            for batch in batches:
                c=hb(f'v10-control-d{d}-r{r}',d,r,micro=batch,ffn=ffn)
                path=a.output/f'config-d{d}-r{r}-b{batch}.json';atomic_json(path,c)
                model=make_model(c,dev);opt=optimizer_for(model,c);torch.npu.reset_peak_memory_stats()
                rows=[]
                for i in range(5):
                    b=synthetic_batch(batch,256,117+i,dev)
                    rows.append(update(model,opt,b,c,8*M,dev))
                seconds=statistics.median(x['seconds'] for x in rows[2:]);mem=memory(dev)
                item=dict(width=d,region_width=r,ffn=ffn,batch=batch,state='completed',config=str(path),
                    parameters=sum(p.numel() for p in model.parameters()),median_seconds=seconds,
                    positions_per_second=batch*256/seconds,peak_mib=mem['resources/npu_peak_allocated_mib'],updates=rows)
                report['cases'].append(item);atomic_json(a.output/'result.json',report)
                metrics.log(len(report['cases']),{'timing/positions_per_second':item['positions_per_second'],
                    'resources/peak_mib':item['peak_mib'],'batch/physical':batch,'shape/width':d,'shape/region_nodes':r})
                del model,opt,b;gc.collect();torch.npu.empty_cache()
        report['state']='completed'
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lens-cli',required=True)
    main(p.parse_args())
