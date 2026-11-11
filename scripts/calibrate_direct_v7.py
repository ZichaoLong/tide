#!/usr/bin/env python3
"""Small finite operator-batch sweep for the already measured direct graph."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens


def main(a):
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model,optimizer_for,update,memory,synthetic_batch
    from scripts.prepare_direct_v7 import config
    a.output.mkdir(parents=True,exist_ok=False);device=runtime(a.device,a.threads,11)
    report=dict(schema='tide-direct-v7-calibration-v1',state='running',started_at=now(),
        source=source_identity(),installation=installed_identity(),cases=[],
        scope='Synthetic complete updates, three warmups then five timed samples; different independent B. No quality or equivalent-speedup claim.')
    run_id='tl-v7-cal-'+('npu' if a.device.startswith('npu') else 'cpu')
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=run_id,
        name=run_id,study_id='TL-DIRECT-RECIPE-v7',group='TL-DIRECT-RECIPE-v7',stage='batch-calibration',
        config=dict(device=a.device,threads=a.threads),created_at=now()))
    metrics=Metrics(a.output,run_id);started=time.monotonic()
    entries=[('hb88',b) for b in (16,32,64,128)] if a.device.startswith('npu') else [('hb88',16),('hb40',32)]
    if a.device.startswith('npu'):entries += [('hb40',32),('tf28',32)]
    try:
        for name,b in entries:
            c=config('transformer' if name.startswith('tf') else 'direct_graph',name,batch=b)
            c.update(device=a.device,threads=a.threads)
            model=make_model(c,device);opt=optimizer_for(model,c);rows=[]
            for i in range(8):
                batch=synthetic_batch(b,128,991+i,device)
                row=update(model,opt,batch,c,(i+1)*b*128,device);row.update(warmup=i<3,memory=memory(device));rows.append(row)
            seconds=statistics.median(r['seconds'] for r in rows if not r['warmup'])
            case=dict(model=name,batch=b,length=128,updates=rows,median_seconds=seconds,
                positions_per_second=b*128/seconds,parameters=sum(p.numel() for p in model.parameters()))
            report['cases'].append(case);atomic_json(a.output/'result.json',report)
            metrics.log(len(report['cases']),{'batch/independent':b,'timing/update_seconds':seconds,
                'throughput/input_positions_per_second':b*128/seconds})
            print(json.dumps({k:v for k,v in case.items() if k!='updates'}),flush=True)
            del model,opt,batch;gc.collect()
        report['state']='completed'
    except BaseException as error:report.update(state='failed',error=repr(error));raise
    finally:
        report.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        metrics.close();atomic_json(a.output/'result.json',report)
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']))
        project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','npu:0'],required=True);p.add_argument('--threads',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lens-cli',required=True)
    main(p.parse_args())
