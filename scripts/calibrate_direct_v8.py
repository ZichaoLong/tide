#!/usr/bin/env python3
"""Bounded full-update batch/backend calibration for actual direct local modules."""
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
    a.output.mkdir(parents=True,exist_ok=False);device=runtime(a.device,a.threads,11)
    report=dict(schema='tide-direct-v8-calibration-v1',state='running',source=source_identity(),
        installation=installed_identity(),started_at=now(),cases=[],
        scope='Synthetic L128 complete loss/backward/clip/AdamW updates, 2 warmups+3 samples; different batches are separate work, no quality or equivalent-speedup claim.')
    name='tl-v8-cal-'+a.tag
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',name=name,run_id=name,
        study_id='TL-DIRECT-COVER-v8',group='TL-DIRECT-COVER-v8',stage='calibration',created_at=now(),
        config=dict(device=a.device,threads=a.threads,tag=a.tag)))
    metrics=Metrics(a.output,name)
    entries=[('hb-anchor',b) for b in ((16,32,64) if a.device.startswith('npu') else (16,32))]
    entries += [(k,32 if a.device.startswith('npu') else 16) for k in ('fc-anchor','hb-mem-affine-state',
        'hb-mem-delta-state','hb-mem-gated_delta-state','hb-mem-ssd-state','hb-source-adapted','hb-send-regional_total-ent1','hb-send-regional_total-bern','hb-send-hard_selection','hb-textbook-mixed')]
    started=time.monotonic()
    try:
        for cid,b in entries:
            if time.monotonic()-started>a.timeout-60:break
            c=json.loads((a.configs/(cid+'-s11.json')).read_text());c.update(device=a.device,threads=a.threads)
            rows=[]
            try:
                model=make_model(c,device);opt=optimizer_for(model,c)
                for i in range(5):
                    batch=synthetic_batch(b,128,881+i,device)
                    row=update(model,opt,batch,c,(i+1)*b*128,device);row.update(warmup=i<2,memory=memory(device));rows.append(row)
                seconds=statistics.median(r['seconds'] for r in rows if not r['warmup'])
                item=dict(id=cid,batch=b,length=128,state='completed',median_seconds=seconds,
                    input_positions_per_second=b*128/seconds,updates=rows,parameters=sum(p.numel() for p in model.parameters()))
            except Exception as e:item=dict(id=cid,batch=b,state='failed',error=repr(e),updates=rows)
            report['cases'].append(item);atomic_json(a.output/'result.json',report)
            print(json.dumps({k:v for k,v in item.items() if k!='updates'}),flush=True)
            metrics.log(len(report['cases']),{'calibration/passed':item['state']=='completed',
                'batch/independent':b,'timing/update_seconds':item.get('median_seconds',0)})
            # A failed device operation can poison the context; keep the failure and stop this calibration.
            if item['state']=='failed' and device.type=='npu':break
            del model,opt;gc.collect()
        report['state']='completed'
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:
        report.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        metrics.close();atomic_json(a.output/'result.json',report)
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']))
        project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('configs','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--threads',type=int,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--lens-cli',required=True)
    p.add_argument('--timeout',type=int,default=1200);main(p.parse_args())
