#!/usr/bin/env python3
"""Finite actual-shape full-update calibration, including accumulated batch64."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,Metrics,now,project_to_lens


def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model,optimizer_for,update,memory,synthetic_batch
    c=json.loads(a.config.read_text());device=runtime(c['device'],c['threads'],c['seed'])
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(state='running',started_at=now(),config_id=c['id'],cases=[],source=source_identity(),installation=installed_identity(),
        axes=dict(performance='exploratory full-update cost',strict_numerical='separate admission',learning='not_checked'))
    name='tl-v9-batch-'+c['id'];metrics=Metrics(a.output,name)
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=name,name=name,
        group='TL-CALIBRATION-v9',study_id='TL-CALIBRATION-v9',config=c,created_at=now(),stage='batch-calibration'))
    try:
        for batch in (2,4,8,16,32,64):
            if time.time()>a.deadline_epoch-60:break
            if report['cases']:
                prev=report['cases'][-1]
                if prev.get('peak_mib',float('inf'))*2.2>30000:
                    report['not_attempted_from_batch']=batch;report['reason']='conservative measured HBM growth estimate';break
            model=make_model(c,device);opt=optimizer_for(model,c)
            if device.type=='npu':torch.npu.reset_peak_memory_stats()
            rows=[]
            try:
                for i in range(5):
                    b=synthetic_batch(batch,c['training']['length'],771+i,device)
                    row=update(model,opt,b,c,9*2**20,device);rows.append(row)
                cost=statistics.median(x['seconds'] for x in rows[2:]);mem=memory(device)
                item=dict(batch=batch,state='completed',median_seconds=cost,positions_per_second=batch*c['training']['length']/cost,
                    peak_mib=mem.get('resources/npu_peak_allocated_mib',mem['resources/peak_rss_mib']),updates=rows,resources=mem)
                metrics.log(batch,{'batch/physical':batch,'timing/update_seconds':cost,'timing/positions_per_second':item['positions_per_second'],
                    'resources/peak_mib':item['peak_mib']})
                del b,row;gc.collect()
            except Exception as error:
                item=dict(batch=batch,state='failed',error=repr(error),updates=rows)
            report['cases'].append(item);atomic_json(a.output/'result.json',report)
            print(json.dumps({k:v for k,v in item.items() if k!='updates'}),flush=True)
            if item['state']=='failed':break
            del model,opt;gc.collect()
            if device.type=='npu':torch.npu.empty_cache()
        good=[x for x in report['cases'] if x['state']=='completed' and x['peak_mib']<=30000]
        if good:
            peak=max(x['positions_per_second'] for x in good)
            choice=max((x for x in good if x['positions_per_second']>=.95*peak),key=lambda x:x['batch'])
            report.update(state='completed',selected_batch=choice['batch'],global_batch=64,
                selection='within5% of best measured useful-position throughput, prefer larger independent microbatch; global64 via accumulation')
        else:report.update(state='failed',error='no safe measured microbatch')
    except BaseException as error:report.update(state='failed',error=repr(error));raise
    finally:
        report.update(finished_at=now(),seconds=time.monotonic()-start);atomic_json(a.output/'result.json',report)
        metrics.close();atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']))
        project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);p.add_argument('--deadline-epoch',type=float,required=True)
    main(p.parse_args())
