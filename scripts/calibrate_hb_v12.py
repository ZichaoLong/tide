#!/usr/bin/env python3
"""Bounded microbatch calibration, constant global64, no scientific ancestry."""
import argparse,copy,gc,json,statistics,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens

def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,synthetic_batch,memory
    c=json.loads(a.config.read_text());device=runtime('npu:0',2,33)
    a.output.mkdir(parents=True,exist_ok=False);run='tl-v12-cal-'+c['id']
    report=dict(schema='tide-hb-v12-calibration-v1',state='running',started_at=now(),source=source_identity(),installation=installed_identity(),cases=[])
    atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',name=run,run_id=run,study_id='TL-HB-CAL-v12',group='TL-HB-CAL-v12',config=c))
    metrics=Metrics(a.output,run);previous=None
    d=c['direct']['plan']['width'];r=c['candidate']['region_width']
    batches=[8,16,32] if d==64 else [4,8,16] if d==128 else [2,4,8]
    try:
        for micro in batches:
            if previous and previous['peak_mib']*1.85>52000:
                report['cases'].append(dict(micro=micro,state='not_attempted',reason='measured peak growth exceeds52000MiB'));break
            if time.time()>a.deadline_epoch-90:raise TimeoutError('calibration deadline')
            cfg=copy.deepcopy(c);cfg['training']['batch']=micro
            torch.npu.empty_cache();model=make_model(cfg,device);opt=optimizer_for(model,cfg);torch.npu.reset_peak_memory_stats()
            rows=[]
            for step in range(4):
                batches64=[synthetic_batch(micro,256,7000+step*64+i,device) for i in range(0,64,micro)]
                row=logical_update(model,opt,batches64,cfg,4*2**20,device);rows.append(row)
            peak=memory(device)['resources/npu_peak_allocated_mib'];rate=statistics.median(x['tokens']/x['seconds'] for x in rows[1:])
            item=dict(micro=micro,state='completed',global_batch=64,accumulation=64//micro,targets_per_second=rate,peak_mib=peak,observations=rows)
            report['cases'].append(item);previous=item
            metrics.log(micro,{'batch/physical':micro,'batch/global':64,'timing/targets_per_second':rate,'resources/peak_mib':peak})
            atomic_json(a.output/'result.json',report)
            del model,opt,batches64;gc.collect();torch.npu.empty_cache()
        good=[r for r in report['cases'] if r['state']=='completed' and r['peak_mib']<52000]
        if not good:raise RuntimeError('no bounded useful physical batch')
        selected=max(good,key=lambda r:r['targets_per_second'])
        report.update(state='completed',selected={k:selected[k] for k in ('micro','global_batch','targets_per_second','peak_mib')},
            scope='synthetic L256 global64 complete updates including VJP/Adam; warmup first update excluded; contention recorded by launcher; not language quality or independent correctness')
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('config','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
