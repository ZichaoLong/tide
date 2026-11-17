#!/usr/bin/env python3
"""Six bounded shape calibrations; compare same-batch qry and two sum kernels."""
import argparse,copy,gc,json,statistics,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens

def main(a):
 import torch
 from tide_lab.train import runtime,source_identity,installed_identity,synchronize
 from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,synthetic_batch,memory
 c=json.loads(a.config.read_text());device=runtime('npu:0',2,33);a.output.mkdir(parents=True,exist_ok=False)
 group=c['mem_comparison']['group'];run='tl-v13-cal-'+group
 report=dict(state='running',started_at=now(),source=source_identity(),installation=installed_identity(),cases=[])
 atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',name=run,run_id=run,group='TL-HB-MEM-CAL',config=c))
 metrics=Metrics(a.output,run)
 d=c['direct']['plan']['width'];batches=[8,16,32,64] if d==64 else [4,8,16,32]
 try:
  for mode in ('qry','matrix','blocked'):
   previous=None
   # qry supplies a matched physical-batch cost anchor, not recipe selection.
   for micro in (batches[:1] if mode=='qry' else batches):
    if previous and previous['peak_mib']*1.85>52000:
     report['cases'].append(dict(implementation=mode,micro=micro,state='not_attempted',reason='measured next-peak estimate exceeds52000MiB'));break
    if time.time()>a.deadline_epoch-120:raise TimeoutError('bounded calibration deadline')
    cfg=copy.deepcopy(c);cfg.pop('planned_inventory',None);cfg['training']['batch']=micro
    if mode=='qry':
     for m in cfg['direct']['plan']['memories']:m['memory_read']='qry'
     cfg['implementation'].pop('sum_implementation',None)
    else:cfg['implementation']['sum_implementation']=mode
    torch.npu.empty_cache();model=make_model(cfg,device);opt=optimizer_for(model,cfg);torch.npu.reset_peak_memory_stats();rows=[]
    for step in range(4):
     full=synthetic_batch(64,256,7000+step,device)
     inputs=[{k:v[i:i+micro] for k,v in full.items()} for i in range(0,64,micro)]
     rows.append(logical_update(model,opt,inputs,cfg,4*2**20,device))
    peak=memory(device)['resources/npu_peak_allocated_mib'];rate=statistics.median(x['tokens']/x['seconds'] for x in rows[1:])
    inference=[];model.eval()
    with torch.no_grad():
     for i in range(4):
      synchronize(device);start=time.monotonic();model(inputs[0]['input_ids']);synchronize(device);inference.append(time.monotonic()-start)
    item=dict(implementation=mode,micro=micro,state='completed',global_batch=64,accumulation=64//micro,
     targets_per_second=rate,peak_mib=peak,inference_seconds=statistics.median(inference[1:]),observations=rows)
    report['cases'].append(item);previous=item
    metrics.log(len(report['cases']),{'batch/physical':micro,'batch/global':64,'timing/targets_per_second':rate,
     'resources/peak_mib':peak,'mode/qry':int(mode=='qry'),'mode/blocked':int(mode=='blocked'),'timing/inference_seconds':item['inference_seconds']})
    atomic_json(a.output/'result.json',report)
    del model,opt,inputs;gc.collect();torch.npu.empty_cache()
  good=[r for r in report['cases'] if r['state']=='completed' and r['implementation']!='qry' and r['peak_mib']<52000]
  if not good:raise RuntimeError('no qualified sum physical batch')
  chosen=max(good,key=lambda r:r['targets_per_second'])
  report.update(state='completed',selected={k:chosen[k] for k in ('implementation','micro','global_batch','targets_per_second','peak_mib')},
   scope='Synthetic L256/global64 full VJP+Adam; first warmup excluded; inference same physical batch. Mem changes function/workload; no equivalent speedup claim. Same-batch and selected-batch observations separate.')
 except BaseException as e:report.update(state='failed',error=repr(e));raise
 finally:
  report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
  atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('config','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
