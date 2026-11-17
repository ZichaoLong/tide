#!/usr/bin/env python3
"""Bounded decomposition of retained L256 FP32 sum VJP discrepancy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,now,Metrics,project_to_lens
from tide_lab.train import runtime,source_identity,installed_identity
from tide_lab.sum_memory import sum_read
from tide_lab.direct_semantic_ops import time_weights
from tide_lab.language_checks import finite_delta

def main(a):
 device=runtime('npu:0',2,44);a.output.mkdir(parents=True,exist_ok=False);run='tl-v13-numeric-triage'
 r=dict(state='running',started_at=now(),source=source_identity(),installation=installed_identity(),rows=[],
  original='runs/v13-check-npu/result.json',strict_thresholds=dict(atol=1e-6,rtol=1e-5),scope='L256 short/long decomposition using identical FP32-rounded inputs; strict failures remain; no universal gradient error certification')
 metrics=Metrics(a.output,run);atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',run_id=run,name=run,group='TL-HB-MEM-CHECK'))
 try:
  w0=torch.rand(2,2,256);w0[0]=0;p0=torch.randn(2,2,256,5);g0=torch.randn_like(p0)
  for component in (0,1):
   results={}
   for label,dev,dtype in [('cpu64',torch.device('cpu'),torch.float64),('cpu32',torch.device('cpu'),torch.float32),('npu32',device,torch.float32)]:
    w=w0.to(dev,dtype).requires_grad_();p=p0.to(dev,dtype).requires_grad_()
    decay=time_weights(torch.ones(256,device=dev,dtype=dtype),7,.099)[0]
    masses=[torch.nn.functional.linear(w,decay),w.cumsum(-1)]
    y=sum_read(w,p,masses,decay,rate=.099,stride=7)[component]
    grad=torch.autograd.grad((y*g0.to(dev,dtype)).sum(),w,retain_graph=True)[0]
    future=torch.autograd.grad(y[...,:128,:].sum(),p)[0][...,128:,:]
    if not torch.isfinite(grad).all() or int(future.count_nonzero()):raise AssertionError('finite/causal VJP mechanism failure')
    results[label]=(y.detach().cpu(),grad.detach().cpu())
   r['rows'].append(dict(component='short' if component==0 else 'long',
    cpu32_vs_cpu64=[finite_delta(x,y,1e-6,1e-5) for x,y in zip(results['cpu32'],results['cpu64'])],
    npu32_vs_cpu64=[finite_delta(x,y,1e-6,1e-5) for x,y in zip(results['npu32'],results['cpu64'])],
    npu32_vs_cpu32=[finite_delta(x,y,1e-6,1e-5) for x,y in zip(results['npu32'],results['cpu32'])],future_input_vjp='exact_zero'))
  r.update(state='completed',admitted_for_finite_exploration=True,
   interpretation='Bounded finite FP32 history/quotient VJP rounding with exact future-input zero on this fixture; original strict failures retained; long-run near-boundary routing/learning unresolved.')
  metrics.log(0,{'check/causal_future_vjp_zero':1,'check/components':2})
 except BaseException as e:r.update(state='failed',error=repr(e));raise
 finally:
  r['finished_at']=now();atomic_json(a.output/'result.json',r);metrics.close()
  atomic_json(a.output/'completed.json',dict(producer_finished=True,state=r['state']));project_to_lens(a.output,a.lens_cli)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
