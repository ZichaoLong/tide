#!/usr/bin/env python3
"""Independent sum history/VJP and unchanged frozen qry source checks."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens
from tide_lab.train import runtime,source_identity,installed_identity
from tide_lab.language_checks import finite_delta
from tide_lab.direct_semantic_ops import time_weights
from tide_lab.sum_memory import sum_read
from tide_lab.direct_specialized import SpecializedSemanticLM
from scripts.prepare_direct_v9 import direct

def main(a):
 device=runtime(a.device,2,11);a.output.mkdir(parents=True,exist_ok=False)
 run='tl-v13-check-'+device.type;metrics=Metrics(a.output,run)
 report=dict(state='running',started_at=now(),source=source_identity(),installation=installed_identity(),
  device=str(device),parent_source_sha256=digest(a.parent_source/'tide_lab/direct_specialized.py'),rows=[],
  strict_thresholds=dict(atol=1e-6,rtol=1e-5),scope='local independent history/VJP; small old/new qry exact regression; formal per-config causality/recovery/learning separate')
 atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',run_id=run,name=run,group='TL-HB-MEM-CHECK'))
 try:
  torch.manual_seed(44)
  for length in (3,35,256):
   w0=torch.rand(2,2,length,dtype=torch.float64);w0[0]=0;p0=torch.randn(2,2,length,5,dtype=torch.float64)
   w0.requires_grad_();p0.requires_grad_();expected=[]
   for beta in (.099,0.):
    parts=[]
    for t in range(length):
     weight=w0[...,:t+1]*torch.exp(-beta*7*torch.arange(t,-1,-1,dtype=torch.float64))
     parts.append((weight.unsqueeze(-1)*p0[...,:t+1,:]).sum(-2)/weight.sum(-1).clamp_min(1).unsqueeze(-1))
    expected.append(torch.stack(parts,-2))
   g=[torch.randn_like(x) for x in expected]
   eg=torch.autograd.grad(sum((v*z).sum() for v,z in zip(expected,g)),(w0,p0))
   dtype=torch.float64 if device.type=='cpu' else torch.float32
   for mode in ('matrix','blocked'):
    w=w0.detach().to(device,dtype).requires_grad_();p=p0.detach().to(device,dtype).requires_grad_()
    decay=time_weights(torch.ones(length,device=device,dtype=dtype),7,.099)[0]
    masses=[torch.nn.functional.linear(w,decay),w.cumsum(-1)]
    y=sum_read(w,p,masses,decay,rate=.099,stride=7,implementation=mode)
    grads=torch.autograd.grad(sum((v*z.to(device,dtype)).sum() for v,z in zip(y,g)),(w,p))
    deltas=[finite_delta(x,y,1e-6,1e-5) for x,y in zip((*y,*grads),(*expected,*eg))]
    finite=all(bool(torch.isfinite(x).all()) for x in (*y,*grads))
    empty=all(int(x[0].count_nonzero())==0 for x in y)
    row=dict(length=length,implementation=mode,deltas=deltas,finite=finite,empty_exact_zero=empty)
    report['rows'].append(row)
    if not finite or not empty:raise AssertionError('sum mechanism failed')
  spec=importlib.util.spec_from_file_location('tide_lab._v12_specialized',a.parent_source/'tide_lab/direct_specialized.py')
  old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  plan=direct('check','hb',2,64,'temperature1',region_width=4,k=2)['direct']['plan']
  for s in plan['regions']:s.pop('activity_schedule',None)
  kwargs=dict(seed=11,attention='math' if device.type=='cpu' else 'npu_fusion_augmented',aggregate_vjp='ordered-v1')
  previous=old.SpecializedSemanticLM(plan,**kwargs).to(device);current=SpecializedSemanticLM(plan,**kwargs).to(device)
  x=torch.tensor([[1,2,3],[4,5,6]],device=device);a0=previous(x);b0=current(x)
  a0.square().mean().backward();b0.square().mean().backward()
  exact=torch.equal(a0,b0)
  for (n,p),(m,q) in zip(previous.named_parameters(),current.named_parameters()):
   exact &= n==m and torch.equal(p,q) and ((p.grad is None and q.grad is None) or (p.grad is not None and q.grad is not None and torch.equal(p.grad,q.grad)))
  report['qry_frozen_regression']=dict(state='passed' if exact else 'failed',forward_and_parameter_vjp_exact=exact)
  if not exact:raise AssertionError('qry numeric path changed')
  strict=all(d['state']=='passed' for row in report['rows'] for d in row['deltas'])
  report.update(state='completed',axes=dict(execution='passed',strict_numerical='passed' if strict else 'failed_finite_retained',
   discrete_semantics='empty_history_exact_zero',qry_regression='passed',recovery='per_chain_required',learning='unchecked'))
  metrics.log(0,{'check/local_strict_passed':int(strict),'check/qry_exact':int(exact),'check/cases':len(report['rows'])})
 except BaseException as e:report.update(state='failed',error=repr(e));raise
 finally:
  report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
  atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('output','parent-source'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--device',required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
