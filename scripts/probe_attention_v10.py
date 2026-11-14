#!/usr/bin/env python3
"""Bounded NPU weighted-fusion forward and receipt-gradient probe."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens


def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.special_attention import weighted_attention
    device=runtime('npu:0',2,11);a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tide-v10-fusion-probe-v1',state='running',source=source_identity(),
        installation=installed_identity(),started_at=now(),cases=[])
    atomic_json(a.output/'result.json',report)
    metrics=Metrics(a.output,'tl-v10-fusion-probe')
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',
        run_id='tl-v10-fusion-probe',name='tl-v10-fusion-probe',group='TL-PERF-v10',study_id='TL-PERF-v10',created_at=now()))
    for i,(d,rate,empty) in enumerate([(32,0.,False),(32,.099,False),(64,0.,False),(128,0.,False),(32,.099,True)]):
        torch.manual_seed(71+i)
        q,k,v=[torch.randn(2,1,32,d) for _ in range(3)]
        w=torch.rand(2,1,32);w[...,::3]=0
        if empty:w.zero_()
        inputs=[q,k,v,w];items=[]
        try:
            for dev,impl in [(torch.device('cpu'),'math'),(device,'npu_fusion_augmented')]:
                args=[x.to(dev).detach().requires_grad_() for x in inputs]
                out=weighted_attention(*args,rate=rate,stride=7,implementation=impl)
                upstream=torch.linspace(-1,1,out.numel()).reshape(out.shape).to(dev)
                grads=torch.autograd.grad((out*upstream).sum(),args)
                items.append([out.detach().cpu(),*[g.detach().cpu() for g in grads]])
            rows=[]
            for name,x,y in zip(('output','q','k','v','receipt'),*items):
                if not torch.isfinite(y).all():raise FloatingPointError('nonfinite '+name)
                rows.append(dict(name=name,max_abs=float((x-y).abs().max()),reference_max=float(x.abs().max()),
                    strict_passed=torch.allclose(x,y,atol=1e-6,rtol=1e-5)))
            receipt_nonzero=bool(items[1][-1].abs().max()>0) if not empty else bool(items[1][-1].abs().max()==0)
            if not receipt_nonzero:raise AssertionError('receipt-gradient/empty-row mechanism failed')
            item=dict(d=d,rate=rate,empty=empty,state='completed',comparisons=rows,receipt_mechanism=True)
            metrics.log(i,{'probe/max_abs':max(r['max_abs'] for r in rows),'probe/strict_passed':all(r['strict_passed'] for r in rows)})
        except Exception as e:item=dict(d=d,rate=rate,empty=empty,state='failed',error=repr(e))
        report['cases'].append(item);atomic_json(a.output/'result.json',report);print(json.dumps(item),flush=True)
    report.update(state='completed' if all(x['state']=='completed' for x in report['cases']) else 'failed',finished_at=now())
    atomic_json(a.output/'result.json',report);metrics.close()
    atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']))
    project_to_lens(a.output,a.lens_cli)
    if report['state']!='completed':raise RuntimeError('fusion probe failed; inspect preserved cases')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lens-cli',required=True)
    main(p.parse_args())
