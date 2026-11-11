#!/usr/bin/env python3
"""Finite actual-config forward/gradient/causal witnesses; no learning claims."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now


def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model
    device=runtime('cpu',1,11);a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tide-direct-v8-implementation-check-v1',state='running',source=source_identity(),
        installation=installed_identity(),started_at=now(),cases=[],scope='Independent scalar/serial formula suite plus actual exported candidate CPU B2/L3 witnesses. No whole-foundation equivalence or long-trajectory qualification.')
    atomic_json(a.output/'result.json',report)
    with (a.output/'formula-tests.log').open('x') as log:
        r=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_direct_semantic.py','-v'],stdout=log,stderr=subprocess.STDOUT)
    report['formula_suite_exit_code']=r.returncode
    if r.returncode:raise RuntimeError('independent formula checks failed')
    for cid in json.loads((a.configs/'catalog.json').read_text())['cases']:
        c=json.loads((a.configs/(cid+'.json')).read_text());started=time.monotonic()
        try:
            model=make_model(c,device);x=torch.tensor([[1,2,3],[4,5,6]])
            y=model(x);changed=x.clone();changed[:,2]+=10
            with torch.no_grad():future=model(changed)
            if not torch.equal(y[:,:2],future[:,:2]):raise AssertionError('future changed earlier logits')
            loss=torch.nn.functional.cross_entropy(y.flatten(0,1),x.flatten());loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not torch.isfinite(y).all() or not gradients or any(not torch.isfinite(g).all() for g in gradients):raise FloatingPointError('nonfinite or missing gradient')
            item=dict(id=cid,state='passed',parameters=sum(p.numel() for p in model.parameters()),
                connected_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
                work=model.last_metrics,seconds=time.monotonic()-started)
            del model,y,future,loss,gradients
        except Exception as e:item=dict(id=cid,state='failed',error=repr(e),seconds=time.monotonic()-started)
        report['cases'].append(item);atomic_json(a.output/'result.json',report)
        print(json.dumps({k:v for k,v in item.items() if k!='work'}),flush=True)
    report.update(state='completed',finished_at=now(),passed=sum(c['state']=='passed' for c in report['cases']),
        failed=sum(c['state']=='failed' for c in report['cases']))
    atomic_json(a.output/'result.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('configs','output'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
