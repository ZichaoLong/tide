#!/usr/bin/env python3
"""Observe the final native sending arithmetic at a captured failure boundary.

No optimizer updates, replacements, clamps or foundation mutations. A bounded
ring stores small candidate-shaped ATen operations so the actual centered
scores, temperature, received weights, shares and caps can be distinguished.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class CandidateHistory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows=deque(maxlen=64)
        self.calls=0

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}));self.calls+=1
        if (str(func) in {'aten.div.Tensor','aten.exp.default','aten.log.default','aten.mul.Tensor'}
                and isinstance(out,torch.Tensor) and out.ndim==2 and out.shape[1]==4
                and out.numel()<=512):
            self.rows.append(dict(operator=str(func),ordinal=self.calls,
                inputs=[x.detach().clone() if isinstance(x,torch.Tensor) else x for x in args],
                output=out.detach().clone()))
        return out

    def record(self):
        import math
        def tensor(t):
            if not isinstance(t,torch.Tensor):return t
            return dict(shape=list(t.shape),dtype=str(t.dtype),values=[
                v if math.isfinite(v) else str(v) for v in t.flatten().tolist()])
        return [dict(operator=r['operator'],ordinal=r['ordinal'],
                     inputs=[tensor(t) for t in r['inputs']],output=tensor(r['output'])) for r in self.rows]


def main(a):
    start=time.monotonic();root=a.root.resolve();parent=root/'runs/v6-positive-direct-received'
    old=root/'.local/source-v6-launch';sys.path.insert(0,str(old))
    from tide_lab.records import atomic_json,digest,now,Metrics
    from tide_lab.train import runtime,installed_identity
    from tide_lab.campaign_runtime import make_model
    manifest=json.loads((root/'runs/v6-cover-hb-direct-received/train/manifest.json').read_text())
    if installed_identity()!=manifest['identity']['installation']:raise ValueError('installation changed')
    native=Path(os.environ['TIDE_LAB_NATIVE_LIBRARY'])
    if digest(native/'_tide_native.so')!=manifest['identity']['native_binary_sha256']:raise ValueError('native changed')
    device=runtime('cpu',1,manifest['config']['seed'])
    model=make_model(manifest['config'],device)
    state=torch.load(parent/'failure-boundary.pt',weights_only=False,map_location='cpu')
    data=torch.load(parent/'failure-input.pt',weights_only=False,map_location='cpu')
    model.load_state_dict(state['model'],strict=True);model.train(data['training'])
    torch.set_rng_state(data['torch_rng'])
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((Path(__file__).resolve().parents[1]/'.tide-source.json').read_text())
    ident=dict(source_commit=source['commit'],source_sha256=source['content']['tree_sha256'],
               parent_boundary_sha256=digest(parent/'failure-boundary.pt'),
               parent_input_sha256=digest(parent/'failure-input.pt'),original_identity=manifest['identity'])
    run_id='tl-v6-positive-direct-boundary-context'
    atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',name=run_id,run_id=run_id,
        group='TL-POSITIVE-v6',study_id='TL-POSITIVE-v6',identity=ident,
        config=dict(stage='single captured forward',batch=8,length=32,device='cpu',threads=1)))
    observer=CandidateHistory();error=None
    try:
        with observer:model(data['input_ids'])
    except RuntimeError as exc:error=repr(exc)
    good=error=="RuntimeError('positive batched attention sending support underflowed')"
    atomic_json(a.output/'operator-context.json',dict(schema='tide-sending-boundary-context-v1',
        identity=ident,error=error,matching_original_exception=good,operator_calls=observer.calls,
        candidate_operation_tail=observer.record(),scope='read-only native ATen boundary trace; no optimizer update'))
    metrics=Metrics(a.output,run_id)
    metrics.log(state['progress']['step'],{'diagnostic/matching_original_exception':good,
        'diagnostic/observed_operators':observer.calls});metrics.close()
    atomic_json(a.output/'result.json',dict(schema='tide-sending-boundary-diagnosis-v1',
        state='completed' if good else 'failed',identity=ident,error=error,
        matching_original_exception=good,elapsed_seconds=time.monotonic()-start,finished_at=now()))
    print(error,flush=True)
    return 0 if good else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):p.add_argument('--'+key,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
