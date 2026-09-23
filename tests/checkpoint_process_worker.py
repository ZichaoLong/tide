"""Fresh-process acceptance client for the three existing checkpoint scopes.

The test driver supplies its fixed workload/update index separately. These files
do not claim to restore a data cursor, RNG or arbitrary training controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, os.environ.get('TIDE_BUILD_DIR', str(Path(__file__).resolve().parents[1]/'build')))
import torch
import _tide_native as core
from durable_records import write_json
from tidegraph.checkpoint import save, load
from tidegraph.checkpoint_io import publish
from tidegraph.checkpoint_values import encode
from tidegraph.checkpoint_ownership import parameter_aliases
from single_graph_training import Case, Encoded
from single_graph_optimizer import make_optimizer
from single_graph_roots import objective
from token_checkpoint_cases import fixture, NativeTwoClock, state, restore
from test_cpp_checkpoint import _fixture, _optimizer, _state_snapshot


def clone(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, dict): return {k:clone(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return type(value)(clone(v) for v in value)
    return value


def run_graph(args, dtype):
    if args.scope == 'single':
        case = Case(dtype,'all-softmax', True); runner = Encoded(case,'hst','native-packed'); app = None
    else:
        case, app = fixture(dtype,'all-softmax',True); runner = NativeTwoClock(case,'hst','packed')
    opt = make_optimizer(case,args.kind); aliases = parameter_aliases(case.owner)
    ids = [id(p) for p in case.owner.parameters()]
    if args.phase == 'suffix':
        with torch.no_grad():
            for p in case.owner.parameters(): p.add_(.03)
        if app: restore(runner,app.load(args.checkpoint,opt))
        else: runner.q = load(args.checkpoint,runner.adapter.graph,runner.adapter.model,opt)
        assert ids == [id(p) for p in case.owner.parameters()]
        assert parameter_aliases(case.owner) == aliases
    stops = (case.period-1, 3*case.period-1, 4*case.period)
    indices = (0,) if args.phase == 'prefix' else (1,2) if args.phase == 'suffix' else (0,1,2)
    records = []
    for i in indices:
        opt.zero_grad(set_to_none=True)
        for x in case.xs: x.value.grad = None
        frame = runner.advance(stops[i]); loss = objective(case,frame); loss.backward()
        grads = {n:clone(p.grad) for n,p in case.variables.items()}
        runner.detach(); opt.step()
        records.append(clone({'update':i, 'body':encode(frame.body.continuation),
            'readout':encode(frame.read.continuation), 'body_outputs':frame.body.outputs,
            'read_outputs':frame.read.outputs, 'buffer':frame.buffer, 'gradients':grads,
            'values':case.owner.state_dict(), 'optimizer':opt.state_dict(), 'aliases':aliases}))
    if args.phase == 'prefix':
        assert frame.buffer, 'The interrupted point must retain a partial readout window'
        if app: app.save(args.checkpoint,state(runner),opt)
        else: save(args.checkpoint,runner.adapter.graph,runner.adapter.model,runner.q,opt)
    return records


def run_native(args,dtype):
    registry,shared,other = _fixture(dtype,10. if args.phase == 'suffix' else 0.)
    opt = _optimizer(registry,args.kind,variant=args.phase == 'suffix')
    if args.phase == 'suffix': core.load_checkpoint(str(args.checkpoint),registry,opt,'fresh-process-named-values-v1')
    indices = (0,) if args.phase == 'prefix' else (1,2) if args.phase == 'suffix' else (0,1,2)
    records = []
    for i in indices:
        opt.zero_grad()
        loss = ((i+1)*shared).square().sum()
        if i: loss = loss + other.sum()*0
        loss.backward(); gradients = clone((shared.grad,other.grad)); opt.step()
        records.append({'update':i,'values':clone((shared,other)),'gradients':gradients,
                        'optimizer':_state_snapshot(opt),'aliases':registry.alias_partitions(),
                        'groups':opt.layout().groups,
                        'hyperparameters':[(g.lr,g.eps,g.weight_decay,g.momentum) for g in opt.groups()]})
    if args.phase == 'prefix': core.save_checkpoint(str(args.checkpoint),registry,opt,'fresh-process-named-values-v1')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',required=True,choices=['cpu'])
    p.add_argument('--dtype',required=True,choices=['float32','float64'])
    p.add_argument('--scope',required=True,choices=['single','application','native'])
    p.add_argument('--kind',required=True,choices=['sgd','adamw'])
    p.add_argument('--phase',required=True,choices=['baseline','prefix','suffix'])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    records = (run_native if args.scope == 'native' else run_graph)(args,getattr(torch,args.dtype))
    publish(args.output,records)
    binary = Path(core.__file__)
    metadata = {'pid':os.getpid(),'phase':args.phase,'scope':args.scope,'exit_code':0,
                'device':'cpu','resolution_reason':'explicit:cpu','dtype':args.dtype,'threads':torch.get_num_threads(),
                'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'dirty':subprocess.check_output(['git','status','--porcelain'],text=True),
                'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint.exists() else None}
    write_json(args.output.with_suffix('.json'),metadata)
    assert json.loads(args.output.with_suffix('.json').read_text()) == metadata


if __name__ == '__main__': main()
