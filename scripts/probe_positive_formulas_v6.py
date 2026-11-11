#!/usr/bin/env python3
"""Small public-API/independent-formula witnesses, not a replacement core.

Reads terminal replay evidence and writes only a separate review evidence file.
Decimal Bernoulli reference is local to a captured four-candidate row.
"""
import argparse
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from scripts.review_io import write_review_json
from tide_lab.records import digest, now
from tide_lab.train import installed_identity
from tidegraph import Graph, Node, Region, External, Continuation, ExecutionPlacement
from tidegraph.weighted import bind, KVConfig, RegionalConfig, allocate, attention_send
from tidegraph.weighted.execution import eager_runner


def bernoulli_reference(record):
    """100-digit scalar mathematical response / root, independent of Torch."""
    raw = {k: v['values'] for k, v in record['tensors'].items()}
    with localcontext() as ctx:
        ctx.prec = 100
        d = lambda x: Decimal.from_float(float(x))
        s, mu = list(map(d, raw['scores'])), list(map(d, raw['caps']))
        tau = d(raw['temperature'][0])
        shift = d(raw.get('shift', [0.])[0])
        branch = record['details']['branch']
        def response(lam):
            return [cap/(1+(-(score-shift-lam)/tau).exp()) for score, cap in zip(s, mu)]
        lam = Decimal(0)
        target = None
        if branch != 'slack':
            target = d(raw['upper' if branch == 'upper' else 'lower'][0])
            lo, hi = min(s)-shift-tau, max(s)-shift+tau
            distance = max(Decimal(1), hi-lo)
            for _ in range(32):
                if sum(response(lo)) >= target >= sum(response(hi)):
                    break
                lo -= distance; hi += distance; distance *= 2
            else:
                raise ValueError('bounded Decimal bracket failed')
            for _ in range(300):
                lam = (lo+hi)/2
                if sum(response(lam)) > target:
                    lo = lam
                else:
                    hi = lam
            lam = (lo+hi)/2
        exact = response(lam)
        return dict(precision=100, branch=branch, multiplier=str(lam),
                    controls=[str(v) for v in exact],
                    target_residual=None if target is None else str(abs(sum(exact)-target)),
                    fp32_rounded=torch.tensor([float(v) for v in exact], dtype=torch.float32).tolist(),
                    mathematically_below_half_fp32_subnormal=[v < Decimal(2)**-150 for v in exact])


def captured_bern(path):
    r = json.loads(path.read_text())
    fields = {k: torch.tensor(v['values'], dtype=torch.float64).reshape(v['shape'])
              for k, v in r['tensors'].items()}
    probes = []
    for dtype in (torch.float32, torch.float64):
        f = {k: v.to(dtype) for k, v in fields.items()}
        try:
            a = allocate(f['scores'], f['caps'], f['lower'], f['upper'], f['temperature'],
                         regularizer='bern', score_shift=f.get('shift'))
            probe = dict(state='completed', controls=a.controls.tolist(), residual=float(a.residual))
        except ArithmeticError as e:
            probe = dict(state='failed', error=repr(e))
        probes.append(dict(dtype=str(dtype), **probe))
    x = fields['response'].float()/fields['temperature'].float()
    stable = fields['caps'].float()*F.logsigmoid(x).exp()
    return dict(captured=r, capture_sha256=digest(path), decimal=bernoulli_reference(r),
                normalized_response=x.tolist(), stable_evaluation_of_captured_response=stable.tolist(),
                public_scalar_probes=probes,
                scope='captured local row only; a stable formula probe is not an implemented solver or accepted model repair')


def captured_direct(root):
    path=root/'runs/v6-positive-direct-boundary-context/operator-context.json'
    context=json.loads(path.read_text())
    observed=json.loads((root/'runs/v6-positive-direct-received/operator-observation.json').read_text())
    event=observed['last_events'][-1]
    row,column=event['lost_coordinates'][0]
    history=context['candidate_operation_tail']
    product=history[-1]
    assert product['operator']=='aten.mul.Tensor' and product['ordinal']==event['ordinal']
    assert product['inputs']==event['inputs'] and product['output']==event['output']
    division=next(x for x in reversed(history) if x['operator']=='aten.div.Tensor')
    logarithm=next(x for x in reversed(history) if x['operator']=='aten.log.default')
    def at(t):
        return t['values'][row*t['shape'][1]:(row+1)*t['shape'][1]]
    scores=at(division['inputs'][0]);tau=at(division['inputs'][1])[0]
    received=at(logarithm['inputs'][0]);caps=at(product['inputs'][0]);shares=at(product['inputs'][1])
    with localcontext() as ctx:
        ctx.prec=100
        d=lambda x:Decimal.from_float(float(x))
        weights=[d(r)*(d(s)/d(tau)).exp() for r,s in zip(received,scores)]
        probabilities=[w/sum(weights) for w in weights]
        controls=[d(c)*p for c,p in zip(caps,probabilities)]
        actual_product=d(caps[column])*d(shares[column])
        independent=dict(precision=100, shares=[str(x) for x in probabilities],
            controls=[str(x) for x in controls], captured_operand_product=str(actual_product),
            below_half_fp32_subnormal=actual_product<Decimal(2)**-150)
    probes=[]
    for dtype in (torch.float32,torch.float64):
        try:
            c=attention_send(torch.tensor(scores,dtype=dtype),torch.tensor(caps,dtype=dtype),
                torch.tensor(received,dtype=dtype),mode='direct_attention',base='received',temperature=tau)
            result=dict(state='completed',controls=c.tolist())
        except ArithmeticError as e:result=dict(state='failed',error=repr(e))
        probes.append(dict(dtype=str(dtype),**result))
    return dict(context_sha256=digest(path),local_batch_row=row,local_candidate_column=column,
        centered_scores=scores,temperature=tau,received=received,caps=caps,shares=shares,
        actual_controls=at(product['output']),decimal=independent,public_scalar_probes=probes,
        scope='actual native max-centered score operands and final product; absolute pre-centering scores/global node and event-time metadata not recorded; no full-model FP64 claim')


def public_aggregate(native):
    graph = Graph(tuple(Node(0) for _ in range(4)), (), (Region(2),), (0,1,2,3), (0,1,2,3))
    rows = []
    for policy, implementation in [('strict','python'), ('strict','native'), ('positive-floor-v1','python')]:
        b = bind(graph, memories=KVConfig(width=2, summary_width=2, adapter_rank=1, ffn_width=3, memory_read='qry'),
                 regions=RegionalConfig(k=2, regularizer='ent', ent_alpha=1., support_policy=policy),
                 dtype=torch.float32, seed=7)
        xs = [External(0,i,0,0,torch.tensor([2.**-149,.25,.25])) for i in range(4)]
        run = eager_runner(b, implementation, ExecutionPlacement(preset='cpu'), packing='packed',
                           native_library=str(native) if implementation == 'native' else None)
        result = run(Continuation(b.graph.identity,1), xs, 1, sealed_until=1)
        state = result.continuation.states[0,0]
        rows.append(dict(policy=policy, implementation=implementation, execution='completed',
                         expected_received_content=[.25,.25], received_content=result.trace[0]['content'].tolist(),
                         raw_kv_weights=state.slots['weights'].tolist(), raw_kv_keys=state.slots['keys'].tolist(),
                         observations=state.observations))
    return rows


def descriptor_cast_witness():
    """Same positive FP64 payload; explicit supported descriptor precisions."""
    graph=Graph(tuple(Node(0) for _ in range(4)),(),(Region(2),),(0,1,2,3),(0,1,2,3))
    rows=[]
    for precision in ('float64','float32'):
        b=bind(graph,memories=KVConfig(width=2,summary_width=2,adapter_rank=1,ffn_width=3),
            regions=RegionalConfig(selection_mode='proportional',k=2),dtype=torch.float64,seed=7)
        xs=[External(0,i,0,0,torch.tensor([1e-46,.25,.25],dtype=torch.float64)) for i in range(4)]
        run=eager_runner(b,'python',ExecutionPlacement(preset='cpu',scoring_dtype=precision),packing='packed')
        result=run(Continuation(b.graph.identity,1),xs,1,sealed_until=1)
        rows.append(dict(payload_dtype='float64',scoring_dtype=precision,execution='completed',
            active=[e['active'] for e in result.trace],controls=[float(e['control']) for e in result.trace],
            kv_observations=[s.observations for s in result.continuation.states.values()]))
    return rows


def formula_probes():
    z = torch.tensor(-90., dtype=torch.float32)
    # This reference and alternative expression never drive a graph execution.
    sigmoid = dict(z=float(z), actual=float(z.sigmoid()), stable=float(F.logsigmoid(z).exp()),
                   float64=float(z.double().sigmoid()), fp32_min_subnormal=2.**-149)
    s = torch.tensor([90.,0.], requires_grad=True)
    a = allocate(s, torch.ones(2), 0., 2., 1., regularizer='bern')
    root = torch.tensor([1e30,0.])
    gradient = torch.autograd.grad(a.controls, s, root)[0]
    with localcontext() as ctx:
        ctx.prec = 100
        tiny = (-Decimal(90)).exp()
        expected = Decimal.from_float(float(root[0]))*tiny/(1+tiny)**2
    bern_vjp = dict(scores=s.tolist(), controls=a.controls.tolist(), root=root.tolist(),
                    actual_gradient=gradient.tolist(), expected_first_gradient=str(expected),
                    scope='public scalar Bernoulli slack VJP; synthetic positive saturated tail, not a training trajectory')
    logits = torch.tensor([0.,-120.]); values = torch.tensor([0.,1e30])
    weights = logits.softmax(0)
    with localcontext() as ctx:
        ctx.prec = 100
        tail = (-Decimal(120)).exp()
        attention_exact = tail/(1+tail)*Decimal.from_float(float(values[1]))
    attention = dict(logits=logits.tolist(), values=values.tolist(), probabilities=weights.tolist(),
        actual=float((weights*values).sum()), expected=str(attention_exact),
        scope='independent forward expression used by KV AttentionMean; no captured training occurrence')
    epsilon = 1e-50
    KVConfig(epsilon=epsilon)
    positive_config = dict(epsilon=epsilon, config_accepted=True, fp32_value=float(torch.tensor(epsilon)),
        scope='configuration positivity is checked as a host number; this probe does not certify the full graph')
    decay = dict(age=2000, rate=.1, fp32_value=float(torch.exp(torch.tensor(-200.))),
                 fp64_value=math.exp(-200.),
                 scope='explicitly allowed short-channel floating decay; raw state existence must remain independent')
    return dict(sigmoid_tail=sigmoid, bern_vjp=bern_vjp, attention_expression=attention,
                positive_configuration_cast=positive_config, passive_decay=decay)


def main(a):
    torch.set_num_threads(1)
    root = a.root.resolve()
    result = dict(schema='tide-positive-local-analysis-v1', created_at=now(),
        installation=installed_identity(), script_sha256=digest(__file__),
        scope='bounded CPU FP32/FP64/Decimal local witnesses; no foundation edits, training repair or all-path qualification',
        local_probes=formula_probes(), aggregate_witness=public_aggregate(root/'.local/native-build-v6'),
        descriptor_cast_witness=descriptor_cast_witness(),
        actual_bern={},actual_direct=captured_direct(root))
    for name in ('regional-total-bern','constrained-bern'):
        path=root/'runs'/('v6-positive-'+name)/'allocation-failure.json'
        if not path.exists():
            raise ValueError('wait for actual replay capture: '+str(path))
        result['actual_bern'][name]=captured_bern(path)
    write_review_json(a.output,result,root=root)
    print(json.dumps({k:{'normalized_response':v['normalized_response'],
                         'decimal_controls':v['decimal']['controls'],
                         'fp32_rounded':v['decimal']['fp32_rounded']} for k,v in result['actual_bern'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
