#!/usr/bin/env python3
"""Audit receipt/sending denominators and small CPU fixed-route mechanisms.

No training, NPU use, throughput benchmark, or application-resume claim.
The forced routes are diagnostic controls, not learned low-activity results.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tidegraph.planning.recipes import prepare
from tide_lab.campaign_config import local_candidates
from tide_lab.direct_semantic import DirectSemanticLM
from tide_lab.direct_semantic_memory import Memory
from tide_lab.direct_semantic_ops import allocation
from tide_lab.records import now
from scripts.resolve_direct_v8 import recipe
from scripts.review_io import write_review_json


def fixed_route(graph, count):
    """Independent Boolean edge propagation, not the model's tensor execution."""
    sent = set()
    rows = []
    for region in sorted(range(graph['regions']), key=lambda j: graph['ranks'][j]):
        members = [v for v, j in enumerate(graph['node_regions']) if j == region]
        received = [v for v in members if v in graph['inputs'] or any(
            src in sent and dst == v for src, dst, _ in graph['edges'])]
        selected = received[:count]
        sent.update(selected)
        rows.append(dict(region=region, received=received, selected=selected))
    return rows


class FixedSend(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def forward(self, received, memory):
        present = received > 0
        sending = present & (present.cumsum(0) <= self.count)
        self.last_support_errors = received.new_zeros(())
        return sending.to(received.dtype), received.new_zeros(())


def receipt_witness(region_width, selected_count, source_recipe):
    spec = recipe(source_recipe, 'hb', 32)
    spec.update(region_width=region_width, depth=2)
    plan, _ = prepare(spec, samples=1, positions=1, seed=11)
    model = DirectSemanticLM(plan).double()
    expected = fixed_route(plan['graph'], selected_count)
    observed = {}
    hooks = []
    for j, memory in enumerate(model.memories):
        def capture(module, args, j=j):
            r = args[1]
            observed[j] = [model.members[j][i] for i in range(r.shape[0]) if r[i, 0, 0] > 0]
        hooks.append(memory.register_forward_pre_hook(capture))
    model.selectors = nn.ModuleList([FixedSend(selected_count) for _ in model.members])
    with torch.no_grad():
        result = model(torch.tensor([[7, 12, 9, 3]]))
    for hook in hooks:
        hook.remove()
    assert torch.isfinite(result).all()
    for row in expected:
        assert row['received'] == observed[row['region']]
    received = sum(len(row['received']) for row in expected)
    full = sum(len(row['selected']) for row in expected)
    nodes = plan['graph']['nodes']
    assert model.last_metrics['work/candidates'] == received * 4
    assert model.last_metrics['work/full'] == full * 4
    assert model.last_metrics['work/physical_full_rows'] == nodes * 4
    return dict(region_width=region_width, selected_per_nonempty_region=selected_count,
                width=32, dtype='float64', device='cpu', batch=1, length=4,
                nodes=nodes, received=received, full=full,
                receive_fraction=received/nodes, full_over_candidates=full/received,
                full_over_graph=full/nodes, physical_full_fraction=1.,
                rows=expected, tensor_execution_matches_independent_edges=True,
                scope='Fixed-route mechanism witness, not trained routing or performance qualification.')


def memory_witness(source_recipe):
    plan, _ = prepare(recipe(source_recipe, 'hb', 32), samples=1, positions=1, seed=11)
    spec = copy.deepcopy(plan['memories'][0])
    spec['memory_read'] = 'sum'
    torch.manual_seed(29)
    memory = Memory(1, spec, plan['stride']).double()
    x = torch.randn(1, 1, 3, 32, dtype=torch.float64, requires_grad=True)
    # Only final Full contributes to loss: the first received event was unsent.
    received = x.new_tensor([[[1., 0., 1.]]])
    content = memory(x, received)['content']
    grad = torch.autograd.grad(content[:, :, 2].square().sum(), x)[0]
    assert grad[:, :, 0].abs().max() > 1e-12
    assert torch.equal(grad[:, :, 1], torch.zeros_like(grad[:, :, 1]))
    altered = x.detach().clone()
    altered[:, :, 1] += 37
    with torch.no_grad():
        original = memory(x.detach(), received)['content'][:, :, 2]
        changed = memory(altered, received)['content'][:, :, 2]
    torch.testing.assert_close(original, changed, rtol=1e-10, atol=1e-12)
    return dict(unsent_received_event_has_future_gradient=float(grad[:, :, 0].abs().max()),
                absent_event_has_zero_input_gradient=True,
                absent_content_perturbation_future_max_abs=float((original-changed).abs().max()),
                scope='CPU FP64 KV/sum, three positions, same block; no continuation claim.')


def inspect():
    torch.set_num_threads(1)
    source_recipe = next(r for r in local_candidates() if r['id'] == 'temperature1')
    previous_path = ROOT/'docs/evidence/direct-v8-results.json'
    previous = json.loads(previous_path.read_text())
    ids = {'hb-anchor-s11', 'hb-temperature1-s11', 'hb-budget-div4-s11',
           'hb-send-hard_selection-s11', 'hb-send-pooled_attention-s11'}
    rows = []
    for row in previous['cases']:
        if row['id'] not in ids:
            continue
        w = row['final_work']
        n, c, a = (w[k] for k in ('work/physical_full_rows', 'work/candidates', 'work/full'))
        rows.append(dict(id=row['id'], record=row['record'], validation_nll=row['validation_nll'],
                         graph_slots=n, candidates=c, logical_full=a,
                         receive_fraction=c/n, full_over_candidates=a/c,
                         full_over_graph=a/n, physical_full_fraction=1.))
    inventories = []
    for region_width, control, depth in ((32, 'context', 2), (64, 'context', 2),
                                       (32, 'fixed', 2), (64, 'fixed', 2),
                                       (32, 'fixed', 3)):
        spec = recipe(source_recipe, 'hb', 64)
        spec.update(region_width=region_width, depth=depth)
        if control == 'fixed':
            spec['sparse_denominator'] = 8 if region_width == 32 else 32
            spec['region_overrides'].update(region_control_mode='fixed', region_evidence=None,
                                           total_fraction=.5)
        plan, _ = prepare(spec, samples=1, positions=1, seed=11)
        with torch.device('meta'):
            model = DirectSemanticLM(plan)
        inventories.append(dict(region_width=region_width, width=64, control=control, depth=depth,
                                k=plan['regions'][0]['k'], recipe=spec,
                                nodes=plan['graph']['nodes'], regions=plan['graph']['regions'],
                                parameters=sum(p.numel() for p in model.parameters()),
                                max_source_slots=max(map(len, model.connections))))
    score = torch.tensor([[2., 0.]], dtype=torch.float64, requires_grad=True)
    cap = torch.ones_like(score)
    budget = torch.ones((1, 1), dtype=torch.float64)
    c, _ = allocation(score, cap, budget, budget, .25, 'quad')
    grad = torch.autograd.grad((c * c.new_tensor([[1., 3.]])).sum(), score)[0]
    assert torch.equal(c, c.new_tensor([[1., 0.]]))
    assert torch.equal(grad, torch.zeros_like(grad))
    score64 = torch.full((1, 64), -4., dtype=torch.float64)
    score64[0, :2] = score64.new_tensor([.1, -.1])
    score64.requires_grad_()
    budget64 = score64.new_tensor([[1.5]])
    c64, error64 = allocation(score64, torch.ones_like(score64), budget64, budget64, 1., 'quad')
    upstream = torch.arange(64, dtype=torch.float64).reshape(1, 64)
    grad64 = torch.autograd.grad((c64 * upstream).sum(), score64)[0]
    assert (c64 > 0).sum() == 2 and (grad64 != 0).sum() == 2
    assert error64.max() < 1e-10
    files = ('scripts/review_hb_activity.py', 'tide_lab/direct_semantic.py',
             'tide_lab/direct_semantic_memory.py', 'tide_lab/direct_semantic_ops.py')
    return dict(schema='tide-hb-activity-alignment-v1', observed_at=now(),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                foundation_selection=json.loads((ROOT/'dependencies/foundation-source.json').read_text()),
                previous_results_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                previous_validation=rows, static_inventories=inventories,
                fixed_route_cpu_witnesses=[receipt_witness(n, a, source_recipe) for n, a in ((4,1),(32,1),(32,3),(64,2))],
                memory_witness=memory_witness(source_recipe),
                singleton_quad_score_vjp=dict(sending=c.tolist(), score_gradient=grad.tolist()),
                two_of_64_quad_score_vjp=dict(sending=c64.tolist(), score_gradient=grad64.tolist(),
                                             scope='Synthetic feasible point, not a learned routing result.'),
                scope='Read-only historical review plus bounded CPU mechanism checks. No new training, NPU, throughput, or recovery evidence.')


if __name__ == '__main__':
    report = inspect()
    write_review_json('docs/evidence/hb-activity-alignment.json', report)
    print(json.dumps(dict(historical=len(report['previous_validation']),
                         cpu_witnesses=len(report['fixed_route_cpu_witnesses']), scope=report['scope'])))
