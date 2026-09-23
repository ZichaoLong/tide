"""Frozen runner contracts: actual owners, semantic clients, updates and accounting."""
import json
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import _tide_native as core
from tidegraph.compare import equivalent
from foundation_workloads import initialize, count, large_config, VARIANTS
from foundation_execute import Execution, loss, observations
from foundation_measure import execute_pass, measure

SUITE=json.loads((Path(__file__).resolve().parents[1]/'benchmarks/foundation-v1.json').read_text())


def small(identity,family=None,dtype=torch.float64):
    config=dict(next(c for c in SUITE['configurations'] if c['id']==identity))
    config.update(width=8,batch=2,sequence=4,dtype=str(dtype).split('.')[-1])
    if config['id']=='P02':config['body_nodes']=16
    if config['training_window']:config['training_window']=2
    if family:config['graph']=family
    return config


@pytest.mark.parametrize('identity',[c['id'] for c in SUITE['configurations']])
def test_count_matches_real_parameter_owners(dtype,identity):
    config=small(identity,'pdg' if identity=='TR01' else None,dtype)
    g,s,m=initialize(config)
    assert count(config)['parameters']==sum(p.numel() for p in m.parameters())
    assert count(config)['encoded_nodes']==len(g.nodes)+(2 if s else 0)


CASES=[(k,v) for k,vs in VARIANTS.items() if k not in {'P02','TR01'} for v in vs]
@pytest.mark.parametrize('identity,variant',CASES)
def test_benchmark_clients_full_semantics(dtype,identity,variant):
    config=small(identity,dtype=dtype)
    reference=Execution(config,'python-settle' if config['graph']=='settle' else 'python-stream',trace=True)
    actual=Execution(config,variant,workers=2,trace=True)
    with torch.no_grad():
        expected=reference.advance(config['sequence']);observed=actual.advance(config['sequence'])
    equivalent(expected,observed)
    assert observed.outputs


@pytest.mark.parametrize('family,variant',[('pdg','native-stream-packed'),('timed-dag','native-frontier'),('settle','native-settle')])
@pytest.mark.parametrize('modes',[['grad-forward'],['backward'],['optimizer'],['grad-forward','backward','optimizer','train-step']])
def test_actual_training_phases_and_instrumentation(dtype,family,variant,modes):
    config=small('TR01',family,dtype)
    a=Execution(config,variant,workers=2);b=Execution(config,variant,workers=2)
    initial={k:p.detach().clone() for k,p in a.model.named_parameters()}
    measured=execute_pass(a,modes);profile=execute_pass(b,modes,instrument=True)
    equivalent(a.q,b.q);equivalent(dict(a.model.named_parameters()),dict(b.model.named_parameters()))
    update=any(mode in {'optimizer','train-step'} for mode in modes)
    assert measured['optimizer_updates']==(2 if update else 0)
    assert all(v>0 for v in measured['seconds'].values())
    assert measured['observations']['logical']['Full']>0
    assert measured['observations']['outputs']==8
    assert all(s.value.grad_fn is None for s in a.q.states.values())
    changed=any(not torch.equal(p,initial[k]) for k,p in a.model.named_parameters())
    assert changed==update
    assert measured['work']=={}
    assert profile['work']['op/state_replay_worker_ns']>0
    assert all(v==0 for v in core.work_metrics().values())


def test_warmup_resets_parameter_trajectory_and_state(dtype):
    config=small('TR01','settle',dtype)
    a=Execution(config,'native-settle');b=Execution(config,'native-settle')
    ma,pa=measure(a,['train-step'],2);mb,pb=measure(b,['train-step'],0)
    equivalent(a.q,b.q);equivalent(dict(a.model.named_parameters()),dict(b.model.named_parameters()))
    assert ma['losses']==mb['losses']==pa['losses']==pb['losses']


def test_prefill_and_counted_fallback(dtype):
    config=small('T02',dtype=dtype)
    result=execute_pass(Execution(config,'native-frontier'),['nograd-forward'],instrument=True)
    stats=result['observations']['stats']
    assert stats['max_state_sequence']>1
    assert any(v>0 for k,v in stats.items() if 'prefill_fallback' in k)


def test_matmul_work_has_analytic_single_node_anchor(dtype):
    config=small('M01',dtype=dtype);config.update(body_nodes=1,batch=1,sequence=1)
    execution=Execution(config,'native-stream-scalar')
    result=execute_pass(execution,['nograd-forward'],instrument=True)
    # One Linear Attention event: Q/K/V projections and output; one tanh Full.
    assert result['work']['op/qkv_flops']==6*8*8
    assert result['work']['op/out_flops']==2*8*8
    assert result['work']['op/full_matmul_flops']==2*8*8
    assert result['observations']['logical']==dict(Aggregate=1,Upd=1,Read=1,Next=1,Full=1)


@pytest.mark.parametrize('preset',SUITE['large_presets'])
@pytest.mark.parametrize('family',['pdg','timed-dag','settle'])
def test_large_preflight_target_is_not_a_smoke_claim(preset,family):
    config=large_config(preset,family)
    counted=count(config)
    target=preset.get('reference_parameters',8500000000)
    assert abs(counted['parameters']-target)<counted['parameters_per_node']*preset['nominal_selection_denominator']
    assert config['batch']==512 and config['sequence']==6
    assert counted['body_nodes']>=4*preset['nominal_selection_denominator']
