"""Complete records, independent roots, shared owners, optimizer and recovery."""
import os
import pytest
import torch
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.weighted import observe_support
from positive_support_case import fixture, execute, start, roots, optimizer
from isolated_cases import vjp


@pytest.fixture
def device():
    name = os.environ.get("TIDE_POSITIVE_SUPPORT_DEVICE","cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401; explicit target, never silent fallback.
        torch.npu.set_device(name)
    return name


@pytest.mark.parametrize("read",["sum","qry"])
def test_complete_records_roots_and_positive_kv(device,read):
    b,xs,stop = fixture(device,read=read)
    with observe_support() as obs:
        expected = execute(b,start(b),xs,stop,False)
        actual = execute(b,start(b),xs,stop)
    assert obs.metrics()["repaired_coordinates"] > 0
    equivalent(expected,actual)
    tiny = actual.continuation.states[0,4]
    assert tiny.observations == 3 and tiny.slots["times"].tolist() == [2,3,5]
    assert tiny.slots["weights"].tolist() == [2.**-149]*3
    assert tiny.slots["values"].abs().max() > .01
    assert len({m.source for m in actual.messages if m.node == 5}) == 4
    assert b.model.nodes[0].kernel.base is b.model.nodes[1].kernel.base
    assert b.model.nodes[0].kernel.adapter_u is not b.model.nodes[1].kernel.adapter_u
    variables = roots(b,xs)
    equivalent(vjp(objective(expected),variables),vjp(objective(actual),variables))
    for a,c in ((expected.outputs[0][-1],actual.outputs[0][-1]),
                (expected.continuation.states[0,4].slots["values"],tiny.slots["values"])):
        equivalent(vjp(a,variables),vjp(c,variables))
        equivalent(vjp(a,variables,zero=True),vjp(c,variables,zero=True))
    bars = vjp(actual.outputs[0][-1],variables)
    assert any(v is None for k,v in bars.items() if k.startswith("input."))


def test_adamw_pending_retained_and_disk_recovery(device,tmp_path):
    a,xa,stop = fixture(device); b,xb,_ = fixture(device)
    oa,ob = optimizer(a),optimizer(b)
    for _ in range(2):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        ref = execute(a,start(a),xa,stop,False); got = execute(b,start(b),xb,stop)
        equivalent(ref,got)
        objective(ref).backward(); objective(got).backward()
        equivalent([p.grad for p in a.model.parameters()],[p.grad for p in b.model.parameters()])
        oa.step(); ob.step()
        equivalent(a.model.state_dict(),b.model.state_dict()); equivalent(oa.state_dict(),ob.state_dict())
    cut=2
    head=execute(b,start(b),[x for x in xb if x.time<cut],cut)
    assert head.continuation.pending
    tail=[x for x in xb if x.time>=cut]
    ref=execute(b,head.continuation,tail,stop,False)
    got=execute(b,head.continuation,tail,stop)
    equivalent(ref,got)
    equivalent(vjp(objective(ref),roots(b,xb)),vjp(objective(got),roots(b,xb)))
    save(tmp_path/"cut.pt",b.graph,b.model,head.continuation,ob)
    c,xc,_=fixture(device); oc=optimizer(c)
    restored=load(tmp_path/"cut.pt",c.graph,c.model,oc)
    equivalent(head.continuation.detach(),restored,atol=0,rtol=0)
    equivalent(ob.state_dict(),oc.state_dict(),atol=0,rtol=0)
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    ref=execute(b,head.continuation.detach(),tail,stop)
    got=execute(c,restored,[x for x in xc if x.time>=cut],stop)
    equivalent(ref,got,atol=0,rtol=0)
    objective(ref).backward(); objective(got).backward(); ob.step(); oc.step()
    equivalent(b.model.state_dict(),c.model.state_dict(),atol=0,rtol=0)
    equivalent(ob.state_dict(),oc.state_dict(),atol=0,rtol=0)


def test_strict_refusal_and_policy_checkpoint_identity(tmp_path):
    b,xs,stop=fixture(policy="strict")
    with pytest.raises(ArithmeticError,match="support underflowed"):
        execute(b,start(b),xs,stop)
    save(tmp_path/"strict.pt",b.graph,b.model,start(b))
    c,_,_=fixture()
    assert c.graph.identity != b.graph.identity
    with pytest.raises(ValueError,match="schema/graph mismatch"):
        load(tmp_path/"strict.pt",c.graph,c.model)


def test_cpu_fp64_independent_complete_reference(device):
    from dataclasses import replace
    from positive_support_case import cast_tree
    b,xs,stop=fixture(device)
    a,xa,_=fixture()
    a.model.double()
    xa=[replace(x,value=x.value.detach().double().requires_grad_()) for x in xa]
    ref=execute(a,start(a),xa,stop,False)
    got=execute(b,start(b),xs,stop)
    equivalent(cast_tree(ref),cast_tree(got))
    equivalent(cast_tree(vjp(objective(ref),roots(a,xa))),
               cast_tree(vjp(objective(got),roots(b,xs))))


def test_fresh_process_exact_next_window(device,tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    import tidegraph
    # Child must import the same selected source/installation as this check.
    env=dict(os.environ,PYTHONPATH=str(Path(tidegraph.__file__).resolve().parent.parent))
    for phase in ("write","resume"):
        subprocess.run([sys.executable,str(Path(__file__).with_name("positive_support_resume.py")),
                        phase,"--device",device,"--directory",str(tmp_path/"fresh")],
                       env=env,check=True,timeout=120)
