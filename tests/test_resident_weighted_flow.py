"""Core-5 public resident adapter: independent schedule and exact raw cuts.

This verifies Python-owned device execution separately from standalone LibTorch.
Local formulas/declared adjoints retain their component-specific anchors.
"""
import os
from dataclasses import replace

import pytest
import torch

from tidegraph import Continuation, External
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.weighted import WeightedResident
from resident_test_target import target_device
from test_weighted_execution import setup
from test_weighted_recurrent_execution import model as recurrent_model
from test_weighted_scoring_execution import configured
from weighted_precision_reference import reference_model, widen, assert_records


def owner(binding, target, *, q=None, prefill=True, **limits):
    return WeightedResident(binding.graph, binding.model, device=target,
        native_library=os.environ["TIDE_BUILD_DIR"],
        resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
        batch_size=1 if q is None else q.batch_size, continuation=q,
        limits=dict(queue=64,stages=64,cache=16,prefill=prefill,**limits))


@pytest.mark.parametrize("storage_dtype",[torch.float32,torch.float16])
@pytest.mark.parametrize("family",["kv","affine","delta","gated_delta","ssd"])
@pytest.mark.parametrize("prefill",[False,True])
def test_public_full_records_raw_checkpoint_and_private_windows(family,prefill,storage_dtype,tmp_path):
    target=target_device()
    b,_,xs,stop=(setup(storage_dtype,"pdg") if family=="kv" else recurrent_model(storage_dtype,family))
    q=Continuation(b.graph.identity,1)
    model=reference_model(b.model) if storage_dtype==torch.float16 else b.model
    with torch.no_grad():
        flow=owner(b,target,prefill=prefill)
        head=[x for x in xs if x.time<3];tail=[x for x in xs if x.time>=3]
        first=flow.advance(head,stop=3,sealed_until=3)
        expected=reference(b.graph,model,q,widen(head),3,sealed_until=3)
        assert_records(first,expected,storage_dtype)
        snapshot=flow.checkpoint()
        save(tmp_path/"weighted.pt",b.graph,b.model,snapshot)
        restored=load(tmp_path/"weighted.pt",b.graph,b.model)
        equivalent(snapshot,restored,rtol=0,atol=0)
        flow.close()
        flow=owner(b,target,q=restored,prefill=prefill)
        actual=flow.advance(tail,stop=stop,sealed_until=stop)
        expected=reference(b.graph,model,expected.continuation,widen(tail),stop,sealed_until=stop)
        assert_records(actual,expected,storage_dtype)
        view=flow.checkpoint()
        for state in view.states.values():state.value.fill_(float("nan"))
        equivalent(actual.continuation,flow.checkpoint())
        manifest=flow.manifest()
        assert manifest["execution"]=="resident" and manifest["effective_device"]==target
        assert 0<manifest["reserved_tensor_bytes"]<=manifest["effective_limits"]["tensor_bytes"]
        assert manifest["training"] is False
        assert manifest["dtype"]==str(storage_dtype).removeprefix("torch.")
        assert manifest["raw_dtype"]==manifest["dtype"] and manifest["compute_dtype"]=="float32"
        assert all(x[-1].dtype==storage_dtype for x in actual.outputs)
        assert all(x.value.dtype==storage_dtype for x in actual.continuation.pending)
        empty=flow.advance([],stop=stop,sealed_until=stop)
        assert not empty.trace
        equivalent(empty.continuation,actual.continuation)
        flow.close()


@pytest.mark.parametrize("case",[3,7,8,11])
def test_heterogeneous_formal_configurations_public_path(case):
    target=target_device();b,xs,stop=configured(torch.float32,case,family="pdg")
    with torch.no_grad():
        flow=owner(b,target)
        actual=flow.advance(xs,stop=stop,sealed_until=stop)
        expected=reference(b.graph,b.model,Continuation(b.graph.identity,1),xs,stop,sealed_until=stop)
        equivalent(actual,expected);flow.close()


def test_boundary_retry_and_parameter_version_guards():
    target=target_device();b,_,xs,stop=setup(torch.float32,"fc")
    with torch.no_grad():
        flow=owner(b,target)
        bad=[replace(xs[0],position=9),*xs[1:]]
        with pytest.raises(ValueError):flow.advance(bad,stop=stop,sealed_until=stop)
        assert flow.checkpoint().cut==0
        actual=flow.advance(xs,stop=stop,sealed_until=stop)
        expected=reference(b.graph,b.model,Continuation(b.graph.identity,1),xs,stop,sealed_until=stop)
        equivalent(actual,expected)
        next(b.model.parameters()).add_(.01)
        with pytest.raises(RuntimeError,match="parameters changed"):flow.advance([],stop=stop,sealed_until=stop)
        flow.close()


def test_weighted_loader_verifies_independent_library(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    from tidegraph.resident_loader import load_resident
    core=tmp_path/"_tide_native.so";core.write_bytes(b"never imported")
    module=tmp_path/"_tide_weighted_resident.so";module.write_bytes(b"also never imported")
    record=dict(runtime="python",core=dict(binary_sha256={core.name:hashlib.sha256(core.read_bytes()).hexdigest()}),
                binary_sha256={module.name:"incorrect"})
    (tmp_path/"control-build.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="binary differs"):
        load_resident(tmp_path,SimpleNamespace(__file__=str(core)),component="weighted")


@pytest.mark.parametrize("topology",["fc","hb"])
def test_general_settle_port_schedule_uses_same_compiled_graph(topology):
    target=target_device();b,spec,xs,stop=setup(torch.float32,topology)
    # General ports may have different global positions. Keep each port's own
    # occurrence indices contiguous, without changing ranks or graph delays.
    xs=[replace(x,time=x.time+spec.stride if x.port==1 else x.time) for x in xs]
    stop+=spec.stride
    with torch.no_grad():
        flow=owner(b,target)
        actual=flow.advance(xs,stop=stop,sealed_until=stop)
        expected=reference(b.graph,b.model,Continuation(b.graph.identity,1),xs,stop,sealed_until=stop)
        equivalent(actual,expected);flow.close()
