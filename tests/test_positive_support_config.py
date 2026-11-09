"""Public configuration domain, identity compatibility and backend refusal."""
from copy import deepcopy
from dataclasses import asdict
import pytest
import torch
from tidegraph.experiment.config import resolve
from tidegraph.planning.model import resolve_model
from tidegraph.planning.topology import layered
from tidegraph.weighted import RegionalConfig, RecurrentConfig, bind, fully_connected
from tidegraph.weighted.native import encode, region_config
from positive_support_case import fixture


def request():
    return dict(schema="tide-experiment-v1",name="support",
        model=dict(recipe=dict(topology="fc",depth=2,region_width=2,width=8,history=4,
            regularizer="ent",ent_alpha=1.,region_overrides=dict(support_policy="positive-floor-v1"))),
        task=dict(samples=2,positions=3,input_pattern="staggered-v1"),
        execution=dict(device="cpu",client="pytorch",dtype="float32"))


def test_resolved_policy_and_default_identity():
    a=request(); b=deepcopy(a)
    b["model"]["recipe"]["region_overrides"]={}
    new,old=resolve(a),resolve(b)
    assert new["model_id"]!=old["model_id"]
    assert all(r["support_policy"]=="positive-floor-v1" for r in new["plan"]["regions"])
    assert all("support_policy" not in r for r in old["plan"]["regions"])
    g=layered("fc",depth=2,region_width=2)
    r=asdict(RegionalConfig())
    first,_=resolve_model(g,{},r)
    r.pop("support_policy")
    assert first==resolve_model(g,{},r)[0]
    assert "support_policy" not in RegionalConfig().identity_fields()


@pytest.mark.parametrize("execution",[
    dict(device="cpu",client="python-native"),dict(device="cpu",client="libtorch"),
    dict(device="cuda:0",client="python-native",placement="resident"),
    dict(device="cpu",client="pytorch",dtype="float16"),
    dict(device="npu:0",client="pytorch",peer_devices=["npu:0","npu:1"])])
def test_unimplemented_execution_refuses(execution):
    r=request();r["execution"]=execution
    with pytest.raises(ValueError,match="positive-floor-v1"):
        resolve(r)


def test_native_encoder_and_non_kv_refuse_before_numeric_execution():
    b,_,_=fixture()
    with pytest.raises(ValueError,match="not implemented"):
        encode(None,b.graph,b.model)
    with pytest.raises(ValueError,match="not implemented"):
        region_config(None,b.regions[0])
    with pytest.raises(ValueError,match="requires KV"):
        bind(fully_connected(2,2).graph,memories=RecurrentConfig(),
             regions=b.regions[0],dtype=torch.float32)
