"""Public raw roots use declared schema; reading values is an explicit action."""
from types import SimpleNamespace
import pytest
import torch
from tidegraph.weighted.resident_blocks import WeightedBlockResidentTraining


def client():
    value=WeightedBlockResidentTraining.__new__(WeightedBlockResidentTraining)
    value._check=lambda:None
    value.module=SimpleNamespace(BlockRoots=SimpleNamespace,BlockRoot=SimpleNamespace)
    def observe(*args):
        raise AssertionError("cotangents must not materialize raw values")
    value.owner=SimpleNamespace(observe=observe)
    return value


def test_raw_root_needs_no_prefix_value_or_decode():
    c=client();present=torch.tensor([True,False,True]);shape=[3,7,4]
    field=SimpleNamespace(values=None,present=present,valid=torch.ones(3,7,dtype=torch.bool),shape=shape)
    window=SimpleNamespace(token=object(),fields={"region.0.old.keys":field})
    result=c.cotangents(window,{"region.0.old.keys":torch.zeros(shape)})
    assert result.token is window.token
    assert result.fields["region.0.old.keys"].connected is present
    with pytest.raises(ValueError,match="shape/device/dtype"):
        c.cotangents(window,{"region.0.old.keys":torch.zeros(3,6,4)})


def test_raw_observation_is_named_and_explicit():
    c=client();window=SimpleNamespace(token=object(),fields={"a":object(),"b":object()})
    calls=[]
    def observe(token,names):
        calls.append((token,names));return {name:window.fields[name] for name in names}
    c.owner.observe=observe
    assert list(c.observe(window,["b"]))==["b"]
    assert list(c.observe(window))==["a","b"]
    assert calls==[(window.token,["b"]),(window.token,["a","b"])]
