"""Pinned FC/HB §1 clock/source/boundary binding, including output existence."""
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.settle_ports import run_ports
from tidegraph.weighted import KVConfig,RegionalConfig,fully_connected,hb_lattice,bind_layered,layered_inputs,readout


@pytest.mark.parametrize("family",["fc","hb"])
def test_layered_formal_binding_prefix_outputs_and_window(dtype,family):
    topology = fully_connected(3,2) if family=="fc" else hb_lattice(2,1,2)
    spec,b = bind_layered(topology,memory=KVConfig(width=2,summary_width=2),
                         regions=RegionalConfig(k=2),window_positions=2,short_rate_per_position=.3,dtype=dtype)
    assert b.source_classes.edges == tuple(edge.source+1 for edge in b.graph.edges)
    assert all(c.window==2*spec.stride and c.beta_short==.3/spec.stride and c.position_scale==spec.stride
               for c in b.memories)
    x = torch.tensor([[[.2,-.4],[.5,.7],[-.3,.1]]],dtype=dtype,requires_grad=True)
    xs = layered_inputs(spec,x)
    expected = run_ports(spec,b.model,Continuation(b.graph.identity,1),xs,3*spec.stride,sealed_until=3*spec.stride)
    # Future changes must preserve both values and presence of all prefix slots.
    future = x.detach().clone();future[:,1:] *= -17
    actual = run_ports(spec,b.model,Continuation(b.graph.identity,1),layered_inputs(spec,future),
                       3*spec.stride,sealed_until=3*spec.stride)
    prefix = [o for o in expected.outputs if o[1]<spec.stride]
    equivalent(prefix,[o for o in actual.outputs if o[1]<spec.stride])
    for (_,v),state in expected.continuation.states.items():
        assert state.slots["times"].tolist()==[spec.stride+spec.rank(v),2*spec.stride+spec.rank(v)]
    coords=[(max(spec.ranks),p) for p in range(len(spec.graph.outputs))]
    y=readout(expected.outputs,coords,x[0,0])
    assert torch.isfinite(y).all()
    gradient=torch.autograd.grad(y.sum(),x)[0]
    assert gradient[:,1:].count_nonzero()==0


def test_readout_absence_and_cancellation_are_distinct(dtype):
    ref=torch.zeros(1,dtype=dtype)
    records=[(0,2,0,torch.tensor([.95,2.],dtype=dtype)),(0,2,1,torch.tensor([.45,-2.],dtype=dtype))]
    torch.testing.assert_close(readout(records,[(2,0),(2,1)],ref),ref.new_tensor([5/7]))
    zero=[(0,2,0,ref.new_tensor([.5,1.])),(0,2,1,ref.new_tensor([.5,-1.]))]
    equivalent(readout(zero,[(2,0),(2,1)],ref),readout([],[(2,0),(2,1)],ref))
    assert len(zero)==2
