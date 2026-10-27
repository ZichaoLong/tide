"""Comparison's scalar probe must not depend on dictionary insertion order."""
from dataclasses import replace
import torch
from tidegraph.compare import objective
from test_weighted_recurrent_execution import model
from tidegraph import Continuation
from tidegraph.reference import run


def test_objective_field_and_owner_permutation_keeps_value_and_vjp(dtype):
    b,_,xs,stop=model(dtype,"ssd","0")
    result=run(b.graph,b.model,Continuation(b.graph.identity,1),xs,stop,sealed_until=stop)
    alternate=replace(result,continuation=replace(result.continuation,states={
        owner:replace(state,slots=dict(reversed(tuple(state.slots.items()))))
        for owner,state in reversed(tuple(result.continuation.states.items()))}))
    a,c=objective(result),objective(alternate)
    assert torch.equal(a,c)
    ga,gc=[torch.autograd.grad(x.square(),xs[2].value,retain_graph=True)[0] for x in (a,c)]
    assert torch.equal(ga,gc)
