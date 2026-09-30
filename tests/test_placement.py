"""Placement leaves graph semantics, parameter ownership and VJP roots intact."""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph import ExecutionOptions, ExecutionPlacement, GraphRuntime, Graph, Node, Region, Continuation, External
from tidegraph.compare import equivalent, objective
from tidegraph.ops import Model
from tidegraph.placement import place_model
from tidegraph.reference import run
from tidegraph.greedy import run as greedy
from tidegraph.readout import ReadProgram
from test_greedy_library import configuration, frozen
from test_library import advance


@pytest.fixture
def device(dtype):
    spec = os.environ.get("TIDE_PLACEMENT_DEVICE", "cpu")
    if spec.startswith("npu"):
        import torch_npu  # Explicitly requested target only.
        if dtype == torch.float64:
            pytest.skip("NPU payload FP64 is outside the contract; CPU Read FP64 has a separate case")
    return torch.device(spec)


def request_for(device):
    return ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c")


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
def test_public_placement_carried_training(dtype, device, implementation, family, schedule, tmp_path):
    cfg = configuration(family, dtype)
    options = ExecutionOptions(implementation=implementation, schedule=schedule, mode="hst", trace=True,
                               placement=request_for(device))
    candidate = GraphRuntime(cfg, device=str(device), options=options)
    baseline = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(schedule="reference", packed=False, mode="hst", trace=True))
    # Execution views must retain checkpoint names and the caller's leaf owners.
    assert baseline.execution_model.state_dict().keys() == candidate.execution_model.state_dict().keys()
    # Settle adds identity boundaries and remaps physical scale names. The
    # trainable body leaves themselves must still be shared exactly.
    assert {id(p) for p in candidate.model.parameters() if p.requires_grad} == {
        id(p) for p in candidate.execution_model.parameters() if p.requires_grad}
    opts = [torch.optim.SGD(r.model.parameters(), lr=.0001, momentum=.25) for r in (baseline, candidate)]
    sessions = [r.session(2) for r in (baseline, candidate)]
    for step in range(3):
        rows = []
        for runtime, session, opt in zip((baseline, candidate), sessions, opts):
            opt.zero_grad(set_to_none=True)
            x = (torch.cos(torch.arange(96, dtype=dtype).reshape(2, 12, 4)*.11)*.2).to(runtime.device).requires_grad_()
            result = advance(session, x, step*4, (step+1)*4)
            loss = objective(result)/100
            loss.backward()
            grads = {name:frozen(p.grad) for name,p in runtime.model.named_parameters()}
            session.detach(); opt.step()
            rows.append((result, grads, frozen(x.grad), frozen(runtime.model.state_dict()), frozen(opt.state_dict())))
        # Device placement is separately checked; comparison normalizes transport.
        equivalent(rows[0], rows[1], check_device=False)
        if step == 0:
            path = tmp_path / "placement.pt";sessions[1].save(path, opts[1])
            restored = GraphRuntime(cfg, device=str(device), options=options)
            opt = torch.optim.SGD(restored.model.parameters(),lr=.0001,momentum=.25)
            session = restored.session(2);session.load(path,opt)
            candidate,sessions[1],opts[1] = restored,session,opt
        if step == 1:
            options = replace(options, schedule="greedy" if schedule=="streaming" else "streaming", prefill=None)
            candidate = GraphRuntime(cfg, device=str(device), options=options, model=candidate.model)
            sessions[1] = candidate.session(2,continuation=sessions[1].continuation)
    manifest = candidate.manifest()
    assert manifest["placement"]["events"] == "cpu"
    assert manifest["requested_options"]["placement"]["preset"] == options.placement.preset


@pytest.mark.parametrize("selector", ["count-v1", "positive-v1", "lh-count-affect-v1", "tensor-history-v1"])
@pytest.mark.parametrize("read", ["content", "old", "proposal"])
def test_python_placement_isolated_roots_and_aliases(dtype, device, selector, read):
    graph = Graph((Node(0),Node(0),Node(1)),(),(Region(1,read_mode=read,selector=selector),Region(1)),(0,1,2),(0,1,2))
    model = Model(graph,width=2,dtype=dtype).to(device)
    placed = place_model(graph,model,request_for(device))
    assert dict(model.named_parameters()).keys() == dict(placed.named_parameters()).keys()
    for name,value in model.named_parameters():
        assert value is dict(placed.named_parameters())[name]
    xs = [torch.full((2,),.25+i/8,dtype=dtype,device=device,requires_grad=True) for i in range(3)]
    inputs = [External(0,i,0,0,x) for i,x in enumerate(xs)]
    q = Continuation(graph.identity,1)
    expected = run(graph,model,q,inputs,1,sealed_until=1,mode="hst")
    got = greedy(graph,placed,q,inputs,1,sealed_until=1,mode="hst")
    equivalent(expected,got)
    leaves = [*xs,*model.parameters()]
    for key in ("descriptor","control","full"):
        # Unselected nodes deliberately have no Full invocation or result.
        index = next((i for i,e in enumerate(expected.trace) if e["node"] < 2 and key in e), None)
        if index is None:
            continue
        root = expected.trace[index][key];actual = got.trace[index][key]
        if root is None or not root.requires_grad:
            continue
        for factor in (1.,0.):
            a=torch.autograd.grad(root.sum()*factor,leaves,allow_unused=True,retain_graph=True)
            b=torch.autograd.grad(actual.sum()*factor,leaves,allow_unused=True,retain_graph=True)
            equivalent(a,b);assert b[2] is None


def test_python_explicit_linear_scoring_dtype_and_custom_refusal(dtype, device):
    graph=Graph((Node(0),),(),(Region(1),),(0,),(0,))
    model=Model(graph,width=2,dtype=dtype).to(device)
    p=ExecutionPlacement(read="cpu",control="cpu",selection="cpu",scoring_dtype="float64")
    placed=place_model(graph,model,p)
    x=torch.tensor([.3,-.2],dtype=dtype,device=device,requires_grad=True)
    got=greedy(graph,placed,Continuation(graph.identity,1),[External(0,0,0,0,x)],1,sealed_until=1)
    descriptor=got.trace[0]["descriptor"]
    assert descriptor.device.type=="cpu" and descriptor.dtype==torch.float64
    proposal=got.trace[0]["proposal"].cpu().double()
    equivalent(descriptor,(proposal*model.nodes[0].read.cpu().double()).sum())
    class Custom(ReadProgram):
        profile="linear-v1"
    model.nodes[0].read_program=Custom()
    with pytest.raises(ValueError,match="custom"):
        place_model(graph,model,p)


def test_placement_configuration_refusals():
    with pytest.raises(ValueError,match="accelerator"):
        ExecutionPlacement(preset="mixed-a").resolve("cpu")
    with pytest.raises(ValueError,match="unknown"):
        ExecutionPlacement(preset="invalid")
    with pytest.raises(ValueError,match="model payload"):
        ExecutionPlacement(read="cuda:1").resolve("cpu")
