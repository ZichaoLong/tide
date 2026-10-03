"""Real eager owners: independent CPU schedule, connected windows and restart."""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph import (Graph, Node, Edge, Region, GraphConfig, GraphRuntime,
                       ExecutionOptions, ExecutionPlacement, External, place_payloads)
from tidegraph.ops import Model
from tidegraph.compare import equivalent, objective
from tidegraph.checkpoint_ownership import parameter_aliases
from tidegraph.ownership import region_reference
from test_greedy_library import frozen


def configuration(family, memory, dtype):
    profile = "lh-add-repeat-v1" if memory == "add" else "lh-fiber-attention-sum-repeat-v1"
    nodes = tuple(Node(r, memory=profile, full="tanh") for r in (0,0,1,1,2))
    edges = (Edge(0,2,2), Edge(0,2,2), Edge(1,3,2), Edge(1,2,2 if family == "settle" else 1))
    if family == "pdg":
        edges += (Edge(2,0,3),)
    graph = Graph(nodes, edges, (Region(1,selector="tensor-history-v1"), Region(1), Region(1)), (0,1,4), (2,3,4))
    return GraphConfig(family,graph,width=4,dtype=str(dtype).split(".")[-1], ranks=(1,3,2) if family=="settle" else ())


def model_for(cfg, devices):
    model = Model(cfg.graph, cfg.width, cfg.seed, getattr(torch,cfg.dtype))
    model.nodes[3].weight = model.nodes[2].weight
    model.input_scale[1] = model.input_scale[0]
    # An alias used by two payload owners must keep one canonical leaf and VJP.
    model.edge_scale[0] = model.agg_scale[0]
    return place_payloads(cfg.graph, model, devices)


def advance(session, x, start, stop):
    runtime = session.runtime
    if runtime.spec:
        return session.advance(x[:,start:stop].to(runtime.device))
    graph, model = runtime.execution_graph, runtime.execution_model
    inputs = [External(b,p,t,t,x[b,t].to(model.nodes[node].bias.device))
              for b in range(len(x)) for p,node in enumerate(graph.inputs) for t in range(start,stop)]
    return session.advance(inputs,stop=stop,sealed_until=stop)


def assert_owners(runtime, result):
    graph, model = runtime.execution_graph, runtime.execution_model
    for event in result.trace:
        device = runtime.model.nodes[event["node"]].bias.device
        for name in ("content","proposal","control","comparison","next"):
            assert event[name].device == device
    for (_, node), state in result.continuation.states.items():
        assert state.value.device == model.nodes[node].bias.device
        assert all(v.device == state.value.device for v in state.slots.values())
    for atom in result.messages + result.continuation.pending:
        assert atom.value.device == model.nodes[atom.node].bias.device
    for (_,r), history in result.continuation.history.items():
        assert all(v.device == region_reference(graph,model,r).device for v in history.tensors.values())


def pytest_generate_tests(metafunc):
    if "preset" in metafunc.fixturenames:
        choices = ["cpu"] if os.environ.get("TIDE_PAYLOAD_BACKEND", "cpu")=="cpu" else ["mixed-a","mixed-b","mixed-c"]
        metafunc.parametrize("preset", choices)


@pytest.fixture
def backend():
    name = os.environ.get("TIDE_PAYLOAD_BACKEND", "cpu")
    if name == "npu":
        import torch_npu
        assert torch.npu.device_count() >= 2, "explicit multi-owner gate requires two NPUs"
    elif name == "cuda":
        assert torch.cuda.device_count() >= 2, "explicit multi-owner gate requires two CUDA devices"
    elif name != "cpu":
        raise ValueError("unknown payload gate backend")
    return name


def devices_for(backend, remap=False):
    return tuple("cpu" if backend=="cpu" else f"{backend}:{i}" for i in
                 ((0,1,0,0,1) if remap else (0,1,1,1,0)))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("memory", ["add", "attention"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
def test_complete_updates_and_remapped_checkpoint(dtype, backend, implementation, family, memory, schedule, preset, tmp_path):
    cfg = configuration(family,memory,dtype)
    owners = devices_for(backend)
    options = ExecutionOptions(implementation=implementation,schedule=schedule,mode="hst",trace=True,
                               workers=2 if implementation=="native" else 1,
                               placement=ExecutionPlacement(preset="cpu" if backend=="cpu" else preset))
    candidate = GraphRuntime(cfg,device=owners[0],options=options,model=model_for(cfg,owners),node_devices=owners)
    cpu_owners = ["cpu"]*5
    reference = GraphRuntime(cfg,device="cpu",options=ExecutionOptions(schedule="reference",packed=False,mode="hst",trace=True),
                             model=model_for(cfg,cpu_owners))
    assert parameter_aliases(reference.model) == parameter_aliases(candidate.model)
    make_opt = lambda r: torch.optim.AdamW(r.model.parameters(),lr=.0001,eps=1e-5,weight_decay=.01,foreach=False)
    runtimes = [reference,candidate]
    sessions = [r.session(1) for r in runtimes]
    optimizers = [make_opt(r) for r in runtimes]
    for step in range(2):
        records = []
        for r, session, opt in zip(runtimes,sessions,optimizers):
            opt.zero_grad(set_to_none=True)
            x = (torch.cos(torch.arange(32,dtype=dtype).reshape(1,8,4)*.17)*.15).requires_grad_()
            a = advance(session,x,step*4,step*4+2)
            b = advance(session,x,step*4+2,(step+1)*4)
            assert_owners(r,a);assert_owners(r,b)
            # A later-window root must reach earlier inputs through the carried
            # state/KV/pending graph, without a replayed CPU route.
            gradient = torch.autograd.grad(objective(b),x,retain_graph=True)[0]
            assert gradient[:,:step*4+2].abs().sum() > 0
            loss = (objective(a)+objective(b))/100
            loss.backward()
            grads = frozen({n:p.grad for n,p in r.model.named_parameters()})
            session.detach();opt.step();r.synchronize()
            records.append((a,b,loss,gradient,frozen(x.grad),grads,frozen(r.model.state_dict()),frozen(opt.state_dict())))
        equivalent(*records,check_device=False)
        if step == 0:
            path = tmp_path / "owners.pt"
            sessions[1].save(path,optimizers[1])
            changed = devices_for(backend,True)
            opts = replace(options,schedule="greedy" if schedule=="streaming" else "streaming",prefill=None)
            restored = GraphRuntime(cfg,device=changed[0],options=opts,model=model_for(cfg,changed),node_devices=changed)
            opt = make_opt(restored);session = restored.session(1);session.load(path,opt)
            equivalent(sessions[1].continuation,session.continuation,check_device=False)
            equivalent(optimizers[1].state_dict(),opt.state_dict(),check_device=False)
            for parameter,state in opt.state.items():
                assert state["exp_avg"].device == parameter.device
            runtimes[1],sessions[1],optimizers[1] = restored,session,opt
    assert len(set(runtimes[1].manifest()["model_storage"]["node_devices"])) == (1 if backend=="cpu" else 2)


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
def test_isolated_remote_roots_preserve_absence(dtype, backend, implementation, schedule, preset):
    cfg = configuration("timed-dag","add",dtype)
    cfg = replace(cfg,graph=replace(cfg.graph,regions=(replace(cfg.graph.regions[0],budget=2),*cfg.graph.regions[1:])))
    records = []
    for candidate in (False, True):
        owners = devices_for(backend) if candidate else ["cpu"]*5
        options = ExecutionOptions(implementation=implementation if candidate else "python",
            schedule=schedule if candidate else "reference",packed=candidate,trace=True,mode="hst",
            placement=ExecutionPlacement(preset=preset) if candidate else None)
        runtime = GraphRuntime(cfg,device=owners[0],options=options,model=model_for(cfg,owners),node_devices=owners)
        xs = [torch.full((4,),.25+i/10,dtype=dtype,requires_grad=True) for i in range(3)]
        external = [External(0,p,0,0,x.to(runtime.model.nodes[node].bias.device))
                    for p,(node,x) in enumerate(zip(cfg.graph.inputs,xs))]
        result = runtime.session(1).advance(external,stop=3,sealed_until=3)
        event = next(e for e in result.trace if e["node"]==0)
        roots = [event["descriptor"], event["control"]]
        roots += [a.value for a in result.messages if a.source==0][:1]
        assert len(roots)==3
        parameters = dict(runtime.model.named_parameters())
        leaves = (*xs,*parameters.values())
        values = []
        for root in roots:
            for factor in (1.,0.):
                grads = torch.autograd.grad(root.sum()*factor,leaves,allow_unused=True,retain_graph=True)
                assert grads[2] is None
                for name,g in zip(parameters,grads[3:]):
                    if name.startswith("nodes.4."):
                        assert g is None
                values.append(grads)
        records.append((result,values))
    equivalent(*records,check_device=False)


def test_alias_conflict_is_preallocation_and_transactional():
    cfg = configuration("pdg","add",torch.float32)
    model = model_for(cfg,["cpu"]*5)
    before = {n:id(p) for n,p in model.named_parameters()}
    # CUDA need not exist: the contradictory map must fail before a device copy.
    with pytest.raises(ValueError,match="co-located"):
        place_payloads(cfg.graph,model,["cuda:0","cuda:1","cuda:0","cuda:1","cuda:0"])
    assert before == {n:id(p) for n,p in model.named_parameters()}
    assert all(p.device.type=="cpu" for p in model.parameters())


def test_configured_owners_and_preflight(backend):
    cfg = configuration("timed-dag","add",torch.float32)
    owners = devices_for(backend)
    runtime = GraphRuntime(cfg,device=owners[0],node_devices=owners)
    record = runtime.manifest()
    assert record["model_storage"]["node_devices"] == list(owners)
    assert [r["payload"] for r in record["eager_owners"]["nodes"]] == list(owners)
    if backend != "cpu":
        assert runtime.options.placement == ExecutionPlacement()
        api = getattr(torch, backend)
        assert api.current_device() == runtime.device.index
    with pytest.raises(ValueError,match="every node"):
        GraphRuntime(cfg,device=owners[0],node_devices=owners[:-1])
    other = Model(cfg.graph,4,dtype=torch.float32)
    if backend != "cpu":
        with pytest.raises(ValueError,match="node tensors"):
            GraphRuntime(cfg,device=owners[0],model=other,node_devices=owners)
        with pytest.raises(ValueError,match="model payload"):
            GraphRuntime(cfg,device=owners[0],node_devices=owners,
                         options=ExecutionOptions(placement=ExecutionPlacement(read=owners[0])))
        unavailable = (owners[0], owners[1], f"{backend}:{api.device_count()}", *owners[3:])
        with pytest.raises(RuntimeError,match="unavailable"):
            GraphRuntime(cfg,device=owners[0],node_devices=unavailable)
        assert api.current_device() == runtime.device.index


def test_custom_region_initial_reference_preserves_values_and_vjp(dtype):
    from tidegraph.region import CountSelector, evaluate
    from tidegraph import Continuation
    class VectorHistory(CountSelector):
        profile = "initial-vector-v1"
        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.tensor(.5,dtype=dtype))
        def initial(self,layout,reference):
            h = super().initial(layout,reference)
            h.tensors["memory"] = reference*self.alpha
            return h
        def validate_history(self,history,layout):
            assert history.tensors["memory"].shape==(3,)
    graph = Graph((Node(0),Node(1)),(),(Region(1),Region(1,selector="initial-vector-v1")),(1,),(1,))
    model = Model(graph,width=3,dtype=dtype,region_programs={1:VectorHistory()})
    _,_,history = evaluate(graph,model,Continuation(graph.identity,1),0,1,0,[(1,model.nodes[1].read.sum())])
    memory = history.tensors["memory"]
    equivalent(memory,model.nodes[0].bias*.5)
    grads = torch.autograd.grad(memory.sum(),(model.nodes[0].bias,model.nodes[1].bias),allow_unused=True)
    equivalent(grads,(torch.full((3,),.5,dtype=dtype),None))
