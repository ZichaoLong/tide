import copy
import pytest
import torch
from cases import ring
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run


def selected(model):
    # Same-shaped parameters made the old positional restore silently succeed.
    return [model.input_scale[0], model.edge_scale[0]]


def optimizer(model, kind, parameters=None):
    parameters = selected(model) if parameters is None else parameters
    if kind == "sgd":
        return torch.optim.SGD(parameters, lr=.01, momentum=.8, weight_decay=.02)
    cls = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
    return cls(parameters, lr=.01, amsgrad=True, weight_decay=.02)


def seed_state(model, opt):
    for value, parameter in enumerate(selected(model), 1):
        parameter.grad = torch.full_like(parameter, value*3.)
    opt.step(); opt.zero_grad(set_to_none=True)


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("mismatch", ["order", "groups", "class", "foreign"])
def test_optimizer_ownership_mismatch_is_rejected_before_mutation(dtype, kind, mismatch, tmp_path):
    graph, model, state, *_ = ring(dtype)
    opt = optimizer(model, kind); seed_state(model, opt)
    path = tmp_path/"state.pt"; save(path, graph, model, state, opt)
    _, restored, *_ = ring(dtype)
    params = selected(restored)
    if mismatch == "order":
        actual = optimizer(restored, kind, params[::-1])
    elif mismatch == "groups":
        actual = optimizer(restored, kind, [{"params": params[:1]}, {"params": params[1:]}])
    elif mismatch == "class":
        actual = optimizer(restored, "adamw" if kind == "sgd" else "sgd")
    else:
        actual = optimizer(restored, kind, [params[0], torch.nn.Parameter(torch.ones_like(params[1]))])
    before = copy.deepcopy((restored.state_dict(), actual.state_dict()))
    with pytest.raises(ValueError, match="optimizer"):
        load(path, graph, restored, actual)
    equivalent(before, (restored.state_dict(), actual.state_dict()))


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("corruption", ["ids", "orphan", "shape", "dtype", "nan", "slots"])
def test_optimizer_payload_preflight(dtype, kind, corruption, tmp_path):
    graph, model, state, *_ = ring(dtype)
    opt = optimizer(model, kind); seed_state(model, opt)
    path = tmp_path/"state.pt"; save(path, graph, model, state, opt)
    record = torch.load(path, weights_only=True)
    stored = record["optimizer"]
    key = "momentum_buffer" if kind == "sgd" else "exp_avg"
    value = stored["state"][0][key]
    if corruption == "ids":
        stored["param_groups"][0]["params"].reverse()
    elif corruption == "orphan":
        stored["state"][999] = {key: value}
    elif corruption == "slots":
        stored["state"][0] = {"wrong": value}
    else:
        stored["state"][0][key] = {
            "shape": torch.ones(2, dtype=dtype), "dtype": value.to(torch.int64),
            "nan": torch.full_like(value, float("nan")),
        }[corruption]
    torch.save(record, path)
    _, restored, *_ = ring(dtype)
    actual = optimizer(restored, kind); seed_state(restored, actual)
    with torch.no_grad():
        restored.nodes[0].weight.add_(.5)  # Failure must not overwrite live values.
    before = copy.deepcopy((restored.state_dict(), actual.state_dict()))
    with pytest.raises(ValueError, match="optimizer"):
        load(path, graph, restored, actual)
    equivalent(before, (restored.state_dict(), actual.state_dict()))


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
def test_shared_subset_optimizer_resumes_the_next_update(dtype, kind, tmp_path):
    graph, model, state, inputs, *_ = ring(dtype)
    model.edge_scale[1] = model.edge_scale[0]
    opt = optimizer(model, kind)
    prefix = run(graph, model, state, inputs, 7, sealed_until=7, mode="hst")
    objective(prefix).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    path = tmp_path/"state.pt"; save(path, graph, model, prefix.continuation, opt)
    _, restored, *_ = ring(dtype); restored.edge_scale[1] = restored.edge_scale[0]
    actual = optimizer(restored, kind); actual.param_groups[0]["lr"] = .8
    resumed = load(path, graph, restored, actual)
    assert restored.edge_scale[0] is restored.edge_scale[1]
    for m, q, o in ((model, prefix.continuation.detach(), opt), (restored, resumed, actual)):
        result = run(graph, m, q, [], 10, sealed_until=10, mode="hst")
        objective(result).backward(); o.step(); o.zero_grad(set_to_none=True)
    equivalent(model.state_dict(), restored.state_dict())
    equivalent(opt.state_dict(), actual.state_dict())


def test_duplicate_optimizer_aliases_and_foreign_parameters_are_not_saved(dtype, tmp_path):
    graph, model, state, *_ = ring(dtype)
    parameter = model.edge_scale[0]
    with pytest.warns(UserWarning, match="duplicate"):
        repeated = torch.optim.SGD([parameter, parameter], lr=.1)
    foreign = torch.optim.SGD([torch.nn.Parameter(torch.ones_like(parameter))], lr=.1)
    for index, opt in enumerate((repeated, foreign)):
        path = tmp_path/f"invalid-{index}.pt"
        with pytest.raises(ValueError, match="optimizer"):
            save(path, graph, model, state, opt)
        assert not path.exists()


def test_conflicting_shared_weights_are_rejected_before_mutation(dtype, tmp_path):
    graph, model, state, *_ = ring(dtype); model.edge_scale[1] = model.edge_scale[0]
    path = tmp_path/"state.pt"; save(path, graph, model, state)
    record = torch.load(path, weights_only=True)
    record["weights"]["edge_scale.1"] = record["weights"]["edge_scale.0"]+1
    torch.save(record, path)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="shared parameter values"):
        load(path, graph, model)
    equivalent(before, model.state_dict())


def test_previous_schema_cannot_implicitly_resume_an_optimizer(dtype, tmp_path):
    graph, model, state, *_ = ring(dtype)
    path = tmp_path/"old.pt"; save(path, graph, model, state)
    record = torch.load(path, weights_only=True); record["schema"] = "tide-continuation-v4"
    torch.save(record, path)
    with pytest.raises(ValueError, match="schema"):
        load(path, graph, model)
