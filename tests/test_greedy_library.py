"""Public online policy, carried training and checkpoint/schedule interchange."""
from dataclasses import replace
import json
import pytest
import torch
from tidegraph import ExecutionOptions, GraphConfig, GraphRuntime
from tidegraph.compare import equivalent, objective
from test_greedy import generated
from test_library import config, advance, gradients


def configuration(family, dtype):
    if family == "pdg":
        graph, _, _, _, _ = generated(dtype, 3, "ssm")
        return GraphConfig(family, graph, width=4, dtype=str(dtype).split(".")[-1])
    return config(family, dtype)


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("packed,prefill", [(True, True), (False, False)])
def test_public_configuration_full_observables_and_vjp(dtype, implementation, family, packed, prefill):
    options = ExecutionOptions(implementation=implementation, schedule="greedy", packed=packed,
                               prefill=prefill, trace=True, mode="hst", max_events=1000)
    cfg = replace(configuration(family, dtype), execution=options)
    restored = GraphConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert cfg.identity == restored.identity
    runtime = GraphRuntime(restored, device="cpu")
    reference = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(
        schedule="reference", packed=False, trace=True, mode="hst"))
    x = (torch.arange(32, dtype=dtype).reshape(2, 4, 4) / 100).requires_grad_()
    y = x.detach().clone().requires_grad_()
    stop = 4 if family == "settle" else 8
    expected = advance(reference.session(2), x, 0, stop)
    actual = advance(runtime.session(2), y, 0, stop)
    equivalent(expected, actual)
    equivalent(gradients(expected, reference, x), gradients(actual, runtime, y))
    assert actual.stats["greedy_stages"] > 0
    # Settle projects away the encoded input/output identity boundaries, while
    # work counters intentionally continue to include their execution.
    boundaries = 2 * x.shape[0] * x.shape[1] if family == "settle" else 0
    assert actual.stats["candidate_events"] == len(actual.trace) + boundaries
    if not packed:
        assert actual.stats["max_full_batch"] == 1
    assert runtime.manifest()["resolved_options"]["schedule"] == "greedy"


def frozen(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: frozen(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(frozen(v) for v in value)
    return value


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_carried_training_checkpoint_and_policy_change(dtype, implementation, family, optimizer, tmp_path):
    cfg = configuration(family, dtype)
    trajectories = []
    for candidate in (False, True):
        options = ExecutionOptions(implementation=implementation if candidate else "python",
            schedule="greedy" if candidate else "reference", packed=candidate, trace=True, mode="hst")
        runtime = GraphRuntime(cfg, device="cpu", options=options)
        session = runtime.session(2)
        make = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
        opt_args = dict(lr=.0001, weight_decay=.01)
        if optimizer == "sgd":
            opt_args.update(momentum=.9)
        else:
            opt_args.update(eps=1e-5)
        opt = make(runtime.model.parameters(), **opt_args)
        trajectory = []
        for step in range(3):
            # The ledger, pending messages, state and history continue across
            # updates; the explicit detach declares the same TBPTT boundary.
            x = (torch.cos(torch.arange(96, dtype=dtype).reshape(2, 12, 4) * .11) * .2).requires_grad_()
            opt.zero_grad(set_to_none=True)
            result = advance(session, x, step*4, (step+1)*4)
            loss = objective(result) / 100
            loss.backward()
            grads = {k: frozen(p.grad) for k, p in runtime.model.named_parameters()}
            session.detach()
            opt.step()
            trajectory.append((loss.detach(), grads, frozen(x.grad), frozen(runtime.model.state_dict()),
                               frozen(opt.state_dict()), session.continuation))
            if step == 0 and candidate:
                checkpoint = tmp_path / "carried.pt"
                session.save(checkpoint, opt)
                other = GraphRuntime(replace(cfg, seed=917), device="cpu", options=options)
                resumed = other.session(2)
                other_opt = make(other.model.parameters(), **opt_args)
                resumed.load(checkpoint, other_opt)
                equivalent(session.continuation, resumed.continuation)
                equivalent(opt.state_dict(), other_opt.state_dict())
                runtime, session, opt = other, resumed, other_opt
            elif step == 1 and candidate:
                # Reuse weights/optimizer; change only the host schedule.
                other = GraphRuntime(cfg, device="cpu", options=replace(options, schedule="streaming"),
                                     model=runtime.model)
                session = other.session(2, continuation=session.continuation)
                runtime = other
        trajectories.append(trajectory)
    equivalent(*trajectories)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_greedy_prefill_default_and_explicit_capacity_rejection(implementation):
    cfg = configuration("pdg", torch.float32)
    runtime = GraphRuntime(cfg, device="cpu", options=dict(implementation=implementation,
                          schedule="greedy", max_events=1))
    assert runtime.options.prefill is True
    session = runtime.session(2)
    before = session.continuation
    with pytest.raises((ValueError, RuntimeError), match="live-fiber capacity"):
        advance(session, torch.ones(2, 3, 4), 0, 3)
    assert session.continuation is before and before.cut == 0
    assert not before.ledger and not before.pending and not before.states
