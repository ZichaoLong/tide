"""Independent dense layer formulas vs scalar graph schedule, not two wrappers."""
import sys
import os
from pathlib import Path
from dataclasses import replace
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from layer_tensor_baseline import fixture, external, run, numerics
from tidegraph import Continuation
from tidegraph.reference import run as reference
from tidegraph.runtime import resolve_device


@pytest.mark.parametrize("family", ["kv", "ssd"])
def test_tensor_layers_outputs_raw_state_parameter_input_and_state_roots(dtype, family, request):
    from dense_checks import Checks
    check = Checks(request.node.nodeid,
        failure_reason="layer tensor strict numerical difference retained; see bounded layer diagnostic")
    device, _ = resolve_device(os.environ.get("TIDE_LAYER_DEVICE", "cpu"))
    numerics(device, os.environ.get("TIDE_LAYER_NUMERICS", "ieee"))
    b, spec = fixture(family, width=4, depth=2, heads=2, dtype=dtype, device=device)
    oracle, oracle_spec = fixture(family, width=4, depth=2, heads=2, dtype=dtype)
    original = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    values = original.detach().to(device).requires_grad_()
    output, states = run(b, spec, values)
    stop = values.shape[1]*spec.stride
    expected = reference(oracle.graph, oracle.model, Continuation(oracle.graph.identity, 2), external(oracle_spec, original),
                         stop, sealed_until=stop)
    indexed = {(batch, (time-spec.rank(len(b.graph.nodes)-1))//spec.stride): p[1:]
               for batch, time, _, p in expected.outputs}
    rows = torch.stack([indexed[i, j] for i in range(2) for j in range(3)]).reshape_as(output)
    check(output, rows, "output")
    pairs = [(output, rows)]
    for v, raw in enumerate(states):
        for batch in range(2):
            state = expected.continuation.states[batch, v]
            for name, value in raw.items():
                actual = value if name in {"time", "times", "conv_times"} else value[batch]
                check(actual, state.slots[name], f"state.{v}.{batch}.{name}")
                if actual.is_floating_point():
                    pairs.append((actual, state.slots[name]))
    parameters = [p for p in b.model.parameters() if p.requires_grad]
    roots = [values, *parameters]
    oracle_roots = [original, *(p for p in oracle.model.parameters() if p.requires_grad)]
    for pair, (actual, scalar) in enumerate(pairs):
        ga = (torch.autograd.grad(actual.square().sum(), roots, allow_unused=True, retain_graph=True)
              if actual.requires_grad else [None]*len(roots))
        gs = (torch.autograd.grad(scalar.square().sum(), oracle_roots, allow_unused=True, retain_graph=True)
              if scalar.requires_grad else [None]*len(roots))
        # Algebraic output projection only: dead graph fields have zero numeric
        # derivatives. This must never replace native None/zero qualification.
        for index, (a, s, root) in enumerate(zip(ga, gs, roots)):
            check(torch.zeros_like(root) if a is None else a,
                  torch.zeros_like(root) if s is None else s, f"root.{pair}.{index}")
    check.finish()


def test_tensor_layer_domain_rejects_other_allocation_before_execution():
    from tidegraph.weighted import bind, RegionalConfig
    b, spec = fixture("kv", width=4, depth=2, heads=2)
    invalid = bind(spec.graph, memories=b.memories, regions=RegionalConfig(k=1), dtype=torch.float32)
    with pytest.raises(ValueError, match="proportional"):
        run(invalid, replace(spec, graph=invalid.graph), torch.ones(2, 3, 4))


@pytest.mark.parametrize("family", ["kv", "ssd"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_same_layer_workload_uses_public_default_entry(dtype, family, implementation, request):
    from dense_checks import Checks
    check = Checks(request.node.nodeid,
        failure_reason="default layer vs tensor strict numerical difference retained; see bounded layer diagnostic")
    from tidegraph import ExecutionPlacement
    from tidegraph.weighted.execution import eager_runner
    device, _ = resolve_device(os.environ.get("TIDE_LAYER_DEVICE", "cpu"))
    numerics(device, os.environ.get("TIDE_LAYER_NUMERICS", "ieee"))
    b, spec = fixture(family, width=4, depth=2, heads=2, dtype=dtype, device=device)
    values = torch.linspace(-.2, .4, 24, dtype=dtype, device=device).reshape(2, 3, 4).requires_grad_()
    expected, _ = run(b, spec, values)
    execute = eager_runner(b, implementation, ExecutionPlacement(
        preset="cpu" if device.type == "cpu" else "mixed-c", scoring_dtype=str(dtype).split(".")[-1]),
        native_library=os.environ.get("TIDE_BUILD_DIR"), trace=False)
    stop = 3*spec.stride
    result = execute(Continuation(b.graph.identity, 2), external(spec, values), stop, sealed_until=stop)
    actual = torch.stack([p[1:] for *_, p in result.outputs]).reshape(3, 2, 4).transpose(0, 1)
    check(actual, expected, "output")
    roots = [values, *(p for p in b.model.parameters() if p.requires_grad)]
    ga = torch.autograd.grad(actual.square().sum(), roots, allow_unused=True)
    ge = torch.autograd.grad(expected.square().sum(), roots, allow_unused=True)
    for index, (a, e, root) in enumerate(zip(ga, ge, roots)):
        check(torch.zeros_like(root) if a is None else a, torch.zeros_like(root) if e is None else e, f"root.{index}")
    assert result.stats["candidate_events"] == 12
    assert result.stats["max_state_sequence"] == 3
    check.finish()
