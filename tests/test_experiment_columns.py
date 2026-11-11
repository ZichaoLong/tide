"""Public task boundary: exact material, physical outputs, and independent train."""
from pathlib import Path
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.experiment.config import resolve
from tidegraph.experiment.eager import Eager
from tidegraph.experiment.column_task import ColumnInputs, loss
from tidegraph.planning.execution import ExecutionSpec
from tidegraph.weighted.window_records import external
from test_experiment_engine import config


def selected(family="kv", client="pytorch"):
    value = config("training", client=client, dtype="float64", scoring_dtype="float64", return_mode="outputs")
    value["model"]["recipe"].update(family=family, region_width=2)
    value["task"].update(positions=3, samples=2)
    value["measurement"]["validation"] = "none"
    if client == "python-native":
        import _tide_native
        value["site"] = dict(paths=dict(native_library=str(Path(_tide_native.__file__).parent)))
    return value


@pytest.mark.parametrize("value_type", [torch.float32, torch.float64])
@pytest.mark.parametrize("pattern", [None, "staggered-v1"])
def test_input_columns_are_bitwise_same_material_and_port_clock(value_type, pattern):
    from tidegraph.planning.hierarchy import general_ports_plan
    from tidegraph.planning.materialize import materialize
    from tidegraph.settle import SettleGraph
    from tidegraph import Continuation
    from tidegraph.experiment.tasks import WeightedOutputTask
    plan, _ = general_ports_plan(samples=3, positions=7, width=4)
    # Regenerate the workload identity if selecting another static pattern.
    from tidegraph.planning.model import workload
    from tidegraph.planning.model import resolve_model
    model, _ = resolve_model(**{k: plan[k] for k in (
        "graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")})
    plan, _ = workload(model, samples=3, positions=7, input_pattern=pattern)
    binding, templates, _ = materialize(plan, dtype=value_type, max_bytes=10**8)
    spec = SettleGraph(binding.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
    column = ColumnInputs(plan, value_type)
    state = Continuation(binding.graph.identity, plan["samples"])
    task = WeightedOutputTask({}, plan)
    for index in (0, 1):
        layout, ports, leaf = column.window(index, "cpu", training=True)
        got = external(spec, layout, ports, state)
        want = task.inputs(templates, index, training=True)
        key = lambda x: (x.batch, x.port, x.position, x.time)
        got, want = sorted(got, key=key), sorted(want, key=key)
        assert [key(x) for x in got] == [key(x) for x in want]
        for a, b in zip(got, want):
            torch.testing.assert_close(a.value, b.value, atol=0, rtol=0)
        # One repeated source row must accumulate every physical port use.
        bar, = torch.autograd.grad(sum(x.value.sum() for x in got), leaf)
        expected = torch.zeros_like(leaf)
        for m in ports.values():
            expected.index_add_(0, m.rows, torch.ones_like(leaf[m.rows]))
        torch.testing.assert_close(bar, expected, atol=0, rtol=0)
        for x in got:
            state.ledger[x.batch, x.port] = max(state.ledger.get((x.batch, x.port), (-1, -1)), (x.position, x.time))


@pytest.mark.parametrize("family", ["kv", "ssd"])
@pytest.mark.parametrize("client", ["pytorch", "python-native"])
def test_default_column_task_matches_independent_scalar_training(family, client):
    resolved = resolve(selected(family, client))
    left, right = Eager(resolved, reference=True), Eager(resolved)
    try:
        assert right.manifest()["effective_input_interface"] == "tensor-window-v1"
        for i in range(2):
            a, x = left.step(i)
            b, y = right.step(i)
            equivalent(a, b)
            assert x == pytest.approx(y, rel=1e-8, abs=1e-10)
            equivalent(left.decoded_state(), right.decoded_state())
            equivalent(left.training_state(), right.training_state())
    finally:
        left.close(); right.close()


def test_default_path_does_not_materialize_event_inputs_or_differentiable_output_rows(monkeypatch):
    from tidegraph.experiment.tasks import WeightedOutputTask
    def forbidden(*a, **kw):
        raise AssertionError("event input materialization on column path")
    monkeypatch.setattr("tidegraph.planning.materialize.materialize_inputs", forbidden)
    monkeypatch.setattr(WeightedOutputTask, "inputs", forbidden)
    owner = Eager(resolve(selected()))
    try:
        actual, _ = owner.step(0)
        assert owner.last_work["input_positions"] == 5  # ragged lengths 3 and 2
        assert actual["outputs"] and all(not x.requires_grad for *_, x in actual["outputs"])
        assert owner.last_inputs.is_leaf and owner.last_inputs.grad is not None
    finally:
        owner.close()


def test_per_port_square_does_not_square_a_combined_readout():
    from types import SimpleNamespace
    from tidegraph.weighted.message_columns import MessageColumns
    c = torch.tensor([.5, 0.], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[2., -1.], [7., 3.]], dtype=torch.float64, requires_grad=True)
    m = MessageColumns(torch.arange(2), c, x)
    value = loss(SimpleNamespace(outputs={0: m, 1: m}), .3, training=True)
    expected = sum((c[i]*x[i]).square().sum()*.3 for _ in range(2) for i in range(2))
    torch.testing.assert_close(value, expected)
    a = torch.autograd.grad(value, (c, x), retain_graph=True)
    b = torch.autograd.grad(expected, (c, x))
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v)


def test_half_port_loss_promotes_before_product_and_square():
    from types import SimpleNamespace
    from tidegraph.weighted.message_columns import MessageColumns
    c = torch.tensor([2.], dtype=torch.float16, requires_grad=True)
    x = torch.tensor([[300., -200.]], dtype=torch.float16, requires_grad=True)
    m = MessageColumns(torch.tensor([0]), c, x)
    value = loss(SimpleNamespace(outputs={0:m, 1:m}), .0001, training=True)
    assert value.dtype == torch.float32
    # Two physical ports, each (2*300)^2+(2*-200)^2; half would overflow.
    torch.testing.assert_close(value, torch.tensor(104.))
    dc, dx = torch.autograd.grad(value, (c,x))
    torch.testing.assert_close(dc, torch.tensor([104.], dtype=torch.float16))
    torch.testing.assert_close(dx, torch.tensor([[.48,-.32]], dtype=torch.float16))


@pytest.mark.parametrize("family", ["kv", "ssd"])
@pytest.mark.parametrize("client", ["pytorch", "python-native"])
def test_half_default_task_independent_master_training_and_exact_next_window(family, client, tmp_path):
    from test_weighted_precision import half_close
    value = selected(family, client)
    value["execution"].update(dtype="float16", scoring_dtype="float32")
    resolved = resolve(value)
    left, right = Eager(resolved, reference=True), Eager(resolved)
    resumed = None
    try:
        assert right.manifest()["effective_input_interface"] == "tensor-window-v1"
        for i in range(2):
            a, x = left.step(i); b, y = right.step(i)
            half_close(a,b)
            assert x == pytest.approx(y, rel=2e-2, abs=1e-3)
            half_close(left.decoded_state(), right.decoded_state())
            half_close(left.training_state(), right.training_state())
        right.save(tmp_path)
        resumed = Eager(resolved, checkpoint=tmp_path)
        a, x = right.step(2); b, y = resumed.step(2)
        assert x == y
        equivalent(a,b,atol=0,rtol=0)
        equivalent(right.decoded_state(),resumed.decoded_state(),atol=0,rtol=0)
        equivalent(right.training_state(),resumed.training_state(),atol=0,rtol=0)
    finally:
        left.close(); right.close()
        if resumed is not None: resumed.close()


def test_policy_has_new_identity_and_old_frozen_records_stay_records():
    now = ExecutionSpec(return_mode="outputs").record()
    old = ExecutionSpec(return_mode="outputs", input_interface="records").record()
    assert now["schema"] == "tide-weighted-execution-v5"
    assert now["execution_id"] != old["execution_id"]
    assert "input_interface" not in old["requested"]
    assert ExecutionSpec.from_record(old).input_interface == "records"
    assert ExecutionSpec.from_record(old).record() == old
    assert ExecutionSpec.from_record(now).record() == now
    with pytest.raises(ValueError, match="input_interface"):
        ExecutionSpec(input_interface="invented")


@pytest.mark.parametrize("extra", [dict(dtype="float16"),
    dict(device="npu:0",peer_devices=("npu:0","npu:1"))])
def test_column_policy_preserves_auto_for_half_and_multiple_owners(extra):
    chosen = ExecutionSpec(return_mode="outputs", **extra).record()
    assert chosen["schema"] == "tide-weighted-execution-v5"
    assert chosen["effective"]["input_interface"] == "auto"
    assert chosen["effective"]["query_execution"] == "auto"
    old = ExecutionSpec(return_mode="outputs", input_interface="records", **extra).record()
    assert old["effective"]["query_execution"] == "legacy"
    assert ExecutionSpec.from_record(old).record() == old
    assert old["execution_id"] != chosen["execution_id"]


@pytest.mark.parametrize("change", ["records", "greedy", "complete", "feedback", "non-aligned"])
def test_non_column_requests_keep_the_general_contract(change):
    value = selected()
    if change in {"feedback", "non-aligned"}:
        value["model"]["transforms"] = [dict(kind=change)]
    elif change == "records":
        value["execution"]["input_interface"] = "records"
    elif change == "greedy":
        value["execution"]["prefill_schedule"] = "greedy"
    else:
        value["execution"]["return_mode"] = "complete"
    owner = Eager(resolve(value))
    try:
        assert owner.manifest()["effective_input_interface"] == "records"
        assert owner.column_reason
        result = owner.forward(0)
        assert hasattr(result, "trace")
    finally:
        owner.close()
