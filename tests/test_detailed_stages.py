"""Timing attribution, reversible hooks and exact CPU training witnesses."""
from contextlib import nullcontext
from copy import deepcopy
import importlib
import inspect
from itertools import count
import json

import pytest
import torch

from benchmarks.baselines import TransformerBaseline
from benchmarks.costs import state_hash
from benchmarks.detailed_stages import (DetailedStageTimes, OperatorTimes,
                                        detailed_stage_hooks)
from tide import ModelConfig, TideModel, TopologyConfig


def test_stage_entry_wait_belongs_to_parent_not_following_publish():
    class PendingDevice:
        now, pending = 0., 11.

        def clock(self):
            return self.now

        def sync(self, device):
            self.now += self.pending
            self.pending = 0.

    device = PendingDevice()
    observer = DetailedStageTimes(torch.device("cpu"), clock=device.clock, sync=device.sync)
    with observer.scope("full"):
        device.pending = 7.  # Earlier, unwrapped asynchronous FFN work.
        with observer.scope("queries.publish"):
            device.pending = 2.
        device.pending = 3.
    rows = {row["path"]: row for row in observer.report()["rows"]}
    assert rows["full/queries.publish"]["inclusive_seconds"] == 2.
    assert rows["full"]["inclusive_seconds"] == 12.
    assert rows["full"]["exclusive_seconds"] == 10.
    assert sum(row["exclusive_seconds"] for row in rows.values()) == 12.
    assert observer.stack == observer.children == []


@pytest.mark.parametrize("body_fails", [False, True])
def test_stage_scope_cleans_up_when_completion_synchronization_fails(body_fails):
    syncs = count()

    def sync(device):
        if next(syncs) == 1:
            raise RuntimeError("completion failed")

    observer = DetailedStageTimes(torch.device("cpu"), sync=sync)
    expected = ValueError if body_fails else RuntimeError
    with pytest.raises(expected, match="body failed" if body_fails else "completion failed"):
        with observer.scope("failed"):
            if body_fails:
                raise ValueError("body failed")
    assert observer.stack == observer.children == []
    assert observer.report()["rows"] == []
    assert observer.report()["boundary_failures"][0]["boundary"] == "completion"
    with observer.scope("next"):
        pass
    assert [row["path"] for row in observer.report()["rows"]] == ["next"]


def test_operator_timer_excludes_entry_sync_and_includes_completion():
    now, events = [0.], []
    waits = iter([17., 2.])

    def sync(device):
        events.append("sync")
        now[0] += next(waits)

    def clock():
        events.append("clock")
        return now[0]

    inputs = torch.ones(2, 3)
    observer = OperatorTimes(torch.device("cpu"), clock=clock, sync=sync)
    with observer, observer.scope("forward"):
        inputs + inputs
    assert events == ["sync", "clock", "sync", "clock"]
    report = observer.report()
    assert report["operator_seconds"] == 2.
    assert report["dispatched_calls"] == 1
    assert report["rows"][0]["path"] == "forward"


def test_operator_groups_path_dtype_shape_stride_and_reports_consistent_totals():
    ticks, syncs = count(), []
    observer = OperatorTimes(torch.device("cpu"), clock=lambda: float(next(ticks)), sync=syncs.append)
    a, b = torch.randn(2, 3), torch.randn(3, 4)
    double_a, double_b = a.double(), b.double()
    larger = torch.randn(4, 3)
    strided = torch.randn(3, 2).T
    with observer:
        with observer.scope("forward"):
            torch.mm(a, b)
            torch.mm(a, b)
            torch.mm(double_a, double_b)
            torch.mm(larger, b)
            torch.mm(strided, b)
        with observer.scope("decode"):
            torch.mm(a, b)
    report = observer.report()
    assert len(report["rows"]) == 5
    assert all(row["operator"] == "aten.mm.default" for row in report["rows"])
    assert sorted(row["calls"] for row in report["rows"]) == [1, 1, 1, 1, 2]
    assert sum(row["calls"] for row in report["rows"]) == report["dispatched_calls"] == 6
    assert sum(row["seconds"] for row in report["rows"]) == report["operator_seconds"] == 6.
    assert len(syncs) == 12
    assert json.loads(json.dumps(report)) == report


def test_failed_dispatch_restores_mode_and_path_and_records_failure():
    a, b = torch.ones(2, 3), torch.ones(4, 2)
    observer = OperatorTimes(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        with observer, observer.scope("bad"):
            torch.mm(a, b)
    assert observer.path == "outside"
    report = observer.report()
    assert report["dispatched_calls"] == 0 and report["failed_calls"] == 1
    assert report["failures"][0]["boundary"] == "dispatch"
    a + a  # The failed context must no longer intercept dispatches.
    assert observer.report() == report
    with observer, observer.scope("next"):
        a + a
    assert observer.report()["rows"][0]["path"] == "next"


def make_model(kind):
    if kind == "transformer":
        return TransformerBaseline(64, 1, window=3, chunk=2)
    return TideModel(ModelConfig(
        topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4",
                                timing="timed" if kind == "tide_batched" else "settle"),
        d=64, memory="X", sharing="independent", memory_sharing="region_shared",
        region_batch_size=4 if kind == "tide_batched" else 1))


def update(model, x, observer=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    scope = (lambda name: nullcontext()) if observer is None else observer.scope
    hooks = nullcontext() if observer is None else detailed_stage_hooks(model, observer)
    dispatch = observer if isinstance(observer, OperatorTimes) else nullcontext()
    with hooks, dispatch:
        with scope("training"):
            with scope("forward"):
                result = model(x)
                values = [result.readout()] if isinstance(model, TideModel) else [result]
            with scope("flush"):
                if isinstance(model, TideModel) and not model.settle:
                    values.append(model.flush(result.state).readout())
            with scope("loss"):
                output = torch.cat(values, 1)
                loss = output.square().mean()
            with scope("backward"):
                loss.backward()
            with scope("optimizer"):
                optimizer.step()
    gradients = {name: p.grad for name, p in model.named_parameters()}
    assert x.grad is not None and x.grad.count_nonzero() > 0
    assert any(g is not None and g.count_nonzero() > 0 for g in gradients.values())
    return output.detach(), x.grad, (state_hash(gradients), state_hash(model.state_dict()),
                                    state_hash(optimizer.state_dict()))


@pytest.mark.parametrize("kind", ["transformer", "tide_serial", "tide_batched"])
@pytest.mark.parametrize("observer_type", [DetailedStageTimes, OperatorTimes])
def test_detailed_hooks_preserve_outputs_input_parameter_gradients_and_updates(kind, observer_type):
    torch.manual_seed(217)
    reference = make_model(kind)
    observed = deepcopy(reference)
    x = torch.randn(1, 3, 64, requires_grad=True)
    expected = update(reference, x)
    observer = observer_type(torch.device("cpu"))
    actual = update(observed, x.detach().clone().requires_grad_(), observer)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[2] == expected[2]
    paths = {row["path"] for row in observer.report()["rows"]}
    assert any(path.startswith("training/backward") for path in paths)
    assert any(path.startswith("training/optimizer") for path in paths)
    if kind == "transformer":
        assert any(path.endswith("block/qkv") for path in paths)
        assert not any("linear.blocks.0.qkv" in path for path in paths)
    else:
        assert any("/ffn/node_linear." in path for path in paths)
        assert any("memory.K.prepare/kv.history.order" in path for path in paths)
        assert any("memory.S.prepare/ssd.summaries" in path for path in paths)
    assert observer.stack == []


@pytest.mark.parametrize("kind", ["transformer", "tide_batched"])
@pytest.mark.parametrize("observer_type", [DetailedStageTimes, OperatorTimes])
def test_detailed_hooks_restore_owned_inherited_and_global_attributes_after_exception(kind, observer_type):
    model = make_model(kind)
    target = model.receive if isinstance(model, TideModel) else model.blocks[0].qkv

    def fail(*args, **kwargs):
        raise RuntimeError("intentional forward failure")

    target.forward = fail  # Exercise restoration of an instance-owned override.
    attributes = [(module, "forward") for module in model.modules()]
    for module in [model, *getattr(model, "memories", {}).values()]:
        attributes.extend((module, name) for name in
                          ("_incoming", "_region_batches", "_evaluate_region", "_score_regions",
                           "_layout", "prepare", "finish", "_finish_tile", "_summaries", "_initial")
                          if hasattr(module, name))
    for name in ("tide.model", "tide.region_batch", "tide.parameters", "tide.kv",
                 "tide.ssd_memory", "tide.reception", "tide.region", "benchmarks.baselines"):
        module = importlib.import_module(name)
        attributes.extend((module, key) for key, value in vars(module).items() if inspect.isfunction(value))
    attributes.append((importlib.import_module("tide.layout").Queries, "publish"))
    before = [(obj, name, getattr(obj, name), name in vars(obj)) for obj, name in attributes]
    observer = observer_type(torch.device("cpu"))
    dispatch = observer if isinstance(observer, OperatorTimes) else nullcontext()
    with pytest.raises(RuntimeError, match="intentional forward failure"):
        with detailed_stage_hooks(model, observer), dispatch, observer.scope("failed"):
            model(torch.randn(1, 3, 64))
    assert observer.stack == []
    for obj, name, original, owned in before:
        assert (name in vars(obj)) == owned
        assert getattr(obj, name) == original
