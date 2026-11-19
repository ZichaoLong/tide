"""CPU logical-device mechanism checks; hardware concurrency needs its own job."""
import copy
from contextlib import ExitStack, contextmanager
import threading
from types import SimpleNamespace

import pytest
import torch

from benchmarks.statistics import RuntimeAudit
from tide import (ModelConfig, TopologyConfig, TideModel, detach_state, load_checkpoint,
                  place_model, placement_adamw, placement_report, save_checkpoint, serial_region_audit)
from tide.precision import ArithmeticAudit
from tide.region_execution import RegionExecutor, device_groups


def model(policy="per_device", *, devices=("cpu:0", "cpu:1"), timing="settle", recompute=False, terminal=False):
    source = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=int(terminal),
                            profile="compact-9-two-inputs", timing=timing, terminal_hubs=terminal), d=64, memory="K",
                            sharing="independent", checkpoint_regions=recompute))
    regions = [(source.graph.nodes[nodes[0]].position+source.graph.nodes[nodes[0]].layer) % 2
               for nodes in source.graph.regions]
    return place_model(source, devices, region_partitions=regions, region_concurrency=policy)


@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference"])
@pytest.mark.parametrize("bf16", [False, True])
def test_real_worker_overlap_and_thread_local_modes(mode, bf16):
    placed = model()
    serial = copy.deepcopy(placed)
    serial._placement_spec["region_concurrency"] = "serial"
    first = next(regions for regions in placed.layer_regions if len(device_groups(placed, regions)) == 2)
    participants = {regions[0] for regions in device_groups(placed, first).values()}
    barrier = threading.Barrier(2, timeout=10)
    original = placed._evaluate_region
    seen, lock = {}, threading.Lock()
    caller = threading.get_ident()
    def observe(region, *args):
        if region in participants:
            # This can only complete if both device tasks really overlap.
            barrier.wait()
        result = original(region, *args)
        with lock:
            seen[region] = (threading.get_ident(), torch.is_grad_enabled(),
                            torch.is_inference_mode_enabled(), torch.is_autocast_enabled("cpu"),
                            torch.get_autocast_dtype("cpu"), result[0].dtype)
        return result
    placed._evaluate_region = observe
    serial_dtypes = {}
    serial_original = serial._evaluate_region
    def observe_serial(region, *args):
        result = serial_original(region, *args)
        serial_dtypes[region] = result[0].dtype
        return result
    serial._evaluate_region = observe_serial
    with ExitStack() as stack:
        if mode == "no_grad":
            stack.enter_context(torch.no_grad())
        elif mode == "inference":
            stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False))
        x = torch.randn(1, 2, 64)
        expected, result = serial(x), placed(x)
    assert len({seen[region][0] for region in participants}) == 2
    assert all(seen[region][0] != caller for region in participants)
    assert all(row[1:5] == (mode == "grad", mode == "inference", bf16, torch.bfloat16)
               for row in seen.values())
    # BF16 is the arithmetic policy; published region tensors may remain FP32.
    assert {region: row[5] for region, row in seen.items()} == serial_dtypes
    torch.testing.assert_close(result.values, expected.values, atol=0, rtol=0)
    assert result.values.requires_grad == (mode == "grad")
    assert result.values.is_inference() == (mode == "inference")


def test_same_device_serial_order_and_failure_joins_workers():
    # Duplicate partition devices collapse into one task; no worker is needed.
    placed = model(devices=("cpu", "cpu"))
    caller = threading.get_ident()
    order = []
    with RegionExecutor(placed) as executor:
        for regions in placed.layer_regions:
            executor.map(regions, lambda region: order.append((region, threading.get_ident())))
    assert order == [(region, caller) for regions in placed.layer_regions for region in regions]
    assert placement_report(placed)["execution"]["maximum_workers"] == 0
    # A sibling task must finish before a failure escapes the executor context.
    fake = SimpleNamespace(_placement_devices=(torch.device("cpu:0"), torch.device("cpu:1")),
                           _region_partitions=(0, 1, 0, 1), layer_regions=((0, 1, 2, 3),),
                           _placement_spec={"region_concurrency": "per_device"})
    barrier, finished = threading.Barrier(2, timeout=10), threading.Event()
    order = []
    def fail(region):
        if region in (0, 1):
            barrier.wait()
        if region == 0:
            raise RuntimeError("deliberate region failure")
        order.append(region)
        if region == 3:
            finished.set()
    with pytest.raises(RuntimeError, match="deliberate region failure"):
        with RegionExecutor(fake) as executor:
            executor.map(fake.layer_regions[0], fail)
    assert finished.is_set() and order == [1, 3]
    assert not any(thread.name.startswith("tide-region") for thread in threading.enumerate())


def test_public_device_stream_context_is_captured_on_caller_and_entered_on_worker(monkeypatch):
    # No accelerator import/allocation: this checks the public context protocol,
    # while a real backend's stream semantics require hardware qualification.
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    fake = SimpleNamespace(_placement_devices=devices, _region_partitions=(0, 1),
                           layer_regions=((0, 1),), _placement_spec={"region_concurrency": "per_device"})
    caller, local, captured, exits = threading.get_ident(), threading.local(), [], []
    streams = {device: object() for device in devices}
    monkeypatch.setattr(torch.cuda, "init", lambda: captured.append(("init", threading.get_ident())))
    def current_stream(device):
        captured.append((device, threading.get_ident()))
        return streams[device]
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    @contextmanager
    def device_context(device):
        previous = getattr(local, "device", None)
        local.device = device
        try:
            yield
        finally:
            local.device = previous
            exits.append(("device", previous))
    @contextmanager
    def stream_context(stream):
        previous = getattr(local, "stream", None)
        local.stream = stream
        try:
            yield
        finally:
            local.stream = previous
            exits.append(("stream", previous))
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    barrier = threading.Barrier(2, timeout=10)
    def observe(region):
        barrier.wait()
        return local.device, local.stream, threading.get_ident()
    with RegionExecutor(fake) as executor:
        observed = executor.map((0, 1), observe)
    assert all(thread == caller for _, thread in captured)
    assert len(captured) == 4
    assert [(device, stream) for device, stream, _ in observed] == [(device, streams[device]) for device in devices]
    assert all(thread != caller for _, _, thread in observed)
    assert sorted(exits) == [("device", None), ("device", None), ("stream", None), ("stream", None)]


@pytest.mark.parametrize("audit_class", [RuntimeAudit, ArithmeticAudit])
def test_dispatch_audit_requires_explicit_recorded_serial_witness(audit_class):
    placed = model()
    serial = copy.deepcopy(placed)
    serial._placement_spec["region_concurrency"] = "serial"
    x = torch.randn(1, 2, 64)
    with pytest.raises(RuntimeError, match="serial_region_audit"), audit_class():
        placed(x)
    with serial_region_audit(placed) as execution, audit_class() as actual:
        result = placed(x, record_work=True)
    with audit_class() as expected:
        reference = serial(x, record_work=True)
    torch.testing.assert_close(result.values, reference.values, atol=0, rtol=0)
    assert result.work == reference.work
    assert actual.report() == expected.report()
    assert execution["requested_region_concurrency"] == "per_device"
    assert execution["effective_region_concurrency"] == "serial"
    # Exiting the witness restores the requested concurrent path.
    assert RegionExecutor(placed).workers == 2
    from torch.overrides import TorchFunctionMode
    class FunctionAudit(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))
    with pytest.raises(RuntimeError, match="dispatch/function modes"), FunctionAudit():
        placed(x)


def test_parallel_serial_three_fp64_updates_preserve_none_gradients():
    torch.manual_seed(971)
    parallel = model(timing="timed", recompute=True, terminal=True).double()
    serial = copy.deepcopy(parallel)
    serial._placement_spec["region_concurrency"] = "serial"
    optimizers = [placement_adamw(current, lr=1e-4, foreach=False) for current in (serial, parallel)]
    states = [None, None]
    x = torch.randn(1, 2, 64, dtype=torch.float64)
    for _ in range(3):
        gradients = []
        for i, (current, optimizer) in enumerate(zip((serial, parallel), optimizers)):
            optimizer.zero_grad(set_to_none=True)
            result = current(x, states[i], record_work=True)
            tail = current.flush(result.state, record_work=True)
            (result.readout().square().mean()+tail.readout().square().mean()).backward()
            gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                              for name, p in current.named_parameters()})
            optimizer.step()
            states[i] = detach_state(tail.state)
        assert any(value is None for value in gradients[0].values())
        assert {key for key, value in gradients[0].items() if value is None} == {
            key for key, value in gradients[1].items() if value is None}
        torch.testing.assert_close(gradients[0], gradients[1], atol=3e-10, rtol=3e-8)
        torch.testing.assert_close(serial.state_dict(), parallel.state_dict(), atol=3e-10, rtol=3e-8)
        torch.testing.assert_close(optimizers[0].state_dict(), optimizers[1].state_dict(), atol=3e-10, rtol=3e-8)


def test_policy_validation_and_old_checkpoint_default(tmp_path):
    source = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", n=1,
                                  profile="compact-5-dd4"), d=64))
    with pytest.raises(ValueError, match="serial or per_device"):
        place_model(source, ["cpu"], region_concurrency="threads")
    with pytest.raises(ValueError, match="requires region_partitions"):
        place_model(source, ["cpu"], region_concurrency="per_device")
    assert not hasattr(source, "_placement_spec")
    placed = model()
    path = tmp_path/"old.pt"
    save_checkpoint(path, placed)
    payload = torch.load(path, weights_only=False)
    payload["placement"].pop("region_concurrency")
    torch.save(payload, path)
    restored, _, _ = load_checkpoint(path)
    assert placement_report(restored)["region_concurrency"] == "serial"
