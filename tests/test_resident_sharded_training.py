"""Public multi-owner consumer paths, independent CPU oracle and disk repartition."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from resident_test_target import target_device, device_api, owner_devices
from dataclasses import replace
from tidegraph import GraphRuntime, ResidentPlacement, ResidentTrainingLimits
from resident_training_cases import configuration, tree_equal
from test_resident_training import training_case
from resident_half_training_cases import runtime, limits, roots, update, payload_master
from test_resident_library import inputs


@pytest.fixture
def target():
    return target_device(cards=2)


def placement(target, count=2, policy="locality"):
    start = torch.device(target).index
    return ResidentPlacement(devices=owner_devices(target, count), policy=policy)


@pytest.mark.parametrize("backend", ["cuda", "npu"])
def test_cpu_safe_sharded_options(backend):
    for fields in (dict(devices=f"{backend}:0"), dict(devices=[backend]),
                   dict(devices=[f"{backend}:0", f"{backend}:0"]),
                   dict(devices=["cpu"]), dict(devices=[f"{backend}:128"]), dict(policy="guess"),
                   dict(devices=["cuda:0", "npu:1"]),
                   dict(full_owners=[0]), dict(devices=[f"{backend}:0"], state_owners=[True])):
        with pytest.raises(ValueError):
            ResidentPlacement(**fields)
    valid = ResidentPlacement(devices=[f"{backend}:0", f"{backend}:1"], full_owners=[0, 1])
    assert ResidentPlacement(**valid.to_dict()) == valid
    r = GraphRuntime(configuration("pdg"), device="cpu", model_device="cpu")
    assert r.manifest()["model_storage"] == dict(requested="cpu", resolved="cpu",
        requested_node_devices=None, node_devices=["cpu"]*len(r.config.graph.nodes))
    with pytest.raises(ValueError, match="model_device"):
        GraphRuntime(configuration("pdg"), device="cpu", model_device="meta")


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_sharded_python_cpu_oracle(target, family, schedule, kind, tmp_path):
    training_case(target, family, schedule, kind, tmp_path,
                  placement=placement(target), model_device="cpu")


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("memory", ["ema", "attention", "lh-fiber-attention-all-softmax-repeat-v1"])
def test_sharded_half_consumer(target, family, schedule, memory, tmp_path):
    r = runtime(family, target, schedule, memory, "softp", model_device="cpu")
    assert all(p.device.type == "cpu" for p in r.execution_model.state_dict().values())
    values = (torch.arange(24).reshape(1, 6, 4)*.005).half()
    values[0, 1].zero_()
    with torch.no_grad(), r.training_session(1, placement=placement(target), limits=replace(limits(), backward_bytes=8*1024**3), optimizer="adamw") as s:
        record = s.manifest()["resident"]
        assert record["devices"] == 2 and record["training"]
        assert record["resolved_training_placement"] == s.placement
        previous = s.checkpoint()
        for step, mode in enumerate(("all", "zero", "none")):
            bars = []
            for start in range(step*2, step*2+2):
                args, kw = inputs(s, values, start, start+1)
                w = s.advance_device(*args, **kw)
                assert w.state_values is None and not w.cache and len(w.states) == 2
                assert sum(len(x.nodes) for x in w.states) == len(r.execution_graph.nodes)
                assert all(str(x.values.device) == d for x, d in zip(w.states, s.placement["devices"]))
                bars.append(roots(s, w, mode))
            gradients = s.backward(bars)
            assert gradients.values is None and len(gradients.parameter_shards) == 2
            assert all(x.values.dtype == torch.float32 for x in gradients.parameter_shards)
            assert s.step().applied
            saved = s.checkpoint()
            payload_master(saved)
            if mode == "none":
                tree_equal(previous["state"], saved["state"])
            previous = saved
        s.save(tmp_path / "sharded.pt")
    # Windows and returned gradients own storage beyond session close.
    assert all(torch.isfinite(x.values).all() for x in gradients.parameter_shards)
    assert w.outputs.values.dtype == torch.float16


def test_sharded_invalid_roots_and_atomic_step(target):
    r = runtime("pdg", target, model_device="cpu")
    values = torch.ones(1, 2, 4).half()*.01
    with torch.no_grad(), r.training_session(1, placement=placement(target), limits=replace(limits(), backward_bytes=8*1024**3)) as s:
        before = s.checkpoint()
        args, kw = inputs(s, values, 0, 1)
        w = s.advance_device(*args, **kw)
        with pytest.raises(ValueError, match="owner-local"):
            s.cotangents(w, final=torch.zeros(1, device=target))
        with pytest.raises(ValueError, match="all owner"):
            s.cotangents(w, states=[])
        root = roots(s, w)
        for state in root.states:
            state.final.fill_(float("nan"))
        s.backward([root])
        refused = s.step()
        assert not refused.applied and refused.refusal_code == 20 and s.generation == 0
        s.detach()
        tree_equal(before["state"], s.checkpoint()["state"])
        for name, value in before["parameters"].items():
            assert torch.equal(value, s.checkpoint()["parameters"][name])
        bad = copy.deepcopy(s.checkpoint())
        bad["state"]["steps"].fill_(-1)
        with pytest.raises(ValueError, match="counter"):
            r.training_session(1, checkpoint=bad, placement=placement(target), limits=replace(limits(), backward_bytes=8*1024**3))


@pytest.mark.parametrize("initial_cards,resume_cards", [(0, 2), (2, 3)])
def test_sharded_checkpoint_new_process(target, tmp_path, initial_cards, resume_cards):
    assert device_api(target).device_count() >= resume_cards
    r = runtime("pdg", target, memory="attention", mode="softp", model_device="cpu")
    values = (torch.arange(16).reshape(1, 4, 4)*.005).half()
    path, output = tmp_path / "prefix.pt", tmp_path / "suffix.pt"
    with torch.no_grad():
        with r.training_session(1, placement=placement(target, initial_cards), limits=replace(limits(), backward_bytes=8*1024**3), optimizer="adamw") as s:
            update(s, values, 0, 2)
            s.save(path)
            checkpoint = s.checkpoint()
        with r.training_session(1, checkpoint=path, placement=placement(target, resume_cards, "memory"), limits=replace(limits(), backward_bytes=8*1024**3)) as s:
            tree_equal(checkpoint, s.checkpoint())
            update(s, values, 2, 4)
            expected = s.checkpoint()
    subprocess.run([sys.executable, str(Path(__file__).with_name("resident_sharded_training_worker.py")),
                    "--device", target, "--cards", str(resume_cards), "--checkpoint", str(path), "--output", str(output)],
                   check=True, timeout=180)
    tree_equal(expected, torch.load(output, weights_only=True))
