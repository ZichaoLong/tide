"""Downstream heads must be real optimizer/checkpoint/independent-root owners."""
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.experiment.config import resolve
from tidegraph.experiment.eager import Eager
from tidegraph.experiment.task_identity import identity, verify
from tidegraph.experiment.task_module import bind, validate_optimizer

EXAMPLE = Path(__file__).resolve().parents[1]/"examples/consumer_core5/configured/src"


@pytest.fixture
def config(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from tidegraph_consumer_core5.configure import configuration
    request = configuration(device="cpu", mode="training", samples=1, positions=1, dtype="float64",
        model=dict(recipe=dict(topology="fc", family="kv", width=4, depth=2, region_width=1, history=4)))
    torch.set_num_threads(1)
    return resolve(request)


def test_task_head_independent_training_disk_checkpoint_and_exact_next(config, tmp_path):
    candidate, reference = Eager(config), Eager(config, reference=True)
    before = candidate.task.module.weight.detach().clone()
    for index in range(2):
        actual, loss = candidate.step(index)
        expected, want_loss = reference.step(index)
        equivalent(actual, expected, atol=1e-9, rtol=1e-8)
        assert loss == pytest.approx(want_loss, rel=1e-8, abs=1e-9)
        equivalent(candidate.parameters(), reference.parameters(), atol=1e-9, rtol=1e-8)
        equivalent(candidate.training_state(), reference.training_state(), atol=1e-9, rtol=1e-8)
    assert not torch.equal(before, candidate.task.module.weight)
    assert candidate.update_counts["task.weight"] == 2
    assert candidate.training_state()["gradients"]["task.weight"] is not None
    assert candidate.module_manifest["parameters"]["weight"]["dtype"] == "torch.float64"
    assert candidate.last_work["actual_record_locations"]["state_values"]["cpu"]["tensors"] > 0
    checkpoint = tmp_path/"checkpoint"
    checkpoint.mkdir()
    candidate.save(checkpoint)
    saved = torch.load(checkpoint/"state.pt", weights_only=True)
    assert "task.weight" in saved["weights"] and any(n.startswith("graph.nodes.") for n in saved["weights"])
    restored = Eager(config, checkpoint=checkpoint)
    equivalent(candidate.decoded_state(), restored.decoded_state(), atol=0, rtol=0)
    actual, loss = candidate.step(2)
    expected, want_loss = restored.step(2)
    equivalent(actual, expected, atol=0, rtol=0)
    equivalent(candidate.parameters(), restored.parameters(), atol=0, rtol=0)
    equivalent(candidate.training_state(), restored.training_state(), atol=0, rtol=0)
    assert loss == want_loss
    candidate.close(); reference.close(); restored.close()


def test_independent_roots_include_disconnected_and_head_loss(config):
    from tidegraph.experiment.checks import eager_roots
    result = eager_roots(config, full=False)
    assert result["state"] == "passed" and "downstream-task-loss" in result["roots"]


def test_task_optimizer_missing_unowned_and_duplicate_owners_are_rejected():
    graph = torch.nn.Linear(2, 2)
    task = SimpleNamespace(module=torch.nn.Linear(2, 1))
    task.module.register_parameter("weight_alias", task.module.weight)
    owner = bind(task, graph)
    names = dict(owner.named_parameters(remove_duplicate=False))
    assert names["task.weight"] is names["task.weight_alias"]
    with pytest.raises(ValueError, match="missing trainable"):
        validate_optimizer(owner, torch.optim.SGD(graph.parameters(), lr=.1))
    external = torch.nn.Parameter(torch.ones(()))
    with pytest.raises(ValueError, match="not owned"):
        validate_optimizer(owner, torch.optim.SGD([*owner.parameters(), external], lr=.1))
    optimizer = torch.optim.SGD(owner.parameters(), lr=.1)
    optimizer.param_groups[0]["params"].append(task.module.weight)
    with pytest.raises(ValueError, match="repeats a shared"):
        validate_optimizer(owner, optimizer)
    task.module.weight = graph.weight
    with pytest.raises(ValueError, match="across graph/task"):
        validate_optimizer(owner, torch.optim.SGD(owner.parameters(), lr=.1))


def test_task_nonfinite_and_midrun_owner_replacement_cannot_be_ignored(config):
    from tidegraph.comparison import NonFiniteMismatch
    candidate = Eager(config)
    with torch.no_grad():
        candidate.task.module.weight.fill_(float("nan"))
    with pytest.raises(NonFiniteMismatch):
        candidate.step(0)
    candidate.close()
    candidate = Eager(config)
    candidate.task.module.weight = torch.nn.Parameter(candidate.task.module.weight.detach().clone())
    with pytest.raises(ValueError, match="ownership changed"):
        candidate.step(0)
    candidate.close()


def test_whole_plugin_package_identity_needs_no_import_and_refuses_changed_helper(monkeypatch, tmp_path):
    package = tmp_path/"task_identity_fixture"
    package.mkdir()
    (package/"__init__.py").write_text("raise AssertionError('identity must not import the package')\n")
    (package/"helper.py").write_text("VERSION = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    task = dict(plugin=dict(module="task_identity_fixture.task", factory="create_task", options={"nested": [1]}))
    old = identity(task)
    verify(task, old)
    task["plugin"]["options"]["nested"].append(2)
    assert old["options"]["nested"] == [1]
    with pytest.raises(ValueError, match="code/configuration"):
        verify(task, old)
    task["plugin"]["options"]["nested"].pop()
    (package/"helper.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="installed task code"):
        verify(task, old)


def test_restore_rejects_task_source_identity_and_missing_head_weight(config, tmp_path):
    candidate = Eager(config)
    candidate.step(0)
    candidate.save(tmp_path)
    path = tmp_path/"task.pt"
    extra = torch.load(path, weights_only=True)
    original = copy.deepcopy(extra)
    extra["task_manifest"]["package_source_sha256"] = "0"*64
    torch.save(extra, path)
    with pytest.raises(ValueError, match="task source"):
        Eager(config, checkpoint=tmp_path)
    torch.save(original, path)
    path = tmp_path/"state.pt"
    state = torch.load(path, weights_only=True)
    del state["weights"]["task.weight"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="parameter keys"):
        Eager(config, checkpoint=tmp_path)
    candidate.close()


def test_example_platform_configuration_and_task_formula(config):
    from tidegraph_consumer_core5.configure import configuration
    from tidegraph_consumer_core5.check import check
    for device in ("cpu", "cuda:0", "npu:4"):
        for mode in ("inference", "training"):
            value = resolve(configuration(device=device, mode=mode))
            assert value["execution"]["effective"]["device"] == device
            assert value["task"]["plugin"]["module"] == "tidegraph_consumer_core5.task"
    assert check()["state"] == "passed"
    with pytest.raises(ValueError, match="standalone LibTorch"):
        configuration(device="cpu", client="libtorch")


def test_inference_independent_check_compares_task_state(config):
    from tidegraph.experiment.validation import independent
    request = copy.deepcopy(config["requested"])
    request["measurement"]["mode"] = "inference"
    resolved = resolve(request)
    owner = Eager(resolved)
    observation = owner.step(0)
    task_state = owner.task_state()
    report = independent(resolved, [observation], None, None, 0, task_state=task_state)
    assert report["strict_passed"]
    task_state["module"]["bias"].add_(.1)
    report = independent(resolved, [observation], None, None, 0, task_state=task_state)
    assert report["numeric"] == "failed" and not report["strict_passed"]
    owner.close()


def test_resident_task_module_is_rejected_before_graph_allocation(config, monkeypatch):
    from tidegraph.experiment.resident import Resident
    from tidegraph.planning import materialize
    def unexpected(*args, **kwargs):
        raise AssertionError("resident rejection must precede materialization")
    monkeypatch.setattr(materialize, "materialize", unexpected)
    with pytest.raises(ValueError, match="resident does not register external"):
        Resident(config)
