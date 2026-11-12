"""Source/material identity and timing/restore boundaries for REQ-006 tools."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from req006_material import configuration, extract_source, synthetic_inputs, tensor_digest
from req006_profile import summarize_trace
from req006_runtime import capture, restore, run_once


@pytest.mark.parametrize("case,nodes,batch", [
    ("hb88-pool-s11", 88, 16), ("hb184-quad-s11", 184, 8),
    ("tree-r64-h2-q3-s11", 661, 4), ("tf-hb184-s11", None, 64)])
def test_retained_original_configuration_identity(case, nodes, batch):
    config, identity = configuration(case)
    assert len(identity["sha256"]) == 64
    assert config["training"]["batch"] == batch
    assert config["training"]["global_batch"] == 64
    assert config["training"]["length"] == 256
    if nodes is not None:
        assert config["direct"]["plan"]["graph"]["nodes"] == nodes
    else:
        assert config["transformer"]["width"] == 512
        assert config["transformer"]["layers"] == 16


def test_source_extraction_uses_git_bytes_not_worktree(tmp_path):
    repository = tmp_path/"repo"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init")
    package = repository/"tide_lab"
    package.mkdir()
    (package/"campaign_runtime.py").write_text("identity = 'fixed'\n")
    git("add", "tide_lab")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixed input")
    revision = git("rev-parse", "HEAD")
    (package/"campaign_runtime.py").write_text("identity = 'active edits'\n")
    record = extract_source(repository, tmp_path/"extracted", revision=revision)
    assert record["commit"] == revision
    assert (tmp_path/"extracted/tide_lab/campaign_runtime.py").read_text() == "identity = 'fixed'\n"
    assert (package/"campaign_runtime.py").read_text() == "identity = 'active edits'\n"
    with pytest.raises(ValueError, match="full immutable"):
        extract_source(repository, tmp_path/"short", revision=revision[:8])


def test_material_is_independent_of_partition_and_restores_populated_optimizer():
    material = synthetic_inputs(5, 3, 19)
    assert tensor_digest(material) == tensor_digest(synthetic_inputs(5, 3, 19))
    assert not torch.equal(material["input_ids"], synthetic_inputs(5, 3, 20)["input_ids"])
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
    x = torch.randn(2, 3)
    def update():
        optimizer.zero_grad(set_to_none=True)
        model(x).square().sum().backward()
        optimizer.step()
    update()
    state = capture(model, optimizer, torch.device("cpu"))
    identity = tensor_digest(state)
    update()
    restore(state, model, optimizer, torch.device("cpu"))
    assert tensor_digest(capture(model, optimizer, torch.device("cpu"))) == identity
    update()
    assert tensor_digest(state) == identity  # In particular, CPU Adam step counters did not alias.


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 3)
        self.head = torch.nn.Linear(3, 12)
        self.last_metrics = {}

    def forward(self, x):
        return self.head(self.embedding(x))


@pytest.mark.parametrize("mode", ["nograd", "grad-forward", "training"])
def test_complete_interval_denominator_and_forward_duties(mode):
    model = ToyModel()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    material = synthetic_inputs(5, 3, 19, vocabulary=12)
    material["labels"][0, 0] = -100
    config = dict(training=dict(batch=2))
    calls = []
    def logical_update(model, optimizer, batches, config, progress, device):
        calls.append((len(batches), progress))
        optimizer.zero_grad(set_to_none=True)
        count = sum(int((b["labels"] != -100).sum()) for b in batches)
        for batch in batches:
            logits = model(batch["input_ids"])
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(),
                                                    reduction="sum", ignore_index=-100)/count
            loss.backward()
        optimizer.step()
        return dict(seconds=.001, tokens=count, work={}, microbatches=len(batches))
    runtime = SimpleNamespace(synchronize=lambda _: None, logical_update=logical_update)
    result = run_once(runtime, model, optimizer, material, config, torch.device("cpu"), mode=mode, progress=123)
    assert result["tokens"] == 14 and result["input_positions"] == 15
    assert result["effective_target_tokens_per_second"] == 14/result["complete_seconds"]
    assert result["microbatches"] == 3
    if mode == "training":
        assert calls == [(3, 123)]
        assert result["update_tokens_per_second"] == 14000
        logits = reference(material["input_ids"])
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), material["labels"].flatten()).backward()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
    else:
        assert not calls and "update_seconds" not in result
        assert tensor_digest(model.state_dict()) == tensor_digest(reference.state_dict())
        assert all(p.grad is None for p in model.parameters())


def test_trace_keeps_interface_counts_device_events_and_scopes_distinct(tmp_path):
    def event(name, ts, dur, **extra):
        return dict(ph="X", pid=1, tid=2, name=name, ts=ts, dur=dur, **extra)
    events = [event("req006::model", 0, 100), event("req006::node_linear", 10, 10),
              event("aten::bmm", 12, 4, args={"Input Dims": [[4, 128, 16], [4, 16, 32]]}),
              event("aten::stack", 30, 2), event("aten::_local_scalar_dense", 40, 5),
              dict(ph="M", name="process_name", pid=9, args={"name": "Ascend Hardware"}),
              dict(ph="X", pid=9, tid=0, name="Index", ts=13, dur=7, args={"Task Type": "KERNEL_AICPU"})]
    path = tmp_path/"trace.json"
    path.write_text(json.dumps(dict(traceEvents=events)))
    result = summarize_trace(path)
    matrix, = [row for row in result["operators"] if row["operator"] == "aten::bmm"]
    assert matrix["stage"] == "req006::node_linear" and matrix["calls"] == 1
    assert result["operator_categories"]["index_and_assembly"] == 1
    assert result["kernel_types"] == {"KERNEL_AICPU": 1}
    assert result["kernel_union_us"] == 7
    assert not result["observed_host_fallback_events"]
