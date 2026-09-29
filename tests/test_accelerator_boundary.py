"""CPU-executable failures that must remain reliable without an accelerator."""
import pytest
import torch
from tidegraph import GraphConfig, GraphRuntime
from tidegraph.runtime import resolve_device
from tidegraph.qualification import qualify
from tidegraph.qualification_inputs import Probe
from tidegraph.qualification_checks import assert_placement, compare_finite


def config():
    return GraphConfig.from_dict(dict(schema_version=1, family="settle",
                                     topology=dict(kind="chain"), model=dict(width=2)))


@pytest.mark.parametrize("index", [-1, True, 1.5, 32768])
def test_bad_index_does_not_initialize_vendor(index, monkeypatch):
    def forbidden():
        raise AssertionError("invalid index initialized a backend")
    monkeypatch.setattr("tidegraph.runtime._load_npu", forbidden)
    with pytest.raises(ValueError, match="index"):
        resolve_device("npu", index)


def test_explicit_missing_cuda_never_runs_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        qualify(config(), device="cuda", output_dir=tmp_path / "unavailable")
    assert not (tmp_path / "unavailable").exists()


def test_cpu_half_csr_rejected_before_execution(tmp_path):
    cfg = GraphConfig.from_dict(dict(schema_version=1, family="timed-dag",
        topology=dict(kind="chain", module=dict(memory="lh-fiber-attention-sum-repeat-v1")),
        model=dict(width=4, dtype="float16"),
        execution=dict(implementation="native", fiber_pooling="csr")))
    with pytest.raises(ValueError, match="CPU FP16 CSR pooling is unsupported"):
        qualify(cfg, device="cpu", output_dir=tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()


def test_probe_transfer_preserves_fixture_and_independent_leaves():
    probe = Probe(config(), positions=3)
    fixture_hash = probe.manifest()["sha256"]
    left, right = probe.clone("cpu"), probe.clone("cpu")
    left.values.sum().backward()
    assert left.values.is_leaf and right.values.is_leaf
    assert right.values.grad is None and probe.values.grad is None
    assert left.values.data_ptr() != right.values.data_ptr() != probe.values.data_ptr()
    assert probe.manifest()["sha256"] == fixture_hash


def test_live_placement_check_catches_host_tensor():
    with pytest.raises(AssertionError, match="expected cuda:0"):
        assert_placement({"state":torch.ones(2)}, torch.device("cuda:0"))
    assert_placement({"state":torch.ones(2)}, torch.device("cpu"))


def test_numerical_comparison_keeps_dtype_and_discrete_identity():
    with pytest.raises(AssertionError):
        compare_finite(torch.ones(2), torch.ones(2, dtype=torch.float64))
    with pytest.raises(AssertionError):
        compare_finite({"pending":None}, {"pending":torch.zeros(())}, atol=1.)


def test_oracle_stays_cpu_and_independent(tmp_path, monkeypatch):
    calls = []
    original = GraphRuntime.__init__
    def observe(self, cfg, **kwargs):
        calls.append((kwargs["device"], kwargs.get("options")))
        original(self, cfg, **kwargs)
    monkeypatch.setattr(GraphRuntime, "__init__", observe)
    # An unavailable request must fail before any oracle is mistaken for a candidate.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        qualify(config(), device="cuda", output_dir=tmp_path / "gate")
    assert calls == [("cuda", None)]


def test_native_backend_mismatch_rejected_before_execution():
    from types import SimpleNamespace
    from tidegraph.native_loader import _check_backend
    cpu = SimpleNamespace(execution_backend=lambda: "cpu")
    with pytest.raises(RuntimeError, match="built for cpu"):
        _check_backend(cpu, "npu")
    with pytest.raises(ValueError, match="manifest"):
        _check_backend(cpu, "cpu", {"backend":"cuda"})
    cuda = SimpleNamespace(execution_backend=lambda: "cuda")
    assert _check_backend(cuda, "cpu") is cuda
