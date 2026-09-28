"""The installed gate must certify actual configurations and preserve failures."""
from pathlib import Path
import json
import os
import pytest
import torch
from tidegraph import GraphConfig, GraphRuntime, External
from tidegraph.qualification import qualify


@pytest.fixture(autouse=True)
def child_import_path(monkeypatch):
    # pytest's pythonpath setting does not propagate to a fresh Python process.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "python"))


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_gate_all_families(dtype, family, implementation, tmp_path):
    topology = dict(kind="ring", size=4) if family == "pdg" else dict(kind="diamond")
    topology.update(module=dict(full="swiglu"), node_overrides={"0":dict(memory="linear"),
                    "1":dict(memory="ssm"), "2":dict(memory="delta"), "3":dict(memory="attention")})
    cfg = GraphConfig.from_dict(dict(schema_version=1, family=family, topology=topology,
        model=dict(width=8, dtype=str(dtype).split(".")[-1]),
        execution=dict(implementation=implementation, mode="hst")))
    build = os.environ.get("TIDE_BUILD_DIR", str(Path(__file__).resolve().parents[1] / "build"))
    report = qualify(cfg, device="cpu", output_dir=tmp_path / "gate", width=4, positions=3, steps=2,
                     optimizer="momentum" if family == "pdg" else "adamw",
                     native_library=build if implementation == "native" else None)
    assert report["state"] == "passed" and report["topology_unchanged"]
    assert report["coverage"]["observed_nodes"] == 4
    assert report["requested_config"]["model"]["width"] == 8
    assert report["effective_config"]["model"]["width"] == 4
    names = {check["name"] for check in report["checks"]}
    assert {"independent-vjps", "optimizer-trajectory", "fresh-process-checkpoint"} <= names
    assert ("independent-direct-settle" in names) == (family == "settle")
    assert json.loads((tmp_path / "gate/report.json").read_text())["state"] == "passed"


def test_caller_external_data_and_no_overwrite(tmp_path):
    cfg = GraphConfig.from_dict(dict(schema_version=1, family="pdg", topology=dict(kind="self_loop", size=1), model=dict(width=2)))
    inputs = [External(0,0,0,0,torch.ones(2)), External(0,0,1,3,torch.zeros(2))]
    report = qualify(cfg, device="cpu", output_dir=tmp_path / "gate", inputs=inputs,
                     batch_size=1, stop=5, steps=2, optimizer="sgd")
    assert report["inputs"]["source"] == "caller-supplied"
    assert report["coverage"]["pending"] > 0
    with pytest.raises(FileExistsError):
        qualify(cfg, device="cpu", output_dir=tmp_path / "gate")
    with pytest.raises(ValueError, match="explicit CPU"):
        qualify(cfg, device="npu", output_dir=tmp_path / "npu")
    assert not (tmp_path / "npu").exists()


def test_bad_candidate_preserves_failed_report(tmp_path, monkeypatch):
    cfg = GraphConfig.from_dict(dict(schema_version=1, family="timed-dag", topology=dict(kind="chain"), model=dict(width=2)))
    original = GraphRuntime._run
    def corrupt(runtime, *args, **kwargs):
        result = original(runtime, *args, **kwargs)
        if runtime.options.schedule == "frontier":
            b,t,p,x = result.outputs[0]
            result.outputs[0] = b,t,p,x + .1
        return result
    monkeypatch.setattr(GraphRuntime, "_run", corrupt)
    with pytest.raises(AssertionError):
        qualify(cfg, device="cpu", output_dir=tmp_path / "gate")
    report = json.loads((tmp_path / "gate/report.json").read_text())
    assert report["state"] == "failed" and "AssertionError" in report["error"]
