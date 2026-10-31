from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from tidegraph.experiment.config import resolve
from tidegraph.experiment.matrix import resolve_matrix
from tidegraph.experiment.site import launch_environment, load
from tidegraph.planning.hierarchy import general_ports_plan
from tidegraph.planning.model import resolve_model


def example():
    return dict(schema="tide-experiment-v1", name="small",
        model=dict(recipe=dict(topology="fc", depth=2, region_width=1, width=8, history=4)),
        task=dict(samples=2, positions=3, input_pattern="staggered-v1"),
        execution=dict(device="cpu", family="settle"))


def test_public_plan_is_allocation_free_and_defaults_keep_original_objective(tmp_path):
    config = tmp_path/"run.json"
    config.write_text(json.dumps(example()))
    source = str(Path(__file__).resolve().parents[1]/"python")
    code = "import sys; sys.path.insert(0, sys.argv[1]); from tidegraph.experiment.config import resolve; r=resolve(sys.argv[2]); assert 'torch' not in sys.modules; print(r['config_id'])"
    proc = subprocess.run([sys.executable, "-S", "-c", code, source, str(config)], capture_output=True, text=True, check=True)
    record = resolve(config)
    assert proc.stdout.strip() == record["config_id"]
    assert record["task"]["objective"] == dict(kind="weighted-output-square", scale=.001)
    assert record["task"]["optimizer"] == dict(kind="sgd", lr=.001, momentum=.3, weight_decay=.01)
    assert record["capacity_budget"]["automatic_admission"] is False
    assert record["resources"]["logical_windows"] == 4
    assert "resident_limits" not in record["resources"]


def test_general_public_model_and_same_model_hierarchy_identity():
    plan, _ = general_ports_plan()
    model, _ = resolve_model(**{key: plan[key] for key in (
        "graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")})
    request = example()
    request["model"] = dict(resolved=model)
    request["task"]["input_origins"] = plan["input_origins"]
    first = resolve(request)
    request["execution"]["family"] = "pdg"
    second = resolve(request)
    assert first["model_id"] == second["model_id"] == model["model_id"]
    assert first["plan"] == second["plan"]
    assert first["execution_id"] != second["execution_id"]
    assert first["plan"]["graph"]["inputs"][-2:] == [2, 2]


def test_model_task_execution_and_frozen_content_are_distinct():
    request = example()
    first = resolve(request)
    request["task"]["samples"] = 8
    larger_batch = resolve(request)
    assert first["model_id"] == larger_batch["model_id"]
    assert first["workload_id"] != larger_batch["workload_id"]
    request["execution"]["intraop_threads"] = 4
    threaded = resolve(request)
    assert larger_batch["task_id"] == threaded["task_id"]
    assert larger_batch["execution_id"] != threaded["execution_id"]
    assert resolve(json.loads(json.dumps(first)))["config_id"] == first["config_id"]
    first["measurement"]["validation"] = "none"
    with pytest.raises(ValueError, match="content has changed"):
        resolve(first)


@pytest.mark.parametrize("change,field", [
    ({"execution": {"dtype": "float32"}}, "execution.device"),
    ({"execution": {"device": "cpu", "worker": 1}}, "execution.worker"),
    ({"measurement": {"validation": "ignore-anything"}}, "measurement.validation"),
    ({"task": {"samples": True}}, "task.samples"),
    ({"resources": {"logical_windows": "4"}}, "resources.logical_windows"),
    ({"resources": {"logical_windows": 1}}, "logical_windows"),
    ({"site": {"environment": {"API_TOKEN": "unprinted"}}}, "site.environment.API_TOKEN"),
    ({"execution": {"device": "cpu", "return_mode": "metrics"}}, "independent validation"),
])
def test_field_qualified_errors(change, field):
    request = example()
    request.update(change)
    with pytest.raises(ValueError, match=field):
        resolve(request)


def test_feedback_changes_the_model_and_not_only_its_name():
    request = example()
    first = resolve(request)
    request["execution"]["family"] = "pdg"
    request["model"]["transforms"] = [dict(kind="feedback", delay=2)]
    feedback = resolve(request)
    assert feedback["model_id"] != first["model_id"]
    assert len(feedback["plan"]["graph"]["edges"]) == len(first["plan"]["graph"]["edges"])+1
    assert feedback["plan"]["graph"]["kind"] == "positive-delay"


def test_multicard_plan_preserves_single_model_and_records_each_owner():
    request = example()
    original = resolve(request)
    request["execution"] = dict(device="cuda:0", client="python-native", placement="mixed-c",
                                peer_devices=["cuda:0", "cuda:1", "cuda:2"])
    record = resolve(request)
    assert record["model_id"] == original["model_id"]
    assert set(record["payload_placement"]["per_device"]) == {"cuda:0", "cuda:1", "cuda:2"}
    assert record["execution"]["effective"]["node_devices"] == tuple(record["payload_placement"]["node_devices"])


def test_resident_resource_expansion_and_no_silent_reduced_return_policy():
    request = example()
    request["execution"] = dict(device="cuda:0", client="python-native", placement="resident")
    request["resources"] = dict(resident_limits=dict(tensor_bytes=2048))
    record = resolve(request)
    assert record["resources"]["resident_limits"]["tensor_bytes"] == 2048
    assert record["resources"]["resident_limits"]["prefill"] is True
    assert record["resources"]["resident_limits"]["cache"] >= record["plan"]["span"]*4+1
    request["resources"]["resident_limits"]["prefill"] = False
    with pytest.raises(ValueError, match="must match execution.schedule"):
        resolve(request)
    request = example()
    request["execution"]["return_mode"] = "outputs"
    request["measurement"] = dict(validation="none")
    assert resolve(request)["measurement"]["validation"] == "none"


def test_site_paths_and_environment_are_explicit_and_bounded(tmp_path):
    site = tmp_path/"site.json"
    site.write_text(json.dumps(dict(schema="tide-site-v1", paths=dict(native_library="build"),
                                   threads=dict(intraop_threads=4))))
    selected = load(site)
    assert selected["paths"]["native_library"] == str(tmp_path/"build")
    execution = resolve(example())["execution"]
    environment = launch_environment(selected, execution, environ={"API_TOKEN": "private", "LD_LIBRARY_PATH": "/explicit"})
    assert "API_TOKEN" not in environment
    assert environment["LD_LIBRARY_PATH"] == "/explicit"
    assert environment["OMP_NUM_THREADS"] == "1"


def test_finite_matrix_resolves_only_named_cases_and_rejects_path_escape():
    matrix = dict(schema="tide-experiment-matrix-v1", cases=[
        dict(name="b2", config=example()), dict(name="b8", config=example(), overrides=dict(task=dict(samples=8)))])
    resolved = resolve_matrix(matrix)
    assert len(resolved["cases"]) == 2
    assert resolved["cases"][0]["config"]["model_id"] == resolved["cases"][1]["config"]["model_id"]
    assert resolved["cases"][1]["config"]["plan"]["samples"] == 8
    matrix["cases"][1]["name"] = "../escape"
    with pytest.raises(ValueError, match="safe name"):
        resolve_matrix(matrix)


def test_b512_and_more_than_eight_measured_windows_have_no_legacy_sweep_cap():
    request = example()
    request["task"]["samples"] = 512
    request["measurement"] = dict(steps=12)
    config = resolve(request)
    assert config["plan"]["samples"] == 512
    assert config["measurement"]["steps"] == 12
    assert config["resources"]["logical_windows"] == 14


def test_disabling_extra_resume_check_still_budgets_durable_checkpoint():
    request = example()
    request["measurement"] = dict(checkpoint_check=False)
    config = resolve(request)
    assert config["resources"]["logical_windows"] == 3
    assert config["capacity_budget"]["checkpoint"] is True
    assert config["capacity_budget"]["checkpoint_policy"]["estimated_file_tensor_bytes"] > 0


def test_declared_unused_device_is_an_error_but_site_inventory_can_be_larger():
    request = example()
    request["resources"] = dict(devices={"cuda:0": {"free_bytes": 1024**3}})
    with pytest.raises(ValueError, match="absent from execution"):
        resolve(request)
    request.pop("resources")
    request["site"] = dict(resources=dict(devices={"cuda:0": {"free_bytes": 1024**3}}))
    assert resolve(request)["resources"]["devices"] == {}


def test_matrix_identity_failure_publishes_terminal_result(tmp_path, monkeypatch):
    from tidegraph.experiment import matrix
    request = dict(schema="tide-experiment-matrix-v1", cases=[dict(name="one", config=example())])
    monkeypatch.setattr(matrix, "capture", lambda paths: {})
    def changed(*args):
        raise ValueError("installation changed")
    monkeypatch.setattr(matrix, "verify", changed)
    with pytest.raises(ValueError, match="installation changed"):
        matrix.run_matrix(request, tmp_path/"matrix")
    result = json.loads((tmp_path/"matrix"/"result.json").read_text())
    assert result["state"] == "failed" and result["executed_cases"] == 0
