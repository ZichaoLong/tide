"""Allocation-free resident policy and unchanged historical execution identity."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from tidegraph.experiment.config import resolve
from tidegraph.planning.execution import ExecutionSpec


def configuration():
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=2, width=4, history=8, memory_overrides=dict(memory_read="qry"))),
        task=dict(samples=2, positions=3, input_pattern="staggered-v1"),
        execution=dict(client="python-native", device="npu:0", placement="resident"))


def test_default_block_identity_and_capacities_are_allocation_free(tmp_path):
    source = str(Path(__file__).resolve().parents[1]/"python")
    path = tmp_path/"request.json"
    path.write_text(json.dumps(configuration()))
    code = ("import sys; sys.path.insert(0,sys.argv[1]); from tidegraph.experiment.config import resolve; "
            "x=resolve(sys.argv[2]); assert 'torch' not in sys.modules; print(x['config_id'])")
    completed = subprocess.run([sys.executable, "-S", "-c", code, source, str(path)],
                               check=True, capture_output=True, text=True)
    record = resolve(path)
    assert completed.stdout.strip() == record["config_id"]
    assert record["execution"]["schema"] == "tide-weighted-execution-v7"
    assert ExecutionSpec.from_record(record["execution"]).record() == record["execution"]
    assert record["resident_dispatch"]["effective"] == "block-v2"
    assert record["resident_dispatch"]["requested_limits"] == {}
    assert record["resident_dispatch"]["effective_limits"] == record["resources"]["resident_block_limits"]
    assert "resident_limits" not in record["resources"]
    budget = record["capacity_budget"]["resident_layout"]
    assert budget["schema"] == "tide-resident-block-reservation-v2"
    assert "queue" not in budget["limits"] and "stages" not in budget["limits"]
    components = record["capacity_budget"]["per_device"]["npu:0"]["components"]
    assert "resident_queue_control" not in components
    assert components["pending_messages"] == components["live_record_estimate"] == 0
    assert components["fp32_masters"] > 0 and components["optimizer_slots"] > 0  # Inference archive retains them.
    assert resolve(json.loads(json.dumps(record))) == json.loads(json.dumps(record))


@pytest.mark.parametrize("execution,model,task,reason", [
    ({"family": "pdg"}, {}, {}, "SettleGraph"),
    ({"schedule": "streaming"}, {}, {}, "prefill"),
    ({"dtype": "float16"}, {}, {}, "FP32"),
    ({"peer_devices": ["npu:0", "npu:1"]}, {}, {}, "one device"),
    ({"client": "libtorch"}, {}, {}, "standalone block requires outputs/metrics"),
    ({}, {"family": "affine", "memory_overrides": {"memory_read": "state"}}, {}, "KV sum/query"),
    ({}, {}, {"plugin": {"module": "unimported_plugin"}}, "plugin"),
])
def test_auto_fallback_is_static_and_explicit_block_refuses(execution, model, task, reason):
    value = configuration()
    value["execution"].update(execution)
    value["model"]["recipe"].update(model)
    value["task"].update(task)
    record = resolve(value)
    assert record["resident_dispatch"]["effective"] == "general"
    assert reason in record["resident_dispatch"]["reason"]
    assert "resident_limits" in record["resources"] and "resident_block_limits" not in record["resources"]
    value["execution"]["resident_executor"] = "block-v2"
    with pytest.raises(ValueError, match=reason):
        resolve(value)


def test_old_limit_semantics_are_never_swallowed():
    value = configuration()
    value["resources"] = dict(resident_limits=dict(queue=81, stages=41, tensor_bytes=2**30))
    record = resolve(value)
    selected = record["resident_dispatch"]
    assert selected["effective"] == "general" and selected["requested_limits"] == value["resources"]["resident_limits"]
    assert selected["effective_limits"]["queue"] == 81 and selected["effective_limits"]["stages"] == 41
    value["resources"]["resident_block_limits"] = dict(cache=8)
    with pytest.raises(ValueError, match="different owners"):
        resolve(value)
    del value["resources"]["resident_limits"]
    value["resources"]["resident_block_limits"]["queue"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        resolve(value)
    del value["resources"]["resident_block_limits"]["queue"]
    value["execution"]["resident_executor"] = "general"
    with pytest.raises(ValueError, match="explicit general"):
        resolve(value)


def test_explicit_capacities_and_historical_records_keep_separate_identity():
    value = configuration()
    value["resources"] = dict(resident_block_limits=dict(cache=3, forward_bytes=1048576, total_bytes=1))
    current = resolve(value)
    assert current["resident_dispatch"]["requested_limits"] == value["resources"]["resident_block_limits"]
    assert current["capacity_budget"]["status"] == "static-refusal"
    old = ExecutionSpec(client="python-native", device="npu:0", placement="resident",
                        resident_executor="general", query_execution="legacy").record()
    assert "resident_executor" not in old["requested"]
    assert ExecutionSpec.from_record(old).resident_executor == "general"
    assert ExecutionSpec.from_record(old).record() == old
    modern = ExecutionSpec.from_record(old["requested"]).record()
    assert modern["schema"] == "tide-weighted-execution-v7" and modern["execution_id"] != old["execution_id"]


def test_frozen_resolved_record_and_selected_runtime_failure_do_not_dispatch_again(monkeypatch):
    from tidegraph.experiment.resident import Resident
    import tidegraph.experiment.block_resident as block
    current = resolve(configuration())
    def refused(*args, **kwargs):
        raise RuntimeError("selected block capacity refusal")
    monkeypatch.setattr(block, "BlockResident", refused)
    with pytest.raises(RuntimeError, match="selected block capacity"):
        Resident(current)
    frozen = deepcopy(current)
    frozen.pop("resident_dispatch")
    frozen["execution"] = ExecutionSpec(client="python-native", device="npu:0", placement="resident",
                                        resident_executor="general").record()
    # Old resolved records do not silently acquire a new dispatch policy.
    assert type(Resident.__new__(Resident, frozen)) is Resident


@pytest.mark.parametrize("read", ["sum", "qry"])
def test_standalone_selected_returns_receive_only_block_capacity_keys(read, tmp_path):
    from tidegraph.experiment.standalone import resident_arguments
    value = configuration()
    value["model"]["recipe"]["memory_overrides"]["memory_read"] = read
    value["execution"].update(client="libtorch", return_mode="outputs")
    value["measurement"] = dict(validation="none")
    resolved = resolve(value)
    assert resolved["resident_dispatch"]["effective"] == "block-v2"
    args = resident_arguments(resolved, tmp_path)
    assert args[:3] == ["--resident-executor", "auto", "--resident-block-limits"]
    assert json.loads(Path(args[3]).read_text()) == resolved["resources"]["resident_block_limits"]
    assert "--resident-limits" not in args
    value["execution"].update(return_mode="complete", resident_executor="general")
    resolved = resolve(value)
    args = resident_arguments(resolved, tmp_path)
    assert args[:3] == ["--resident-executor", "general", "--resident-limits"]
    frozen = deepcopy(resolved)
    frozen.pop("resident_dispatch")
    assert resident_arguments(frozen, tmp_path) == args
