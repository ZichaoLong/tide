"""Selected scale identities and client-specific resource refusals; no GPU work."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tidegraph.planning.inference_budget import GIB, fc_inference_budget
from tidegraph.planning.study import case, command, expand

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT/"benchmarks/core5-cuda-inference-v1.json"


def catalog():
    return json.loads(CATALOG.read_text())


def test_fp64_cells_keep_client_and_model_identity_without_inventing_resident_support():
    study = catalog()
    assert "fc-1b-b8-kv" in expand(study) and "fc-boundary-b1-kv" in expand(study)
    first = case(study, "fc-1b-b1-kv", "native-cuda")
    assert first["plan"]["sha256"] == "841ad97772f531310e96f9effe8adcaa3b788726dbebdc221c624e637e35385e"
    for execution in ("python-cuda-fp64", "native-cuda-fp64", "cpp-cuda-fp64", "native-cpu-fp64"):
        row = case(study, "fc-1b-b8-kv", execution)
        assert row["plan"]["model_id"] == first["plan"]["model_id"]
        assert row["plan"]["sha256"] != first["plan"]["sha256"]
        assert row["inventory"] == first["inventory"]
        e = row["execution"]["requested"]
        assert e["dtype"] == e["scoring_dtype"] == "float64"
        assert e["placement"] != "resident" and not e["peer_devices"]
        args = command(row, python="PY", root=ROOT, output="OUT", workload="PLAN",
                       native_library="CORE", standalone_binary="CPP", fixture="FIXTURE")
        assert args[args.index("--dtype")+1] == args[args.index("--scoring-dtype")+1] == "float64"


@pytest.mark.parametrize("name,execution,trainable,bank", [
    ("fc-1b-b1-kv", "native-cuda-fp64", 1010339302, 8620240912),
    ("fc-boundary-fp64-b1-kv", "native-cuda-fp64", 2270465510, 19372666896),
    ("fc-boundary-b1-kv", "native-cuda", 4033908198, 17210030088)])
def test_static_parameter_banks_keep_frozen_scales_separate(name, execution, trainable, bank):
    budget = fc_inference_budget(case(catalog(), name, execution))
    assert budget["counts"]["trainable_parameters"] == trainable
    assert budget["counts"]["frozen_parameters"] == 12
    assert budget["counts"]["unique_parameters"] == trainable+12
    assert budget["static_bytes"]["one_model_bank"] == bank
    assert not budget["automatic_admission"]
    assert budget["static_bytes"]["inference_optimizer_grad_master"] == 0


def test_standalone_near_boundary_resume_is_refused_by_two_bank_budget():
    study = catalog()
    native = fc_inference_budget(case(study, "fc-boundary-fp64-b1-kv", "native-cuda-fp64"))
    cpp = fc_inference_budget(case(study, "fc-boundary-fp64-b1-kv", "cpp-cuda-fp64"))
    assert native["checkpoint"]["device_model_banks_during_restore"] == 1
    assert cpp["checkpoint"]["device_model_banks_during_restore"] == 2
    assert native["resource_reservation"]["cuda_after_restore_models_and_return"] > 0
    assert cpp["resource_reservation"]["cuda_after_restore_models_and_return"] < 0
    assert native["checkpoint"]["weight_tensor_bytes"] > cpp["checkpoint"]["weight_tensor_bytes"]
    assert native["checkpoint"]["suggested_max_file_bytes"] == 20*GIB
    assert not cpp["checkpoint"]["native_tensor_limit_exceeded"]


def test_batch_budget_counts_real_input_activity_but_does_not_predict_routing():
    one = fc_inference_budget(case(catalog(), "fc-1b-b1-kv", "native-cuda-fp64"))
    eight = fc_inference_budget(case(catalog(), "fc-1b-b8-kv", "native-cuda-fp64"))
    assert one["static_bytes"] == eight["static_bytes"]
    assert (one["counts"]["input_atoms"], eight["counts"]["input_atoms"]) == (6, 39)
    assert (one["counts"]["input_roots"], eight["counts"]["input_roots"]) == (4, 25)
    assert eight["counts"]["candidate_upper_per_window"] == 128
    assert eight["return_tensor_reservation"]["cpu_all_results"] == 8*one["return_tensor_reservation"]["cpu_all_results"]
    assert "actual_full_calls" not in eight["counts"]
    cpu = fc_inference_budget(case(catalog(), "fc-1b-b1-kv", "native-cpu-fp64"))
    assert cpu["checkpoint"]["device_model_bytes_during_restore"] == 0
    assert cpu["resource_reservation"]["cuda_after_restore_models_and_return"] is None


def test_budget_requires_its_declared_scope_and_imports_without_torch():
    wrong = case(catalog(), "hb-small-b8-kv", "native-cuda")
    with pytest.raises(ValueError, match="four-node"):
        fc_inference_budget(wrong)
    wrong = deepcopy(case(catalog(), "fc-1b-b1-kv", "native-cuda-fp64"))
    wrong["measurement"]["mode"] = "training"
    with pytest.raises(ValueError, match="inference"):
        fc_inference_budget(wrong)
    script = ("import sys; sys.path.insert(0, 'python'); "
              "from tidegraph.planning.inference_budget import fc_inference_budget; "
              "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
