"""Public native codec -> independent standalone execution and strict damage refusal."""
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_topology import topology
from weighted_workload_plan import make_plan, canonical
from durable_records import write_json

ROOT = Path(__file__).resolve().parents[1]


def binary():
    path = os.environ.get("TIDE_WEIGHTED_BENCHMARK")
    if not path: pytest.skip("standalone benchmark qualification provides TIDE_WEIGHTED_BENCHMARK")
    assert Path(path).is_file()
    return path


def command(*args):
    return subprocess.run(list(map(str,args)), capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("family,selection", [("kv","regional_total"), ("affine","regional_total_zero"),
    ("delta","hard_selection"), ("gated_delta","regional_total"), ("ssd","regional_total")])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_five_families_binary_material_and_complete_training(tmp_path, family, selection, precision):
    executable = binary()
    plan, _ = make_plan(topology("fc", depth=2, region_width=2), family=family, selection=selection,
                         width=4, samples=1, positions=2)
    plan["alpha_agg"] = .31
    plan["sha256"] = hashlib.sha256(canonical({k:v for k,v in plan.items() if k != "sha256"})).hexdigest()
    write_json(tmp_path/"plan.json", plan)
    made = command(sys.executable, ROOT/"scripts/export_weighted_benchmark.py", "--native-library", os.environ["TIDE_BUILD_DIR"],
                   "--workload", tmp_path/"plan.json", "--binary", "--output", tmp_path/"fixture.json")
    assert made.returncode == 0, made.stderr
    assert json.loads((tmp_path/"fixture.json").read_text())["alpha_agg"] == .31
    result = command(executable,"--fixture",tmp_path/"fixture.json","--device","cpu","--dtype",precision,"--steps","1", "--output-dir",tmp_path/"run")
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path/"run/result.json").read_text())
    assert record["state"] == "passed" and record["optimizer_values_checked"] and record["post_measurement_complete_records"]
    assert record["dtype"] == precision and record["final_parameter_gradients_checked"]
    assert record["phase"] == "complete"
    if family != "kv" or precision != "float32": return
    data = json.loads((tmp_path/"fixture.json").read_text())
    # Distinct corruptions must fail with stable reasons before any timed step.
    for label, edit, expected in (
        ("hash", lambda x: x["binary"].update(sha256="0"*64), "SHA256"),
        ("range", lambda x: next(t for t in x["tensors"] if t["encoding"] == "binary").update(offset=4), "range/shape"),
        ("alias", lambda x: x.update(parameter_aliases=[]), "alias partitions"),
        ("budget", lambda x: x.update(max_materialize_bytes=1), "ceiling")):
        copy = json.loads(json.dumps(data)); edit(copy); write_json(tmp_path/(label+".json"),copy)
        failed = command(executable,"--fixture",tmp_path/(label+".json"),"--device","cpu","--steps","1", "--output-dir",tmp_path/label)
        assert failed.returncode and expected in failed.stderr, failed.stderr
        assert json.loads((tmp_path/label/"result.json").read_text())["phase"] == "construction"
    # Original FP32 material fits this ceiling; explicit FP64 storage does not.
    actual_bytes = sum(4*math.prod(t["shape"]) for t in data["tensors"])
    data["max_materialize_bytes"] = 2*actual_bytes-1
    write_json(tmp_path/"double-budget.json", data)
    refused = command(executable,"--fixture",tmp_path/"double-budget.json","--device","cpu","--dtype","float64",
                      "--steps","1","--output-dir",tmp_path/"double-budget")
    assert refused.returncode and "ceiling" in refused.stderr


def test_standalone_persists_phase_before_blocking_fixture_read(tmp_path):
    # A FIFO blocks fixture preparation without a large model or timing race.
    # Termination leaves a running partial result; the outer job owns exit status.
    executable = binary()
    fixture = tmp_path/"blocked.json"
    os.mkfifo(fixture)
    out = tmp_path/"run"
    with (tmp_path/"process.log").open("w") as log:
        process = subprocess.Popen([executable, "--fixture", str(fixture), "--device", "cpu",
                                    "--output-dir", str(out)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+30
            while not (out/"result.json").exists():
                assert process.poll() is None, "benchmark exited before persisting progress"
                assert time.monotonic() < deadline, "benchmark did not persist progress"
                time.sleep(.02)
            record = json.loads((out/"result.json").read_text())
            assert record["state"] == "running" and record["phase"] == "fixture-preparation"
            assert record["metrics"] == []
            assert process.poll() is None
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert json.loads((out/"result.json").read_text()) == record
