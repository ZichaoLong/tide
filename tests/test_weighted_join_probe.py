"""Private join diagnostics have an independent check and a real resume test."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))


def test_join_literal_cancellation_broadcast_custom_none_and_restoration():
    result = subprocess.run([sys.executable, str(ROOT/"scripts/diagnose_weighted_joins.py"), "--self-check"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["state"] == "passed" and report["counts"]["joined_slots"] > 0


def test_join_small_graph_two_steps_checkpoint_and_no_overwrite(tmp_path):
    from weighted_topology import topology
    from weighted_workload_plan import make_plan
    from durable_records import write_json
    plan, _ = make_plan(topology("fc", depth=2, region_width=2), width=4, samples=1, positions=2)
    write_json(tmp_path/"plan.json", plan)
    out = tmp_path/"check"
    command = [sys.executable, str(ROOT/"scripts/diagnose_weighted_joins.py"), "--workload", str(tmp_path/"plan.json"),
               "--dtype", "float32", "--implementation", "python", "--checkpoint-check", "--output-dir", str(out),
               "--arithmetic", "linear_reduce_norm64"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads((out/"result.json").read_text())
    assert report["state"] == "passed" and report["steps"][0]["checkpoint_exact_decode"]
    assert report["steps"][1]["exact_resume"] == [dict(records=None, gradients=None, parameters=None, optimizer=None)]*2
    witness = (out/"join-probe.json").read_bytes()
    assert json.loads(witness)["counts"]["joined_slots"] > 0
    retry = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert retry.returncode != 0 and (out/"join-probe.json").read_bytes() == witness
