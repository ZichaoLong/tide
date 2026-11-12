"""CPU admission lifecycle and public metadata adapter; no device qualification."""
from pathlib import Path
import subprocess
from types import SimpleNamespace


def test_planned_budget_refuses_before_arena_and_recording(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/device_online/resident_program_admission_check.cpp"
    binary = tmp_path / "admission-check"
    subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    assert "planned_workspace=true post_lowering=true exact_limit=true no_services=true" in result.stdout


def test_public_manifest_reads_current_resource_inventory():
    from tidegraph.weighted.resident_blocks import WeightedBlockResidentTraining
    current = dict(programs=0, worker_programs=0, streams=0, notifications=0, boundary_events=0)
    client = WeightedBlockResidentTraining.__new__(WeightedBlockResidentTraining)
    client._check = lambda: None
    client.owner = SimpleNamespace(resource_usage=lambda: current, optimizer_groups=lambda: [])
    client.device, client.initialization, client.requested_optimizer = "cuda:0", "fresh", "sgd"
    client.requested_limits, client.effective_limits, client.build_record = {}, {}, {}
    client.requested_groups = None
    old = client.manifest()["resource_usage"]
    current.update(programs=1, streams=1, boundary_events=2)
    assert old["programs"] == 0
    assert client.resource_usage() == client.manifest()["resource_usage"] == current
    assert current["worker_programs"] == current["notifications"] == 0
