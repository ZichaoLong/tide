"""Independent integer inventory and capacity checks; no device qualification."""
from pathlib import Path
import subprocess


def test_optimizer_storage_inventory_aliases_and_refusal(tmp_path):
    source = Path(__file__).resolve().parents[1]/"tools/device_online/optimizer_storage_check.cpp"
    binary = tmp_path/"optimizer-storage-check"
    subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    assert "old_layout=59625113 old_allowance=50331648" in result.stdout
