"""CPU truth/restore coverage for the same explicit target qualification entry."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_device_cases import CASES
from weighted_device_checks import check_case
from weighted_device_precision import Precision
from tidegraph.placement import ExecutionPlacement


@pytest.mark.parametrize("payload,optimizer,case", [
    (torch.float32, "adamw", "kv-fc"),
    (torch.float16, "adamw", "kv-fc"),
    (torch.float16, "sgd", "gated-hb"),
])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_target_entry_full_records_roots_and_four_master_updates(payload, optimizer, case, implementation, tmp_path):
    result = check_case(next(c for c in CASES if c.name == case), torch.device("cpu"), implementation,
                        ExecutionPlacement(scoring_dtype="float32"), tmp_path,
                        os.environ.get("TIDE_BUILD_DIR", str(ROOT/"build")),
                        precision=Precision(payload, optimizer))
    assert result["training_updates"] == 4 and result["cpu_checkpoint_handoff"]


@pytest.mark.parametrize("extra,message", [
    (["--dtype", "float16"], "requires explicit --scoring-dtype"),
    (["--loss-scale", "16"], "requires explicit float16"),
    (["--loss-scale", "nan"], "positive and finite"),
])
@pytest.mark.parametrize("script", ["check_weighted_device.py", "profile_weighted_device.py"])
def test_precision_rejection_precedes_output_and_device_initialization(script, extra, message, tmp_path):
    output = tmp_path/"must-not-exist"
    proc = subprocess.run([sys.executable, str(ROOT/"scripts"/script),
                           "--device", "invalid", "--implementation", "python", "--placement", "mixed-c",
                           "--output-dir", str(output), *extra], capture_output=True, text=True)
    assert proc.returncode and message in proc.stderr
    assert not output.exists()
