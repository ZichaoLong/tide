"""Independent executable owns graph fixtures, scheduling, VJP and persistence."""
import json
from pathlib import Path
import subprocess
import pytest


def test_standalone_weighted_complete_training_checkpoint(dtype, tmp_path):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-weighted-training-check")
    result = subprocess.run([str(binary), "--device=cpu", "--dtype="+str(dtype).split(".")[-1],
                             "--output-dir", str(tmp_path/"native-weighted")],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == dict(state="passed", cases=24, updates_per_case=4, complete_native_checkpoint=True,
                          cpu_reference=True, placement="native", payload_dtype=str(dtype).split(".")[-1],
                          scoring_dtype="profile", optimizer="adamw", loss_scale=1, fp32_masters=False)


@pytest.mark.parametrize("optimizer,cases", [("sgd", []), ("adamw", ["fc/kv", "feedback/delta"])])
def test_standalone_half_master_training_and_complete_checkpoint(optimizer, cases, tmp_path):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-weighted-training-check")
    command = [str(binary), "--device=cpu", "--dtype=float16", "--scoring-dtype", "float32",
               "--optimizer", optimizer, "--loss-scale", "16", "--output-dir", str(tmp_path/"half")]
    for case in cases:
        command += ["--case", case]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["cases"] == (len(cases) or 24)
    assert report["payload_dtype"] == "float16" and report["scoring_dtype"] == "float32"
    assert report["optimizer"] == optimizer and report["loss_scale"] == 16 and report["fp32_masters"]


@pytest.mark.parametrize("arguments,message", [
    ([], "requires explicit scoring dtype"),
    (["--scoring-dtype", "float32", "--loss-scale", "nan"], "positive and finite"),
    (["--scoring-dtype", "float32", "--loss-scale", "16junk"], "invalid loss scale"),
])
def test_standalone_half_precision_rejects_before_output_or_device(arguments, message, tmp_path):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-weighted-training-check")
    output = tmp_path/"absent"
    result = subprocess.run([str(binary), "--device=invalid", "--dtype=float16", "--output-dir", str(output),
                             *arguments], capture_output=True, text=True)
    assert result.returncode and message in result.stderr
    assert not output.exists()
