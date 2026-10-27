"""Independent executable owns graph fixtures, scheduling, VJP and persistence."""
import json
from pathlib import Path
import subprocess


def test_standalone_weighted_complete_training_checkpoint(dtype, tmp_path):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-weighted-training-check")
    result = subprocess.run([str(binary), "--device=cpu", "--dtype="+str(dtype).split(".")[-1],
                             "--output-dir", str(tmp_path/"native-weighted")],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == dict(state="passed", cases=24, updates_per_case=4, complete_native_checkpoint=True,
                          cpu_reference=True, placement="native")
