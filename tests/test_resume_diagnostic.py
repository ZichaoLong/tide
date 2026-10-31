"""The diagnostic compares full states and next records without a reference run."""
import importlib.util
from pathlib import Path
import json


def test_diagnostic_replay_uses_new_owner_and_preserves_public_checkpoint(tmp_path):
    path = Path(__file__).resolve().parents[1]/"scripts/diagnose_experiment_resume.py"
    spec = importlib.util.spec_from_file_location("resume_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=1, width=4)), execution=dict(device="cpu"), measurement=dict(steps=1, warmup=0),
        resources=dict(logical_windows=2))
    result = module.diagnose(config, tmp_path/"first", windows=1)
    assert result["raw_boundary_exact"] and result["next_window_exact"]
    assert result["original_next_work"]["candidate_updates"] == result["restored_next_work"]["candidate_updates"]
    assert result["original_layouts"]["sampled_states"]
    repeated = module.diagnose(config, tmp_path/"repeat", checkpoint=tmp_path/"first", windows=1)
    assert repeated["state"] == "passed" and repeated["independent_reference"] == "not_run"
    assert json.loads((tmp_path/"repeat/result.json").read_text())["performance_qualification"] is False
