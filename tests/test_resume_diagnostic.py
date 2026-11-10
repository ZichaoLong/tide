"""The diagnostic compares full states and next records without a reference run."""
import importlib.util
from pathlib import Path
import json
import pytest


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


@pytest.mark.parametrize("corrupt", [False, True])
def test_selected_diagnostic_keeps_execution_and_detects_unreturned_raw_state(tmp_path, monkeypatch, corrupt):
    from dataclasses import replace
    from tidegraph.experiment.eager import Eager
    path = Path(__file__).resolve().parents[1]/"scripts/diagnose_experiment_resume.py"
    spec = importlib.util.spec_from_file_location("selected_resume_diagnostic", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_init, original_step = Eager.__init__, Eager.step
    def init(self, *args, **kw):
        original_init(self, *args, **kw)
        self.diagnostic_restored = kw.get("checkpoint") is not None
    def step(self, *args, **kw):
        assert not self.trace_records
        assert kw.get("return_mode") is None  # No hidden switch to the traced algorithm.
        result = original_step(self, *args, **kw)
        if corrupt and self.diagnostic_restored:
            key = next(iter(self.state.states)); state = self.state.states[key]
            self.state.states[key] = replace(state, value=state.value+1)
        return result
    monkeypatch.setattr(Eager, "__init__", init); monkeypatch.setattr(Eager, "step", step)
    config = dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=3,
        region_width=1, width=4)), execution=dict(device="cpu", return_mode="outputs"),
        measurement=dict(steps=1, warmup=0, validation="none"), resources=dict(logical_windows=2))
    result = module.diagnose(config, tmp_path/"selected", windows=1)
    assert result["raw_boundary_exact"] and result["return_mode"] == "outputs"
    assert result["state"] == ("strict-failure" if corrupt else "passed")
