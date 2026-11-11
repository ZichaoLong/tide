"""Evidence must retain failed inner commands and independent check status."""
import json
import pytest
from collect_fc_suites import collect


def fixture(tmp_path):
    (tmp_path/"ok").mkdir()
    (tmp_path/"commands.json").write_text(json.dumps([
        dict(name="ok", exit_code=0), dict(name="failed", exit_code=7)]))
    record = dict(schema="tide-dense-query-cost-v1", phases=[dict(warmup=False,
        tokens=8, complete_wall_seconds=.5, input_positions_per_second=16.)],
        resolved=dict(device="cpu"), construction_seconds=.1, profile="none",
        execution="completed", strict_numerical="failed", discrete="not_checked")
    (tmp_path/"ok/result.json").write_text(json.dumps(record))
    return record


def test_cost_is_not_numerical_qualification_and_missing_result_stays_visible(tmp_path):
    fixture(tmp_path)
    result = collect([tmp_path])
    assert result['costs'][0]['strict_numerical'] == "failed"
    assert result['costs'][0]['discrete'] == "not_checked"
    assert result['costs'][1]['execution'] == "missing_result"
    assert result['costs'][1]['command_exit_code'] == 7


def test_inconsistent_rate_or_live_job_is_rejected(tmp_path):
    record = fixture(tmp_path)
    record['phases'][0]['input_positions_per_second'] = 17
    (tmp_path/"ok/result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="numerator"):
        collect([tmp_path])
    job = tmp_path/"job.json";job.write_text(json.dumps(dict(state="running")))
    with pytest.raises(ValueError, match="still live"):
        collect([], [job])
