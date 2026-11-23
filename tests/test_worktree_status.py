"""Local foundation job receipts retain their recorded scope."""
import json

from scripts import status_runtime as runtime


def test_empty_checkout_stays_unavailable(tmp_path):
    assert runtime.observe(tmp_path)["kind"] == "unavailable"
    assert not list(tmp_path.iterdir())


def test_preserves_job_fields_and_reports_unreadable_receipts(tmp_path):
    directory = tmp_path / "artifacts/instances"
    bad = directory / "a/status.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{")
    assert runtime.observe(tmp_path)["kind"] == "unavailable"

    good = directory / "b/status.json"
    good.parent.mkdir()
    value = {"state": "running", "source": "abc", "elapsed": 3, "exit_code": None}
    good.write_text(json.dumps(value))
    view = runtime.observe(tmp_path)
    assert view["kind"] == "snapshot_only"
    unreadable, receipt = view["records"]
    assert unreadable["record"] == str(bad.relative_to(tmp_path)) and unreadable["error"]
    assert receipt == {"record": str(good.relative_to(tmp_path)), **value}
