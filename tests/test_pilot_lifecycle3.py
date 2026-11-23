import json

import pytest

from pilotlab.lifecycle3 import cancelled_before_allocation


def setup_queue(tmp_path):
    queue = {"status": "cancelled", "job_id": "job-1"}
    events = [{"event": "ascend_queue", "job_id": "job-1", "status": status,
               "timestamp": f"2026-11-20T01:00:0{index}+00:00"}
              for index, status in enumerate(("queued", "cancelled"))]
    (tmp_path / "stdout.log").write_text("\n".join(json.dumps(e) for e in events))
    return queue, events


def test_cancelled_waiter_has_no_required_worker_result(tmp_path):
    queue, _ = setup_queue(tmp_path)
    assert cancelled_before_allocation(tmp_path, queue)


@pytest.mark.parametrize("damage", ["acquired", "wrong_job", "missing_log", "heartbeat", "result", "physical"])
def test_missing_result_is_not_enough_to_infer_no_execution(tmp_path, damage):
    queue, events = setup_queue(tmp_path)
    if damage == "acquired":
        events[0]["status"] = "running"
    elif damage == "wrong_job":
        events[0]["job_id"] = "different-job"
    elif damage == "physical":
        queue["physical_ids"] = [2]
    elif damage in ("heartbeat", "result"):
        (tmp_path / f"{damage}.json").write_text("{}")
    (tmp_path / "stdout.log").write_text("\n".join(json.dumps(e) for e in events))
    if damage == "missing_log":
        (tmp_path / "stdout.log").unlink()
    assert not cancelled_before_allocation(tmp_path, queue)
