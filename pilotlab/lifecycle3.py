"""Read-only witnesses for queue cancellation before a worker was launched."""
from datetime import datetime
import json
from pathlib import Path


def cancelled_before_allocation(directory, queue):
    """Require positive queue evidence; missing results alone prove nothing."""
    directory = Path(directory)
    if (queue.get("status") != "cancelled" or not queue.get("job_id")
            or queue.get("physical_ids") or queue.get("physical_id") is not None):
        return False
    if any((directory / name).exists() for name in
           ("result.json", "heartbeat.json", "checkpoint.json", "failure.pt")):
        return False
    if any((directory / "checkpoints").glob("*.pt")):
        return False
    events = []
    try:
        for line in (directory / "stdout.log").read_text().splitlines():
            if '"ascend_queue"' not in line:
                continue
            event = json.loads(line)
            if event.get("event") != "ascend_queue":
                continue
            if event.get("job_id") != queue["job_id"]:
                return False
            stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            if (stamp.tzinfo is None or event.get("status") not in {"queued", "cancelled"}
                    or event.get("physical_ids") or event.get("physical_id") is not None):
                return False
            events.append((event["status"], stamp.timestamp()))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return (len(events) >= 2 and events[-1][0] == "cancelled"
            and all(status == "queued" for status, _ in events[:-1])
            and all(a[1] <= b[1] for a, b in zip(events, events[1:])))


def empty_dispatch_ticket_closed(entry, group, queue):
    """Verify a completed, withdrawn cooperative ticket that never held a device."""
    return (entry.get("dispatch_lease") is True and entry.get("task_ids") == []
            and group.get("workers") == {} and group.get("admission", {}).get("close_reason") == "controller_close"
            and group.get("admission", {}).get("closed") is True and group.get("status") == "completed"
            and group.get("cancelled_before_allocation") is True and group.get("queue_released_unix")
            and group.get("physical_id") is None and group.get("allocation_acquired_unix") is None
            and group.get("allocation_released_unix") is None and group.get("physical_device_seconds") == 0
            and queue.get("physical_ids") == [] and queue.get("physical_id") is None
            and queue.get("job_id") == entry["job_id"] and queue.get("status") == "completed" and queue.get("return_code") == 0)
