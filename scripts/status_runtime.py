"""Read the foundation runner's local job receipts without importing Tide."""
import json


REPOSITORY = "graph-execution-foundation"
JOBS = "artifacts/instances/*/status.json"


def observe(root):
    records = []
    sources = []
    valid = 0
    for path in sorted(root.glob(JOBS)):
        name = str(path.relative_to(root))
        sources.append(name)
        row = {"record": name}
        try:
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict) or not isinstance(value.get("state"), str) or not value["state"]:
                raise ValueError("job record needs a state")
            fields = {key: value.get(key) for key in ("state", "source", "elapsed", "exit_code")}
            json.dumps(fields, allow_nan=False)
        except (OSError, ValueError) as error:
            row["error"] = str(error)
        else:
            row.update(fields)
            valid += 1
        records.append(row)
    return {
        "kind": "snapshot_only" if valid else "unavailable",
        "summary": f"本机实例作业记录 {valid} 份可读，{len(records) - valid} 份不可读",
        "sources": sources or [JOBS],
        "next_check": "python scripts/check_upstream.py",
        "records": records,
    }
