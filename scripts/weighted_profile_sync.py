"""Offline profile compatibility entry; public installed implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from tidegraph.experiment.profile_sync import summarize_cuda_sync


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    document = json.loads(raw)
    events = document if isinstance(document, list) else document["traceEvents"]
    result = summarize_cuda_sync(events)
    result["trace_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))
