"""Small authoritative records, independent of any dashboard SDK."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class Metrics:
    def __init__(self, directory, run_id):
        self.path = Path(directory) / 'metrics.jsonl'
        self.stream = self.path.open('x')
        self.run_id, self.sequence, self.start = run_id, 0, time.monotonic()

    def log(self, step, values):
        values = {k: float(v) for k, v in values.items()}
        if not values or any(not math.isfinite(v) for v in values.values()):
            raise ValueError('Metrics must be nonempty finite scalar observations')
        record = dict(schema_version=1, run_id=self.run_id, sequence=self.sequence,
                      timestamp=now(), step=step, elapsed_seconds=time.monotonic()-self.start,
                      metrics=values)
        self.stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.stream.flush()
        self.sequence += 1

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
