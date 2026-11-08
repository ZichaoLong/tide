#!/usr/bin/env python3
"""Compute-only cap inside a queued service, also bounded by a common deadline."""
import argparse
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.npu_learning_stage import bounded_command, CleanupFailure
from tide_lab.records import atomic_json, now
from tide_lab.supervisor import group_processes


def remaining_seconds(max_seconds, deadline_epoch, started):
    return min(max_seconds-(time.monotonic()-started), deadline_epoch-time.time())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not 20 < a.max_seconds <= 28800:
        raise ValueError('outside finite authorized compute cap (maximum eight hours)')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    a.output.mkdir(parents=True, exist_ok=False)
    record = dict(state='starting', command=command, started_at=now(), max_seconds=a.max_seconds,
                  deadline_epoch=a.deadline_epoch)
    def publish():
        atomic_json(a.output/'result.json', record)
    def interrupted(signum, _):
        raise InterruptedError(f'workload wrapper received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    started = time.monotonic()
    try:
        wall = remaining_seconds(a.max_seconds, a.deadline_epoch, started)
        if wall <= 15:
            raise TimeoutError('common deadline leaves no compute/cleanup window')
        bounded_command(command, a.output/'workload.log', wall-10, record, publish)
    except BaseException as error:
        record.update(state='failed', error=repr(error)); publish()
        if isinstance(error, CleanupFailure):
            while group_processes(record.get('process_group')):
                time.sleep(.2)  # keep queue locks until the service kills all descendants
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
