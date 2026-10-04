"""Run an already leased, finite group with admission, cleanup and raw records.

The caller owns accelerator leases and the common measurement lock. This runner
owns only its child process groups; it never scans/signals unrelated processes.
"""
import ctypes
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from durable_records import write_json
from foundation_lifecycle import group_pids, reap_exited_group, rss, terminate_group
from measurement_lanes import admit, check_placement, numa_command, process_placement


def run_group(cases, *, output, reserve_bytes, interval=1.0, teardown_seconds=2.0):
    if not cases or not 0.1 <= interval <= 5:
        raise ValueError('cases and bounded monitoring interval required')
    if not 0 <= teardown_seconds <= 10:
        raise ValueError('finite natural teardown allowance between zero and ten seconds required')
    if any(not 0 < c['timeout_seconds'] <= 100000 for c in cases):
        raise ValueError('each case requires a finite positive timeout')
    admission = admit([c['lane'] for c in cases], reserve_bytes=reserve_bytes)
    numactl = shutil.which('numactl')
    if numactl is None:
        raise RuntimeError('numactl is required; no unbound fallback')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Adopt orphaned grandchildren for group-scoped cleanup on all exits.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'cannot enable child subreaper')
    record = dict(state='running', admission=admission, cases=[],
                  monitor_interval_seconds=interval, placement_interval_seconds=5,
                  natural_teardown_seconds=teardown_seconds,
                  peak_combined_rss_bytes=0, samples=[])
    children, streams = [], []
    started = time.monotonic()

    def interrupted(signum, frame):
        raise InterruptedError('measurement group received signal ' + str(signum))

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for case in cases:
            lane = case['lane']
            stream = (output / (lane['name'] + '.log')).open('x')
            streams.append(stream)
            argv = numa_command(case['command'], lane, executable=numactl)
            child = subprocess.Popen(argv, cwd=case['cwd'], env=case['environment'],
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            record['cases'].append(dict(name=lane['name'], pid=child.pid, command=argv,
                timeout_seconds=case['timeout_seconds'], state='running',
                started_elapsed_seconds=time.monotonic()-started,
                peak_rss_bytes=0, placement_samples=[], reaped_descendants=[]))
        write_json(output / 'result.json', record)
        last_placement = -5.
        while True:
            elapsed = time.monotonic() - started
            total, running = rss(os.getpid()), False
            for case, child, row in zip(cases, children, record['cases']):
                if child.returncode is None:
                    pid, status, used = os.wait4(child.pid, os.WNOHANG)
                    if pid:
                        child.returncode = os.waitstatus_to_exitcode(status)
                        row.update(exit_code=child.returncode,
                            state='finishing' if child.returncode == 0 else 'failed',
                            process_finished_elapsed_seconds=elapsed,
                            process_peak_rss_bytes=used.ru_maxrss*1024)
                        if row['process_peak_rss_bytes'] > case['lane']['memory_bytes']:
                            raise MemoryError('terminal RSS exceeded lane budget: ' + row['name'])
                if child.returncode is not None:
                    if child.returncode != 0:
                        raise RuntimeError('child failed: ' + row['name'])
                    # Exited adopted helpers are zombies until the subreaper
                    # collects them. Their presence alone is not a live leak.
                    reaped = reap_exited_group(child.pid)
                    row['reaped_descendants'].extend(reaped)
                    if any(item['exit_code'] != 0 for item in reaped):
                        raise RuntimeError('adopted descendant failed: ' + row['name'])
                    pending = group_pids(child.pid)
                    if not pending:
                        if row['state'] != 'passed':
                            row.update(state='passed', finished_elapsed_seconds=elapsed,
                                natural_teardown_elapsed_seconds=elapsed-row['process_finished_elapsed_seconds'])
                        continue
                    observed = []
                    for pid in pending:
                        try:
                            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                            observed.append(dict(pid=pid, state=fields[0]))
                        except FileNotFoundError:
                            pass
                    row['remaining_after_parent_exit'] = observed
                    if elapsed-row['process_finished_elapsed_seconds'] >= teardown_seconds:
                        raise RuntimeError('child exited with remaining live group processes: ' + row['name'])
                running = True
                pids = group_pids(child.pid)
                current = sum(rss(pid) for pid in pids)
                total += current
                row['peak_rss_bytes'] = max(row['peak_rss_bytes'], current)
                if current > case['lane']['memory_bytes']:
                    raise MemoryError('lane RSS budget exceeded: ' + row['name'])
                if elapsed-row['started_elapsed_seconds'] > case['timeout_seconds']:
                    raise TimeoutError('lane time bound exceeded: ' + row['name'])
                if elapsed-last_placement >= 5:
                    observed = []
                    for pid in pids:
                        # numactl can be observed just before it establishes the
                        # mask/execs. Inspect only after the initial second.
                        if elapsed-row['started_elapsed_seconds'] < 1:
                            continue
                        placement = process_placement(pid)
                        check_placement(placement, case['lane'])
                        if placement is not None:
                            observed.append(placement)
                    row['placement_samples'].append(dict(elapsed_seconds=elapsed, processes=observed))
            record['peak_combined_rss_bytes'] = max(record['peak_combined_rss_bytes'], total)
            if total + reserve_bytes > admission['host']['memory_budget_bytes']:
                raise MemoryError('combined RSS plus reserve exceeded admitted memory budget')
            if elapsed-last_placement >= 5 or not running:
                last_placement = elapsed
                record['samples'].append(dict(elapsed_seconds=elapsed, combined_rss_bytes=total))
                write_json(output / 'result.json', record)
            if not running:
                break
            time.sleep(interval)
        record['state'] = 'passed'
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for child, row in zip(children, record['cases']):
                terminate_group(child)
                row['remaining_group_pids'] = group_pids(child.pid)
                if row['state'] in ('running', 'finishing'):
                    row.update(state='cancelled', exit_code=child.returncode)
        except BaseException as error:
            record.update(state='failed', cleanup_error=repr(error))
            raise
        finally:
            for stream in streams:
                stream.close()
            record['wall_seconds'] = time.monotonic()-started
            write_json(output / 'result.json', record)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return record
