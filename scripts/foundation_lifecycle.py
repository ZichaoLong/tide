"""Linux child-group lifetime and aggregate RSS monitoring for bounded runs."""
import ctypes
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def rss(pid):
    try:
        return int(next(s.split()[1] for s in Path(f'/proc/{pid}/status').read_text().splitlines()
                        if s.startswith('VmRSS:')))*1024
    except (OSError, StopIteration):
        return 0


def descendants(pid):
    pending=[pid];seen=set()
    while pending:
        current=pending.pop()
        if current in seen:continue
        seen.add(current)
        try:
            for task in Path(f'/proc/{current}/task').iterdir():
                try:pending.extend(map(int,(task/'children').read_text().split()))
                except OSError:pass
        except OSError:pass
    return seen-{pid}


def group_pids(group):
    # Subreaper owns both live descendants and orphans: no full-host /proc scan.
    result=[]
    for pid in descendants(os.getpid()):
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group:result.append(pid)
        except (OSError,ValueError,IndexError):pass
    return result


def reap_exited_group(group):
    """Reap adopted group descendants after the direct child was collected.

    WNOHANG never waits for a live helper. The caller owns this process group;
    this must not be used to collect an unrelated or still-awaited direct child.
    """
    reaped = []
    while True:
        try:
            pid, status = os.waitpid(-group, os.WNOHANG)
        except ChildProcessError:
            break
        if not pid:
            break
        reaped.append(dict(pid=pid, exit_code=os.waitstatus_to_exitcode(status)))
    return reaped


def storage_bytes(directory):
    total = 0
    for parent, _, files in os.walk(directory, followlinks=False):
        for name in files:
            path = Path(parent)/name
            try:
                if not path.is_symlink(): total += path.stat().st_size
            except FileNotFoundError:
                pass  # An active profiler can rotate files.
    return total


def terminate_group(child):
    """Reap adopted grandchildren as well as the direct child, including zombies."""
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 20)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic()+grace
        while True:
            if child.returncode is None:
                child.poll()
            if child.returncode is not None:
                try:
                    while os.waitpid(-child.pid, os.WNOHANG)[0]:
                        pass
                except ChildProcessError:
                    pass
            if not group_pids(child.pid):
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
    raise RuntimeError(f'process group {child.pid} still exists after SIGKILL/reap')


def run_child(command, *, cwd, env, log, affinity, timeout, memory_budget, audit,
              storage_path=None, storage_budget=None):
    if storage_path is not None and (storage_budget is None or storage_budget <= 0):
        raise ValueError('positive storage budget required')
    taskset = shutil.which('taskset')
    if taskset is None:
        raise RuntimeError('Linux taskset is required for bounded CPU affinity')
    # Adopt orphaned worker descendants so their lifetime can be proved terminal.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'cannot enable child subreaper')
    argv = [taskset, '--cpu-list', ','.join(map(str, affinity)), *command]
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    start = time.monotonic()
    audit.update(pid=child.pid, launch_argv=argv, peak_combined_rss_bytes=0, samples=[])
    last_storage_check = -5.
    try:
        while True:
            pid, status, used = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                audit['process_peak_rss_bytes'] = used.ru_maxrss*1024
                if storage_path is not None:
                    stored = storage_bytes(storage_path)
                    audit['peak_observed_storage_bytes'] = max(audit.get('peak_observed_storage_bytes', 0), stored)
                    if stored > storage_budget:
                        raise OSError('profile output exceeded storage budget')
                break
            current = sum(rss(pid) for pid in descendants(os.getpid()))+rss(os.getpid())
            audit['peak_combined_rss_bytes'] = max(audit['peak_combined_rss_bytes'], current)
            elapsed = time.monotonic()-start
            if storage_path is not None and elapsed-last_storage_check >= 1:
                last_storage_check = elapsed
                stored = storage_bytes(storage_path)
                audit['peak_observed_storage_bytes'] = max(audit.get('peak_observed_storage_bytes', 0), stored)
                if stored > storage_budget:
                    raise OSError('profile output exceeded storage budget')
            if not audit['samples'] or elapsed-audit['samples'][-1]['elapsed_seconds'] >= 1:
                audit['samples'].append(dict(elapsed_seconds=elapsed, combined_rss_bytes=current))
            if current > memory_budget:
                raise MemoryError('worker group plus coordinator RSS exceeded aggregate budget')
            if elapsed > timeout:
                raise TimeoutError('bounded workload timeout')
            time.sleep(.1)
    finally:
        handlers={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            terminate_group(child)
        finally:
            audit.update(worker_exit_code=child.returncode, remaining_group_pids=group_pids(child.pid),
                         process_seconds=time.monotonic()-start)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return child.returncode
