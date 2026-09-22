"""Bounded subprocess groups with durable logs and child-specific Linux rusage."""
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


class ProcessFailure(RuntimeError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def execute(command, cwd, env, logfile, timeout, memory_gib=0, on_line=None):
    start = time.monotonic(); child = None; usage = None; error = None
    pending = b''

    def limits():
        if memory_gib:
            resource.setrlimit(resource.RLIMIT_AS, (memory_gib*2**30,)*2)

    def reap():
        nonlocal usage
        if child.returncode is None:
            pid, status, measured = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status); usage = measured

    def drain(reader):
        nonlocal pending
        pending += reader.read()
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            if on_line:
                on_line(line.decode('utf-8', errors='replace'))

    try:
        with Path(logfile).open('wb') as output, Path(logfile).open('rb') as reader:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=limits)
            heartbeat = start
            while child.returncode is None:
                drain(reader); reap(); now = time.monotonic()
                if child.returncode is not None:
                    break
                if now-start >= timeout:
                    raise TimeoutError('subprocess exceeded '+str(timeout)+' seconds')
                if now-heartbeat >= 30:
                    print(f'RUNNING {Path(logfile).name}: {now-start:.0f}s; log={logfile}', flush=True)
                    heartbeat = now
                time.sleep(.1)
            drain(reader)
    except BaseException as exception:
        error = type(exception).__name__+': '+str(exception)
    finally:
        if child is not None and child.returncode is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic()+10
                while child.returncode is None and time.monotonic() < deadline:
                    reap(); time.sleep(.1)
                if child.returncode is not None:
                    break
    result = dict(argv=list(command), cwd=str(cwd), exit_code=None if child is None else child.returncode,
                  elapsed_seconds=time.monotonic()-start, log=Path(logfile).name,
                  peak_rss_bytes=None if usage is None else usage.ru_maxrss*1024,
                  user_seconds=None if usage is None else usage.ru_utime,
                  system_seconds=None if usage is None else usage.ru_stime,
                  unreaped_child_pid=child.pid if child is not None and child.returncode is None else None)
    if error or result['exit_code'] != 0:
        result['error'] = error or 'native exit '+str(result['exit_code'])
        raise ProcessFailure(result['error'], result)
    return result
