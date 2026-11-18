"""Run a frozen workload with a sampled process-tree RSS bound and durable status."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import time


def atomic(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".status.")
    try:
        with os.fdopen(fd,"w") as stream:
            json.dump(payload,stream,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _process_table():
    processes={}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                fields=Path(entry.path,"stat").read_text().rsplit(")",1)[1].split()
                processes[int(entry.name)]=(int(fields[1]),int(fields[2]),int(fields[3]),
                                            int(fields[19]),int(fields[21]),fields[0])
            except (OSError,ValueError,IndexError):
                continue  # A process may exit while its /proc entry is read.
    return processes


def enable_subreaper():
    """Adopt this dedicated supervisor's orphan descendants, including setsid.

    Invoke only in a runner process that owns its workload children. Adoption
    closes the interval in which a short-lived parent can exit between samples.
    It does not alter another process, session, cgroup or site's device queue.
    """
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0) != 0:  # Linux PR_SET_CHILD_SUBREAPER.
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error))


class ProcessTree:
    """Retain PID/start identities after a descendant reparents or calls setsid."""
    def __init__(self,pid,*,supervisor_pid=None):
        self.pid,self.supervisor_pid=pid,supervisor_pid
        self.identities={}
        initial=_process_table().get(pid)
        self.root_start=None if initial is None else initial[3]
        self.refresh()

    def refresh(self):
        processes=_process_table()
        owned={pid for pid,start in self.identities.items()
               if pid in processes and processes[pid][3]==start}
        root=processes.get(self.pid)
        if root is not None and (self.root_start is None or root[3]==self.root_start):
            self.root_start=root[3]
            owned.add(self.pid)
        # The runner created a fresh session. Orphan members of that original
        # session are owned, provided the root PID has not been reused.
        if root is None or root[3]==self.root_start:
            owned.update(pid for pid,p in processes.items() if p[1]==self.pid and p[2]==self.pid)
        if self.supervisor_pid is not None:
            # Only dedicated subreaper runners use this seed. Other direct
            # children cannot exist in their single-workload lifecycle.
            owned.update(pid for pid,p in processes.items() if p[0]==self.supervisor_pid)
        while True:
            children={pid for pid,p in processes.items() if p[0] in owned}
            if children<=owned:
                break
            owned.update(children)
        self.identities.update((pid,processes[pid][3]) for pid in owned)
        return {pid:processes[pid] for pid in owned if processes[pid][5]!="Z"}

    def rss(self):
        live=self.refresh()
        return sum(max(0,p[4]) for p in live.values())*os.sysconf("SC_PAGE_SIZE"),len(live)

    def record(self):
        return [dict(pid=pid,start_ticks=start) for pid,start in sorted(self.identities.items())]


def tree_rss(pid,tracker=None):
    """Sample RSS sum; shared pages count per process, not a cgroup hard limit."""
    return (ProcessTree(pid) if tracker is None else tracker).rss()


def _signal_identity(pid,start,signum):
    """Check birth identity and use a pidfd when supported to avoid PID reuse."""
    descriptor=None
    try:
        if hasattr(os,"pidfd_open") and hasattr(signal,"pidfd_send_signal"):
            try:
                descriptor=os.pidfd_open(pid)
            except OSError as error:
                if error.errno not in (22,38):  # Older kernel: checked PID fallback.
                    raise
        fields=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        if int(fields[19])!=start:
            return False
        if descriptor is None:
            os.kill(pid,signum)
        else:
            signal.pidfd_send_signal(descriptor,signum)
        return True
    except (FileNotFoundError,ProcessLookupError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stop_group(child,grace,tracker=None):
    """Stop only observed descendant identities, including nested sessions."""
    tracker=ProcessTree(child.pid) if tracker is None else tracker
    sent=set()
    def send(signum):
        live=tracker.refresh()
        # Descendants usually have newer PIDs; identity tracking remains valid
        # if their parents exit before the remaining signals are delivered.
        for pid in sorted(live,reverse=True):
            identity=pid,live[pid][3]
            if signum==signal.SIGKILL or identity not in sent:
                _signal_identity(*identity,signum)
                sent.add(identity)
        return live
    send(signal.SIGTERM)
    deadline=time.monotonic()+grace
    while tracker.refresh() and time.monotonic()<deadline:
        child.poll()
        send(signal.SIGTERM)  # Include children discovered during shutdown.
        time.sleep(min(.05,max(0,deadline-time.monotonic())))
    escalated=bool(tracker.refresh())
    if escalated:
        send(signal.SIGKILL)
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return escalated,False
    deadline=time.monotonic()+1
    while tracker.refresh() and time.monotonic()<deadline:
        send(signal.SIGKILL)
        time.sleep(.02)
    return escalated,not bool(tracker.refresh())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--status",required=True)
    parser.add_argument("--source",required=True)
    parser.add_argument("--max-rss-gib",type=float,default=64)
    parser.add_argument("--poll-interval",type=float,default=.25)
    parser.add_argument("--terminate-grace",type=float,default=5)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command:
        parser.error("a workload command is required")
    for name in ("max_rss_gib","poll_interval","terminate_grace"):
        value=getattr(args,name)
        if not math.isfinite(value) or value<0 or (value==0 and name!="terminate_grace"):
            parser.error(f"{name.replace('_','-')} must be finite and positive (grace may be zero)")
    start=time.monotonic()
    record=dict(state="running",source=args.source,command=command,pid=os.getpid(),
                started=datetime.now(timezone.utc).isoformat(),max_rss_bytes=int(args.max_rss_gib*2**30),
                peak_rss_bytes=0,max_processes=0,rss_sample_count=0,poll_interval=args.poll_interval)
    atomic(args.status,record)
    stopped=None
    def terminated(signum,frame):
        nonlocal stopped
        stopped=signum
    signal.signal(signal.SIGTERM,terminated)
    signal.signal(signal.SIGINT,terminated)
    child=tracker=None
    result=1
    try:
        enable_subreaper()
        record["descendant_supervision"]="Linux child subreaper; retained PID/start identities across nested sessions"
        child=subprocess.Popen(command,start_new_session=True)
        tracker=ProcessTree(child.pid,supervisor_pid=os.getpid())
        record.update(child_pid=child.pid,process_group=child.pid)
        atomic(args.status,record)
        published=time.monotonic()
        while True:
            rss,count=tree_rss(child.pid,tracker)
            record.update(peak_rss_bytes=max(rss,record["peak_rss_bytes"]),
                          max_processes=max(count,record["max_processes"]),
                          rss_sample_count=record["rss_sample_count"]+1)
            if stopped is not None or rss>record["max_rss_bytes"]:
                record.update(state="terminated" if stopped is not None else "failed",
                              reason="signal" if stopped is not None else "max_rss_exceeded")
                if stopped is not None:
                    record["signal"]=stopped
                    result=128+stopped
                record["sent_sigkill"],record["cleanup_complete"]=stop_group(child,args.terminate_grace,tracker)
                break
            code=child.poll()
            if code is not None and count==0:
                record["state"]="passed" if code==0 else "failed"
                result=code if code>=0 else 128-code
                break
            if time.monotonic()-published>=1:
                record["elapsed"]=time.monotonic()-start
                atomic(args.status,record)
                published=time.monotonic()
            time.sleep(args.poll_interval)
    except BaseException as exc:
        record.update(state="failed",error=repr(exc))
        if child is not None:
            record["sent_sigkill"],record["cleanup_complete"]=stop_group(child,args.terminate_grace,tracker)
    finally:
        if tracker is not None:
            record["descendant_identities"]=tracker.record()
        record.update(exit_code=None if child is None else child.returncode,
                      elapsed=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
        atomic(args.status,record)
    return result


if __name__=="__main__":
    raise SystemExit(main())
