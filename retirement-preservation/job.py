"""Run a frozen workload with a sampled process-tree RSS bound and durable status."""
import argparse
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


def tree_rss(pid):
    """Sum resident pages, including nested children and orphaned group members.

    Shared pages are counted in each process's RSS. This is a sampled bound,
    not a cgroup allocation limit; processes must stay in the launched group.
    """
    processes={}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                fields=Path(entry.path,"stat").read_text().rsplit(")",1)[1].split()
                processes[int(entry.name)]=(int(fields[1]),int(fields[2]),int(fields[21]),fields[0])
            except (OSError,ValueError,IndexError):
                continue  # A process may exit while its /proc entry is read.
    owned={p for p,(_,group,_,_) in processes.items() if p==pid or group==pid}
    while True:
        children={p for p,(parent,_,_,_) in processes.items() if parent in owned}
        if children<=owned:
            break
        owned.update(children)
    live=[processes[p] for p in owned if processes[p][3]!="Z"]
    return sum(max(0,p[2]) for p in live)*os.sysconf("SC_PAGE_SIZE"),len(live)


def stop_group(child,grace):
    """Only signal the session/group created for this runner's own child."""
    def send(signum):
        try:
            os.killpg(child.pid,signum)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    deadline=time.monotonic()+grace
    while tree_rss(child.pid)[1] and time.monotonic()<deadline:
        child.poll()
        time.sleep(min(.05,max(0,deadline-time.monotonic())))
    escalated=bool(tree_rss(child.pid)[1])
    if escalated:
        send(signal.SIGKILL)
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return escalated,False
    deadline=time.monotonic()+1
    while tree_rss(child.pid)[1] and time.monotonic()<deadline:
        time.sleep(.02)
    return escalated,not bool(tree_rss(child.pid)[1])


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
    child=None
    result=1
    try:
        child=subprocess.Popen(command,start_new_session=True)
        record.update(child_pid=child.pid,process_group=child.pid)
        atomic(args.status,record)
        published=time.monotonic()
        while True:
            rss,count=tree_rss(child.pid)
            record.update(peak_rss_bytes=max(rss,record["peak_rss_bytes"]),
                          max_processes=max(count,record["max_processes"]),
                          rss_sample_count=record["rss_sample_count"]+1)
            if stopped is not None or rss>record["max_rss_bytes"]:
                record.update(state="terminated" if stopped is not None else "failed",
                              reason="signal" if stopped is not None else "max_rss_exceeded")
                if stopped is not None:
                    record["signal"]=stopped
                    result=128+stopped
                record["sent_sigkill"],record["cleanup_complete"]=stop_group(child,args.terminate_grace)
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
            record["sent_sigkill"],record["cleanup_complete"]=stop_group(child,args.terminate_grace)
    finally:
        record.update(exit_code=None if child is None else child.returncode,
                      elapsed=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
        atomic(args.status,record)
    return result


if __name__=="__main__":
    raise SystemExit(main())
