"""Run one frozen workload with durable status; invoke inside background.slice."""
import argparse
from datetime import datetime,timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import time


def atomic(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".status.")
    with os.fdopen(fd,"w") as stream:
        json.dump(payload,stream,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--status",required=True)
    parser.add_argument("--source",required=True)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    record=dict(state="running",source=args.source,command=command,pid=os.getpid(),
                started=datetime.now(timezone.utc).isoformat())
    atomic(args.status,record)
    start=time.monotonic()
    def terminated(signum,frame):
        raise RuntimeError(f"job terminated by signal {signum}")
    signal.signal(signal.SIGTERM,terminated)
    signal.signal(signal.SIGINT,terminated)
    try:
        code=subprocess.run(command,check=False).returncode
    except BaseException as exc:
        record.update(state="failed",error=repr(exc));atomic(args.status,record);raise
    record.update(state="passed" if code==0 else "failed",exit_code=code,
                  elapsed=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
    atomic(args.status,record)
    return code


if __name__=="__main__":
    raise SystemExit(main())
