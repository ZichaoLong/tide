"""Conversation-authorized admission layered on the site's existing queue.

FIFO, group locks, double sampling and child lifetime remain site-owned. A
borrow binds an existing exclusive holder and inode, uses a separate mutex,
and terminates this adapter if the master's identity or authorization expires.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time


def identity(pid):
    directory=Path("/proc")/str(pid)
    text=(directory/"stat").read_text()
    return {"pid":pid,"uid":directory.stat().st_uid,
            "start_ticks":int(text.rsplit(")",1)[1].split()[19])}


def admitted(device,policy,now=None):
    now=time.time() if now is None else now
    if not valid_time(policy,now):
        return False
    allowed=policy["devices"].get(str(device.physical_id))
    if allowed is None or device.health!="OK":
        return False
    numbers=(device.aicore_percent,device.hbm_used_mib,device.hbm_total_mib,
             policy["max_aicore_percent"],policy["max_existing_hbm_mib"],policy["required_free_mib"])
    if not all(math.isfinite(value) and value>=0 for value in numbers):
        return False
    if device.aicore_percent>policy["max_aicore_percent"]:
        return False
    if device.hbm_used_mib>policy["max_existing_hbm_mib"]:
        return False
    if device.hbm_total_mib-device.hbm_used_mib<policy["required_free_mib"]:
        return False
    try:
        actual=[identity(pid) for pid in sorted(device.process_ids)]
    except (OSError,ValueError):
        return False
    return actual==allowed and all(item["uid"]==os.getuid() for item in actual)


def valid_time(policy,now=None):
    now=time.time() if now is None else now
    try:
        issued,expires=policy["issued_epoch"],policy["expires_epoch"]
        return all(math.isfinite(x) for x in (issued,expires,now)) and issued<=now<expires
    except (KeyError,TypeError,ValueError):
        return False


def lock_holder(path):
    """Read a whole-file kernel FLOCK owner; never trust lock-file annotations.

    The returned record is suitable for policy['existing_locks'][physical_id]:
    pid, uid, start_ticks, lock_dev (st_dev integer), lock_inode (st_ino).
    Missing, replaced, nonregular, foreign-user and ambiguous locks fail closed.
    """
    try:
        descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            metadata=os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid!=os.getuid() or metadata.st_nlink!=1:
                return None
            key=(os.major(metadata.st_dev),os.minor(metadata.st_dev),metadata.st_ino)
            owners=[]
            for line in Path("/proc/locks").read_text().splitlines():
                fields=line.split()
                if len(fields)!=8 or fields[1:4]!=["FLOCK","ADVISORY","WRITE"] or fields[6:]!=["0","EOF"]:
                    continue
                major,minor,inode=fields[5].split(":")
                if (int(major,16),int(minor,16),int(inode))==key:
                    owners.append(int(fields[4]))
            if len(owners)!=1 or owners[0]<=0:
                return None
            holder=identity(owners[0])
            current=Path(path).lstat()
            if holder["uid"]!=os.getuid() or (current.st_dev,current.st_ino)!=(metadata.st_dev,metadata.st_ino):
                return None
            return {**holder,"lock_dev":metadata.st_dev,"lock_inode":metadata.st_ino}
        finally:
            os.close(descriptor)
    except (OSError,ValueError,IndexError):
        return None


def _busy_without_writing(path):
    """An ordinary exclusive try-lock, opened read-only to preserve annotations."""
    try:
        descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            try:
                fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False
        finally:
            os.close(descriptor)
    except OSError:
        return False


def _shared_mutex(path):
    try:
        descriptor=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
        handle=os.fdopen(descriptor,"r+")
        metadata=os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid!=os.getuid() or metadata.st_nlink!=1:
            handle.close()
            return None
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=path.lstat()
        if (current.st_dev,current.st_ino)!=(metadata.st_dev,metadata.st_ino):
            handle.close()
            return None
        return handle
    except OSError:
        if "handle" in locals():
            handle.close()
        elif "descriptor" in locals():
            os.close(descriptor)
        return None


class BorrowedLock:
    """A site-compatible handle whose close unregisters its monitored lease."""
    def __init__(self,adapter,physical_id,handle):
        self.adapter,self.physical_id,self.handle=adapter,physical_id,handle

    def fileno(self):
        return self.handle.fileno()

    def close(self):
        with self.adapter.guard:
            if self.adapter.borrowed.get(self.physical_id) is self:
                self.adapter.borrowed.pop(self.physical_id)
        self.handle.close()


class SharedAdmission:
    def __init__(self,site,policy,lock_dir=None,mapping=None,monitor_seconds=2.0,terminate=None):
        if not isinstance(policy.get("existing_locks",{}),dict):
            raise ValueError("existing_locks must be a per-device mapping")
        self.site,self.policy,self.mapping=site,policy,mapping
        runtime=Path(os.environ.get("XDG_RUNTIME_DIR",f"/run/user/{os.getuid()}"))
        self.lock_dir=Path(lock_dir) if lock_dir is not None else runtime/"ascend-device-locks"
        self.ordinary_lock,self.snapshot=site.try_lock,site.snapshot
        self.latest,self.borrowed={},{}
        self.guard,self.stop=threading.Lock(),threading.Event()
        self.monitor_seconds,self.monitor=monitor_seconds,None
        self.terminate=terminate or (lambda:os.kill(os.getpid(),signal.SIGTERM))

    def expected(self,physical_id):
        return self.policy.get("existing_locks",{}).get(str(physical_id))

    def master_matches(self,physical_id):
        expected=self.expected(physical_id)
        return expected is not None and lock_holder(self.lock_dir/f"physical-{physical_id}.lock")==expected

    def check(self,device):
        self.latest[device.physical_id]=device
        try:
            return (admitted(device,self.policy) and
                    (str(device.physical_id) not in self.policy.get("existing_locks",{})
                     or self.master_matches(device.physical_id)))
        except (KeyError,TypeError,ValueError,OSError):
            return False

    def lease_violation(self,now=None):
        """Lifetime checks deliberately exclude utilization caused by our child."""
        with self.guard:
            physical_ids=tuple(self.borrowed)
        if not physical_ids:
            return None
        if not valid_time(self.policy,now):
            return "authorization_expired_or_not_yet_valid"
        for physical_id in physical_ids:
            if not self.master_matches(physical_id):
                return f"master_lock_changed:physical-{physical_id}"
        return None

    def _monitor(self):
        while not self.stop.wait(self.monitor_seconds):
            reason=self.lease_violation()
            if reason is not None:
                print(json.dumps({"shared_borrow_release_trigger":reason,"adapter_pid":os.getpid(),
                                  "action":"SIGTERM own adapter; site reaps own child before unlock"}),flush=True)
                self.terminate()
                return

    def try_lock(self,lock_dir,physical_id):
        if Path(lock_dir)!=self.lock_dir or not valid_time(self.policy):
            return None
        if str(physical_id) not in self.policy.get("existing_locks",{}):
            return self.ordinary_lock(lock_dir,physical_id)
        original=self.lock_dir/f"physical-{physical_id}.lock"
        if not self.master_matches(physical_id) or not _busy_without_writing(original):
            return None
        if not self.master_matches(physical_id):
            return None
        handle=_shared_mutex(self.lock_dir/f"shared-physical-{physical_id}.lock")
        if handle is None:
            return None
        try:
            # This fresh sample is in addition to the site's two sampling
            # phases; it closes the interval spent acquiring the borrow mutex.
            current=self.snapshot(self.mapping)[physical_id]
            if not self.check(current):
                handle.close()
                return None
        except (KeyError,TypeError,ValueError,OSError,RuntimeError):
            handle.close()
            return None
        borrowed=BorrowedLock(self,physical_id,handle)
        with self.guard:
            self.borrowed[physical_id]=borrowed
            if self.monitor_seconds is not None and self.monitor is None:
                self.monitor=threading.Thread(target=self._monitor,daemon=True)
                self.monitor.start()
        print(json.dumps({"shared_borrow_acquired":physical_id,"master":self.expected(physical_id)}),flush=True)
        return borrowed

    def close(self):
        self.stop.set()
        if self.monitor is not None:
            self.monitor.join(timeout=self.monitor_seconds+1)


def install(site,policy):
    """Patch only this imported module instance, leaving the public file intact."""
    adapter=SharedAdmission(site,policy)
    ordinary_schedule=site.attempt_schedule
    def schedule(**kwargs):
        adapter.lock_dir=Path(kwargs["device_lock_dir"])
        adapter.mapping=kwargs["mapping"]
        try:
            return ordinary_schedule(**kwargs)
        except BaseException:
            with adapter.guard:
                handles=list(adapter.borrowed.values())
            site.release_locks(handles)
            raise
    site.attempt_schedule=schedule
    site.DeviceStatus.is_free=lambda self,**kwargs:adapter.check(self)
    site.try_lock=adapter.try_lock
    return adapter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--site-helper",required=True)
    parser.add_argument("--authorization",required=True)
    args,remaining=parser.parse_known_args()
    policy=json.loads(Path(args.authorization).read_text())
    if policy.get("scope")!="this-conversation-explicit-shared-npu":
        raise SystemExit("explicit expiring conversation authorization required")
    spec=importlib.util.spec_from_file_location("tide_site_queue",args.site_helper)
    site=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=site
    spec.loader.exec_module(site)
    adapter=install(site,policy)
    print(json.dumps({"shared_admission":policy},ensure_ascii=False),flush=True)
    try:
        return site.main(remaining)
    finally:
        adapter.close()


if __name__=="__main__":
    raise SystemExit(main())
