"""Conversation-authorized admission layered on the site's existing queue.

Only changes the device predicate; FIFO, group locks, double sampling and child
lifetime remain owned by the site helper. An expiring PID/start policy is required.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def identity(pid):
    directory=Path("/proc")/str(pid)
    text=(directory/"stat").read_text()
    return {"pid":pid,"uid":directory.stat().st_uid,
            "start_ticks":int(text.rsplit(")",1)[1].split()[19])}


def admitted(device,policy,now=None):
    now=time.time() if now is None else now
    if now>policy["expires_epoch"] or now<policy["issued_epoch"]:
        return False
    allowed=policy["devices"].get(str(device.physical_id))
    if allowed is None or device.health!="OK":
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
    return actual==allowed


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
    site.DeviceStatus.is_free=lambda self,**kwargs:admitted(self,policy)
    print(json.dumps({"shared_admission":policy},ensure_ascii=False),flush=True)
    return site.main(remaining)


if __name__=="__main__":
    raise SystemExit(main())
