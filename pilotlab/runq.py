"""Account-wide durable CPU and physical-NPU job queue."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from .runq_devices import external, free_bytes, observe, process_unit
from .runq_place import Placement, estimated_spec, place, ram_available
from .runq_runtime import change, defer, hold_card, run_job, service, state_of
from .runq_store import (command_env, connect, default_root, device, event, lookup, manual_grant, packed,
                         pin, prefix, put, setting, sockpath, submit, unit, wake, write)
from .shared_npu3 import load_queue_helper


def dispatcher_unit(root):
    return prefix(root)+"-dispatcher.service"


def frozen_source(root):
    files = {p.name:p.read_bytes() for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    digest = hashlib.sha256()
    for name,data in files.items():
        digest.update(name.encode()+b"\0"+hashlib.sha256(data).digest())
    versions = root/"versions"
    versions.mkdir(exist_ok=True)
    dest = versions/digest.hexdigest()
    if not dest.exists():
        staging = Path(tempfile.mkdtemp(dir=versions))
        try:
            (staging/"pilotlab").mkdir()
            for name,data in files.items():
                (staging/"pilotlab"/name).write_bytes(data)
            try:
                staging.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dest


def start(root, restart=False):
    name = dispatcher_unit(root)
    if restart:
        subprocess.run(["systemctl","--user","stop",name], capture_output=True)
    if state_of(name) in ("activating","active"):
        return name
    result = service(root,name,["_dispatch"],source=frozen_source(root),restart=True)
    if result.returncode and state_of(name) not in ("activating","active"):
        raise RuntimeError(result.stderr.strip())
    return name


def sampling(root):
    db = connect(root)
    helper, mapping = None, None
    while True:
        if not db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','reserved','running','starting') LIMIT 1").fetchone():
            time.sleep(1)
            continue
        try:
            if helper is None:
                helper = load_queue_helper()
            if mapping is None:
                mapping = helper.parse_mapping(helper.run_npu_smi("info","-m"))
            sample = observe(helper, mapping)
            with write(db):
                put(db, "samples", [*setting(db,"samples",[])[-1:], sample])
                put(db, "sample_error", None)
            wake(root)
        except Exception as error:
            mapping = None
            with write(db):
                put(db,"sample_error",str(error))
        time.sleep(2)


def launch_pending(db, root):
    now = time.time()
    for row in db.execute("SELECT * FROM leases WHERE state IN ('opening','held','closing')").fetchall():
        ident = row["id"]
        name = unit(root,"lease",ident)
        if row["pid"] and process_unit(row["pid"]) == name:
            continue
        if row["tried"] and now-row["tried"] < 12:
            continue
        if row["tried"] and state_of(name) in ("active","activating","deactivating"):
            db.execute("UPDATE leases SET tried=? WHERE id=?", (now,ident))
            continue
        if row["state"] == "opening":
            db.execute("UPDATE leases SET tried=? WHERE id=?", (now,ident))
            result = service(root,name,["_lease",str(ident)])
            if result.returncode:
                db.execute("UPDATE leases SET reason=? WHERE id=?", (result.stderr.strip(),ident))
        else:
            db.execute("UPDATE leases SET state='lost',released=?,reason='lease service exited' WHERE id=?", (now,ident))
    for row in db.execute("SELECT * FROM jobs WHERE state IN ('reserved','starting','running')").fetchall():
        name = unit(root, "job", row["id"], row["generation"])
        if row["wrapper"] and process_unit(row["wrapper"]) == name:
            continue
        if row["tried"] and now-row["tried"] < 12:
            continue
        if row["tried"] and state_of(name) in ("active","activating","deactivating"):
            continue
        if row["wrapper"]:
            from .audit3 import cgroup_observation, systemd_observation
            if cgroup_observation(systemd_observation(name))["released"] is not True:
                continue
        with write(db):
            current = lookup(db, row["id"])
            if current["generation"] != row["generation"] or current["state"] != row["state"] or current["wrapper"] != row["wrapper"]:
                continue
            if current["state"] != "reserved":
                change(db,row["id"],"failed","worker service exited without a terminal receipt",rc=-1,finished=now)
                continue
            lease = db.execute("SELECT state FROM leases WHERE id=?", (row["lease"],)).fetchone() if row["lease"] else None
            if lease and lease[0] != "held":
                defer(db,row,"lease unavailable before launch")
                continue
            spec = json.loads(row["spec"])
            if spec.get("deadline") and now >= spec["deadline"]:
                change(db,row["id"],"cancelled","deadline before launch",finished=now)
                continue
            if spec.get("owner") and setting(db,"permit."+spec["owner"],0) < now:
                defer(db,row,"waiting for campaign controller")
                continue
            if not current["command"]:
                continue
            db.execute("UPDATE jobs SET tried=? WHERE id=?", (now,row["id"]))
        result = service(root,name,["_worker",str(row["id"]),str(row["generation"])],deadline=spec.get("deadline"))
        if result.returncode:
            db.execute("UPDATE jobs SET reason=? WHERE id=? AND state='reserved'", (result.stderr.strip(),row["id"]))


def dispatch(root):
    db = connect(root)
    with (root/"dispatcher.lock").open("a+") as lock, socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as listener:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        addr = Path(sockpath(root))
        addr.unlink(missing_ok=True)
        listener.bind(str(addr))
        os.chmod(addr,0o600)
        threading.Thread(target=sampling,args=(root,),daemon=True).start()
        while True:
            place(db,root,setting(db,"samples",[]))
            launch_pending(db,root)
            if select.select([listener],[],[],1)[0]:
                while select.select([listener],[],[],0)[0]:
                    listener.recv(16)


def rows_for_status(db, state=None):
    where = " WHERE state=?" if state else ""
    rows = db.execute("""SELECT j.id,job_key,target,manual,fill,j.state,backend,l.card,created AS submitted,
      assigned,started,first_step,finished,rc,j.reason,j.pid,generation FROM
      jobs j LEFT JOIN
      (SELECT id AS lease_id,card FROM leases) l ON l.lease_id=j.lease"""+where+" ORDER BY j.id",
      (state,) if state else ())
    return [dict(r) for r in rows]


def status_resources(db, root):
    samples = setting(db,"samples",[])
    with db:
        view = Placement(db,root,samples,ram_available(),time.time())
    by_backend = Counter((j["backend"],bool(j["manual"])) for j in view.active.values() if j["state"]=="running")
    workers = {b:{"auto":by_backend[b,False],"manual":by_backend[b,True]} for b in ("npu","cpu")}
    cards = []
    for number,lease in sorted(view.leases.items()):
        card = view.cards.get(str(number))
        jobs = view.by_card[lease["id"]]
        info = {k:lease[k] for k in ("id","card","role","state","reason")}
        info["running_workers"] = sum(j["state"]=="running" for j in jobs)
        if card:
            info.update(hbm_free_bytes=card["total"]-card["used"],hbm_after_reservations_bytes=free_bytes(card,jobs),
                        aicore_percent=card["core"],external_pids=[p["pid"] if p else None
                        for p in external(card,{j["unit"] for j in jobs})])
        cards.append(info)
    reasons = {row[0] or "awaiting placement":row[1] for row in db.execute(
        "SELECT reason,COUNT(*) FROM jobs WHERE state='queued' GROUP BY reason ORDER BY COUNT(*) DESC")}
    return dict(workers=workers,leases=cards,wait_reasons=reasons,
                reserved_host_bytes_by_owner={str(k):v for k,v in view.scoped.items() if v})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=default_root(),help="state directory (default RUNQ_HOME or account state)")
    sub = parser.add_subparsers(dest="action",required=True)
    for name in ("start","restart","_dispatch"):
        sub.add_parser(name)
    for name in ("_lease","_worker"):
        p = sub.add_parser(name)
        p.add_argument("id",type=int)
        if name == "_worker":
            p.add_argument("generation",type=int)
    p=sub.add_parser("limits")
    p.add_argument("--cards",type=int)
    p.add_argument("--cpu",type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--auto-card-ids", nargs="+", type=int, metavar="ID",
                       help="restrict automatic placement to these physical NPU IDs")
    group.add_argument("--all-card-ids", action="store_true", help="clear the automatic physical card filter")
    p = sub.add_parser("auto-share", help="allow automatic refill beside the currently observed NPU residents")
    p.add_argument("cards", nargs="+", type=int, metavar="ID")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--until-unix", type=int, help="expire this card's observed-resident grant at this Unix time")
    group.add_argument("--clear", action="store_true", help="withdraw the grants for these physical cards")
    p=sub.add_parser("submit")
    p.add_argument("--key",required=True)
    p.add_argument("--target",default="auto",help="auto, cpu or npu:PHYSICAL_ID")
    p.add_argument("--backend",choices=["npu","cpu","either"],default=None,help="supported backends for automatic placement")
    p.add_argument("--hbm-mib",type=int,default=0,help="peak device allocation")
    p.add_argument("--ram-mib",type=int,required=True,help="peak host RAM")
    p.add_argument("--timeout",type=int,help="overall seconds including queue wait")
    p.add_argument("--allow-auto-share",action="store_true",help="lend this card even with the observed external occupants")
    p.add_argument("argv",nargs=argparse.REMAINDER)
    p=sub.add_parser("pin")
    p.add_argument("id")
    p.add_argument("target")
    p.add_argument("--allow-auto-share",action="store_true")
    p=sub.add_parser("status")
    p.add_argument("--state",choices=["queued","reserved","starting","running","completed","failed","cancelled"])
    p.add_argument("--json",action="store_true")
    p.add_argument("--jobs",action="store_true")
    p=sub.add_parser("cancel",help="cancel one waiting job or stop its own worker unit")
    p.add_argument("id")
    p=sub.add_parser("show")
    p.add_argument("id")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    db = connect(root)
    if args.action in ("start","restart"):
        print(start(root, args.action=="restart"))
    elif args.action == "_dispatch":
        dispatch(root)
    elif args.action == "_lease":
        hold_card(root,args.id)
    elif args.action == "_worker":
        run_job(root,args.id,args.generation)
    elif args.action == "limits":
        with write(db):
            limits = setting(db,"limits",{"cards":8,"cpu":40})
            for k in ("cards","cpu"):
                value = getattr(args,k)
                if value is not None:
                    if value < 0:
                        raise ValueError("automatic targets must be nonnegative")
                    limits[k] = value
            if args.auto_card_ids is not None:
                if min(args.auto_card_ids) < 0:
                    raise ValueError("physical card IDs must be nonnegative")
                limits["auto_card_ids"] = sorted(set(args.auto_card_ids))
                if "auto_share_grants" in limits:
                    limits["auto_share_grants"] = {k:v for k,v in limits["auto_share_grants"].items()
                                                   if int(k) in limits["auto_card_ids"]}
            elif args.all_card_ids:
                limits.pop("auto_card_ids", None)
            put(db,"limits",limits)
        wake(root)
        print(packed(limits))
    elif args.action == "auto-share":
        if min(args.cards) < 0 or (not args.clear and args.until_unix <= time.time()):
            parser.error("expected nonnegative physical card IDs and a future expiry for a grant")
        observed = {} if args.clear else {
            str(card): manual_grant(db, root, f"npu:{card}", args.until_unix) for card in sorted(set(args.cards))}
        with write(db):
            limits = setting(db, "limits", {"cards":8, "cpu":40})
            if observed and not set(args.cards) <= set(limits.get("auto_card_ids", args.cards)):
                parser.error("the requested cards must be eligible for automatic placement")
            grants = limits.setdefault("auto_share_grants", {})
            for card in args.cards:
                grants.pop(str(card), None)
            grants.update(observed)
            put(db, "limits", limits)
        wake(root)
        print(packed(grants))
    elif args.action == "pin":
        row = lookup(db,args.id)
        deadline = json.loads(row["spec"]).get("deadline") or time.time()+86400
        grant = manual_grant(db,root,args.target,deadline)
        print(pin(db,args.id,args.target,args.allow_auto_share,grant)["id"])
        start(root)
        wake(root)
    elif args.action == "submit":
        kind = device(args.target)
        backend = args.backend or ("npu" if kind=="auto" else kind)
        allowed = ["npu","cpu"] if backend=="either" else [backend]
        cmd = args.argv[1:] if args.argv[:1]==["--"] else args.argv
        if not cmd or (args.timeout is not None and args.timeout <= 0):
            parser.error("executable argv and positive timeout required")
        exe = shutil.which(cmd[0])
        if not exe:
            raise ValueError("executable not found: "+cmd[0])
        submitted = db.execute("SELECT spec,created FROM jobs WHERE job_key=?", (args.key,)).fetchone()
        origin = json.loads(submitted["spec"]).get("submitted") if submitted else time.time()
        spec = {"allowed":allowed,"ram":{k:args.ram_mib*2**20 for k in allowed},"hbm":args.hbm_mib*2**20,
                "submitted":origin,"deadline":origin+args.timeout if args.timeout else None,
                "command":{"argv":[os.path.abspath(exe),*cmd[1:]],"cwd":os.getcwd(),"env":command_env()}}
        if not submitted:
            from .prepare_campaign3 import storage_observation
            storage_observation(root,reserve=0,snapshot_bytes=0)
            if Path.cwd().stat().st_dev != root.stat().st_dev:
                storage_observation(Path.cwd(),reserve=0,snapshot_bytes=0)
        grant = manual_grant(db,root,args.target,spec["deadline"] or time.time()+86400) if not submitted else None
        print(submit(db,args.key,spec,args.target,args.allow_auto_share,grant))
        start(root)
        wake(root)
    elif args.action == "cancel":
        with write(db):
            row = lookup(db,args.id)
            target = None
            if row["state"] in ("queued","reserved"):
                change(db,row["id"],"cancelled","cancelled by user",finished=time.time())
            elif row["state"] in ("running","starting"):
                target = unit(root,"job",row["id"],row["generation"])
                event(db,row["id"],"stop_requested",target)
        if target:
            subprocess.run(["systemctl","--user","stop",target],check=True)
        wake(root)
        print(lookup(db,args.id)["state"])
    elif args.action == "show":
        row = lookup(db,args.id)
        for k in ("spec","command","grant","original"):
            row[k] = json.loads(row[k]) if row[k] else None
        effective = estimated_spec(row["spec"])
        row["placement_host_bytes"] = effective["ram"]
        row["prior_host_measurement"] = effective.get("prior_host_measurement")
        row["worker_unit"] = unit(root,"job",row["id"],row["generation"]) if row["generation"] else None
        row["events"] = [dict(r) for r in db.execute("SELECT at,event,detail FROM events WHERE job=? ORDER BY id",(row["id"],))]
        print(json.dumps(row,indent=2))
    else:
        rows = rows_for_status(db,args.state)
        info = dict(dispatcher=state_of(dispatcher_unit(root)),limits=setting(db,"limits",{"cards":8,"cpu":40}),
                    counts=dict(Counter(r["state"] for r in rows)), measured_at=time.time(),
                    **status_resources(db,root),
                    sample_error=setting(db,"sample_error"), sample_at=(setting(db,"samples",[]) or [{}])[-1].get("at"))
        if args.json:
            print(json.dumps({**info,"jobs":rows},indent=2,ensure_ascii=False))
        else:
            print(f"dispatcher {info['dispatcher']}   auto targets {info['limits']['cards']} NPU / {info['limits']['cpu']} CPU")
            print("jobs", dict(Counter(r["state"] for r in rows)))
            for backend, counts in info["workers"].items():
                print(f"{backend.upper()} workers  {counts['auto']} automatic + {counts['manual']} manual")
            for lease in info["leases"]:
                free = lease.get("hbm_after_reservations_bytes")
                capacity = f", {free/2**30:.1f} GiB after reservations" if free is not None else ""
                print(f"NPU {lease['card']:2}   {lease['role']:6}  {lease['state']}   {lease['running_workers']} workers{capacity}")
            for reason,count in info["wait_reasons"].items():
                print(f"waiting  {count:4}   {reason}")
            if info["sample_error"]:
                print("NPU measurement:", info["sample_error"])
            if args.jobs or args.state:
                print("ID      STATE       PLACEMENT   KEY / WAIT")
                for row in rows:
                    place = "npu:"+str(row["card"]) if row["card"] is not None else row["backend"] or row["target"]
                    print(f"{row['id']:<7} {row['state']:11} {place:10}  {row['job_key']}  {row['reason'] or ''}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        if any(k in sys.argv for k in ("_dispatch","_worker","_lease")):
            traceback.print_exc()
        sys.exit(str(error))
