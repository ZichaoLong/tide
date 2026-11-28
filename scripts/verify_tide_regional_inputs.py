"""Hash R3's delivered Guiyang inputs on an existing allocation and save per-version receipts."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3

from tide_cloud_allocation import REPO, Store
from tide_cloud_dispatch import object_key
from tide_cloud_submit import pinned_ssh, verify_object
from tide_portable import read, require, write
from ma_ledger import timestamp
import ma_oms
from ma_profile import state_dir


def incoming(storage):
    manifest = read(storage / "oms-service-delivery.json")
    require(manifest["all_transferred"], "OMS transfer is still in progress")
    targets = {}
    for group in manifest["operations"]:
        operation = read(ma_oms.receipt_path(group["id"]))
        objects = operation["objects"]
        require(operation["operation_id"] == group["id"] and str(operation["task_id"]) == str(group["task"])
                and operation["source"] == "east" and operation["destination"] == "guiyang"
                and group["status"] == 5 and group["failed_num"] == group["skipped_num"] == 0
                and group["n"] == group["total_num"] == len(objects)
                and group["bytes"] == group["complete_size"] == sum(o["size"] for o in objects),
                "OMS task identity or completion counters differ")
        for obj in objects:
            h = obj["sha256"]
            source = read(storage / "receipts/east" / (h + ".json"))
            require(h not in targets and source["sha256"] == h and source["verified"]
                    and obj["source_key"] == obj["destination_key"] == object_key(h)
                    and source["size"] == obj["size"] and source["etag"] == obj["source_etag"],
                    "OMS member differs from its verified source")
            targets[h] = dict(operation=group["id"], oms_task=str(group["task"]), key=obj["destination_key"],
                              size=obj["size"], sha256=h)
    require(len(targets) == manifest["total_objects"] and sum(x["size"] for x in targets.values()) == manifest["total_bytes"],
            "OMS totals do not match their individual objects")
    return targets


def prior_receipts(storage, region, targets):
    receipts = {}
    project_ops = {row["operation"] for row in targets.values()}
    with sqlite3.connect((state_dir() / "catalog.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        for op in project_ops:
            for locator, size, detail, observed in db.execute(
                    "SELECT locator,bytes_used,details,seen_at FROM resources WHERE kind='obs' AND run_id=?", (op,)):
                meta = json.loads(detail)
                h = meta["sha256"]
                require(h in targets and targets[h]["operation"] == op and targets[h]["size"] == size
                        and meta["side"] == "guiyang" and meta["key"] == targets[h]["key"]
                        and locator == region.uri(meta["key"]), "unexpected object in OMS ledger")
                receipts[h] = {**targets[h], "etag":meta["etag"], "version":meta.get("version"),
                               "verified_at":observed, "method":"maflow verified OMS destination"}
    for group in ("guiyang-object", "guiyang-node"):
        for file in (storage / "receipts" / group).glob("*.json"):
            h = file.stem
            record = read(file)
            require(h in targets and record["sha256"] == h and record["key"] == targets[h]["key"]
                    and record["size"] == targets[h]["size"] and str(record["oms_task"]) == targets[h]["oms_task"]
                    and record["etag"] and record["verified_at"], "unbound individual destination receipt")
            receipts[h] = dict(record, method=record.get("method") or record.get("origin") or "individual destination SHA-256")
    return receipts


def matches_version(left, right):
    return bool(left and right and left["etag"] and left["etag"] == right["etag"]
                and left["size"] == right["size"] and left.get("version") == right.get("version"))


def verify(storage, node, workers):
    require(node["platform"]["phase"] == "Running" and node["platform"]["id"] == node["id"]
            and node["platform"]["name"] == node["name"] and pinned_ssh(node), "a running, SSH-pinned Guiyang node is required")
    region = Store.named("guiyang")
    targets = incoming(storage)
    directory = storage / "receipts/guiyang-node"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "verify.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = prior_receipts(storage, region, targets)

        def check(item):
            h, row = item
            key, size = row["key"], row["size"]
            before = region.head(key)
            require(before and before["size"] == size and before.get("sha256") in (None, h), "destination changed: " + h)
            saved = prior.get(h)
            if matches_version(saved, before):
                receipt = {**saved, **row}
                outcome = "reused"
            else:
                accepted = verify_object(node, region, key, h, size, background=True)
                after = region.head(key)
                require(accepted == {"sha256":h, "size":size} and matches_version(before, after)
                        and after.get("sha256") in (None, h), "destination changed during regional readback: " + h)
                receipt = {**row, "etag":after["etag"], "version":after.get("version"), "verified_at":timestamp(),
                           "method":"regional full SHA-256 over signed GET on pinned ModelArts SSH",
                           "job":node["name"], "platform_id":node["id"]}
                outcome = "node"
            write(directory / (h + ".json"), receipt)
            return outcome, size

        # Check still-unread large objects first. Every foreground training worker keeps its own allocation.
        ordered = sorted(targets.items(), key=lambda it: (it[0] in prior, -it[1]["size"]))
        counts, sizes = {}, {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for method, size in executor.map(check, ordered):
                counts[method] = counts.get(method, 0) + 1
                sizes[method] = sizes.get(method, 0) + size
        return dict(completed=timestamp(), destination=region.side, job=node["name"], platform_id=node["id"],
                    workers=workers, objects=len(ordered), bytes=sum(sizes.values()), by_method=counts, bytes_by_method=sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=Path, default=REPO / ".local/modelarts-r3")
    parser.add_argument("--job", required=True, help="正在运行、已有 maflow SSH 主机绑定的贵阳训练作业名")
    parser.add_argument("--workers", type=int, default=2, help="节点上并行的低优先级 SHA-256 读流")
    args = parser.parse_args()
    require(args.workers >= 1, "worker count must be positive")
    node = read(args.area / "control/status.json")["nodes"][args.job]
    storage = args.area / "storage"
    result = verify(storage, node, args.workers)
    write(storage / "guiyang-node-verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
