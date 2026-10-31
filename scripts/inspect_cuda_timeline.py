#!/usr/bin/env python3
"""Summarize an Nsight Systems CUDA SQLite export without reading environment data.

Graph-node activity is an observation, not an automatic residency or correctness
certificate. Review kernel roles, the consumer, source and application result.
Conditional graph kernels may have no joinable host launch correlation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from durable_records import write_json


FOCUS = (
    "tide_closure", "tide_ready_pack", "tide_weighted_node_pack",
    "tide_weighted_region_pack", "tide_weighted_allocate",
    "tide_weighted_region_result", "tide_weighted_owner_adopt",
    "tide_broadcast_route", "tide_queue_propose",
    "tide_weighted_reverse_route", "tide_optimizer_commit",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inspect(path):
    path = path.resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        processes = [row[0] for row in db.execute(
            "select distinct globalPid from CUPTI_ACTIVITY_KIND_KERNEL order by globalPid")]
        devices = [row[0] for row in db.execute(
            "select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by deviceId")]
        # Node IDs and runtime correlation IDs are not globally unique across
        # processes. Keep this initial observer explicitly single-process.
        if len(processes) > 1 or len(devices) > 1:
            raise ValueError("select a single-process, single-device CUDA trace")
        total, graphed, nodes = db.execute(
            "select count(*),sum(graphNodeId>0),count(distinct case when graphNodeId>0 "
            "then graphNodeId end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        if not total:
            raise ValueError("trace has no actual CUDA kernel activities")
        if db.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where end<start").fetchone()[0]:
            raise ValueError("invalid CUDA activity interval")
        if db.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s "
                      "on s.id=k.shortName where s.value is null").fetchone()[0]:
            raise ValueError("CUDA kernel has no recorded name")
        kernels = [dict(name=name, count=count, graph_node_count=graph_count or 0,
                        summed_duration_us=duration / 1000.)
                   for name, count, graph_count, duration in db.execute(
                       "select s.value,count(*),sum(k.graphNodeId>0),sum(k.end-k.start) "
                       "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                       "group by s.value order by count(*) desc,s.value")]
        runtime = dict(db.execute(
            "select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds s on s.id=r.nameId where (r.globalTid & -16777216)=? "
            "group by s.value order by s.value", (processes[0],)))
        # Do not guess a launch from the nearest timestamp: concurrent streams
        # and conditional graph activities make that attribution ambiguous.
        joined = db.execute(
            "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k where k.graphNodeId>0 "
            "and exists(select 1 from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "where r.correlationId=k.correlationId and "
            "(r.globalTid & -16777216)=k.globalPid)").fetchone()[0]
        intervals = db.execute(
            "select start,end from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId>0 order by start")
        end, union = None, 0
        for first, last in intervals:
            union += last-first if end is None else max(0, last-max(first, end))
            end = last if end is None else max(end, last)
        marks = ",".join("?" for _ in FOCUS)
        events = [dict(start_ns=a, end_ns=b, stream=stream, graph_node_id=node, kernel=name)
                  for a, b, stream, node, name in db.execute(
                      "select k.start,k.end,k.streamId,k.graphNodeId,s.value "
                      "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where k.graphNodeId>0 and s.value in ("+marks+") order by k.start limit 64", FOCUS)]
        return dict(schema="tide-cuda-timeline-observation-v1", database_sha256=digest(path),
                    device_ids=devices, process_ids=processes,
                    analyzer_sha256=digest(Path(__file__)), kernel_activities=total,
                    graph_node_kernel_activities=graphed or 0, distinct_executed_graph_nodes=nodes,
                    graph_node_kernel_union_us=union/1000., host_correlated_graph_activities=joined,
                    graph_launch_calls=sum(v for k, v in runtime.items() if "cudaGraphLaunch" in k),
                    kernels=kernels, runtime_calls=runtime, focus_sample=events,
                    scope="observed activities only; no automatic correctness, residency or throughput claim",
                    caveats=["profile includes setup, warmup, validation and checkpoint work",
                             "kernel sums overlap across streams; union is covered device time",
                             "globalPid/globalTid use Nsight Systems' 24-bit thread field",
                             "missing host correlations are retained; no per-launch assignment inferred"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output must be new")
    result = inspect(args.database)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
