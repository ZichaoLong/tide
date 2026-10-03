"""Driver/RSS observations around static eager admission, outside step timers."""
import os
from pathlib import Path
import resource
import sys
import torch
from .eager_capacity import plan


def cpu_memory_info():
    path=Path("/proc/meminfo")
    if path.exists():
        values={row.split(':')[0]:int(row.split()[1])*1024 for row in path.read_text().splitlines() if ':' in row}
        free,total=values["MemAvailable"],values["MemTotal"]
    else:
        free=os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
        total=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    if free<=0 or total<=0:
        raise ValueError("cannot query CPU available memory")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
    return dict(device="cpu",free_bytes=free,total_bytes=total,allocated_bytes=rss,
                counter="process-lifetime peak RSS; incremental growth is a proxy, not allocator accounting")


def prepare(packet, devices, *, budget=0, **options):
    if type(budget) is not int or not 0<=budget<2**63:
        raise ValueError("device-memory-bytes must be a nonnegative int64")
    samples=[];budgets=[]
    for d in devices:
        d=torch.device(d)
        if d.type=="cpu":
            row=cpu_memory_info();limit=row["free_bytes"]//2
        else:
            backend=getattr(torch,d.type)
            with backend.device(d):
                free,total=backend.mem_get_info(d)
                row=dict(device=str(d),free_bytes=free,total_bytes=total,
                         allocated_bytes=backend.memory_allocated(d),counter="framework allocator")
                limit=free
        samples.append(row);budgets.append(min(limit,budget) if budget else limit)
    result=plan(packet,budgets=budgets,**options)
    result.update(initial_devices=samples,requested_device_memory_bytes=budget,
                  cpu_budget_policy="at most half currently available host memory")
    if result["state"]!="admitted":
        from flow_failure import RecordedFailure
        raise RecordedFailure("eager physical sample/head memory envelope exceeds budget before model allocation",
            dict(failure_phase="preallocation_memory_admission",memory_admission=result))
    return result


def observed(admission, memory):
    peaks=[]
    for initial in admission["initial_devices"]:
        if initial["device"]=="cpu":
            peak=max(p["cpu_peak_rss_bytes"] for p in memory["phases"])
        else:
            peak=max(d["peak_allocated_bytes"] for phase in memory["phases"]
                     for d in phase["devices"] if d["device"]==initial["device"])
        peaks.append(max(0,peak-initial["allocated_bytes"]))
    admission["observed_peak_growth_bytes"]=peaks
    admission["allocator_within_estimate"]=all(v<=d["estimated_peak_bytes"] for v,d in zip(peaks,admission["devices"]))
    return admission["allocator_within_estimate"]
