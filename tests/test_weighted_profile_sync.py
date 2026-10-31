"""Independent interval witnesses for offline CUDA host-sync accounting."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_profile_sync import summarize_cuda_sync


def event(name, start, duration, category="cuda_runtime", **kwargs):
    return dict(ph="X", name=name, ts=start, dur=duration, cat=category, **kwargs)


def witness():
    return [event("kernel", 0, 100, "kernel"),
        event("tide.forward", 0, 50, "user_annotation"),
        event("tide.forward", 30, 50, "user_annotation"),
        event("tide.detach_return_sync", 80, 20, "user_annotation"),
        event("cudaStreamSynchronize", 10, 30, tid=1),
        event("cudaEventSynchronize", 20, 40, tid=2),
        event("cudaDeviceSynchronize_v3020", 90, 20, tid=1),
        event("cuCtxSynchronize", 95, 5, "cuda_driver", tid=2),
        event("cudaStreamSynchronize", 0, 100, "cpu_op"),
        event("cudaStreamWaitEvent", 1, 1), event("cudaEventQuery", 2, 1),
        dict(ph="M", name="process_name", ts=-100000, args={"name": "CPU"}),
        dict(ph="B", name="unmatched", ts=-200000)]


def test_overlaps_and_partial_phase_intersections_have_explicit_denominators():
    events = witness()
    before = copy.deepcopy(events)
    result = summarize_cuda_sync(events)
    assert events == before
    assert result["observed_cuda_api_events"] == 6 and result["unclassified_sync_name_events"] == 1
    profile = result["profile"]
    assert profile["duration_us"] == 110
    assert profile["synchronization"] == dict(events=4, inclusive_us=95, union_us=70,
        inclusive_fraction=95/110, union_fraction=70/110)
    phases = result["phases"]
    forward = phases["tide.forward"]
    assert forward["events"] == 2 and forward["inclusive_us"] == 100 and forward["union_us"] == 80
    assert forward["synchronization"] == dict(events=2, inclusive_us=70, union_us=50,
        inclusive_fraction=70/80, union_fraction=50/80)
    assert phases["tide.detach_return_sync"]["synchronization"] == dict(events=2,
        inclusive_us=15, union_us=10, inclusive_fraction=.75, union_fraction=.5)
    assert result["phase_union"]["duration_us"] == 100
    assert result["phase_union"]["synchronization"]["union_us"] == 60
    assert result["outside_phases_sync_union_us"] == 10
    assert result["apis"]["cuCtxSynchronize"]["events"] == 1
    assert "cudaStreamWaitEvent" not in result["apis"] and "cudaEventQuery" not in result["apis"]


def test_multithread_inclusive_can_exceed_elapsed_without_inflating_union():
    result = summarize_cuda_sync([event("cudaStreamSynchronize_ptsz", 0, 100, tid=1),
        event("cuStreamSynchronize_v7000", 0, 100, "cuda_driver", tid=2),
        event("tide.forward", 0, 100, "user_annotation"),
        event("tide.forward", 0, 100, "user_annotation")])
    assert result["profile"]["synchronization"]["inclusive_fraction"] == 2
    assert result["profile"]["synchronization"]["union_fraction"] == 1
    phase = result["phases"]["tide.forward"]
    assert phase["inclusive_us"] == 200 and phase["union_us"] == 100
    assert phase["synchronization"]["inclusive_fraction"] == 2


def test_disjoint_phase_occurrences_clip_once_per_api_and_keep_decimal_precision():
    base = 10000000000000000
    result = summarize_cuda_sync([
        event("kernel", f"{base}.125", "2", "kernel"),
        event("cudaThreadSynchronize", f"{base}.250", ".5"),
        event("cuEventSynchronize", f"{base}.500", ".5", "cuda_driver"),
        event("tide.forward", f"{base}.250", ".125", "user_annotation"),
        event("tide.forward", f"{base}.625", ".125", "user_annotation")])
    profile = result["profile"]
    assert profile["duration_us"] == 2
    assert profile["synchronization"]["inclusive_us"] == 1
    assert profile["synchronization"]["union_us"] == .75
    assert profile["synchronization"]["union_fraction"] == .375
    phase = result["phases"]["tide.forward"]["synchronization"]
    assert phase["events"] == 2 and phase["inclusive_us"] == .375 and phase["union_us"] == .25


@pytest.mark.parametrize("events, duration", [([], 0), ([event("kernel", 0, 10, "kernel")], 10)])
def test_absent_api_events_and_zero_denominator_are_explicit(events, duration):
    result = summarize_cuda_sync(events)
    assert result["observed_cuda_api_events"] == 0 and result["apis"] == {}
    assert result["profile"]["duration_us"] == duration
    sync = result["profile"]["synchronization"]
    assert sync["events"] == 0 and sync["union_us"] == sync["inclusive_us"] == 0
    assert sync["union_fraction"] == (0 if duration else None)


@pytest.mark.parametrize("field, value", [("ts", "nan"), ("ts", "inf"), ("dur", -1), ("dur", "invalid")])
def test_invalid_complete_times_reject_instead_of_inventing_coverage(field, value):
    row = event("cudaDeviceSynchronize", 0, 1)
    row[field] = value
    with pytest.raises(ValueError, match="profiler time"):
        summarize_cuda_sync([row])


@pytest.mark.parametrize("document_list", [False, True])
def test_existing_inspector_and_stdlib_only_cli_leave_trace_unchanged(tmp_path, document_list):
    from profile_weighted_device import inspect_trace
    events = witness()
    path = tmp_path/"trace.json"
    path.write_text(json.dumps(events if document_list else dict(traceEvents=events)))
    before = path.read_bytes()
    summary = inspect_trace(path)
    assert summary["kernel_events"] == 1 and summary["kernel_union_us"] == 100
    assert summary["cuda_host_synchronization"]["profile"]["synchronization"]["events"] == 4
    process = subprocess.run([sys.executable, "-S", str(ROOT/"scripts/weighted_profile_sync.py"), str(path)],
                             check=True, capture_output=True, text=True, timeout=10)
    observed = json.loads(process.stdout)
    assert observed.pop("trace_sha256") == hashlib.sha256(before).hexdigest()
    assert observed == summary["cuda_host_synchronization"]
    assert path.read_bytes() == before
