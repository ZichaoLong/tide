"""Offline CUDA host synchronization coverage; no Torch or device initialization."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import re


_CATEGORIES = {"cuda_runtime", "cuda_driver"}
_API = re.compile(r"(?:cuda(?:Device|Event|Stream|Thread)Synchronize|"
                  r"cu(?:Ctx|Event|Stream)Synchronize)(?:_(?:v[0-9]+|ptsz|ptds))*")


def _interval(event):
    try:
        start, duration = (Decimal(str(event[key])) for key in ("ts", "dur"))
    except (KeyError, InvalidOperation) as error:
        raise ValueError("invalid complete-event profiler time") from error
    if not start.is_finite() or not duration.is_finite() or duration < 0:
        raise ValueError("invalid complete-event profiler time")
    return start, start+duration


def _merge(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], stop)
        else:
            merged.append((start, stop))
    return merged


def _duration(intervals):
    return sum((stop-start for start, stop in intervals), Decimal(0))


def _fraction(value, denominator):
    return float(value/denominator) if denominator else None


def _stats(intervals, denominator):
    inclusive, union = _duration(intervals), _duration(_merge(intervals))
    return dict(events=len(intervals), inclusive_us=float(inclusive), union_us=float(union),
                inclusive_fraction=_fraction(inclusive, denominator),
                union_fraction=_fraction(union, denominator))


def _within(intervals, windows):
    # Each API contributes once to the event count even if it intersects several
    # disjoint occurrences of a phase. Phase windows are already merged.
    clipped, count = [], 0
    for start, stop in intervals:
        pieces = [(max(start, a), min(stop, b)) for a, b in windows
                  if max(start, a) < min(stop, b) or start == stop and a <= start < b]
        if pieces:
            count += 1
            clipped.extend(pieces)
    result = _stats(clipped, _duration(windows))
    result["events"] = count
    return result


def summarize_cuda_sync(events):
    """Summarize complete Chrome events without changing the input document.

    API durations are inclusive host calls, not measured CPU-idle durations.
    Phase membership is wall-clock overlap across threads, not causal nesting.
    """
    complete = [(event, _interval(event)) for event in events if event.get("ph") == "X"]
    intervals = [interval for _, interval in complete]
    span = max(stop for _, stop in intervals)-min(start for start, _ in intervals) if intervals else Decimal(0)
    sync, apis, phases = [], defaultdict(list), defaultdict(list)
    cuda_events, unclassified = 0, 0
    for event, interval in complete:
        name = str(event.get("name", ""))
        category = {part.strip() for part in str(event.get("cat", "")).split(",")}
        cuda = bool(category & _CATEGORIES)
        cuda_events += cuda
        if _API.fullmatch(name):
            if cuda:
                sync.append(interval)
                apis[name].append(interval)
            else:
                unclassified += 1
        if name.startswith("tide."):
            phases[name].append(interval)
    phase_rows = {}
    for name, rows in sorted(phases.items()):
        windows = _merge(rows)
        phase_rows[name] = dict(events=len(rows), inclusive_us=float(_duration(rows)),
            union_us=float(_duration(windows)), synchronization=_within(sync, windows))
    windows = _merge([interval for rows in phases.values() for interval in rows])
    phase_sync = _within(sync, windows)
    profile = _stats(sync, span)
    return dict(schema="tide-cuda-host-sync-v1", time_unit="us",
        scope="complete X CUDA runtime/driver explicit host synchronization API events",
        observed_cuda_api_events=cuda_events, unclassified_sync_name_events=unclassified,
        profile=dict(duration_us=float(span), synchronization=profile,
            denominator="latest complete X event end minus earliest complete X event start; includes capture overhead"),
        apis={name: _stats(rows, span) for name, rows in sorted(apis.items())},
        phases=phase_rows,
        phase_union=dict(duration_us=float(_duration(windows)), synchronization=phase_sync),
        outside_phases_sync_union_us=profile["union_us"]-phase_sync["union_us"],
        phase_scope="wall-clock intersection across captured host threads; denominator is each phase interval union",
        caveat="inclusive durations may overlap and fractions may exceed one; union is covered wall time, "
               "not critical-path attribution, CPU utilization or all host idle time; no implicit copy waits, "
               "query polling, device-side waits, missing events or uncaptured time inferred")
