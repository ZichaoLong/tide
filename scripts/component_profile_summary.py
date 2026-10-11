"""Shared validation of collected component traces; no Torch imports."""
from collections import Counter
import json
import math
from pathlib import Path

from source_identity import digest


def raw_identity(directory):
    directory = Path(directory)
    paths = sorted(directory.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('profile raw input must not contain symlinks')
    files = {str(p.relative_to(directory)): digest(p) for p in paths if p.is_file()}
    if not files:
        raise ValueError('profile raw collection is empty')
    return files


def summarize_trace(path, device_type):
    document = json.loads(Path(path).read_text())
    events = document if isinstance(document, list) else document['traceEvents']
    # TorchNPU offline text exports encode timestamps as decimal strings;
    # Torch/CUDA exports commonly use JSON numbers. Normalize only profiler
    # microseconds, never model clocks/counts or the retained raw document.
    for event in events:
        if event.get('ph') == 'X':
            for key in ('ts', 'dur'):
                value = float(event[key])
                if not math.isfinite(value) or (key == 'dur' and value < 0):
                    raise ValueError('invalid complete-event profiler time')
                event[key] = value
    device_pids = {e['pid'] for e in events if e.get('name') == 'process_name'
                   and e.get('args', {}).get('name') == 'Ascend Hardware'}
    kernels = [e for e in events if e.get('ph') == 'X' and
               ('kernel' in str(e.get('cat', '')).lower() or
                (e.get('pid') in device_pids and
                 str(e.get('args', {}).get('Task Type', '')).startswith('KERNEL')))]
    if device_type != 'cpu' and not kernels:
        raise RuntimeError('profile contains no actual accelerator kernel events')
    fallback = [e.get('name') for e in events if any(t in str(e.get('name', '')).lower()
                for t in ('cpu_fallback', 'fallback_to_cpu', 'fallback to cpu'))]
    if fallback:
        raise RuntimeError('unexpected CPU fallback events: ' + repr(fallback[:10]))
    stages = Counter()
    for e in events:
        if e.get('ph') == 'X' and e.get('name') in {'batched_read', 'scalar_read_replay',
                'packed_numeric', 'batch_graph', 'scalar_state_replay', 'component_backward'}:
            stages[e['name']] += e['dur']
    return dict(host_stage_us=dict(stages), kernel_events=len(kernels),
        kernel_names=dict(Counter(e['name'] for e in kernels)),
        kernel_total_us=sum(e['dur'] for e in kernels),
        kernel_span_us=(max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)) if kernels else 0,
        trace_sha256=digest(path), observed_fallback_events=fallback,
        trace_event_counts=dict(Counter(e.get('name') for e in events if e.get('ph') == 'X')))
