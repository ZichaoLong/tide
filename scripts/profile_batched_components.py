#!/usr/bin/env python3
"""Finite State/Read composition comparison; one process owns one variant.

Scalar replay reproduces the pre-batching attachment path using the same packed
numeric forward. This is a component experiment, not a graph throughput claim.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import torch
from durable_records import write_json
from source_identity import source_state, digest


def make_case(kind, rows, width, dtype, device, read_profile):
    from tidegraph import Graph, Node, Region
    from tidegraph.ops import Model
    from tidegraph.content import Content
    from tidegraph.records import State
    from tidegraph.packing import PackedSequence
    memory = 'attention' if kind == 'state-attention' else 'lh-add-repeat-v1'
    graph = Graph((Node(0, memory=memory, query_heads=2, kv_heads=2, readout=read_profile),), (), (Region(1),), (0,), (0,))
    model = Model(graph, width, dtype=dtype).to(device)
    weights = model.nodes[0]
    generator = torch.Generator().manual_seed(19)
    def tensor(shape):
        return (torch.randn(shape, generator=generator, dtype=dtype)*.1).to(device).requires_grad_()
    old, views = [], []
    for _ in range(rows):
        s = weights.initial()
        slots = {k:tensor((8, *v.shape[1:])) for k, v in s.slots.items()}
        old.append(State(tensor((width,)), 0, 8 if slots else 1, slots))
        views.append(Content(tensor((width,))))
    times = [3]*rows
    batch = PackedSequence(torch.stack([v.value for v in views]), list(range(rows+1)),
                           [(i, 0) for i in range(rows)], times, views)
    # Inputs are fixed leaves; every invocation constructs a fresh result graph.
    leaves = list(model.parameters()) + [t for s, v in zip(old, views) for t in (s.value, *s.slots.values(), v.value)]
    return weights, old, views, times, batch, leaves


def operation(case, kind, variant):
    from tidegraph import autograd, state_vjp
    from tidegraph.readout import ReadInput, evaluate, validate
    weights, old, views, times, batch, leaves = case
    if kind == 'read':
        requests = [ReadInput(s, t, v) for s, v, t in zip(old, views, times)]
        if variant == 'batched':
            with torch.profiler.record_function('batched_read'):
                values = evaluate(weights, requests, packed=True)
        else:
            with torch.profiler.record_function('scalar_read_replay'):
                with torch.no_grad():
                    numeric = weights.read_program.batch(weights, requests)
                    for y, r in zip(numeric, requests):
                        validate(y, r, weights.read_program.precision)
                semantic = [weights.read_program.step(weights, r) for r in requests]
                for y, r in zip(semantic, requests):
                    validate(y, r, weights.read_program.precision)
                values = [autograd.value(y, s) for y, s in zip(numeric, semantic)]
    else:
        with torch.profiler.record_function('packed_numeric'), torch.no_grad():
            # Stack inside the invocation: the component includes packing cost.
            current = replace(batch, contents=torch.stack([v.value for v in views]))
            numeric = weights.kernel.packed_sequence(weights, old, current)[0]
        with torch.profiler.record_function('batch_graph' if variant == 'batched' else 'scalar_state_replay'):
            if variant == 'batched':
                states = state_vjp.bind_batch(weights, old, views, times, numeric)
            else:
                states = [autograd.state(n, weights.kernel.step(weights, s, v, t))
                          for n, s, v, t in zip(numeric, old, views, times)]
            values = [s.value for s in states]
    with torch.profiler.record_function('component_backward'):
        loss = torch.stack(values).square().mean()
        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    return values, grads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True)
    p.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
    p.add_argument('--kind', choices=('read', 'state-add', 'state-attention'), required=True)
    p.add_argument('--variant', choices=('replay', 'batched'), required=True)
    p.add_argument('--read-profile', choices=('linear-v1', 'norm-fp32-v1'), default='linear-v1')
    p.add_argument('--rows', type=int, default=32)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.rows <= 512 or not 2 <= a.width <= 2048 or a.width % 2:
        p.error('finite component profile requires rows1..512 and even width2..2048')
    root = Path(__file__).resolve().parents[1];sys.path.insert(0, str(root/'python'))
    from tidegraph.runtime import resolve_device, synchronize
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device, reason = resolve_device(a.device)
    out = a.output_dir.resolve();out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(schema='tide-batched-components-v1', state='running', source=source, dirty=dirty,
                  command=sys.argv, kind=a.kind, variant=a.variant, rows=a.rows, width=a.width,
                  dtype=a.dtype, device=str(device), resolution_reason=reason,
                  read_profile=a.read_profile,
                  torch=torch.__version__, input_seed=19, model_seed=7, warmup=1, measured_steps=1,
                  scope='pure PyTorch component composition; no native/LibTorch graph throughput claim',
                  timing='packing + numeric forward + gradient attachment + Read finite checks + backward; synchronized; no oracle in timed step',
                  tracking='existing local records only; Trackio off by user contract')
    write_json(out/'result.json', record)
    try:
        case = make_case(a.kind, a.rows, a.width, getattr(torch, a.dtype), device, a.read_profile)
        expected = operation(case, a.kind, 'replay')
        actual = operation(case, a.kind, 'batched')
        tolerance = dict(atol=2e-3, rtol=2e-2) if a.dtype == 'float16' else dict(atol=2e-5, rtol=2e-4)
        for xs, ys in zip(expected, actual):
            for x, y in zip(xs, ys):
                if (x is None) != (y is None):
                    raise AssertionError('component gradient connectivity differs')
                if x is not None:
                    torch.testing.assert_close(x, y, **tolerance)
        del expected, actual
        operation(case, a.kind, a.variant);synchronize(device)
        start = time.perf_counter();operation(case, a.kind, a.variant);synchronize(device)
        record.update(seconds=time.perf_counter()-start, correctness='forward and first-order VJP/None parity', tolerances=tolerance)
        # Save the timing before the separate, untimed profiler pass.
        write_json(out/'measurement.json', record)
        if a.profile:
            os.chdir(out)
            if device.type == 'npu':
                import torch_npu
                profiler = torch_npu.profiler
                profiler.tensorboard_trace_handler(str(out/'raw'), analyse_flag=False)
                activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
            else:
                profiler = torch.profiler
                activities = [profiler.ProfilerActivity.CPU]
                if device.type == 'cuda':activities.append(profiler.ProfilerActivity.CUDA)
            with profiler.profile(activities=activities, record_shapes=True) as trace:
                operation(case, a.kind, a.variant);synchronize(device)
            trace.export_chrome_trace(str(out/'trace.json'))
            document = json.loads((out/'trace.json').read_text())
            events = document if isinstance(document, list) else document['traceEvents']
            device_pids = {e['pid'] for e in events if e.get('name') == 'process_name'
                           and e.get('args', {}).get('name') == 'Ascend Hardware'}
            kernels = [e for e in events if e.get('ph') == 'X' and
                       ('kernel' in str(e.get('cat', '')).lower() or
                        (e.get('pid') in device_pids and
                         str(e.get('args', {}).get('Task Type', '')).startswith('KERNEL')))]
            if device.type != 'cpu' and not kernels:
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
            record.update(host_stage_us=dict(stages), kernel_events=len(kernels),
                kernel_names=dict(Counter(e['name'] for e in kernels)),
                kernel_total_us=sum(e['dur'] for e in kernels),
                kernel_span_us=(max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)) if kernels else 0,
                trace_sha256=digest(out/'trace.json'), observed_fallback_events=fallback,
                trace_event_counts=dict(Counter(e.get('name') for e in events if e.get('ph') == 'X')))
        if source_state(root) != (source, dirty):raise RuntimeError('source changed during component experiment')
        record['state'] = 'passed'
    except BaseException as error:
        record.update(state='failed', error=repr(error));raise
    finally:
        write_json(out/'result.json', record)


if __name__ == '__main__':
    main()
