#!/usr/bin/env python3
"""Bounded phase/operator diagnosis through the installed public eager API.

No execution core or local mathematical function is replaced. Synchronized
phase timings and instrumented traces are deliberately separate observations.
"""
import argparse
import cProfile
import functools
import importlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json, digest, now, Metrics, project_to_lens


def annotate_python():
    """Add observation ranges only; do not synchronize or alter arguments."""
    import torch
    targets = {
        'weighted.query_vjp': ['evaluate', '_signature'],
        'weighted.aggregate_vjp': ['evaluate', '_signature'],
        'weighted.read_batch': ['evaluate'],
        'weighted.full_batch': ['evaluate'],
        'weighted.query_batch': ['kv', 'weighted_read', 'memory_read'],
        'weighted.region_batch': ['evaluate'],
    }
    for module, names in targets.items():
        owner = importlib.import_module('tidegraph.' + module)
        for name in names:
            original = getattr(owner, name, None)
            if not callable(original):
                continue
            def wrap(fn, label):
                @functools.wraps(fn)
                def observed(*args, **kwargs):
                    with torch.profiler.record_function(label):
                        return fn(*args, **kwargs)
                return observed
            setattr(owner, name, wrap(original, 'foundation::' + module + '.' + name))


def main(a):
    import torch
    from torch.nn import functional as F
    from tide_lab.train import runtime, source_identity, installed_identity, synchronize
    from tide_lab.campaign_runtime import make_model, optimizer_for, synthetic_batch, memory, learning_rate
    from tide_lab.language import sequence_readout

    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.config.read_text())
    config.update(device=a.device, threads=a.threads)
    config['training'].update(batch=a.batch, length=a.length)
    config['core'].update(eager_implementation=a.implementation,
                          placement_preset=a.placement)
    if a.native_library:
        config['core']['native_library'] = str(a.native_library)
    if a.width:
        config['core']['width'] = config['core']['recipe']['width'] = a.width
        config.pop('planned_inventory', None)
    run_id = 'tl-' + a.output.name
    state = dict(schema='tide-execution-diagnosis-v1', state='running',
        started_at=now(), source=source_identity(), config_sha256=digest(a.config),
        resolved=config, physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        phases=[], profile=dict(state='not_requested', kind=a.profile),
        acceptance=dict(execution='running', strict_numerical='unchecked',
            discrete_semantics='not_independently_checked_here', recovery='unchecked',
            learning='not_a_learning_study'),
        scope='Synthetic inputs; each phase synchronized; no data transfer, construction or logging in phase time. Profiling and cProfile overhead excluded from unprofiled observations. No equivalent-speedup claim.')
    atomic_json(a.output / 'result.json', state)
    metrics = Metrics(a.output, run_id)
    start = time.monotonic()
    try:
        device = runtime(a.device, a.threads, config['seed'])
        state['installation'] = installed_identity()
        model = make_model(config, device)
        opt = optimizer_for(model, config)
        state['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        atomic_json(a.output / 'manifest.json', dict(schema='tide-lab-run-v1',
            project='tide-learning-lab', run_id=run_id, name=run_id,
            study_id='TL-EXEC-v6-PROFILE', group='TL-EXEC-v6-PROFILE',
            stage='diagnostic', created_at=now(), config=config,
            identity=dict(source=state['source'], installation=state['installation'])))

        def step(index):
            batch = synthetic_batch(a.batch, a.length, 991 + index, device)
            model.train()
            lr = learning_rate(config, (index+1)*a.batch*a.length)
            for group in opt.param_groups:
                group['lr'] = lr
            phases = {}
            def phase(label, fn):
                synchronize(device)
                begin = time.monotonic()
                with torch.profiler.record_function('task::' + label):
                    value = fn()
                    synchronize(device)
                phases[label] = time.monotonic() - begin
                return value
            phase('zero_grad', lambda: opt.zero_grad(set_to_none=True))
            features = phase('embedding', lambda: model.embedding(batch['input_ids']))
            result = phase('core_forward', lambda: model.encoder.run_features(features, trace=False))
            hidden = phase('readout', lambda: sequence_readout(model.encoder, result, features))
            loss = phase('head_loss', lambda: F.cross_entropy(
                model.head(hidden).flatten(0, 1), batch['labels'].flatten()))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite diagnostic loss')
            phase('backward', loss.backward)
            norm = phase('clip', lambda: torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
            phase('optimizer', opt.step)
            row = dict(step=index, phases_seconds=phases, seconds=sum(phases.values()),
                loss=float(loss.detach()), gradient_norm=float(norm), tokens=a.batch*a.length, lr=lr,
                work=dict(result.stats), memory=memory(device))
            return row

        for i in range(a.updates):
            row = step(i)
            row['warmup'] = i == 0
            state['phases'].append(row)
            atomic_json(a.output / 'result.json', state)
            metrics.log(i, {**{'timing/' + k: v for k, v in row['phases_seconds'].items()},
                'timing/step_seconds': row['seconds'], 'diagnostic/loss': row['loss'],
                'diagnostic/gradient_norm': row['gradient_norm']})
            print(json.dumps(row), flush=True)

        if a.profile != 'none':
            state['profile'] = dict(state='running', kind=a.profile)
            atomic_json(a.output / 'result.json', state)
            if a.implementation == 'python':
                annotate_python()
            host = cProfile.Profile()
            if a.profile == 'cpu':
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True, with_stack=a.stack)
            else:
                import torch_npu
                prof = torch_npu.profiler.profile(activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True, with_stack=a.stack,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(a.output / 'npu-profile'), analyse_flag=True))
            with prof:
                host.enable()
                row = step(a.updates)
                host.disable()
            host.dump_stats(str(a.output / 'host.pstats'))
            stats = pstats.Stats(host)
            functions = []
            for (file, line, fn), (primitive, calls, own, cumulative, callers) in stats.stats.items():
                functions.append(dict(file=file, line=line, function=fn, calls=calls,
                    primitive_calls=primitive, own_seconds=own, cumulative_seconds=cumulative))
            functions.sort(key=lambda r: r['cumulative_seconds'], reverse=True)
            atomic_json(a.output / 'host-functions.json', functions)
            if a.profile == 'cpu':
                averages = prof.key_averages(group_by_input_shape=True)
                events = [dict(operator=e.key, count=e.count,
                    self_cpu_ms=e.self_cpu_time_total/1000, total_cpu_ms=e.cpu_time_total/1000,
                    input_shapes=e.input_shapes) for e in averages]
                events.sort(key=lambda r: r['self_cpu_ms'], reverse=True)
                atomic_json(a.output / 'operators.json', events)
                prof.export_chrome_trace(str(a.output / 'trace.json'))
            state['profile'] = dict(state='completed', kind=a.profile, observation=row,
                scope='Instrumented full step; not comparable with unprofiled timings; CPU durations are not NPU kernel durations.')
        state['state'] = state['acceptance']['execution'] = 'completed'
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        state['acceptance']['execution'] = 'failed'
        raise
    finally:
        metrics.close()
        state.update(finished_at=now(), elapsed_seconds=time.monotonic()-start)
        atomic_json(a.output / 'result.json', state)
        atomic_json(a.output / 'completed.json', dict(producer_finished=True, state=state['state']))
        if a.lens_cli:
            project_to_lens(a.output, a.lens_cli)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'npu:0'], required=True)
    p.add_argument('--implementation', choices=['python', 'native'], required=True)
    p.add_argument('--native-library', type=Path)
    p.add_argument('--placement', choices=['cpu', 'mixed-a', 'mixed-b', 'mixed-c'])
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--width', type=int)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--updates', type=int, default=3)
    p.add_argument('--profile', choices=['none', 'cpu', 'npu'], default='none')
    p.add_argument('--stack', action='store_true')
    p.add_argument('--lens-cli')
    args = p.parse_args()
    if min(args.batch, args.length, args.threads, args.updates) < 1:
        p.error('batch/length/threads/updates must be positive')
    main(args)
