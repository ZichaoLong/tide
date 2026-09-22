#!/usr/bin/env python3
"""Record the parameterized original LH test, using its original Think timer."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
import uuid
from build_lh_original import audit_source, digest
from experiment_record import LocalTrackio, atomic_json, utc_now


def parse_log(text):
    observations, pending, parameters = [], None, None
    for line in text.splitlines():
        if match := re.fullmatch(r'Think took (\d+) ms\s*', line):
            if pending is not None:
                raise ValueError('Think timer without its token completion')
            pending = int(match[1])
        elif match := re.fullmatch(r't: (\d+)\s*', line):
            if pending is None or int(match[1]) != len(observations):
                raise ValueError('original token/timer sequence mismatch')
            observations.append(pending)
            pending = None
        elif match := re.search(r'参数量 = (\d+)', line):
            parameters = int(match[1])
    return observations, parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, choices=['cpu'])
    p.add_argument('--dtype', required=True, choices=['float32'])
    p.add_argument('--prepared', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mode', required=True, choices=['nograd', 'grad-forward'])
    p.add_argument('--threads', type=int, default=160)
    p.add_argument('--blas-threads', type=int, default=1)
    p.add_argument('--memory-gib', type=int, default=768)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    p.add_argument('--tracking', choices=['best-effort', 'required', 'off'], default='best-effort')
    args = p.parse_args()
    if not 1 <= args.threads <= 160 or not 1 <= args.blas_threads <= 160 or not 16 <= args.memory_gib <= 1280 or not 30 <= args.timeout_seconds <= 7200:
        p.error('outside bounded original-test resources')
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        p.error('run from a clean frozen Tide commit')
    prepared, out = Path(args.prepared).resolve(), Path(args.output_dir).resolve()
    build = json.loads((prepared / 'build-manifest.json').read_text())
    source = Path(build['source'])
    def verify():
        if audit_source(source, build['source_revision'], build['parameter_files_sha256']) != build['source_files_sha256']:
            raise ValueError('original source changed')
        for name, expected in build['artifacts_sha256'].items():
            if digest(prepared / name) != expected:
                raise ValueError('binary changed: ' + name)
        for name, expected in build['graph_files_sha256'].items():
            if digest(source / 'graph-data' / name) != expected:
                raise ValueError('graph changed: ' + name)
    verify()
    batch, steps = build['batch'], build['steps']
    out.mkdir(parents=True, exist_ok=False)
    run_id = out.name + '-' + uuid.uuid4().hex[:8]
    track = LocalTrackio(args.tracking, out.parent / 'trackio', 'tide-graph-execution', run_id, vars(args))
    executable = Path(build['build_dir']) / ('test-cortexnet-nograd' if args.mode == 'nograd' else 'test-cortexnet-grad')
    command = ['/usr/bin/time', '-v', '-o', str(out / 'resource.txt'), str(executable)]
    now = utc_now()
    record = dict(schema_version=1, run_id=run_id, project='tide-graph-execution', name=run_id,
        status='running', created_at=now, started_at=now, ended_at=None,
        source=dict(repository='tide/graph-execution-foundation', commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), dirty=False, build=build),
        command=dict(argv=command, wrapper_argv=sys.argv, working_directory=build['build_dir']),
        inputs=dict(config_sha256=build['config_sha256'], graph=build['graph_files_sha256']),
        runtime=dict(resolved_device='cpu', resolution_reason='explicit:cpu', dtype='float32',
            host_arch=platform.machine(), threads=args.threads, openblas_threads=args.blas_threads,
            interop_threads='original default; no interop parallel tasks added',
            cpu_affinity=sorted(os.sched_getaffinity(0)), load_average_before=list(os.getloadavg()),
            memory_limit_gib=args.memory_gib),
        experiment=dict(config=vars(args), **{'class': 'benchmark'},
            global_step_semantics=f'original test token index 0..{steps-1}, batch{batch}; includes cold first step',
            primary_metric='perf/ms_per_sample_token', stop_condition=f'{args.timeout_seconds}s; {steps} original steps',
            includes='original Think timer, state preparation and inner timer printing; no backward or optimizer',
            timestamp_semantics='events projected after subprocess exit; elapsed_seconds is projection time',
            seed_policy='unchanged original test; no added manual seed'),
        tracking=track.record, artifacts=dict(metrics='metrics.jsonl', stdout='stdout.log',
            summary='summary.json', resources='resource.txt'))
    atomic_json(out / 'run.json', record)
    code, native_code, error, observations, parameters, events = 1, None, None, [], None, []
    child = None
    started = time.monotonic()
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 1024**3,) * 2)
    def interrupted(signum, _):
        raise InterruptedError('cancelled by signal ' + str(signum))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        track.start()
        atomic_json(out / 'run.json', record)
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS=str(args.blas_threads), MKL_NUM_THREADS=str(args.threads))
        with (out / 'stdout.log').open('w') as log:
            child = subprocess.Popen(command, cwd=build['build_dir'], env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limits)
            native_code = child.wait(timeout=args.timeout_seconds)
        if native_code != 0:
            raise RuntimeError('original test exited ' + str(native_code))
        observations, parameters = parse_log((out / 'stdout.log').read_text())
        if len(observations) != steps or parameters is None:
            raise ValueError('original step inventory incomplete')
        if parameters != build['expected_parameters']:
            raise ValueError('native parameter count differs from configuration preflight')
        verify()
        code = 0
    except BaseException as exception:
        error = f'{type(exception).__name__}: {exception}'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for s, handler in previous.items():
            signal.signal(s, handler)
        native_code = None if child is None else child.returncode
        elapsed = time.monotonic() - started
        try:
            observations, parameters = parse_log((out / 'stdout.log').read_text())
            with (out / 'metrics.jsonl').open('w') as metrics:
                for step, ms in enumerate(observations):
                    event = dict(schema_version=1, run_id=run_id, sequence=step, timestamp=utc_now(), step=step,
                        elapsed_seconds=elapsed, metrics={'perf/think_ms_per_batch': ms,
                        'perf/ms_per_sample_token': ms / batch}, context={'phase': 'forward', 'token_step': step})
                    metrics.write(json.dumps(event, allow_nan=False) + '\n')
                    events.append(event)
            track.project_events(events)
        except Exception as exception:
            code = 1
            error = error or str(exception)
        try:
            track.finish()
        except Exception as exception:
            code = 1
            error = error or str(exception)
        rss = None
        if (out / 'resource.txt').exists():
            match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', (out / 'resource.txt').read_text())
            rss = int(match[1]) * 1024 if match else None
        windows = {}
        for name, lo, hi in [('all', 0, steps), ('after-first-4', 4, steps), ('early-4-11', 4, 12), ('late-80-99', 80, 100)]:
            times = observations[lo:hi]
            if hi > lo and len(times) == hi - lo and sum(times) > 0:
                windows[name] = dict(token_steps=[lo, hi-1], count=len(times),
                    mean_ms_per_sample_token=sum(times) / (batch * len(times)),
                    aggregate_tokens_per_second=1000 * batch * len(times) / sum(times),
                    min_ms_per_sample_token=min(times) / batch, max_ms_per_sample_token=max(times) / batch,
                    median_ms_per_sample_token=statistics.median(times) / batch)
        record.update(status='completed' if code == 0 else 'failed', ended_at=utc_now())
        record['runtime']['load_average_after'] = list(os.getloadavg())
        atomic_json(out / 'summary.json', dict(schema_version=1, run_id=run_id, status=record['status'],
            exit_code=code, native_exit_code=native_code, error=error, observations=len(observations),
            parameters=parameters, process_seconds=elapsed, process_peak_rss_bytes=rss,
            windows=windows, tracking=track.record))
        atomic_json(out / 'run.json', record)
    print(json.dumps(dict(state=record['status'], summary=str(out / 'summary.json'), error=error)))
    return code


if __name__ == '__main__':
    sys.exit(main())
