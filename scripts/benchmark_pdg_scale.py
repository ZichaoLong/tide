#!/usr/bin/env python3
"""Bounded comparable-scale native PDG Attention timing, with fresh random weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from build_identity import revision, source_hash
from experiment_record import LocalTrackio, atomic_json, read_events, utc_now


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reap(child, timeout):
    deadline = time.monotonic()+timeout
    while True:
        pid, status, usage = os.wait4(child.pid, os.WNOHANG)
        if pid:
            child.returncode = os.waitstatus_to_exitcode(status)
            return child.returncode, usage
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(child.args, timeout)
        time.sleep(.1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, choices=['cpu'])
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--topology', required=True)
    p.add_argument('--build-dir', default='build')
    p.add_argument('--output-dir', required=True)
    for key, default in [('width', 64), ('batch', 4), ('steps', 12), ('warmup', 4), ('workers', 1),
                         ('threads', 1), ('head-workers', 1), ('vocab', 50304), ('seed', 7), ('timeout-seconds', 1800), ('memory-gib', 1280)]:
        p.add_argument('--'+key, type=int, default=default)
    for key, default in [('packed', 1), ('grad', 0), ('check', 0), ('profile', 0), ('parallel-regions', 0), ('compact-events', 0)]:
        p.add_argument('--'+key, type=int, choices=[0, 1], default=default)
    p.add_argument('--emission', choices=['row', 'slot'], default='row')
    p.add_argument('--tracking', choices=['best-effort', 'off', 'required'], default='best-effort')
    args = p.parse_args()
    if (not 4 <= args.width <= 4096 or args.width%4 or not 1 <= args.batch <= 1024
            or not 0 <= args.warmup < args.steps <= 1000 or not 1 <= args.workers <= 160
            or not 1 <= args.threads <= 160 or args.workers*args.threads > 160 or not 2 <= args.vocab <= 100000
            or not 1 <= args.head_workers <= 160 or args.head_workers*args.threads > 160
            or not 0 <= args.seed < 2**64 or not 1 <= args.timeout_seconds <= 7200
            or not 1 <= args.memory_gib <= 1280 or (args.check and (args.width > 64 or args.batch > 8 or args.steps > 12))):
        p.error('invalid bounded scale configuration')
    root = Path(__file__).resolve().parents[1]
    build, topology, out = Path(args.build_dir).resolve(), Path(args.topology).resolve(), Path(args.output_dir).resolve()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()
    if dirty:
        p.error('run from a clean frozen source commit')
    identity, commit = source_hash(root), revision(root)
    manifest = json.loads((build/'build-manifest.json').read_text())
    binary = build/'tidegraph-scale-bench'; binary_hash = digest(binary); graph_hash = digest(topology)
    if manifest['cpp_source_sha256'] != identity or manifest['binary_sha256'].get(binary.name) != binary_hash:
        p.error('source/build mismatch; rebuild the immutable source')
    config = {k: getattr(args, k) for k in ('device', 'dtype', 'width', 'batch', 'steps', 'warmup', 'workers',
              'threads', 'head_workers', 'vocab', 'seed', 'packed', 'grad', 'check', 'emission', 'profile',
              'parallel_regions', 'compact_events')}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(topology, out/'topology.txt')
    run_id = out.name+'-'+uuid.uuid4().hex[:8]; now = utc_now()
    track = LocalTrackio(args.tracking, out.parent/'trackio', 'tide-graph-execution', run_id, config)
    command = [str(binary), '--topology', str(out/'topology.txt'), '--run-id', run_id, '--output-dir', str(out/'native')]
    for key, value in config.items(): command += ['--'+key.replace('_', '-'), str(value)]
    record = dict(schema_version=1, run_id=run_id, project='tide-graph-execution', name=run_id,
        status='running', created_at=now, started_at=now, ended_at=None,
        source=dict(repository='tide/graph-execution-foundation', commit=commit, dirty=False,
                    cpp_source_sha256=identity, binary_sha256=binary_hash, build=manifest),
        command=dict(argv=command, wrapper_argv=sys.argv, working_directory=str(root)),
        inputs=dict(topology_sha256=graph_hash, weights='fresh seeded normal std .02; no LH weights'),
        runtime=dict(resolved_device='cpu', resolution_reason='explicit:cpu', dtype=args.dtype,
                     host_arch=platform.machine(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                     load_average_before=list(os.getloadavg()), aten_threads=args.threads,
                     node_workers=args.workers, head_workers=args.head_workers, openblas_num_threads_env=1,
                     effective_thread_counts='native runtime/* metrics; -1 means unavailable',
                     interop_threads=1, address_space_limit_gib=args.memory_gib),
        experiment=dict(config=config, **{'class': 'benchmark'}, primary_metric='perf/ms_per_sample_token',
                        global_step_semantics='token index; warmup prefix retained, caches/history grow; fixed external token IDs',
                        stop_condition=f'{args.steps} steps or {args.timeout_seconds} seconds',
                        backward=False, optimizer=False, detach=False,
                        comparison='comparable scale and modules; no exact LH numerical equivalence'),
        tracking=track.record, artifacts=dict(metrics='metrics.jsonl', stdout='stdout.log', summary='summary.json', topology='topology.txt'))
    def save(): atomic_json(out/'run.json', record)
    save(); code = 1; child = None; native_code = None; usage = None; error = None; events = []; cancelled = None
    def interrupt(signum, _):
        nonlocal cancelled
        cancelled = signum
        raise InterruptedError('interrupted by signal')
    handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    try:
        track.start(); save()
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
        def limits(): resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib*2**30, args.memory_gib*2**30))
        with (out/'stdout.log').open('w') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=limits)
            native_code, usage = reap(child, args.timeout_seconds)
        if native_code != 0: raise RuntimeError(f'native exit {native_code}')
        events = read_events(out/'native/metrics.jsonl', run_id)
        if len(events) != args.steps: raise ValueError('incomplete token inventory')
        if (revision(root) != commit or source_hash(root) != identity or digest(binary) != binary_hash
                or digest(topology) != graph_hash or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip()):
            raise ValueError('source/input/binary changed')
        track.project_events(events); code = 0
    except BaseException as exception:
        error = f'{type(exception).__name__}: {exception}'
    finally:
        # Always publish terminal records, even if a resource-starved child takes
        # too long to reap. A still-unreaped child makes the pipeline stop.
        for s in handlers: signal.signal(s, signal.SIG_IGN)
        if child is not None and child.returncode is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                    native_code, usage = reap(child, 30)
                    break
                except ProcessLookupError: break
                except subprocess.TimeoutExpired: continue
        source_metrics = out/'native/metrics.jsonl'
        if source_metrics.exists():
            shutil.copyfile(source_metrics, out/'metrics.jsonl')
            try: events = read_events(out/'metrics.jsonl', run_id)
            except Exception as exc: code = 1; error = (error or '')+'; metrics: '+str(exc)
        try: track.finish()
        except Exception as exc: code = 1; error = (error or '')+'; tracking: '+str(exc)
        if cancelled: code = 128+cancelled
        record.update(status='cancelled' if cancelled else 'completed' if code == 0 else 'failed', ended_at=utc_now())
        measured = [e['metrics'] for e in events if e['step'] >= args.warmup]
        summaries = {}
        if measured:
            for key in measured[0]:
                values = [m[key] for m in measured if key in m]
                summaries[key] = dict(mean=statistics.mean(values), median=statistics.median(values),
                                      min=min(values), max=max(values), population_stdev=statistics.pstdev(values))
        atomic_json(out/'summary.json', dict(schema_version=1, run_id=run_id, status=record['status'],
            ended_at=record['ended_at'], exit_code=code, native_exit_code=native_code, error=error,
            observations=len(events), measured_observations=len(measured), metrics=summaries,
            process_seconds=time.monotonic()-started, process_peak_rss_bytes=usage.ru_maxrss*1024 if usage else None,
            unreaped_child_pid=child.pid if child is not None and child.returncode is None else None, tracking=track.record))
        save()
        for s, handler in handlers.items(): signal.signal(s, handler)
    print(json.dumps(dict(state=record['status'], summary=str(out/'summary.json'), error=error)))
    return code


if __name__ == '__main__': sys.exit(main())
