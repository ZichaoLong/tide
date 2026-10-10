#!/usr/bin/env python3
"""Bounded complete LibTorch mixed/resident CANN trace with memory calibration.

Uses the ordinary continuous consumer, exact packet and public placement options.
Collection is separate from performance timing; no reference trajectory is used.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from durable_records import write_json, replace_text
from foundation_lifecycle import run_child
from flow_resident_options import validate as validate_resident_options
from profile_flow_support import check_build, check_result, collect_profiles
from run_execution_flow import make_parser, validate_arguments
from run_flow_native import prepare
from source_identity import source_state, digest


def main():
    p = make_parser(standalone=True)
    p.description = __doc__
    p.set_defaults(steps=1, warmup=1)
    p.add_argument('--build-dir', type=Path, required=True, help='build_online_consumer.py output')
    p.add_argument('--prepare-only', action='store_true', help='validate identities and write commands; no device execution')
    p.add_argument('--timeout-seconds', type=int, default=300, help='total collection/export budget, without retries')
    p.add_argument('--storage-limit-mb', type=int, default=512, help='raw CANN collection limit')
    p.add_argument('--storage-budget-mb', type=int, default=2048, help='raw plus exported files, monitored throughout')
    p.add_argument('--host-memory-gib', type=int, default=12)
    a = p.parse_args()
    if not re.fullmatch(r'npu:\d+', a.device) or a.preset == 'cpu':
        p.error('CANN profiling requires explicit logical npu:N and a mixed/resident preset')
    if (a.steps != 1 or not 0 <= a.warmup <= 1 or not 1 <= a.windows_per_step <= 4
            or not 1 <= a.devices <= 3 or not 1 <= a.threads*a.workers <= 8
            or not 30 <= a.timeout_seconds <= 1800 or not 1 <= a.host_memory_gib <= 32
            or not 200 <= a.storage_limit_mb <= a.storage_budget_mb <= 4096
            or a.device_memory_bytes <= 0 or a.diagnostics):
        p.error('invalid bounded profile configuration; require explicit memory cap and no tensor diagnostics')
    root, build = Path(__file__).resolve().parents[1], a.build_dir.resolve()
    source, dirty = source_state(root)
    if dirty:
        p.error('profile requires a clean immutable source')
    record, binary = check_build(root, build, a.preset)
    if a.native_binary is not None and a.native_binary.resolve() != binary:
        p.error('native binary must be the recorded build consumer')
    a.native_binary = binary
    packet = validate_arguments(p, a)
    validate_resident_options(a)
    msprof = shutil.which('msprof')
    if not a.prepare_only and msprof is None:
        p.error('msprof unavailable in the active CANN environment')
    loader = subprocess.check_output(['ldd', str(binary)], text=True)
    if any(x in loader.lower() for x in ('not found', 'libtorch_python', 'libpython', '/stub/', '/stubs/', '/simulator/')):
        p.error('invalid standalone loader closure')
    out = a.output_dir = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    _, path, packet_text, native = prepare(packet, a)
    replace_text(out/'loader.txt', loader)
    command = [msprof or 'msprof', '--output='+str(out/'raw'), '--runtime-api=on', '--task-time=l1',
               '--aicpu=on', f'--storage-limit={a.storage_limit_mb}MB', *native]
    report = dict(schema='tide-complete-flow-profile-v1', state='prepared' if a.prepare_only else 'running',
        source=source, dirty=dirty, tracking='off; project-owned records', build=record,
        workload_sha256=packet['sha256'], packet_json_sha256=digest(a.packet), native_input_sha256=digest(path),
        binary_sha256=digest(binary), loader_sha256=digest(out/'loader.txt'),
        command=command, native_command=native, executions=[],
        budget=dict(seconds=a.timeout_seconds, host_memory_bytes=a.host_memory_gib*2**30,
                    storage_bytes=a.storage_budget_mb*2**20, devices=a.devices),
        scope='complete standalone consumer, fresh seeded inputs/weights, continuous windows; collection only, not throughput',
        config={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        load_before=list(os.getloadavg()), physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    write_json(out/'result.json', report)
    if a.prepare_only:
        print(json.dumps(dict(state='prepared', command=command)));return
    deadline = time.monotonic()+a.timeout_seconds

    def execute(argv, name):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('total profile budget exhausted')
        audit = dict(command=argv);report['executions'].append(audit)
        write_json(out/'result.json', report)
        with (out/name).open('x') as log:
            code = run_child(argv, cwd=out, env=dict(os.environ), log=log,
                affinity=sorted(os.sched_getaffinity(0)), timeout=remaining,
                memory_budget=a.host_memory_gib*2**30, audit=audit,
                storage_path=out, storage_budget=a.storage_budget_mb*2**20)
        audit['log_sha256'] = digest(out/name)
        if code:
            raise RuntimeError(name+' exited '+str(code))

    try:
        execute(command, 'profile.log')
        result = json.loads((out/'consumer/result.json').read_text())
        check_result(result, packet, a, (out/'profile.log').read_text())
        for i, session in enumerate(sorted((out/'raw').glob('PROF_*'))):
            if not list(session.rglob('op_summary_*.csv')):
                execute([msprof, '--export=on', '--output='+str(session)], f'export-{i}.log')
        report.update(collect_profiles(out, a.preset))
        if (source_state(root) != (source, dirty) or check_build(root, build, a.preset)[0] != record
                or digest(a.packet) != report['packet_json_sha256'] or path.read_text() != packet_text):
            raise RuntimeError('source/build/input changed during profiling')
        report.update(state='passed', native_result_sha256=digest(out/'consumer/result.json'),
                      memory=result['memory'], memory_admission=result['memory_admission'], runtime=result['runtime'])
    except BaseException as error:
        report.update(state='failed', error=repr(error));raise
    finally:
        report['load_after'] = list(os.getloadavg())
        write_json(out/'result.json', report)
    print(json.dumps(dict(state=report['state'], device_queue_tasks=report['device_queue_tasks'])))


if __name__ == '__main__':
    main()
