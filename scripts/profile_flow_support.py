"""Identity and acceptance boundaries for complete standalone CANN profiles."""
import csv
import json
import math
from pathlib import Path
from build_device_control import component_hash
from build_identity import source_hash
from source_identity import digest
from summarize_ascend_profile import summarize, operator_files
from verify_resident_target import check_consumer_sources


def check_build(root, build, preset):
    record = json.loads((build/'result.json').read_text())
    core = record['core']
    if (record.get('state') != 'passed' or core['backend'] != 'npu'
            or core.get('npu_runtime') != 'standalone' or core['cpp_source_sha256'] != source_hash(root)):
        raise ValueError('profile requires a matching standalone NPU core build')
    check_consumer_sources(root, record, 'online_bench')
    binary = build/'consumer/tidegraph-online-bench'
    if digest(binary) != record['binary_sha256'].get(binary.name):
        raise ValueError('profile consumer binary changed')
    if preset == 'resident':
        resident = record.get('resident') or {}
        if (resident.get('core') != core or resident.get('backend', 'npu') != 'npu'
                or resident.get('runtime', resident.get('npu_runtime')) != 'standalone'
                or resident.get('component_sha256') != component_hash(root)):
            raise ValueError('resident profile requires matching resident/core sources and runtime')
    return record, binary


def check_result(result, packet, args, log):
    # msprof can exit zero despite a failed/segfaulted application. Require the
    # independently published native result and its calibration, not a marker.
    if 'An exception has occurred in process App' in log:
        raise ValueError('profiler reported failed application termination')
    if any(s in log.lower() for s in ('fall back to run on the cpu', 'npu_cpu_fallback', 'fallback_to_cpu')):
        raise ValueError('unexpected host CPU fallback')
    runtime = result.get('runtime', {})
    if (result.get('state') != 'passed' or result.get('implementation') != 'libtorch'
            or result.get('workload_sha256') != packet['sha256'] or result.get('family') != args.family
            or result.get('training') != args.training
            or runtime.get('device') != args.device or runtime.get('backend') != 'npu'
            or runtime.get('preset') != args.preset or runtime.get('schedule') != args.schedule
            or runtime.get('dtype') != args.dtype):
        raise ValueError('native profile result state/workload/runtime mismatch')
    if (result.get('measured_steps') != args.steps or result.get('warmup_steps') != args.warmup
            or result.get('windows_per_step') != args.windows_per_step
            or result.get('batch_execution', {}).get('logical_batch') != packet['workload']['batch']):
        raise ValueError('native profile result changed run boundaries')
    if args.training and result.get('optimizer') != args.optimizer:
        raise ValueError('native profile result changed optimizer')
    for name in ('seconds', 'outputs', 'losses', 'statistics'):
        if len(result.get(name, [])) != args.steps:
            raise ValueError('incomplete native profile result: ' + name)
    if not any(result['outputs']) or any(x is None or not math.isfinite(x) for x in result['losses']):
        raise ValueError('native profile has no finite complete workload loss/output')
    admission = result.get('memory_admission', {})
    peaks = admission.get('observed_peak_growth_bytes', [])
    if (admission.get('allocator_within_estimate') is not True or len(peaks) != args.devices
            or any(type(x) is not int or x < 0 for x in peaks)
            or not result.get('memory', {}).get('phases')):
        raise ValueError('native profile lacks accepted per-device memory calibration')


def collect_profiles(out, preset):
    directories = sorted({p.parent for p in (out/'raw').rglob('op_summary_*.csv')})
    if not directories:
        raise ValueError('no real device operator records')
    summaries, queue_tasks = [], 0
    for directory in directories:
        summaries.append(dict(path=str(directory.relative_to(out)), **summarize(directory)))
        for path in operator_files(directory):
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    if 'tide_queue_propose' in ' '.join(str(x) for x in row.values()):
                        queue_tasks += 1
    if preset == 'resident' and not queue_tasks:
        raise ValueError('resident profile contains no device queue task')
    return dict(summaries=summaries, device_queue_tasks=queue_tasks,
                limits=['Full process includes setup/warmup and window boundaries; not a throughput measurement.',
                        'Allocator observations exclude untracked vendor/driver allocations.',
                        'Kernel presence alone does not prove device residency; inspect host APIs and raw timeline.',
                        'Overlapping device tasks and nested host APIs must not be added into wall time.'])
