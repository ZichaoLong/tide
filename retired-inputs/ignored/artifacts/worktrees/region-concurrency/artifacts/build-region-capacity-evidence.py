"""Read-only snapshot of root-owned regional qualifications/capacity/performance."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
WORK = Path(__file__).resolve().parents[1]
OUT = WORK/'docs/evidence/region-capacity-20261118'
JOBS = ['region-placement-npu-serial-v1', 'region-placement-npu-concurrent-v1',
        'capacity-region-9b-npu-v1', 'capacity-region-17b-npu-v1', 'region-placement-perf-npu-v1']
GIB = 2**30


def atomic_text(path, text):
    fd, staging = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(staging, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify_source(folder, revision):
    paths = ['src/tide', 'scripts/check_placement.py', 'scripts/job.py',
             'configs/hb-multiscale-x-settle.json', 'configs/hb-multiscale-x-timed.json',
             'configs/scale/hb-9b.json', 'configs/scale/hb-17b.json']
    records = subprocess.check_output(['git', 'ls-tree', '-r', revision, *paths], cwd=ROOT, text=True).splitlines()
    matched, mismatches = [], []
    for record in records:
        metadata, name = record.split('\t'); _, _, oid = metadata.split()
        expected = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        actual = (folder/'source'/name).read_bytes()
        if actual != expected:
            mismatches.append(name)
        matched.append(dict(path=name, git_blob=oid, sha256=hashlib.sha256(actual).hexdigest()))
    lock = read(folder/'source/docs/upstream.json')
    math_mismatches = [name for name, expected in lock['files'].items()
                       if hashlib.sha256((folder/'source'/lock['snapshot']/name).read_bytes()).hexdigest() != expected]
    return dict(state='passed' if not mismatches and not math_mismatches else 'failed', files=matched,
                mismatches=mismatches, math=dict(revision=lock['revision'], files=len(lock['files']), mismatches=math_mismatches))


def peaks(updates):
    result = {}
    for update in updates:
        for device, values in update.get('peak_device_memory', {}).items():
            row = result.setdefault(device, dict(allocated_bytes=0, reserved_bytes=0))
            for key in row:
                row[key] = max(row[key], values[key])
    return result


evidence = dict(schema='tide-region-capacity-evidence-1', generated_at=datetime.now(timezone.utc).isoformat(),
    mathematical_revision='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), jobs={}, qualification=[], capacity=[], performance=[],
    partial_or_failed_results=[],
    cpu_checks=dict(source='4f7ea32a878451bcc9ac9511e4bf9904e442d36d', distinct_passed=64,
        log_scope='development session tool output; reproducible committed tests, not a detached hardware qualification',
        batches=[dict(command='pytest -q tests/test_region_concurrency.py tests/test_region_placement.py', passed=33,
                      note='before the subsequent public-device-context test was added'),
                 dict(command='pytest -q tests/test_region_concurrency.py -k public_device', passed=1),
                 dict(command='pytest -q tests/test_placement.py tests/test_region_capacity.py tests/test_capacity_placement.py', passed=30)],
        environment='private .venv-instances, TORCH_DEVICE_BACKEND_AUTOLOAD=0, affinity80-87, OMP/MKL/OpenBLAS limits4; pytest tiny ATen threads1'),
    boundaries=dict(qualification='same-schedule reference/placed, gradients, three updates, split schedule and real restored update; qualification timings are not performance samples',
        performance='one physical device / two serial devices / two per-device workers; all retain two identical logical shards; 6 real continued updates, cold index 0 separate and indices 1..5 as five non-IID samples',
        timing='complete training includes zero_grad, preparation, forward/readout/necessary Timed flush/loss, backward, host-stage transport, AdamW, detach, all participating devices sync and intervening persisted stage-record IO',
        exclusions='CPU construction, hashes, placement, optimizer/input setup, record_work replay, serial operator audit, finite/storage/update scans',
        memory='per-step max_memory_allocated/max_memory_reserved, despite short field names; reserved may retain prior audit caches; supervisor RSS is sampled process-tree sum, separate from process ru_maxrss',
        precision='FP32 parameters/gradients/master/AdamW; BF16 operator policy and KV payload; controls/SSD FP32, logical times int64',
        instrumentation='per_device operator witness explicitly runs serial_region_audit; record_work runs requested scheduler; unknown caller dispatch/function modes are rejected, hooks/profilers do not propagate',
        claim='no peer DMA/HCCL/replicas/CUDA hardware claim; finite deviations retained with original tolerances; later support/work differences prevent strict equivalent-trajectory speedup'))

for job in JOBS:
    folder = ROOT/'artifacts'/job
    launch = read(folder/'launch.json')
    if launch is None:
        evidence['jobs'][job] = dict(state='not_submitted')
        continue
    records = {}
    for name in ['launch.json', 'status.json', 'queue.json', 'environment.json', 'children.json', 'programme.json', 'dependency.json']:
        path = folder/name
        if path.exists():
            records[name] = dict(identity=identity(path), value=read(path))
    status = read(folder/'status.json') or {}
    evidence['jobs'][job] = dict(state=status.get('state', 'unknown'), source=launch['source'], records=records,
                               source_verification=verify_source(folder, launch['source']),
                               logs=[identity(path) for path in sorted(folder.glob('*.log'))])
    result_files = sorted(folder.glob('*.json')) if 'perf' in job else [folder/'fp32.json', folder/'bf16.json']
    for path in result_files:
        data = read(path)
        if not data or 'observations' not in data:
            continue
        if data.get('state') != 'completed':
            evidence['partial_or_failed_results'].append(dict(job=job, result=identity(path), document=data))
            continue
        for row in data['observations']:
            item = dict(job=job, result=identity(path), source=data['source'], arguments=data['arguments'],
                        physical_devices=data.get('physical_devices'), cpu_affinity=data.get('cpu_affinity'),
                        torch=data.get('torch'), observation={key:value for key,value in row.items() if key != 'progress'})
            if job.startswith('region-placement-npu-'):
                evidence['qualification'].append(item)
            elif job.startswith('capacity-'):
                item['maximum_timed_peaks'] = peaks(row['updates'])
                evidence['capacity'].append(item)
            else:
                item['identity'] = path.stem
                item['cold_update_seconds'] = row['updates'][0]['complete_seconds']
                continued = row['updates'][1:]
                item['continued_indices'] = [value['index'] for value in continued]
                item['continued_seconds'] = [value['complete_seconds'] for value in continued]
                item['continued_median_seconds'] = statistics.median(item['continued_seconds'])
                item['continued_host_gap_seconds'] = [value['between_phase_host_seconds'] for value in continued]
                item['continued_median_host_gap_seconds'] = statistics.median(item['continued_host_gap_seconds'])
                item['continued_input_positions_per_second'] = row['batch']*row['length']/item['continued_median_seconds']
                item['maximum_timed_peaks'] = peaks(row['updates'])
                evidence['performance'].append(item)

for item in evidence['performance']:
    identity_prefix = item['identity'].split('-single')[0] if item['identity'].endswith('-single') else item['identity'].rsplit('-dual_',1)[0]
    base = next((r for r in evidence['performance'] if r['identity'] == identity_prefix+'-single'), None)
    if base is None:
        continue
    actual, reference = item['observation'], base['observation']
    same_work = [a['work'] == b['work'] for a,b in zip(actual['updates'], reference['updates'])]
    item['single_device_comparison'] = dict(
        initial_parameter_sha256_equal=actual['initial_parameter_sha256'] == reference['initial_parameter_sha256'],
        input_sha256_equal=actual['input_sha256'] == reference['input_sha256'],
        region_partitions_equal=actual['placement']['region_partitions'] == reference['placement']['region_partitions'],
        work_equal_per_update=same_work,
        time_ratio_to_single=item['continued_median_seconds']/base['continued_median_seconds'],
        scope='paired initial data and owner layout; actual workload equality is checked, full later numerical trajectory equality is not established')
    serial = next((r for r in evidence['performance'] if r['identity'] == identity_prefix+'-dual_serial'), None)
    if serial is not None:
        item['dual_serial_comparison'] = dict(
            work_equal_per_update=[a['work'] == b['work'] for a,b in zip(actual['updates'], serial['observation']['updates'])],
            time_ratio_to_dual_serial=item['continued_median_seconds']/serial['continued_median_seconds'],
            scope='equality of aggregate work counts does not establish the same active identities or numerical trajectory')

evidence['state'] = ('reviewed_terminal' if all(job['state'] in {'passed','failed','stopped','timeout'}
                                              for job in evidence['jobs'].values()) else 'partial_snapshot')
atomic_text(OUT.with_suffix('.json'), json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(state=evidence['state'], qualification=len(evidence['qualification']), capacity=len(evidence['capacity']),
                     performance=len(evidence['performance']), output=str(OUT.with_suffix('.json')))))
