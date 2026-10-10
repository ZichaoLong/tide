"""A profiler's successful exit must not certify stale or partial native work."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import profile_flow_support as support
from source_identity import digest
from run_execution_flow import make_parser


def fixture():
    args = SimpleNamespace(family='pdg', device='npu:0', preset='mixed-c', schedule='prefill',
                           dtype='float32', training=True, optimizer='adamw', steps=1, warmup=1,
                           windows_per_step=2, devices=1)
    packet = dict(sha256='packet', workload=dict(batch=4))
    result = dict(state='passed', implementation='libtorch', workload_sha256='packet', family='pdg',
                  training=True, optimizer='adamw', measured_steps=1, warmup_steps=1, windows_per_step=2,
                  batch_execution=dict(logical_batch=4), seconds=[1.], outputs=[8], losses=[2.], statistics=[{}],
                  runtime=dict(device='npu:0', backend='npu', preset='mixed-c', schedule='prefill', dtype='float32'),
                  memory=dict(phases=[dict(phase='measured')]),
                  memory_admission=dict(allocator_within_estimate=True, observed_peak_growth_bytes=[1024]))
    return result, packet, args


def test_native_result_calibration_and_boundaries():
    result, packet, args = fixture()
    support.check_result(result, packet, args, '')
    for field, value in [('state', 'failed'), ('workload_sha256', 'changed'), ('losses', []),
                         ('outputs', [0]), ('losses', [float('nan')]), ('windows_per_step', 1),
                         ('optimizer', 'sgd'), ('implementation', 'native')]:
        bad = dict(result, **{field: value})
        with pytest.raises(ValueError):
            support.check_result(bad, packet, args, '')


@pytest.mark.parametrize('change', ['device', 'dtype', 'preset', 'uncalibrated', 'missing-card'])
def test_rejects_changed_runtime_or_unaccepted_memory(change):
    result, packet, args = fixture()
    if change in result['runtime']:
        result['runtime'][change] = 'changed'
    elif change == 'uncalibrated':
        result['memory_admission']['allocator_within_estimate'] = False
    else:
        args.devices = 2
    with pytest.raises(ValueError):
        support.check_result(result, packet, args, '')


@pytest.mark.parametrize('log', ['An exception has occurred in process App',
                                'fall back to run on the CPU', 'npu_cpu_fallback'])
def test_profiler_zero_exit_does_not_hide_application_failure(log):
    result, packet, args = fixture()
    with pytest.raises(ValueError):
        support.check_result(result, packet, args, log)


def test_requires_real_operator_trace_and_resident_queue(tmp_path):
    with pytest.raises(ValueError, match='no real device'):
        support.collect_profiles(tmp_path, 'mixed-c')
    out = tmp_path/'raw/PROF_test/device_0';out.mkdir(parents=True)
    with (out/'api_statistic_1.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['Level','API Name','Time(us)','Count']);w.writerow(['runtime','Launch','2','1'])
    path = out/'op_summary_1.csv'
    def write(kind):
        with path.open('w') as f:
            w=csv.writer(f);w.writerow(['Task Type','OP Type','Task Duration(us)','Input Data Types'])
            w.writerow(['AI_CORE',kind,'3','FLOAT'])
    write('MatMul')
    assert support.collect_profiles(tmp_path, 'mixed-c')['summaries'][0]['operators'] == 1
    with pytest.raises(ValueError, match='no device queue'):
        support.collect_profiles(tmp_path, 'resident')
    write('tide_queue_propose')
    assert support.collect_profiles(tmp_path, 'resident')['device_queue_tasks'] == 1


def test_rejects_stale_consumer_or_wrong_runtime(tmp_path, monkeypatch):
    root, build = tmp_path/'source', tmp_path/'build'
    src = root/'tools/online_bench';src.mkdir(parents=True)
    (src/'main.cpp').write_text('int main() {}')
    binary = build/'consumer/tidegraph-online-bench';binary.parent.mkdir(parents=True);binary.write_bytes(b'native')
    record = dict(state='passed', core=dict(backend='npu', npu_runtime='standalone', cpp_source_sha256='core'),
                  binary_sha256={binary.name:digest(binary)}, consumer_sources={'main.cpp':digest(src/'main.cpp')})
    path = build/'result.json';path.write_text(json.dumps(record))
    monkeypatch.setattr(support, 'source_hash', lambda _: 'core')
    assert support.check_build(root, build, 'mixed-a')[1] == binary
    with pytest.raises(ValueError, match='resident'):
        support.check_build(root, build, 'resident')
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='binary changed'):
        support.check_build(root, build, 'mixed-a')
    binary.write_bytes(b'native');(src/'main.cpp').write_text('int main() {return 1;}')
    with pytest.raises(ValueError, match='compiled sources differ'):
        support.check_build(root, build, 'mixed-a')
    bad = copy.deepcopy(record);bad['core']['npu_runtime'] = 'python';path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='standalone'):
        support.check_build(root, build, 'mixed-a')


def test_profile_cli_keeps_standalone_identity_and_explicit_device():
    p = make_parser(standalone=True)
    base = ['--packet','input.json','--output-dir','output','--preset','mixed-c',
            '--schedule','prefill','--family','pdg']
    with pytest.raises(SystemExit):
        p.parse_args(base)
    assert p.parse_args(base+['--device','npu:0']).implementation == 'libtorch'
    with pytest.raises(SystemExit):
        p.parse_args(base+['--device','npu:0','--implementation','native'])


@pytest.mark.parametrize('mode', ['prepare', 'collected', 'missing-trace'])
def test_complete_entry_lifecycle_with_synthetic_collector(tmp_path, monkeypatch, mode):
    # This tests orchestration/record acceptance only, never vendor execution.
    import profile_execution_flow as entry
    from flow_topology import ranked_graph
    from flow_protocol import make_continuous_packet
    packet = make_continuous_packet(graph=ranked_graph(layers=2, region_width=2, fanout=2, local_span=2),
                                    width=4, batch=2, tokens=2, vocab=7)
    source = tmp_path/'packet.json';source.write_text(json.dumps(packet))
    binary = tmp_path/'native';binary.write_bytes(b'synthetic collector fixture')
    out = tmp_path/'profile'
    monkeypatch.setattr(entry, 'source_state', lambda _: ('fixed', ''))
    monkeypatch.setattr(entry, 'check_build', lambda *args: ({'fixture': 'synthetic'}, binary))
    monkeypatch.setattr(entry.shutil, 'which', lambda _: '/synthetic/msprof')
    monkeypatch.setattr(entry.subprocess, 'check_output', lambda *args, **kwargs: 'libtorch.so => /runtime/libtorch.so')
    calls = []
    def collect(command, **kwargs):
        calls.append(command)
        assert mode != 'prepare', 'prepare-only executed a workload'
        result, _, _ = fixture()
        result.update(workload_sha256=packet['sha256'], batch_execution=dict(logical_batch=2))
        (out/'consumer').mkdir();(out/'consumer/result.json').write_text(json.dumps(result))
        if mode == 'collected':
            raw = out/'raw/PROF_test/device_0';raw.mkdir(parents=True)
            (raw/'api_statistic_1.csv').write_text('Level,API Name,Time(us),Count\nruntime,Launch,2,1\n')
            (raw/'op_summary_1.csv').write_text('Task Type,OP Type,Task Duration(us),Input Data Types\nAI_CORE,MatMul,3,FLOAT\n')
        return 0
    monkeypatch.setattr(entry, 'run_child', collect)
    argv = ['profile_execution_flow', '--packet', str(source), '--output-dir', str(out),
            '--build-dir', str(tmp_path/'build'), '--device', 'npu:0', '--family', 'pdg',
            '--preset', 'mixed-c', '--schedule', 'prefill', '--training', '--optimizer', 'adamw',
            '--device-memory-bytes', '1073741824']
    if mode == 'prepare':argv.append('--prepare-only')
    monkeypatch.setattr(sys, 'argv', argv)
    if mode == 'missing-trace':
        with pytest.raises(ValueError, match='no real device'):
            entry.main()
    else:
        entry.main()
    report = json.loads((out/'result.json').read_text())
    assert report['state'] == {'prepare':'prepared', 'collected':'passed', 'missing-trace':'failed'}[mode]
    assert len(calls) == (0 if mode == 'prepare' else 1)
    assert (out/'topology.txt').is_file() and report['native_input_sha256'] == digest(out/'topology.txt')
