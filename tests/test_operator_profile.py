"""Timers must be optional, thread-safe, and leave the full-state gate intact."""
import json
import subprocess
import pytest
from test_pdg_scale import binary, topology


def test_exclusive_nested_thread_timers():
    result = subprocess.run([str(binary().with_name('tidegraph-profile-check'))],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'PROFILE passed' in result.stdout


@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('packing', ['exact', 'single'])
def test_profiling_preserves_outputs_state_and_work(dtype, workers, packing, tmp_path):
    graph = tmp_path/'graph.txt'; topology(graph)
    runs = []
    for enabled in (0, 1):
        out = tmp_path/str(enabled)
        argv = [str(binary()), '--device', 'cpu', '--dtype', str(dtype).split('.')[-1],
                '--topology', str(graph), '--width', '8', '--batch', '4', '--vocab', '17',
                '--steps', '4', '--warmup', '1', '--workers', str(workers), '--packed', '1',
                '--work-count', '1', '--check', '1', '--attention-packing', packing,
                '--operator-profile', str(enabled), '--parallel-regions', '1', '--compact-events', '1',
                '--run-id', 'profile', '--output-dir', str(out)]
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        assert 'CHECK passed' in result.stdout
        runs.append([json.loads(line)['metrics'] for line in (out/'metrics.jsonl').read_text().splitlines()])
    for off, on in zip(*runs):
        assert not any(k.startswith('detail/') for k in off)
        assert on['check/logits_sum'] == off['check/logits_sum']
        for key in off:
            if key.startswith(('op/', 'model/', 'work/')):
                assert on[key] == off[key], key
        assert on['detail/qkv_calls'] == on['op/qkv_calls']
        assert on['detail/output_calls'] == on['op/out_calls']
        assert on['detail/pooling_calls'] == on['op/out_rows']
        for key, value in on.items():
            if key.startswith('detail/'):
                assert value >= 0
                if key.endswith('_max_seconds'):
                    assert value <= on[key.replace('_max_seconds', '_worker_seconds')]


@pytest.mark.parametrize('extra', [['--grad', '1'], ['--packed', '0'], ['--emission', 'slot']])
def test_rejects_uncovered_profile(extra, tmp_path):
    graph = tmp_path/'graph.txt'; topology(graph)
    out = tmp_path/'out'
    result = subprocess.run([str(binary()), '--device', 'cpu', '--topology', str(graph),
                             '--run-id', 'bad', '--output-dir', str(out),
                             '--operator-profile', '1', *extra], capture_output=True)
    assert result.returncode != 0 and not out.exists()
