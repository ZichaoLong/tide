"""Diagnostic counters must not change the grad-enabled computation."""
import json
import subprocess
import pytest
from test_pdg_scale import binary, topology


@pytest.mark.parametrize('memory', ['add', 'attention'])
@pytest.mark.parametrize('policy', ['replay', 'batched'])
def test_grad_forward_profile_preserves_values_and_roots(dtype, memory, policy, tmp_path):
    graph = tmp_path/'graph.txt'
    topology(graph)
    runs = []
    for enabled in (0, 1):
        out = tmp_path/str(enabled)
        argv = [str(binary()), '--device', 'cpu', '--dtype', str(dtype).split('.')[-1],
                '--topology', str(graph), '--width', '8', '--batch', '4', '--vocab', '17',
                '--steps', '4', '--warmup', '1', '--workers', '3', '--packed', '1',
                '--grad', '1', '--full-autograd', policy, '--memory', memory,
                '--work-count', str(enabled), '--operator-profile', str(enabled),
                '--check', '1', '--parallel-regions', '1', '--compact-events', '1',
                '--run-id', 'grad-profile', '--output-dir', str(out)]
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        if policy == 'batched':
            assert 'CHECK gradient roots passed' in result.stdout
        runs.append([json.loads(line)['metrics'] for line in (out/'metrics.jsonl').read_text().splitlines()])
    for off, on in zip(*runs):
        assert on['check/logits_sum'] == off['check/logits_sum']
        assert on['check/logits_requires_grad'] == 1
        for key in off:
            if key.startswith(('model/', 'work/')):
                assert off[key] == on[key], key
        for part in ('aggregate', 'state', 'read'):
            assert on[f'op/{part}_replay_worker_ns'] > 0
        assert (on['op/full_replay_worker_ns'] > 0) == (policy == 'replay')
        assert on['detail/aggregate_worker_seconds'] > 0
        assert on['detail/read_worker_seconds'] > 0
        assert on['op/head_flops'] == 2*4*8*17
