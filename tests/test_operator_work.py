"""Work accounting must survive node scheduling and preserve the full state gate."""
import json
import subprocess
import pytest
from test_pdg_scale import binary, topology


def test_work_counts_match_shapes_and_schedules(dtype, tmp_path):
    graph = tmp_path / 'graph.txt'
    topology(graph)
    results = []
    for packed, workers, counted in [(0, 1, 1), (1, 1, 1), (1, 3, 1), (1, 3, 0)]:
        out = tmp_path / f'run-{packed}-{workers}-{counted}'
        command = [str(binary()), '--device', 'cpu', '--dtype', str(dtype).split('.')[-1],
                   '--topology', str(graph), '--width', '8', '--batch', '4', '--vocab', '17',
                   '--steps', '4', '--warmup', '1', '--work-count', str(counted),
                   '--packed', str(packed), '--workers', str(workers), '--check', '1',
                   '--parallel-regions', '1', '--compact-events', '1',
                   '--run-id', 'counts', '--output-dir', str(out)]
        run = subprocess.run(command, capture_output=True, text=True)
        assert run.returncode == 0, run.stdout+run.stderr
        results.append([json.loads(line)['metrics'] for line in (out/'metrics.jsonl').read_text().splitlines()])
    for step in range(4):
        serial, packed, parallel, off = [r[step] for r in results]
        assert not any(key.startswith('op/') for key in off)
        assert parallel['check/logits_sum'] == off['check/logits_sum']
        for m in (serial, packed, parallel):
            assert m['op/qkv_flops'] == 6*8**2*m['work/source_rows']
            assert m['op/out_flops'] == 2*8**2*m['work/candidate_events']
            assert m['op/head_flops'] == 2*4*8*17
            assert m['op/emit_flops'] == 2*8**2*m['op/emit_edge_rows']
            assert m['op/body_candidates'] == m['work/candidate_events']-m['work/readout_rows']
            assert m['op/body_selected'] == m['work/selected_events']-m['work/readout_rows']
            assert m['op/executed_score_elements'] == m['op/valid_score_elements']
            assert m['op/executed_attention_flops'] == m['op/valid_score_elements']*8
        for key in serial:
            if key.startswith('op/') and not key.endswith('_calls'):
                assert serial[key] == packed[key] == parallel[key], key


@pytest.mark.parametrize('args', [['--grad', '1'], ['--emission', 'slot']])
def test_work_count_rejects_uncovered_execution(args, tmp_path):
    graph = tmp_path/'graph.txt'; topology(graph)
    run = subprocess.run([str(binary()), '--device', 'cpu', '--topology', str(graph),
                          '--run-id', 'bad', '--output-dir', str(tmp_path/'out'),
                          '--work-count', '1', *args], capture_output=True)
    assert run.returncode != 0 and not (tmp_path/'out').exists()


def test_single_group_counts_only_add_padding_and_reduce_calls(dtype, tmp_path):
    graph = tmp_path/'graph.txt'; topology(graph); observations = {}
    for packing in ('exact', 'single'):
        out = tmp_path/packing
        cmd = [str(binary()), '--device', 'cpu', '--dtype', str(dtype).split('.')[-1],
               '--topology', str(graph), '--width', '8', '--batch', '4', '--vocab', '17',
               '--steps', '4', '--warmup', '1', '--workers', '3', '--packed', '1',
               '--work-count', '1', '--check', '1', '--attention-packing', packing,
               '--run-id', packing, '--output-dir', str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        events = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        assert all(e['context']['attention_packing'] == packing for e in events)
        observations[packing] = [e['metrics'] for e in events]
    changed = False
    for exact, single in zip(observations['exact'], observations['single']):
        assert single['op/attention_calls'] == single['op/qkv_calls']
        assert single['op/attention_calls'] <= exact['op/attention_calls']
        assert single['op/executed_score_elements'] >= exact['op/executed_score_elements']
        changed |= single['op/attention_calls'] < exact['op/attention_calls']
        assert single['check/logits_sum'] == pytest.approx(exact['check/logits_sum'], abs=1e-6, rel=1e-5)
        varied = {'op/attention_calls', 'op/executed_score_elements', 'op/executed_attention_flops', 'op/executed_matmul_flops'}
        for k in exact:
            if k.startswith('op/') and k not in varied:
                assert single[k] == exact[k], k
    assert changed
