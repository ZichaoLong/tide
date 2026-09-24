import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import pytest


def binary():
    import _tide_native
    return Path(_tide_native.__file__).with_name('tidegraph-scale-bench')


def topology(path):
    # Independent small four-block graph: hubs, one local region and parallel edges.
    edges = []
    for src, dst in [(0, 0), (4, 4), (0, 4), (4, 0)]:
        edges += [(src, dst+i) for i in range(4)]
        edges += [(src+i, dst) for i in range(1, 4)]
    edges += [(0, 3)]
    path.write_text('TIDE_PDG_SCALE_1\n4 2 2 1 1 2 '+str(len(edges))+'\n'+''.join(f'{a} {b}\n' for a, b in edges))
    return len(edges)


@pytest.mark.parametrize('grad', [0, 1])
@pytest.mark.parametrize('profile', [0, 1])
@pytest.mark.parametrize('optimized', [0, 1])
@pytest.mark.parametrize('memory', ['attention', 'add'])
def test_scale_full_state_and_schedule_parity(dtype, grad, profile, optimized, memory, tmp_path):
    graph = tmp_path/'graph.txt'; edges = topology(graph); out = tmp_path/'native'
    cmd = [str(binary()), '--device', 'cpu', '--dtype', str(dtype).split('.')[-1], '--topology', str(graph),
           '--width', '8', '--batch', '4', '--steps', '4', '--warmup', '1', '--vocab', '17', '--memory', memory,
           '--workers', '3', '--packed', '1', '--grad', str(grad), '--check', '1', '--profile', str(profile),
           '--head-workers', '3' if optimized else '1', '--parallel-regions', str(optimized),
           '--compact-events', str(optimized),
           '--run-id', 'test', '--output-dir', str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert 'CHECK passed' in run.stdout
    events = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(events) == 4
    for i, e in enumerate(events):
        m = e['metrics']
        assert e['step'] == e['sequence'] == i
        assert m['model/parameters'] == ((4*9 if memory == 'attention' else 0)+edges)*8**2+8*8+edges+1+2+2*17*8
        assert e['context']['memory'] == memory
        assert m['model/physical_edges'] == 2*edges+2
        assert m['check/logits_requires_grad'] == grad
        assert m['runtime/aten_threads'] == m['runtime/interop_threads'] == 1
        assert m['runtime/head_workers'] == (3 if optimized else 1)
        assert m['runtime/openblas_reported_threads'] >= -1
        assert m['work/source_rows'] >= m['work/candidate_events'] >= m['work/selected_events'] > 0
        assert 1 <= m['work/max_node_batch'] <= 4
        assert m['work/semantic_state_replays'] > 0 if grad else m['work/semantic_state_replays'] == 0
        assert m['perf/ms_per_sample_token'] == pytest.approx(m['perf/token_seconds']*1000/4)
        phases = ['events', 'update', 'select', 'full', 'commit', 'cleanup']
        if profile:
            times = [m['profile/'+phase+'_seconds'] for phase in phases]
            assert all(t >= 0 for t in times)
            assert sum(times) == pytest.approx(m['profile/tick_seconds'], abs=1e-8)
            assert 0 < m['profile/tick_seconds'] <= m['perf/advance_seconds']
            assert m['profile/input_seconds'] >= 0 and m['profile/head_seconds'] >= 0
            assert m['profile/input_seconds']+m['perf/advance_seconds']+m['profile/head_seconds'] == pytest.approx(m['perf/token_seconds'], abs=1e-8)
        else:
            assert not any(key.startswith('profile/') for key in m)
    before = (out/'metrics.jsonl').read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert (out/'metrics.jsonl').read_bytes() == before


def load_exporter():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('pdg_scale_topology', root/'scripts/pdg_scale_topology.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_csr_export_retains_parallel_edges_and_rejects_damage(tmp_path):
    module = load_exporter(); n = 4
    # Last data array gives physical IDs, deliberately different from CSR order.
    data = [n, n, 3, 0, 3, 3, 3, 3, 2, 2, 3, 2, 0, 1]
    encoded = struct.pack('<'+'q'*len(data), *data)
    for name in ('inputA', 'outputA', 'ioA', 'oiA'): (tmp_path/name).write_bytes(encoded)
    hashes = {name: hashlib.sha256(encoded).hexdigest() for name in ('inputA', 'outputA', 'ioA', 'oiA')}
    source = dict(graph_config=dict(base_num_or_hpnums=[0, 1, 1, 2], localnum=2),
                  selectnum=1, with_lead_point=True, graph_dir=str(tmp_path), graph_files_sha256=hashes)
    manifest = tmp_path/'input.json'; manifest.write_text(json.dumps(source))
    out = tmp_path/'topology.txt'; record = module.export(manifest, out)
    assert record['logical_edges'] == 12 and record['pdg_nodes'] == 9
    assert out.read_text().splitlines()[2:5] == ['0 2', '0 3', '0 2']
    with pytest.raises(ValueError, match='already exists'): module.export(manifest, out)
    (tmp_path/'inputA').write_bytes(encoded[:-1])
    with pytest.raises(ValueError, match='hash changed'): module.export(manifest, tmp_path/'other.txt')
    with pytest.raises(ValueError, match='truncated'): module.read_csr(tmp_path/'inputA', n)


@pytest.mark.parametrize('args', [[], ['--device', 'npu'], ['--device', 'cpu', '--dtype', 'float16'],
                                ['--device', 'cpu', '--width', '7'], ['--device', 'cpu', '--workers', '161'],
                                ['--device', 'cpu', '--profile', '2'],
                                ['--device', 'cpu', '--attention-packing', 'unknown'],
                                ['--device', 'cpu', '--memory', 'unknown'],
                                ['--device', 'cpu', '--memory', 'add', '--attention-packing', 'single'],
                                ['--device', 'cpu', '--parallel-regions', '2'],
                                ['--device', 'cpu', '--compact-events', '2'],
                                ['--device', 'cpu', '--head-workers', '0'],
                                ['--device', 'cpu', '--head-workers', '161'],
                                ['--device', 'cpu', '--head-workers', '160', '--threads', '2']])
def test_scale_rejects_invalid_cli(args, tmp_path):
    graph = tmp_path/'graph.txt'; topology(graph); out = tmp_path/'native'
    result = subprocess.run([str(binary()), '--run-id', 'bad', '--topology', str(graph), '--output-dir', str(out), *args], capture_output=True)
    assert result.returncode != 0 and not out.exists()
