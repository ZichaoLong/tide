"""Public native worker/transport controls preserve complete training semantics."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from test_online_consumer import packet
from test_online_consumer_npu import target
from online_consumer_support import observer, same
from flow_protocol import native_text
from tools.online_bench.host import run


# Directed interactions, not every worker/transport/family permutation.
CASES = [
    ('pdg', 'add', 'streaming', 2, False, True, 'sgd', 'float64', False),
    ('pdg', 'add', 'prefill', 3, True, False, 'adamw', 'float32', True),
    ('timed-dag', 'attention', 'prefill', 3, True, True, 'adamw', 'float32', False),
    ('settle', 'attention', 'streaming', 2, True, True, 'sgd', 'float64', True),
]


def compare(case, implementation, device, preset, tmp_path):
    family, memory, schedule, workers, packed_sources, batch_next, optimizer, dtype, clear = case
    if str(device) != 'cpu':
        dtype = 'float32'
    p = packet(memory, clear=clear)
    wanted, actual = [], []
    reference = run(p, family=family, implementation='python', device='cpu', dtype=dtype,
                    schedule='streaming', training=True, optimizer=optimizer, steps=2, warmup=0,
                    windows_per_step=2, diagnostics=True, observer=observer(wanted))
    policy = dict(workers=workers, packed_sources=packed_sources, batch_next=batch_next)
    if implementation == 'libtorch':
        binary = os.environ.get('TIDE_ONLINE_BINARY')
        if not binary:
            pytest.skip('standalone consumer not explicitly selected')
        path = tmp_path / 'topology.txt'
        path.write_text(native_text(p))
        out = tmp_path / 'consumer'
        command = [binary, '--packet=' + str(path), '--output-dir=' + str(out), '--device=' + str(device),
                   '--dtype=' + dtype, '--family=' + family, '--preset=' + preset, '--schedule=' + schedule,
                   '--workers=' + str(workers), '--training', '--optimizer=' + optimizer,
                   '--steps=2', '--warmup=0', '--windows-per-step=2', '--diagnostics']
        if packed_sources:
            command.append('--packed-sources')
        if batch_next:
            command.append('--batch-next')
        done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
        assert done.returncode == 0, done.stdout
        actual = [json.loads(line) for line in (out / 'diagnostics.jsonl').read_text().splitlines()]
        result = json.loads((out / 'result.json').read_text())
    else:
        library = os.environ.get('TIDE_BUILD_DIR')
        if not library:
            pytest.skip('native consumer not explicitly selected')
        result = run(p, family=family, implementation='native', device=device, dtype=dtype,
                     schedule=schedule, preset=preset, training=True, optimizer=optimizer, steps=2, warmup=0,
                     windows_per_step=2, diagnostics=True, observer=observer(actual), native_library=library, **policy)
    same(actual, wanted)
    assert result['host_execution'] == policy
    assert result['final_cut'] == reference['final_cut'] and result['outputs'] == reference['outputs']
    torch.testing.assert_close(torch.tensor(result['losses']), torch.tensor(reference['losses']), atol=1e-6, rtol=1e-5)
    (tmp_path / 'host-execution.json').write_text(json.dumps(result))


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('implementation', ['native', 'libtorch'])
def test_cpu_host_execution(case, implementation, tmp_path):
    compare(case, implementation, 'cpu', 'cpu', tmp_path)


@pytest.mark.parametrize('case,preset', list(zip(CASES, ['mixed-a', 'mixed-b', 'mixed-c', 'mixed-a'])))
@pytest.mark.parametrize('implementation', ['native', 'libtorch'])
def test_npu_host_execution(case, preset, implementation, tmp_path):
    compare(case, implementation, target(), preset, tmp_path)


def test_host_controls_refuse_before_model_construction(monkeypatch):
    from tools.online_bench import host
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported host options reached model construction')
    monkeypatch.setattr(host, 'runtime_for', forbidden)
    for option in [dict(workers=2), dict(packed_sources=True), dict(batch_next=True)]:
        for implementation, preset in [('python', 'cpu'), ('native', 'resident')]:
            with pytest.raises(ValueError, match='require an eager native consumer'):
                run(packet(), family='pdg', implementation=implementation, preset=preset, device='cpu', **option)
    for option in [dict(workers=0), dict(workers=True), dict(workers=1025), dict(packed_sources=1), dict(batch_next=1)]:
        with pytest.raises(ValueError, match='invalid host'):
            run(packet(), family='pdg', implementation='native', device='cpu', **option)
