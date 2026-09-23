"""Real process boundaries, existing schemas and exact fixed training trajectories."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph.compare import equivalent


@pytest.mark.parametrize('scope',['single','application','native'])
@pytest.mark.parametrize('kind',['sgd','adamw'])
def test_save_exit_fresh_process_load_and_continue(dtype,scope,kind,tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path/('values.tidenck' if scope == 'native' else 'continuation.pt')
    pids, records, metadata = [], {}, []
    for phase in ('baseline','prefix','suffix'):
        output = tmp_path/f'{phase}.pt'
        command = [sys.executable,str(root/'tests/checkpoint_process_worker.py'),'--device','cpu',
                   '--dtype',str(dtype).removeprefix('torch.'),'--scope',scope,'--kind',kind,
                   '--phase',phase,'--checkpoint',str(checkpoint),'--output',str(output)]
        env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        with (tmp_path/f'{phase}.log').open('w') as log:
            result = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        assert result.returncode == 0, (tmp_path/f'{phase}.log').read_text()
        info = json.loads(output.with_suffix('.json').read_text()); metadata.append(info); pids.append(info['pid'])
        assert info['exit_code'] == 0 and info['threads'] == 1
        assert info['output_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
        records[phase] = torch.load(output,weights_only=True)
    assert len(set(pids)) == 3 and os.getpid() not in pids
    assert metadata[1]['checkpoint_sha256'] == metadata[2]['checkpoint_sha256']
    assert len({x['binary_sha256'] for x in metadata}) == 1
    equivalent(records['baseline'],records['prefix']+records['suffix'])
