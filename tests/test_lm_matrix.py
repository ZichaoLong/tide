import json
from pathlib import Path
import subprocess
import sys
import pytest
from tidegraph.lm.matrix import manifest
from tidegraph.lm.config import resolve
from tidegraph.lm.report import render


def test_finite_matrix_identities_budget_and_authorized_counts():
    m=manifest();cases=m['cases']
    assert len(cases)==80  # 71 configured performance/observations + nine small correctness anchors.
    assert len({x['name'] for x in cases})==len(cases)
    assert sum(x['kind'] in {'main','stage2'} for x in cases)==44
    assert sum(x['kind']=='train' for x in cases)==14
    assert len(m['conditional_unqueued'])==2
    for item in cases:
        r=resolve(item['config'])
        assert r['budget']['per_device_estimate_bytes']>r['budget']['weight_bytes']/item['config']['execution']['devices']
    for backend in ('cpu','npu'):
        for key in ('qwen15','olmoe'):
            assert {x['config']['task']['batch'] for x in cases if x['kind']=='main' and x['config']['model']==key and x['config']['execution']['backend']==backend}>={128,512}


@pytest.mark.parametrize('command',['catalog','matrix','plan','run','check','training-check','resume','results','queue'])
def test_help_without_torch(command):
    source=str(Path(__file__).resolve().parents[1]/'python')
    code='import sys; sys.path.insert(0,sys.argv[1]); from tidegraph.__main__ import main; main(sys.argv[2:])'
    result=subprocess.run([sys.executable,'-S','-c',code,source,'lm',command,'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_lm_html_uses_explicit_token_metrics_and_escapes_untrusted_name():
    text=render([dict(name='<script>bad</script>',model='qwen15',platform='cpu',parameters=10,output_tokens_per_s=20)],'html')
    assert 'output tokens/s' in text and '<script>bad</script>' not in text
    assert 'External dense / MoE' in text


def test_three_device_budget_is_uneven_and_does_not_divide_workspace():
    c=dict(schema='tide-lm-experiment-v1',model='qwen15',execution=dict(backend='npu',devices=3,placement='layer_split'))
    b=resolve(c)['budget']
    assert [x['layers'] for x in b['per_device']]==[10,9,9]
    assert min(x['estimated_bytes'] for x in b['per_device'])>b['workspace_bytes']
