from pathlib import Path
import json
from tidegraph.lm.queue import placeholder, asset_status
from tidegraph.lm.matrix import case
from tidegraph.lm.report import load


def test_dependency_failure_is_not_model_failure_or_throughput(tmp_path):
    placeholder(case('qwen15','cpu'),tmp_path,'dependency_failed','network')
    row=load(tmp_path)[0]
    assert row['execution_status']=='dependency_failed'
    assert row['numerical']=='not_checked' and row['output_tokens_per_s'] is None


def test_missing_asset_never_means_ready(tmp_path):
    assert asset_status('qwen05',tmp_path)=='waiting'


def test_check_failure_has_its_own_schema(tmp_path):
    placeholder(case('qwen15','cpu',kind='check'),tmp_path,'dependency_failed','network')
    assert json.loads((tmp_path/'check.json').read_text())['schema']=='tide-lm-check-v1'
    assert not load(tmp_path)
