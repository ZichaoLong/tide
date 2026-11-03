from copy import deepcopy
import pytest
from tidegraph.lm.recovery import manifest
from tidegraph.lm.selection import validate
from tidegraph.lm.matrix import manifest as original
from tidegraph.experiment.records import fingerprint


def test_remaining_plan_preserves_dependency_configs_and_bounds_new_work():
    value=validate(manifest());old={x['name']:x for x in original()['cases']}
    assert len(value['cases'])==43
    dependency=[x for x in value['cases'] if x['config']['model'] in {'olmoe','qwen7','qwenmoe'}]
    assert len(dependency)==32
    for x in dependency:assert x['config']==old[x['name']]['config']
    extended=[x for x in value['cases'] if x['name'].startswith('extended-')]
    assert len(extended)==4
    for x in extended:
        assert x['config']['measurement']['timeout_seconds']==7200
        assert x['config']['task']==old[x['name'][9:]]['config']['task']
    assert sum(len(manifest(x)['cases']) for x in ('ready','small','moe'))==43


@pytest.mark.parametrize('change',['path','identity','unexpanded','action'])
def test_selection_rejects_unsafe_or_ambiguous_manifest(change):
    value=manifest('ready')
    if change=='path':value['cases'][0]['name']='../bad'
    if change=='identity':value['identity']='bad'
    if change=='unexpanded':del value['cases'][0]['config']['execution']['threads']
    if change=='action':value['cases'][0]['action']='arbitrary-command'
    with pytest.raises(ValueError):validate(value)


def test_selected_queue_does_not_add_default_cases_or_recalibrate_threads(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tidegraph.lm import queue
    from tidegraph.experiment.records import write_json,read_json
    value=manifest('ready');value['cases']=[x for x in value['cases'] if x['name'].startswith('extended-')][:1]
    value['cases'][0]['config']['execution']['threads']=7
    value['identity']=fingerprint(value['cases']);path=tmp_path/'selection.json';write_json(path,value)
    monkeypatch.setattr(queue,'asset_status',lambda *_:'ready')
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=3))
    result=queue.execute(tmp_path,'cpu',selection=path,attempt='selected')
    assert len(result['results'])==1
    case=read_json(tmp_path/'runs/cpu-selected'/value['cases'][0]['name']/'case.json')
    assert case['config']['execution']['threads']==7


def test_prefix_diagnostics_compare_independent_tiny_trajectories():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from tidegraph.lm.diagnostics import full_prefix,comparison
    from tidegraph.lm.inference import generate
    torch.set_num_threads(2);torch.manual_seed(11)
    owner=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,attn_implementation='eager')).eval()
    c=deepcopy(manifest('ready')['cases'][0]['config']);c['task'].update(batch=2,input_tokens=8,output_tokens=3)
    c['execution'].update(prefill_chunk=4,devices=1)
    ids=torch.arange(16).reshape(2,8);mask=torch.ones_like(ids)
    a=full_prefix(owner,ids,mask,3,torch.device('cpu'))
    b=generate(owner,ids,mask,c,torch.device('cpu'),None,diagnostics=True)
    assert comparison(a,b)['numeric']=='passed'
    b['logits'][0]=b['logits'][0]+1
    result=comparison(a,b)
    assert result['numeric']=='failed' and result['first_numeric_difference']['field']=='logits.0'
    b['logits'][0][0,0]=float('nan')
    with pytest.raises(FloatingPointError):comparison(a,b)
