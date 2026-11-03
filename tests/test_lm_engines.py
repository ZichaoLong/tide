import pytest
from tidegraph.lm.engine_matrix import manifest
from tidegraph.lm.selection import validate
from tidegraph.lm.config import resolve


def test_engine_plan_keeps_real_implementation_and_placement_names():
    value=validate(manifest())
    assert len(value['cases'])==14
    runs=[x for x in value['cases'] if x['action']=='run']
    assert len(runs)==10
    assert sum(x['config']['execution']['engine']=='llama_cpp' for x in runs)==4
    for x in value['cases']:
        e=x['config']['execution']
        if e['backend']=='npu':
            assert e['engine']=='transformers' and e['placement']=='layer_split'
            assert e['devices']==(4 if x['config']['model']=='qwenmoe' else 2)


@pytest.mark.parametrize('field,value',[('dtype','float32'),('backend','npu'),('attention','sdpa')])
def test_llama_never_silently_changes_precision_backend_or_operator(field,value):
    c=manifest('llama')['cases'][0]['config'];c['execution'][field]=value
    with pytest.raises(ValueError,match='llama_cpp'):resolve(c)


def test_missing_or_changed_engine_artifacts_are_explicit_failure(tmp_path):
    from tidegraph.lm.llama_adapter import artifacts
    c=manifest('llama')['cases'][0]['config']
    with pytest.raises(FileNotFoundError):artifacts(c,tmp_path/'models')


def test_llama_material_uses_exact_existing_distinct_token_generator(tmp_path):
    from tidegraph.lm.llama_adapter import material
    from tidegraph.lm.inputs import create
    from tidegraph.lm.catalog import model
    from types import SimpleNamespace
    c=manifest('llama')['cases'][0]['config']
    ids,mask,meta=material(c,tmp_path)
    other=create(c['task'],model(c['model'])['config'],SimpleNamespace(pad_token_id=0,eos_token_id=0))
    assert meta==other[2] and meta['distinct_samples']==2
    lines=(tmp_path/'input.txt').read_text().splitlines()
    assert list(map(int,lines[0].split()))==[2,16,4,16,16,0,1,0]
    assert list(map(int,lines[1].split()))==ids[0].tolist()
