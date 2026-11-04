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


def test_resume_uses_original_binary_after_default_build_changes(tmp_path,monkeypatch):
    from tidegraph.lm import llama_adapter as adapter
    from tidegraph.lm.catalog import model
    from tidegraph.experiment.records import write_json,digest,fingerprint
    c=manifest('llama')['cases'][0]['config']
    root=tmp_path/'experiment';root.mkdir();out=root/'run';out.mkdir()
    binary=root/'old-binary';binary.write_bytes(b'original executable')
    library=root/'old-library';library.write_bytes(b'original shared library')
    weights=root/'weights';weights.write_bytes(b'fixed model')
    build=dict(state='passed',binary=str(binary),binary_sha256=digest(binary),shared_libraries={str(library):digest(library)})
    conversion=dict(gguf=str(weights),bytes=weights.stat().st_size,mtime_ns=weights.stat().st_mtime_ns,
                    model_manifest_id=fingerprint(model(c['model'])))
    recorded=dict(build=build,conversion=conversion)
    engine=root/'engines/llama';engine.mkdir(parents=True)
    write_json(engine/'ready.json',{'state':'failed','binary':str(root/'different-default-binary')})
    for name in ('state.bin','tokens.bin','expected-state.bin','expected-logits.bin'):(out/name).write_bytes(name.encode())
    write_json(out/'result.json',dict(configuration=c,runtime=recorded,validation={'checkpoint':'awaiting_fresh_process'},
        checkpoint_hashes={p.name:digest(p) for p in out.glob('*.bin')}))
    def invoke(command,**kwargs):
        assert command[0]==str(binary) and command[-1]=='restore'
        assert kwargs['cwd']==out/'engine-work'
        for field in ('state','logits'):(out/('restored-'+field+'.bin')).write_bytes((out/('expected-'+field+'.bin')).read_bytes())
    monkeypatch.setattr(adapter.subprocess,'run',invoke)
    assert adapter.resume(out,root/'models')['state']=='passed'
    library.write_bytes(b'changed')
    assert adapter.resume(out,root/'models')['state']=='failed'
    with pytest.raises(ValueError,match='library identity changed'):adapter.artifacts(c,root/'models',recorded)


def test_progress_keeps_completed_stages_when_timeout_interrupts_log(tmp_path):
    from tidegraph.lm.llama_adapter import progress_events
    path=tmp_path/'run.log'
    path.write_text('ordinary engine log\ntide-lm-progress {"phase":"warmup","stage":"prefill","completed":64}\n'
                    'tide-lm-progress {"phase":"measurement"')
    events=progress_events(path)
    assert events[0]['completed']==64 and events[0]['phase']=='warmup'
    assert events[1]==dict(phase='log',stage='incomplete',line=3)
