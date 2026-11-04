"""Restart only unfinished work; retain strict failures and workload identity."""
import json
from pathlib import Path
from tidegraph.lm.retry import manifest,isolation_manifest
from tidegraph.lm.selection import validate
from tidegraph.lm.config import resolve


def audit():
    path=Path(__file__).resolve().parents[1]/'docs/evidence/llm-terminal-audit-20261104.json'
    return {x['name']:x for x in json.loads(path.read_text())['cases']}


def test_only_nineteen_dependencies_and_two_explicit_successors():
    value=validate(manifest());old=audit()
    assert len(value['cases'])==21
    assert {k:len(manifest(k)['cases']) for k in ('compute','llama','placement')}==dict(compute=11,llama=6,placement=4)
    unchanged=[x for x in value['cases'] if x['name'] in old]
    assert len(unchanged)==19
    assert {x['name'] for x in unchanged}=={n for n,x in old.items() if x['state']=='dependency_failed'}
    for x in unchanged:
        assert x['config']==old[x['name']]['configuration']
    assert not {x['name'] for x in value['cases']} & {n for n,x in old.items() if x['state']=='completed'}


def test_timeout_extension_keeps_workload_and_capacity_alternative_is_explicit():
    old=audit();new={x['name']:x for x in manifest()['cases']}
    cpu=new['extended-main-olmoe-cpu-b512-l512']['config']
    before=old['main-olmoe-cpu-b512-l512']['configuration']
    for section in ('task','execution','resources'):assert cpu[section]==before[section]
    assert cpu['measurement']=={**before['measurement'],'timeout_seconds':7200}
    npu=new['capacity-2-olmoe-npu-b512-l512']['config']
    before=old['main-olmoe-npu-b512-l512']['configuration']
    for section in ('task','measurement','resources'):assert npu[section]==before[section]
    assert npu['execution']=={**before['execution'],'devices':2,'placement':'layer_split'}
    assert npu['task']['batch']==512 and npu['execution']['prefill_chunk']==64
    assert resolve(npu)['budget']['per_device_estimate_bytes'] < 37*2**30
    assert resolve(before)['budget']['per_device_estimate_bytes'] > 63*2**30


def test_dispatch_explicit_manifest_never_requeues_completed_models(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from tidegraph.experiment.records import write_json,read_json
    path=Path(__file__).resolve().parents[1]/'scripts/dispatch_lm_remaining.py'
    spec=importlib.util.spec_from_file_location('lm_retry_dispatch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'recovery').mkdir()
    selection=tmp_path/'manifest.json';write_json(selection,manifest('compute'))
    calls=[]
    monkeypatch.setattr(module,'ready',lambda *_:True)
    monkeypatch.setattr(module.subprocess,'run',lambda command,**kw:calls.append(command))
    monkeypatch.setattr(sys,'argv',['dispatch',str(tmp_path),'--attempt','retry-test','--small-job','small-new',
        '--moe-job','moe-new','--manifest',str(selection)])
    assert module.main()==0
    assert len(calls)==5
    for group,expected in [('ready',{'olmoe'}),('small',{'qwen7'}),('moe',{'qwenmoe'})]:
        generated=validate(read_json(tmp_path/'recovery/retry-test'/(group+'.json')))
        assert {x['config']['model'] for x in generated['cases']}==expected


def test_site_adapter_pins_policy_before_invoking_scheduler(tmp_path):
    import subprocess
    import sys
    import time
    script=Path(__file__).resolve().parents[1]/'scripts/make_lm_npu_adapter.py'
    helper=tmp_path/'helper.py';helper.write_text('raise AssertionError("must not run")\n')
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(schema='tide-npu-sharing-v1',reason='explicit test grant',
        expires_unix=time.time()+60,cards={'9':dict(max_hbm_mib=8192,max_aicore_percent=5,
        required_free_mib=53248,processes=[dict(pid=1,start_ticks=1,uid=1003)])})))
    output=tmp_path/'adapter.py'
    subprocess.run([sys.executable,str(script),'--policy',str(policy),'--helper',str(helper),
                    '--output',str(output)],check=True,capture_output=True)
    compile(output.read_text(),str(output),'exec')
    policy.write_text('{}')
    result=subprocess.run([sys.executable,str(output),'--state-file',str(tmp_path/'allocation.json')],
                          capture_output=True,text=True)
    assert result.returncode!=0 and 'active sharing policy changed' in result.stderr


def test_isolation_round_only_unexecuted_and_two_timed_out_successors():
    path=Path(__file__).resolve().parents[1]/'docs/evidence/llm-restart-terminal-20261104.json'
    old={x['name']:x for x in json.loads(path.read_text())['cases']}
    value=validate(isolation_manifest())
    assert len(value['cases'])==15
    assert {k:len(isolation_manifest(k)['cases']) for k in ('compute','llama','placement')}==dict(compute=10,llama=2,placement=3)
    for item in value['cases']:
        c=item['config']
        if c['model'] in {'qwen7','qwenmoe'}:
            assert c==old[item['name']]['configuration']
            assert old[item['name']]['audit_state'] in {'not_submitted','dependency_failed'}
        else:
            previous=old[item['name'].removeprefix('extended-')]
            assert previous['audit_state']=='timeout'
            prior=previous['configuration']
            assert c['execution']['engine']=='llama_cpp' and c['task']['batch']==32
            for field in ('task','execution','resources'):assert c[field]==prior[field]
            assert c['measurement']=={**prior['measurement'],'timeout_seconds':7200}
            assert item['restart']['original_config_id']==resolve(prior)['config_id']
