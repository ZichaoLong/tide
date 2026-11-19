"""Cross-check the reduction against hashed raw records; no workloads."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
summary=json.loads(Path('artifacts/reception-validation-summary.json').read_text())
checks=Counter()


def sha(data): return hashlib.sha256(data).hexdigest()


def record_hashes(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes'} <= value.keys():
            p=ROOT/value['path']
            data=p.read_bytes()
            # During a preliminary snapshot, only running lifecycle files may advance.
            if not summary['all_terminal'] and p.name in ('stdout.log','stderr.log','status.json','queue.json'):
                return
            assert sha(data)==value['sha256'] and len(data)==value['bytes'],p
            checks['original_hashes']+=1
        for v in value.values(): record_hashes(v)
    elif isinstance(value,list):
        for v in value: record_hashes(v)


def path_check(raw, small):
    c=raw['comparison']
    assert c['exact']==small['exact']
    assert c['tolerance']==small['tolerance']
    assert len(c['threshold_failures'])==small['threshold_failed_fields']
    assert sha(json.dumps(c['threshold_failures'],separators=(',',':')).encode())==small['threshold_field_list_sha256']
    assert sum(small['failure_families'].values())==len(c['threshold_failures'])
    for flag in ('metadata_failures','dtype_differences','nonfinite','support_differences','work_differences'):
        assert c[flag]==small[flag]
    assert (c['largest_errors'][0] if c['largest_errors'] else None)==small['largest_scaled_row']
    assert {k:v.get('field') for k,v in raw['first_differences'].items()}==small['first_returned_fields']
    assert raw['first_differences'].get('support')==small['first_support_witness']
    checks['path_comparisons']+=1


def install_check(raw,small):
    for k in ('exact','within_threshold','finite','atol','rtol','tensors','metadata_failures','nonfinite','support_differences'):
        assert raw[k]==small[k]
    assert len(raw['threshold_failures'])==small['threshold_failed_fields']
    assert sha(json.dumps(raw['threshold_failures'],separators=(',',':')).encode())==small['threshold_field_list_sha256']
    assert (raw['largest'][0] if raw['largest'] else None)==small['largest_absolute_row']
    checks['installed_comparisons']+=1


def placement_check(raw,small):
    for k,v in raw.items():
        if k!='numerical_failures':assert v==small[k]
    fields=[v['path'] for v in raw['numerical_failures']]
    assert len(fields)==small['threshold_failed_fields']
    assert sha(json.dumps(fields,separators=(',',':')).encode())==small['threshold_field_list_sha256']
    checks['placement_comparisons']+=1


record_hashes(summary)
for job in summary['jobs']:
    root=ROOT/'artifacts'/job['name']
    if job['terminal']:
        current=json.loads((root/'status.json').read_text())
        assert current['state']==job['status']['state'] and current['exit_code']==job['status']['exit_code']
        assert not job['recorded_descendants_still_alive']
    result=job.get('result')
    if result is None:continue
    raw=json.loads((ROOT/result['original']['path']).read_text())
    if job['name'].startswith('agg-paths-'):
        r=raw['result'];assert r['state']==result['state'];allchecks=[]
        assert len(r['candidates'])==len(result['candidates'])
        for rc,sc in zip(r['candidates'],result['candidates']):
            assert rc['name']==sc['name'] and rc['state']==sc['state'] and rc['config']==sc['config']
            base=r['candidates'][0]
            for n,v in rc['checks'].items():path_check(v,sc['checks'][n]);allchecks.append(v)
            for n,v in rc['observations'].items():
                assert v['work']==sc['work_by_mode'][n]
                assert (v['work']==base['observations'][n]['work'])==sc['baseline_work_equal_by_mode'][n]
            assert len(rc['updates'])==len(sc['updates'])
            for ru,su in zip(rc['updates'],sc['updates']):
                assert ru['index']+1==su['index'] and ru['work']==su['work']
                assert (ru['work']==base['updates'][ru['index']]['work'])==su['baseline_work_equal']
                if 'vs_baseline' in ru:path_check(ru['vs_baseline'],su['comparison']);allchecks.append(ru['vs_baseline'])
                else:assert su['comparison'] is None
            for n,v in rc.get('restore',{}).get('checks',{}).items():path_check(v,sc['restore']['checks'][n]);allchecks.append(v)
        agg=result['aggregate']
        assert len(allchecks)==agg['comparison_count']
        assert sum(c['comparison']['exact'] for c in allchecks)==agg['exact_comparison_count']
        assert sum(bool(c['comparison']['threshold_failures']) for c in allchecks)==agg['threshold_failed_comparison_count']
        assert sum(len(c['comparison']['threshold_failures']) for c in allchecks)==agg['threshold_failed_field_occurrences']
        for flag,total in agg['flags'].items():assert sum(len(c['comparison'][flag]) for c in allchecks)==total
    elif job['name']=='agg-installed-final-v1':
        assert raw['state']==result['state']
        assert len(raw['cases'])==len(result['cases'])
        for rc,sc in zip(raw['cases'],result['cases']):
            for key in ('path','precision','config','parameter_count','state','updates','original_fourth_update'):
                assert rc[key]==sc[key]
            for name in ('inference_grad_forward','saved_checkpoint_data'):install_check(rc[name],sc[name])
            for name in ('checkpoint_data','next_forward','gradients','updated'):install_check(rc['resume'][name],sc['resume']['checks'][name])
            assert rc['resume']['actual_fourth_update']==sc['resume']['actual_fourth_update']
            assert rc['resume']['state']==sc['resume']['state']
    elif job['name']=='agg-placement-light64-fp32-v1':
        assert raw['state']==result['state']
        assert len(raw['observations'])==len(result['observations'])
        for rc,sc in zip(raw['observations'],result['observations']):
            assert rc['state']==sc['state']
            for n,v in rc.get('comparisons',{}).items():placement_check(v,sc['comparisons'][n])
            for ru,su in zip(rc.get('updates',[]),sc.get('updates',[])):
                if 'parameters' in ru:placement_check(ru['parameters'],su['parameters'])

if '--final' in sys.argv:
    assert summary['all_terminal']
    for name in ('docs/evidence/reception-validation-20261118.md','docs/evidence/reception-numerics-20261118.md'):
        path=Path(name)
        assert 'TODO' not in path.read_text() and 'DRAFT' not in path.read_text()
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text()):
            if re.match(r'[a-zA-Z]+:',target):continue
            target=target.split('#',1)[0]
            if target:assert (path.parent/target).exists(),(name,target);checks['local_links']+=1
    assert json.loads(Path('docs/evidence/reception-validation-20261118.json').read_text())==summary
print(json.dumps(dict(state='passed',all_terminal=summary['all_terminal'],checks=dict(checks))))
