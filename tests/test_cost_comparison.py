"""Pairing cannot turn failed runs or different workloads into measured speedups."""
import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cost_comparison import collect


def test_cost_roles_failure_and_identity_guards(tmp_path):
    record=dict(schema='tide-layer-chain-cost-v1',execution='completed',source='fixed',
        config=dict(batch=2,length=3,mode='training',device='cpu',threads=1,dtype='float32',
                    family='kv',width=4,depth=2,heads=2,implementation='native'),
        observations=[dict(warmup=True,seconds=10),dict(warmup=False,seconds=2,phases={},actual_work={})],
        strict_numerical='failed_at_original_tolerance',recovery='not_checked',timing='complete',
        parameters=10,numerical_policy={},initial_parameters_sha256='a')
    for name in ('candidate','reference'):(tmp_path/(name+'.json')).write_text(json.dumps(record))
    manifest=dict(schema='tide-cost-comparison-v1',question='finite question',
        records=[dict(name=n,path=n+'.json') for n in ('candidate','reference')],
        comparisons=[dict(candidate='candidate',reference='reference',relation='same-function')])
    result=collect(manifest,tmp_path)
    assert result['records'][0]['input_positions_per_second']==3
    assert result['records'][0]['strict_numerical']=='failed_at_original_tolerance'
    assert result['comparisons'][0]['equivalence']=='not_inferred_from_costs'
    other=copy.deepcopy(record);other['config']['length']=4
    (tmp_path/'candidate.json').write_text(json.dumps(other))
    with pytest.raises(ValueError,match='axes'):collect(manifest,tmp_path)
    other=copy.deepcopy(record);other['initial_parameters_sha256']='different'
    (tmp_path/'candidate.json').write_text(json.dumps(other))
    with pytest.raises(ValueError,match='starting parameters'):collect(manifest,tmp_path)
    manifest['comparisons'][0]['relation']='different-function'
    assert collect(manifest,tmp_path)['comparisons'][0]['state']=='cost_observed'
    other.update(execution='failed',error='capacity')
    (tmp_path/'candidate.json').write_text(json.dumps(other))
    assert collect(manifest,tmp_path)['comparisons'][0]['state']=='unavailable'
