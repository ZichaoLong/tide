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


def test_dense_ttft_actual_work_and_phase_scope(tmp_path):
    record=dict(schema='tide-dense-query-cost-v1',execution='completed',
        resolved=dict(training=dict(batch=2,length=3),device='cpu',core=dict(dtype='float32',
            eager_implementation='native',placement_preset='cpu')),
        mode='inference',effective_threads=1,fc_shape='fc4',query_execution='auto',baseline='graph',
        phase_fences='step',scope='test',trainable_parameters=10,precision={},
        initial_parameters_sha256='a',measured_parameters_sha256='a',prefill_schedule='auto',
        trace_records='minimal',diagnostic_boundary='public',downstream='fixed',downstream_archive_sha256='abc',
        affinity=[0],visibility=None,strict_numerical='not_checked',recovery='not_checked',
        phases=[dict(warmup=False,complete_wall_seconds=2.,tokens=6,first_tokens=[4,5],
                     first_token_seconds=1.5,ttft_scope='returned tokens',phases_seconds={'core_forward':1.},
                     work={'selected_events':5})])
    ref=copy.deepcopy(record);ref['phases'][0]['work']['selected_events']=6
    for name,value in [('candidate',record),('reference',ref)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    manifest=dict(schema='tide-cost-comparison-v1',question='measured numerator and boundaries',
        records=[dict(name=n,path=n+'.json') for n in ('candidate','reference')],
        comparisons=[dict(candidate='candidate',reference='reference',relation='same-function')])
    result=collect(manifest,tmp_path)
    assert result['records'][0]['first_token_seconds']==1.5
    assert result['records'][0]['first_outputs_per_second']==2/1.5
    pair=result['comparisons'][0]
    assert pair['work_count_differences']=={'selected_events':[5,6]}
    assert 'phase_seconds_delta' not in pair
    assert 'first_return_seconds' not in result['records'][0]


def test_diagnostic_failures_and_opaque_graphs_remain_separate(tmp_path):
    from execution_diagnostics import attachments,census
    (tmp_path/'tests.xml').write_text('<testsuites><testsuite tests="1" failures="1"><testcase name="root">'
        '<failure message="strict numerical mismatch">details</failure></testcase></testsuite></testsuites>')
    (tmp_path/'job.json').write_text(json.dumps(dict(state='passed',source='fixed',exit_code=0)))
    entries=[dict(name='qualification',kind='junit',path='tests.xml'),dict(name='job',kind='job',path='job.json')]
    rows=attachments(entries,tmp_path)
    assert rows[0]['nonpasses'][0]['results'][0]['attributes']['message']=='strict numerical mismatch'
    assert rows[1]['job']['state']=='passed'
    (tmp_path/'job.json').write_text(json.dumps(dict(state='running')))
    with pytest.raises(ValueError,match='terminal'):attachments(entries,tmp_path)
    graph=census(dict(python=dict(total={'unique_nodes':7}),
                     native={'total/unique_nodes':20,'graphs/outer/returned-and-task/unique_nodes':7}))
    assert graph['native']['total/unique_nodes']==20  # not 27 from adding the outer twice
    (tmp_path/'census.json').write_text(json.dumps(dict(autograd={'native':{'total/unique_nodes':20}})))
    (tmp_path/'profile.json').write_text(json.dumps(dict(schema='tide-dense-query-cost-v1',profile='stages',
        profile_state='completed',autograd_census='census.json',instrumented_stages={'detail/query_gradient_calls':3})))
    observed=attachments([dict(name='small-profile',kind='profile',path='profile.json')],tmp_path)[0]
    assert observed['autograd']['native']['total/unique_nodes']==20
    assert observed['instrumented_stages']['detail/query_gradient_calls']==3
    assert 'seconds' not in observed  # instrumented run cannot supply a cost observation
