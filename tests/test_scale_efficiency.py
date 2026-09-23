"""Same-process policies preserve scale workload, full states and model owners."""
import json
import subprocess
import pytest
from test_pdg_scale import binary, topology


@pytest.mark.parametrize('packing', ['exact','single'])
def test_independent_ablation_options_preserve_work_and_state(dtype, packing, tmp_path):
    graph=tmp_path/'graph.txt';topology(graph)
    variants=[{}, {'fiber-pooling':'csr'}, {'fiber-cache':'owned'}, {'defer-state-release':1},
              {'projection-layout':'linear'}, {'attention-layout':'head'}, {'fiber-pooling':'csr','fiber-cache':'owned',
                                             'defer-state-release':1,'projection-layout':'linear','attention-layout':'head'}]
    baseline=None
    for i,variant in enumerate(variants):
        out=tmp_path/str(i)
        cmd=[str(binary()),'--device','cpu','--dtype',str(dtype).split('.')[-1],'--topology',str(graph),
             '--width','8','--batch','4','--steps','4','--warmup','1','--vocab','17','--workers','3',
             '--packed','1','--check','1','--work-count','1','--operator-profile','1',
             '--parallel-regions','1','--compact-events','1','--attention-packing',packing,
             '--run-id','ablation','--output-dir',str(out)]
        for k,v in variant.items():cmd+=['--'+k,str(v)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert 'CHECK passed' in result.stdout
        events=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
        if baseline is None:baseline=events
        for a,b in zip(baseline,events):
            for key in a['metrics']:
                if key.startswith(('model/','work/','op/')):assert a['metrics'][key]==b['metrics'][key],key
            assert b['metrics']['check/logits_sum']==pytest.approx(a['metrics']['check/logits_sum'],abs=1e-6,rel=1e-5)
            pooling=variant.get('fiber-pooling','event')
            assert b['metrics']['detail/pooling_calls']==b['metrics']['op/qkv_calls' if pooling=='csr' else 'op/out_rows']
            assert b['context']['fiber_pooling']==pooling
            assert b['context']['fiber_cache']==variant.get('fiber-cache','cloned')
