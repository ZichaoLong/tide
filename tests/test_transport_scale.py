"""Native executable paths report true reuse counts and retain complete traces."""
import json
import subprocess
import pytest
from test_pdg_scale import binary, topology


@pytest.mark.parametrize('packing', ['exact','single'])
def test_optional_transport_work_inventory(dtype, packing, tmp_path):
    graph=tmp_path/'graph.txt';topology(graph);baseline=None
    for sources,nxt in ((0,0),(1,0),(0,1),(1,1)):
        out=tmp_path/f'{sources}-{nxt}'
        cmd=[str(binary()),'--device','cpu','--dtype',str(dtype).split('.')[-1],
             '--topology',str(graph),'--width','8','--batch','4','--steps','4','--warmup','1','--vocab','17',
             '--workers','3','--packed','1','--check','1','--work-count','1','--operator-profile','1',
             '--parallel-regions','1','--compact-events','1','--attention-packing',packing,
             '--packed-sources',str(sources),'--batch-next',str(nxt),'--run-id','transport','--output-dir',str(out)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert 'CHECK passed' in result.stdout
        events=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        if baseline is None:baseline=events
        for a,b in zip(baseline,events):
            expected,actual=a['metrics'],b['metrics']
            for key in expected:
                if key.startswith(('model/','work/','op/')) and key not in ('op/fiber_scale_elements','op/fiber_reused_elements'):
                    assert actual[key]==expected[key],key
            count=expected['op/fiber_scale_elements'];assert count>0
            assert actual['op/fiber_scale_elements']==(0 if sources else count)
            assert actual['op/fiber_reused_elements']==(count if sources else 0)
            assert actual['check/logits_sum']==pytest.approx(expected['check/logits_sum'],abs=1e-6,rel=1e-5)
            if sources:assert actual['work/packed_source_batches']>0
            if nxt:assert actual['work/next_batches']>0 and actual['work/next_reset_batches']>0
            assert b['context']['packed_sources']==bool(sources)
            assert b['context']['batch_next']==bool(nxt)
