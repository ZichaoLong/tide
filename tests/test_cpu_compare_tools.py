"""Portable launch records, timing windows, relocation and bounded failure paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools/cpu_compare'), str(ROOT/'scripts')]
from compare_metrics import Collector, summarize
from compare_config import digest, verify_packet
from compare_process import execute, ProcessFailure
from compare_build import prepare


def test_live_lh_metrics_match_original_window_and_keep_cold_prefix(tmp_path):
    a = SimpleNamespace(batch=4, warmup=1, steps=3, work_count=0)
    c = Collector('lh', a, tmp_path, 'test', 23)
    c.line('构造 ionet, 参数量 = 23')
    for i, ms in enumerate([100, 120, 160]):
        c.line('Think took 999 ms extra')  # Not the outer timer protocol.
        c.line(f'Think took {ms} ms '); c.line(f't: {i}')
        c.line('WORK '+json.dumps(dict(step=i,metrics={'perf/token_seconds':ms/1000,'check/logits_sum':1})))
    r = c.finish()
    assert r['raw_ms'] == [30,40] and r['mean_ms_per_sample_token'] == 35
    assert r['sample_tokens_per_second'] == pytest.approx(1000/35)
    assert r['token_indices'] == [1,2] and r['observations'] == 3
    assert len((tmp_path/'metrics.jsonl').read_text().splitlines()) == 3


def test_partial_pdg_tail_is_preserved_and_never_passes(tmp_path):
    a = SimpleNamespace(batch=4,warmup=0,steps=2,work_count=0)
    c = Collector('pdg',a,tmp_path,'test',23)
    native=tmp_path/'native';native.mkdir()
    first=dict(schema_version=1,run_id='test',sequence=0,step=0,timestamp='2026-09-23T00:00:00Z',elapsed_seconds=1,
               metrics={'model/parameters':23,'perf/ms_per_sample_token':10})
    path=native/'metrics.jsonl';path.write_text(json.dumps(first)+'\n{"step":')
    c.drain_native(); assert len(c.events)==1
    with pytest.raises(ValueError,match='incomplete'):c.finish()
    second=dict(first,sequence=1,step=1,elapsed_seconds=2)
    path.write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n')
    assert c.finish()['observations']==2


def test_out_of_order_nonfinite_and_parameter_mismatch_rejected(tmp_path):
    a=SimpleNamespace(batch=1,warmup=0,steps=1,work_count=0)
    for value in (float('nan'),24):
        c=Collector('pdg',a,tmp_path,'test',23)
        with pytest.raises(ValueError):
            c.add(dict(schema_version=1,run_id='test',step=0,sequence=0,
                       metrics={'model/parameters':value,'perf/ms_per_sample_token':1}))


def test_packet_integrity_is_location_independent(tmp_path):
    packet=tmp_path/'packet';packet.mkdir();(packet/'data').write_text('fixed')
    manifest=dict(schema='tide-lh-cpu-compare-v1',files_sha256={'data':digest(packet/'data')})
    (packet/'manifest.json').write_text(json.dumps(manifest))
    assert verify_packet(packet)==manifest
    moved=tmp_path/'relocated with spaces';packet.rename(moved)
    assert verify_packet(moved)==manifest
    (moved/'data').write_text('changed')
    with pytest.raises(ValueError,match='hash'):verify_packet(moved)


def test_timeout_reaps_child_and_preserves_log(tmp_path):
    command=[sys.executable,'-c','import time; print("began",flush=True); time.sleep(30)']
    with pytest.raises(ProcessFailure,match='TimeoutError') as failed:
        execute(command,tmp_path,os.environ.copy(),tmp_path/'log',.4)
    assert failed.value.result['unreaped_child_pid'] is None
    assert failed.value.result['exit_code'] != 0
    assert (tmp_path/'log').read_text()=='began\n'


@pytest.mark.parametrize('engine',['lh','pdg'])
def test_cli_help_and_invalid_backend_without_torch(engine,tmp_path):
    script=ROOT/'tools/cpu_compare'/('run_'+engine+'.py')
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--torch-prefix' in result.stdout
    assert ('--attention-packing' in result.stdout) == (engine == 'pdg')
    result=subprocess.run([sys.executable,str(script),'--device','npu','--output-dir',str(tmp_path/'bad')],capture_output=True)
    assert result.returncode!=0 and not (tmp_path/'bad').exists()
