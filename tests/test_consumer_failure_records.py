"""Underestimation must fail while preserving every device and completed work."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from test_consumer_capacity import geometry
from tools.online_bench.capacity_runtime import observed
from flow_failure import RecordedFailure

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def observation_probe(tmp_path_factory):
    binary = tmp_path_factory.mktemp('observation-probe')/'probe'
    subprocess.run(['c++','-std=c++17','-O1',str(ROOT/'tests/consumer_observation_probe.cpp'),'-o',str(binary)],check=True)
    return binary


@pytest.mark.parametrize('estimates',[(100,200,300),(99,200,299),(100,199,300)])
def test_all_peaks_survive_first_refusal(estimates,observation_probe):
    # Peak live storage can be freed before sampling; device order in each
    # phase need not match the planner. The first miss must not hide later cards.
    initial = [23,41,17]
    memory = dict(phases=[dict(phase=phase,devices=[dict(device=f'npu:{i}',
        peak_allocated_bytes=initial[i]+value,allocated_bytes=initial[i])
        for i,value in reversed(list(enumerate(values)))])
        for phase,values in [('construction',[10,200,5]),('measured',[100,20,300])]])
    admission = dict(devices=[dict(estimated_peak_bytes=v) for v in estimates],
        initial_devices=[dict(device=f'npu:{i}',allocated_bytes=v) for i,v in enumerate(initial)])
    fits = observed(admission,memory)
    assert admission['observed_peak_growth_bytes']==[100,200,300]
    assert fits == (estimates==(100,200,300))
    text=' '.join(map(str,[3,*(v for pair in zip(estimates,[100,200,300]) for v in pair)]))
    run=subprocess.run([str(observation_probe)],input=text,text=True,capture_output=True,timeout=10)
    assert run.returncode==(0 if fits else 2)
    assert json.loads(run.stdout)=={k:admission[k] for k in ('observed_peak_growth_bytes','allocator_within_estimate')}


def failure(packet):
    return dict(state='failed',error='allocator peak exceeds estimate',workload_sha256=packet['sha256'],
        failure_phase='post_run_memory_calibration',seconds=[1.25],outputs=[16],final_cut=123,
        memory=dict(phases=[dict(phase='construction',devices=[dict(device='npu:0',peak_allocated_bytes=101)])]),
        memory_admission=dict(observed_peak_growth_bytes=[101],allocator_within_estimate=False))


def arguments(packet_path,out,implementation='libtorch'):
    return ['--packet',str(packet_path),'--output-dir',str(out),'--device','npu:0',
        '--family','timed-dag','--implementation',implementation,'--preset','resident','--schedule','prefill']


@pytest.mark.parametrize('case',['matching','wrong_identity','false_success','invalid_json','non_object'])
def test_standalone_failure_is_retained_without_certifying_it(case,tmp_path):
    packet,_=geometry(width=4);path=tmp_path/'packet.json';path.write_text(json.dumps(packet))
    payload=failure(packet)
    if case=='wrong_identity':payload['workload_sha256']='wrong'
    if case=='false_success':payload['state']='passed'
    text='invalid' if case=='invalid_json' else json.dumps(payload)
    if case=='non_object':text='[]'
    # This deliberately failing executable stands in for the independent
    # process boundary. No accelerator runtime is loaded by the launcher.
    fake=tmp_path/'consumer'
    fake.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\n'
        'out=Path(next(x.split("=",1)[1] for x in sys.argv if x.startswith("--output-dir=")))\n'
        'out.mkdir()\n(out/"result.json").write_text('+repr(text)+')\nsys.exit(2)\n')
    fake.chmod(0o700);out=tmp_path/'run'
    command=[sys.executable,str(ROOT/'scripts/run_execution_flow.py'),*arguments(path,out),'--native-binary',str(fake)]
    done=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert done.returncode!=0
    result=json.loads((out/'result.json').read_text())
    assert result['state']=='failed' and result['workload_sha256']==packet['sha256']
    if case=='matching':
        assert all(result[k]==v for k,v in payload.items())
        assert result['native_exit_code']==2 and result['process_wall_seconds']>0
        assert result['command'][0]==str(fake) and len(result['binary_sha256'])==64
    else:
        assert 'memory_admission' not in result and 'standalone consumer failed (2)' in result['error']


def test_python_failure_reaches_the_same_durable_record(tmp_path,monkeypatch):
    import run_execution_flow
    packet,_=geometry(width=4);path=tmp_path/'packet.json';path.write_text(json.dumps(packet));out=tmp_path/'run'
    expected=failure(packet)
    def failed(*_):
        raise RecordedFailure(expected['error'],expected)
    monkeypatch.setattr(run_execution_flow,'python_run',failed)
    monkeypatch.setattr(sys,'argv',['run_execution_flow',*arguments(path,out,'native')])
    with pytest.raises(RecordedFailure):run_execution_flow.main()
    assert json.loads((out/'result.json').read_text())==expected
