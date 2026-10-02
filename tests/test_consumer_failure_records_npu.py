"""Real completed training remains FAILED when allocator calibration rejects it."""
import os
import pytest
import torch
from flow_failure import RecordedFailure
from test_online_consumer import packet
from test_online_consumer_npu import target
from tidegraph import ResidentPlacement
from tools.online_bench.host import run
from tools.online_bench import resident


def test_failed_calibration_keeps_complete_training_and_releases_owner(monkeypatch,tmp_path):
    import json
    device=target();p=packet('attention')
    common=dict(family='timed-dag',training=True,optimizer='adamw',steps=2,warmup=0,windows_per_step=2)
    reference=run(p,implementation='python',device='cpu',schedule='streaming',**common)
    real_observed=resident.observed_capacity
    def underestimate_after_execution(record,memory):
        assert real_observed(record,memory)
        # Inject a planner miss only at the post-run reporting boundary. The
        # actual run is still admitted using its unchanged safe estimate.
        record['devices'][0]['estimated_peak_bytes']=record['observed_peak_growth_bytes'][0]-1
        return real_observed(record,memory)
    monkeypatch.setattr(resident,'observed_capacity',underestimate_after_execution)
    options=dict(implementation='native',device=device,schedule='prefill',preset='resident',
        native_library=os.environ['TIDE_BUILD_DIR'],resident_library=os.environ['TIDE_RESIDENT_LIBRARY'],
        resident_placement=ResidentPlacement(devices=(str(device),f'npu:{device.index+1}')))
    with pytest.raises(RecordedFailure,match='underestimated allocator peak') as failure:
        run(p,**options,**common)
    result=failure.value.record
    assert result['state']=='failed' and result['failure_phase']=='post_run_memory_calibration'
    assert result['outputs']==reference['outputs'] and result['final_cut']==reference['final_cut']
    torch.testing.assert_close(torch.tensor(result['losses']),torch.tensor(reference['losses']),atol=1e-6,rtol=1e-5)
    assert len(result['seconds'])==2 and len(result['memory_admission']['observed_peak_growth_bytes'])==2
    assert result['memory_admission']['allocator_within_estimate'] is False
    assert [v['phase'] for v in result['memory']['phases']]==['initial','construction','measured']
    (tmp_path/'observed-failure.json').write_text(json.dumps(result))
    monkeypatch.setattr(resident,'observed_capacity',real_observed)
    # A fresh owner is usable after the refused run's finally/close path.
    retry=run(p,**options,**common)
    assert retry['memory_admission']['allocator_within_estimate']
    assert retry['losses']==result['losses'] and retry['final_cut']==result['final_cut']
