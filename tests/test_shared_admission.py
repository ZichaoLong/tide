import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location("shared_queue",Path(__file__).parents[1]/"scripts/npu_shared_queue.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_expiring_pid_bound_admission(monkeypatch):
    known={"pid":17,"uid":1003,"start_ticks":42}
    monkeypatch.setattr(module,"identity",lambda pid:known)
    policy=dict(issued_epoch=10,expires_epoch=20,devices={"0":[known]},
                max_aicore_percent=70,max_existing_hbm_mib=30000,required_free_mib=24000)
    device=SimpleNamespace(physical_id=0,health="OK",aicore_percent=20,hbm_used_mib=20000,
                           hbm_total_mib=65536,process_ids=(17,))
    assert module.admitted(device,policy,15)
    assert not module.admitted(device,policy,21)
    assert not module.admitted(device,policy,9)
    monkeypatch.setattr(module,"identity",lambda pid:{**known,"start_ticks":43})
    assert not module.admitted(device,policy,15)
    device.process_ids=(17,18)
    assert not module.admitted(device,policy,15)
    device.hbm_used_mib=50000
    assert not module.admitted(device,policy,15)
