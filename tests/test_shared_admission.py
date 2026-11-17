import importlib.util
from contextlib import contextmanager, ExitStack
import fcntl
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location("shared_queue",Path(__file__).parents[1]/"scripts/npu_shared_queue.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_expiring_pid_bound_admission(monkeypatch):
    known={"pid":17,"uid":os.getuid(),"start_ticks":42}
    monkeypatch.setattr(module,"identity",lambda pid:known)
    policy=dict(issued_epoch=10,expires_epoch=20,devices={"0":[known]},
                max_aicore_percent=70,max_existing_hbm_mib=30000,required_free_mib=24000)
    device=SimpleNamespace(physical_id=0,health="OK",aicore_percent=20,hbm_used_mib=20000,
                           hbm_total_mib=65536,process_ids=(17,))
    assert module.admitted(device,policy,15)
    assert not module.admitted(device,policy,21)
    assert not module.admitted(device,policy,20)
    assert not module.admitted(device,policy,9)
    monkeypatch.setattr(module,"identity",lambda pid:{**known,"start_ticks":43})
    assert not module.admitted(device,policy,15)
    device.process_ids=(17,18)
    assert not module.admitted(device,policy,15)
    device.hbm_used_mib=50000
    assert not module.admitted(device,policy,15)


@contextmanager
def master_lock(path):
    path.write_text("authorized master annotation\n")
    handle=path.open("r+")
    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


def policy_and_device(lock_dir,physical_id=0):
    known=module.identity(os.getpid())
    device=SimpleNamespace(physical_id=physical_id,health="OK",aicore_percent=10,
                           hbm_used_mib=1000,hbm_total_mib=65536,process_ids=(os.getpid(),))
    policy=dict(issued_epoch=time.time()-5,expires_epoch=time.time()+60,
                devices={str(physical_id):[known]},max_aicore_percent=40,
                max_existing_hbm_mib=30000,required_free_mib=24000,
                existing_locks={str(physical_id):module.lock_holder(lock_dir/f"physical-{physical_id}.lock")})
    assert policy["existing_locks"][str(physical_id)] is not None
    return policy,device


def adapter_for(lock_dir,policy,device,**kwargs):
    def ordinary(*args):
        pytest.fail("bound master must be probed without invoking the site's annotating lock writer")
    site=SimpleNamespace(try_lock=ordinary,snapshot=lambda mapping:{device.physical_id:device})
    kwargs.setdefault("monitor_seconds",None)
    return module.SharedAdmission(site,policy,lock_dir=lock_dir,mapping={},**kwargs)


def test_kernel_holder_binding_and_two_competing_borrowers(tmp_path):
    path=tmp_path/"physical-0.lock"
    with master_lock(path):
        original=path.read_bytes(),path.stat().st_ino
        policy,device=policy_and_device(tmp_path)
        first=adapter_for(tmp_path,policy,device)
        second=adapter_for(tmp_path,policy,device)
        assert first.check(device) and second.check(device)
        handle=first.try_lock(tmp_path,0)
        assert handle is not None
        assert second.try_lock(tmp_path,0) is None
        assert module._busy_without_writing(path)
        assert (path.read_bytes(),path.stat().st_ino)==original
        handle.close()
        successor=second.try_lock(tmp_path,0)
        assert successor is not None
        successor.close()
        assert module.lock_holder(path)==policy["existing_locks"]["0"]
        assert (path.read_bytes(),path.stat().st_ino)==original


@pytest.mark.parametrize("field",["pid","uid","start_ticks","lock_dev","lock_inode"])
def test_unexpected_master_identity_or_inode_refused(tmp_path,field):
    path=tmp_path/"physical-0.lock"
    with master_lock(path):
        policy,device=policy_and_device(tmp_path)
        policy["existing_locks"]["0"][field]+=1
        adapter=adapter_for(tmp_path,policy,device)
        assert not adapter.check(device)
        assert adapter.try_lock(tmp_path,0) is None
        assert path.read_text()=="authorized master annotation\n"
        assert not (tmp_path/"shared-physical-0.lock").exists()


def test_expired_or_released_master_never_becomes_an_ordinary_allocation(tmp_path):
    path=tmp_path/"physical-0.lock"
    with master_lock(path) as master:
        policy,device=policy_and_device(tmp_path)
        adapter=adapter_for(tmp_path,policy,device)
        policy["expires_epoch"]=time.time()-1
        assert not adapter.check(device) and adapter.try_lock(tmp_path,0) is None
        policy["expires_epoch"]=time.time()+60
        fcntl.flock(master.fileno(),fcntl.LOCK_UN)
        assert not adapter.check(device) and adapter.try_lock(tmp_path,0) is None
        assert path.read_text()=="authorized master annotation\n"


def test_explicit_unknown_master_is_not_treated_as_an_unbound_device(tmp_path):
    path=tmp_path/"physical-0.lock"
    with master_lock(path):
        policy,device=policy_and_device(tmp_path)
        policy["existing_locks"]["0"]=None
        adapter=adapter_for(tmp_path,policy,device)
        assert not adapter.check(device)
        assert adapter.try_lock(tmp_path,0) is None


def test_replaced_path_and_symlink_mutex_fail_closed(tmp_path):
    path=tmp_path/"physical-0.lock"
    with master_lock(path):
        policy,device=policy_and_device(tmp_path)
        adapter=adapter_for(tmp_path,policy,device)
        shared=tmp_path/"shared-physical-0.lock"
        shared.symlink_to(path)
        assert adapter.try_lock(tmp_path,0) is None
        assert path.read_text()=="authorized master annotation\n"
        shared.unlink()
        path.unlink()
        path.write_text("replacement inode\n")
        assert not adapter.check(device) and adapter.try_lock(tmp_path,0) is None


@pytest.mark.parametrize("change",["occupant","resource","master"])
def test_fresh_check_after_borrow_mutex_releases_on_change(tmp_path,change):
    path=tmp_path/"physical-0.lock"
    with master_lock(path) as master:
        policy,device=policy_and_device(tmp_path)
        adapter=adapter_for(tmp_path,policy,device)
        assert adapter.check(device)
        def changed(mapping):
            if change=="occupant":
                device.process_ids=()
            elif change=="resource":
                device.hbm_used_mib=60000
            else:
                fcntl.flock(master.fileno(),fcntl.LOCK_UN)
            return {0:device}
        adapter.snapshot=changed
        assert adapter.try_lock(tmp_path,0) is None
        available=module._shared_mutex(tmp_path/"shared-physical-0.lock")
        assert available is not None
        available.close()
        assert not adapter.borrowed


def test_monitor_checks_only_lifetime_and_sigterms_own_adapter(tmp_path,monkeypatch,capsys):
    path=tmp_path/"physical-0.lock"
    with master_lock(path) as master:
        policy,device=policy_and_device(tmp_path)
        signalled=threading.Event()
        calls=[]
        def signal_own(pid,signal_number):
            calls.append((pid,signal_number))
            signalled.set()
        monkeypatch.setattr(module.os,"kill",signal_own)
        adapter=adapter_for(tmp_path,policy,device,monitor_seconds=.01)
        handle=adapter.try_lock(tmp_path,0)
        try:
            assert handle is not None and adapter.lease_violation() is None
            device.aicore_percent=100
            device.hbm_used_mib=65000
            assert adapter.lease_violation() is None
            assert adapter.lease_violation(now=policy["expires_epoch"])=="authorization_expired_or_not_yet_valid"
            fcntl.flock(master.fileno(),fcntl.LOCK_UN)
            assert signalled.wait(2)
            assert calls==[(os.getpid(),signal.SIGTERM)]
            assert "shared_borrow_release_trigger" in capsys.readouterr().out
            # Signalling does not unlock: the site helper must reap its child first.
            assert module._shared_mutex(tmp_path/"shared-physical-0.lock") is None
        finally:
            if handle is not None:
                handle.close()
            adapter.close()


def load_site():
    default=Path.home()/".agents/skills/use-local-ascend/scripts/run_on_free_npu.py"
    path=Path(os.environ.get("TIDE_SITE_NPU_HELPER",default))
    if not path.is_file():
        pytest.skip("local site queue integration requires TIDE_SITE_NPU_HELPER")
    spec=importlib.util.spec_from_file_location("shared_queue_site_fixture",path)
    site=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=site
    spec.loader.exec_module(site)
    return site


def test_site_fifo_and_both_sampling_phases_are_preserved(tmp_path,monkeypatch):
    site=load_site()
    device_dir=tmp_path/"devices"
    device_dir.mkdir()
    queue_dir=tmp_path/"queue"
    with master_lock(device_dir/"physical-0.lock"):
        policy,device=policy_and_device(device_dir)
        device=site.DeviceStatus(0,0,0,device.health,device.aicore_percent,device.hbm_used_mib,
                                 device.hbm_total_mib,device.process_ids)
        samples=[]
        def snapshot(mapping):
            samples.append(len(samples))
            return {0:device if len(samples)<3 else site.dataclasses.replace(device,process_ids=())}
        monkeypatch.setattr(site,"snapshot",snapshot)
        adapter=module.install(site,policy)
        adapter.monitor_seconds=None
        handles=[]
        for job_id in ("first","second"):
            handle,_=site.register_job(queue_dir,job_id=job_id,project="test",priority=0,
                                       npu_count=1,candidate_ids={0})
            handles.append(handle)
        args=dict(queue_dir=queue_dir,device_lock_dir=device_dir,mapping={},candidate_ids={0},npu_count=1,
                  max_hbm_mib=0,max_aicore_percent=0,sample_seconds=0,stop_event=threading.Event())
        try:
            waiting=site.attempt_schedule(job_id="second",**args)
            assert waiting.reason=="waiting_turn" and not samples
            changed=site.attempt_schedule(job_id="first",**args)
            assert changed.reason=="device_changed_during_sampling" and len(samples)==3
            assert not changed.device_locks and not adapter.borrowed
            available=module._shared_mutex(device_dir/"shared-physical-0.lock")
            assert available is not None
            available.close()
        finally:
            site.release_locks(handles)
            adapter.close()


def test_site_group_failure_releases_every_partial_borrow(tmp_path,monkeypatch):
    site=load_site()
    device_dir=tmp_path/"devices"
    device_dir.mkdir()
    queue_dir=tmp_path/"queue"
    with ExitStack() as masters:
        for physical_id in (0,1):
            masters.enter_context(master_lock(device_dir/f"physical-{physical_id}.lock"))
        policy,first=policy_and_device(device_dir,0)
        second_policy,second=policy_and_device(device_dir,1)
        policy["devices"].update(second_policy["devices"])
        policy["existing_locks"].update(second_policy["existing_locks"])
        devices={item.physical_id:site.DeviceStatus(0,item.physical_id,item.physical_id,item.health,
                 item.aicore_percent,item.hbm_used_mib,item.hbm_total_mib,item.process_ids)
                 for item in (first,second)}
        monkeypatch.setattr(site,"snapshot",lambda mapping:devices)
        adapter=module.install(site,policy)
        adapter.monitor_seconds=None
        contender=module._shared_mutex(device_dir/"shared-physical-1.lock")
        job_lock,_=site.register_job(queue_dir,job_id="pair",project="test",priority=0,npu_count=2,candidate_ids={0,1})
        try:
            result=site.attempt_schedule(queue_dir=queue_dir,device_lock_dir=device_dir,job_id="pair",
                   mapping={},candidate_ids={0,1},npu_count=2,max_hbm_mib=0,max_aicore_percent=0,
                   sample_seconds=0,stop_event=threading.Event())
            assert result.reason=="insufficient_unlocked_devices"
            assert not adapter.borrowed
            available=module._shared_mutex(device_dir/"shared-physical-0.lock")
            assert available is not None
            available.close()
            assert module._shared_mutex(device_dir/"shared-physical-1.lock") is None
            for physical_id in (0,1):
                assert module.lock_holder(device_dir/f"physical-{physical_id}.lock")==policy["existing_locks"][str(physical_id)]
        finally:
            site.release_locks([job_lock])
            contender.close()
            adapter.close()
