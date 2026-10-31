"""Sharing admits only the explicitly inspected owner and resource envelope."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("shared_npu", Path(__file__).resolve().parents[1] / "scripts/run_shared_npu.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def material():
    owner = dict(pid=123, start_ticks=456, uid=789)
    policy = dict(schema="tide-npu-sharing-v1", reason="user authorized low-use sharing",
                  expires_unix=1000, cards={"9": dict(processes=[owner], max_hbm_mib=8192,
                       max_aicore_percent=5, required_free_mib=24576)})
    device = SimpleNamespace(physical_id=9, health="OK", hbm_used_mib=5656,
                             hbm_total_mib=65536, aicore_percent=0, process_ids=(123,))
    return policy, device, owner


@pytest.mark.parametrize("change", [dict(physical_id=8), dict(health="BAD"),
    dict(hbm_used_mib=8193), dict(hbm_total_mib=10000), dict(aicore_percent=6),
    dict(process_ids=(123, 321))])
def test_unapproved_or_busy_device_is_rejected(change):
    policy, device, owner = material()
    assert shared.shareable(device, policy, identity=lambda p: owner, now=lambda: 999)
    device.__dict__.update(change)
    assert not shared.shareable(device, policy, identity=lambda p: owner, now=lambda: 999)


def test_pid_reuse_uid_change_unreadable_owner_and_expiry_fail_closed():
    policy, device, owner = material()
    for altered in (dict(owner, start_ticks=457), dict(owner, uid=790)):
        assert not shared.shareable(device, policy, identity=lambda p: altered, now=lambda: 999)
    def unreadable(_): raise PermissionError("not observable")
    assert not shared.shareable(device, policy, identity=unreadable, now=lambda: 999)
    assert not shared.shareable(device, policy, identity=lambda p: owner, now=lambda: 1000)
    device.process_ids = ()
    assert shared.shareable(device, policy, identity=unreadable, now=lambda: 999)


@pytest.mark.parametrize("change", [dict(expires_unix="tomorrow"), dict(expires_unix=float("inf")),
                                   dict(expires_unix=float("nan")), dict(cards={}), dict(reason="")])
def test_invalid_authorization_is_rejected(change):
    policy, _, _ = material()
    shared.validate_policy(policy)
    policy.update(change)
    with pytest.raises(ValueError): shared.validate_policy(policy)


def test_process_identity_with_spaces_in_name(tmp_path):
    folder = tmp_path / "123"; folder.mkdir()
    fields = ["S"] + ["0"] * 18 + ["456"] + ["0"] * 5
    (folder / "stat").write_text("123 (worker name) " + " ".join(fields))
    assert shared.process_identity(123, tmp_path) == dict(pid=123, start_ticks=456, uid=folder.stat().st_uid)


def test_unreviewed_scheduler_never_executes(tmp_path):
    helper = tmp_path / "scheduler.py"
    helper.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="hash changed"):
        shared.load_helper(helper, "0" * 64)
