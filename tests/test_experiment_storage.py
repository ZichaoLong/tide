"""Tmpfs is charged to both storage and host; fsync does not make it persistent."""
import json
import pytest
from tidegraph.experiment.config import resolve
from tidegraph.experiment import admission
from tidegraph.experiment.storage import filesystem


def request(**resources):
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=2,
        region_width=1, width=4)), execution=dict(device="cpu"),
        measurement=dict(steps=1, warmup=0), resources=resources)


def test_static_tmpfs_files_add_to_host_once_excluding_unallocated_disk_reserve():
    disk = resolve(request())["capacity_budget"]
    memory = resolve(request(storage_medium="tmpfs"))["capacity_budget"]
    files = disk["disk"]["estimated_new_bytes"]-disk["disk"]["reserve_bytes"]
    assert memory["host"]["future_tmpfs_file_bytes"] == files > 0
    assert memory["host"]["estimated_tensor_reservation_bytes"] == disk["host"]["estimated_tensor_reservation_bytes"]+files
    assert memory["disk"] == disk["disk"]


def test_live_tmpfs_is_detected_without_user_setting_and_refuses_before_allocation(tmp_path, monkeypatch):
    config = resolve(request())
    host = config["capacity_budget"]["host"]["estimated_tensor_reservation_bytes"]
    monkeypatch.setattr(admission, "host_memory", lambda: dict(available_bytes=host+1))
    monkeypatch.setattr(admission, "filesystem", lambda path: dict(memory_backed=True, filesystem="tmpfs", free_bytes=2**40))
    budget = admission.preflight(config, tmp_path)
    assert budget["host"]["future_tmpfs_file_bytes"] > 1
    assert budget["refused"] == ["host: planned reservation exceeds live free memory/storage"]
    assert budget["storage"]["requested_medium"] is None
    assert budget["storage"]["effective_medium"] == "tmpfs"
    from tidegraph.experiment import engine
    monkeypatch.setattr(engine, "create", lambda *a, **kw: pytest.fail("must refuse before constructing"))
    with pytest.raises(ValueError, match="capacity preflight: host"):
        engine.run(config, tmp_path/"refused")
    record = json.loads((tmp_path/"refused/result.json").read_text())
    assert record["execution_status"] == "failed"


def test_declared_medium_mismatch_and_unknown_mount_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "filesystem", lambda path: dict(memory_backed=False, filesystem="ext4", free_bytes=2**40))
    rejected = admission.preflight(resolve(request(storage_medium="tmpfs")), tmp_path)
    assert any("requested tmpfs, output mount is disk" in s for s in rejected["refused"])
    monkeypatch.setattr(admission, "filesystem", lambda path: dict(memory_backed=None, filesystem="unknown", free_bytes=2**40))
    assert any("could not establish" in s for s in admission.preflight(resolve(request()), tmp_path)["refused"])


def test_mount_resolution_handles_nested_paths_and_escaped_spaces(tmp_path):
    mount = tmp_path/"memory space"
    mount.mkdir()
    table = tmp_path/"mountinfo"
    encoded = str(mount).replace(" ", r"\040")
    table.write_text(f"1 0 0:1 / / rw - ext4 /dev/test rw\n2 1 0:2 / {encoded} rw - tmpfs shm rw\n")
    result = filesystem(mount/"future"/"run", mountinfo=table)
    assert result["filesystem"] == "tmpfs" and result["memory_backed"]
    assert result["mount"] == str(mount)
    assert filesystem(tmp_path, mountinfo=table)["filesystem"] == "ext4"


def test_checkpoint_cost_observation_preserves_failure_and_is_scoped():
    from tidegraph.checkpoint_metrics import capture, phase
    with capture() as outer:
        with pytest.raises(RuntimeError, match="original"):
            with phase("write"):
                raise RuntimeError("original")
        with capture() as inner:
            with phase("read"):
                pass
    assert outer["spans"]["write"]["failed_calls"] == 1
    assert "read" not in outer["spans"] and "write" not in inner["spans"]
    assert outer["total_seconds"] >= outer["spans"]["write"]["seconds"] >= 0
