import pytest

from pilotlab.hbm_profile3 import GIB, HbmProfile, frozen_profile, hbm_shape, oom_kind
from pilotlab.io import atomic_json, sha256


def test_hbm_evidence_keeps_measured_peaks_separate_from_requested_bytes(tmp_path):
    task = dict(id="a1", source_identity={"snapshot":"s1"}, model_id="m", model={"d":128},
                precision="fp32", batch_size=4, sequence_length=128, stream_chunks=4)
    profile = HbmProfile(tmp_path)
    shape = hbm_shape(task)
    assert hbm_shape({**task, "phase":"b", "seed":11}) == shape
    assert all(hbm_shape({**task, k:v}) != shape for k,v in (
        ("source_identity",{"snapshot":"s2"}), ("batch_size",2), ("model",{"d":256}), ("sequence_length",64)))
    success = dict(id="a1", device="npu:0", source_identity=task["source_identity"], task_sha256="task",
                   status="completed", device_memory={"npu_peak_reserved_bytes":8*GIB,
                                                       "npu_startup_peak_reserved_bytes":12*GIB})
    failed = {**success, "status":"failed", "failure_class":"capacity_error",
              "device_memory":{"npu_peak_allocated_bytes":11*GIB},
              "error":"NPU out of memory. Tried to allocate 2.00 GiB (NPU 0; 61.27 GiB total capacity; "
                      "11.00 GiB already allocated; 200.00 MiB free; 16.00 GiB reserved in total by PyTorch)"}
    with profile.db:
        assert not profile.record_result(task, "task", tmp_path/"cpu/result.json", {**success,"device":"cpu"}, "cpu")
        assert not profile.record_result(task, "wrong", tmp_path/"other/result.json", success, "x")
        profile.record_result(task, "task", tmp_path/"ok/result.json", success, "ok")
        profile.record_result(task, "task", tmp_path/"oom/result.json", failed, "oom")
        profile.record_driver(tmp_path/"ok", shape, "m", 4, 13*GIB, True)
        profile.record_driver(tmp_path/"ok", shape, "m", 4, 7*GIB, False)
        profile.record_result(task, "task", tmp_path/"ok/result.json", success, "ok")
    stat = profile.summaries()[shape]
    assert stat["terminal"] == 2 and stat["successful"] == 1
    assert stat["success_peak_bytes"] == 13*GIB and stat["observed_peak_bytes"] == 16*GIB
    assert stat["oom_sizing_bytes"] == 18*GIB and stat["reservation_bytes"] == 28*GIB
    snapshot = tmp_path/"frozen.json"
    atomic_json(snapshot, profile.snapshot())
    ref = dict(path=str(snapshot), sha256=sha256(snapshot))
    assert frozen_profile(ref)[shape] == 28*GIB
    snapshot.write_text("{}")
    with pytest.raises(ValueError):
        frozen_profile(ref)


def test_opaque_aclnn_failure_requires_driver_oom_in_same_attempt(tmp_path):
    result = dict(status="failed", error="aclnnBatchMatMul ERR00100")
    log = tmp_path/"stdout.log"
    assert oom_kind(result, log) is None
    log.write_text("Memory_Allocation_Failure(EL0004)")
    assert oom_kind(result, log) is None
    log.write_text("Memory_Allocation_Failure(EL0004)\ndriver error:out of memory")
    assert oom_kind(result, log) == "driver"
    assert oom_kind({**result, "failure_class":"numerical_error"}, log) is None


def test_refrozen_workers_match_only_with_the_same_files_and_runtime(tmp_path):
    def identity(root, foundation="bbb", package="v1"):
        return dict(source_files={f"/{root}/source/pilotlab/worker3.py":"aaa",
                                  f"/{root}/foundation/src/tide/core.py":foundation},
                    runtime_packages={"torch":package}, data_manifest_sha256="data")

    task = dict(id="task", model_id="model", batch_size=2, model={"d":64}, source_identity=identity("a"))
    attempt = tmp_path/"attempt"
    atomic_json(attempt/"task.json", task)
    profile = HbmProfile(tmp_path)
    with profile.db:
        profile.record_driver(attempt, hbm_shape(task), "model", 2, 16*GIB, False)
    path = tmp_path/"profile.json"
    atomic_json(path, profile.snapshot())
    ref = dict(path=str(path), sha256=sha256(path))
    target = identity("b")
    assert frozen_profile(ref, target)[hbm_shape({**task, "source_identity":target})] == 25*GIB
    for other in (identity("b","changed"), identity("b",package="v2")):
        assert hbm_shape({**task, "source_identity":other}) not in frozen_profile(ref, other)
