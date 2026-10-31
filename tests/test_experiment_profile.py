"""Actual CANN operator spelling must survive portable trace summarization."""
import hashlib
import json
import pytest

from tidegraph.experiment.profile_trace import inspect_trace


@pytest.mark.parametrize("as_list", [False, True])
def test_cann_matmul_shape_and_nested_interfaces_are_preserved(tmp_path, as_list):
    # CANN 9.1 records matmul directly (no aten::mm child), with string dims.
    dims = "32,4096;\r\n4096,160"
    events = [
        dict(ph="M", name="process_name", pid=1, args={"name": "Ascend Hardware"}),
        dict(ph="X", name="aten::matmul", pid=2, tid=1, cat="cpu_op", ts=0, dur=10,
             args={"Input Dims": dims}),
        dict(ph="X", name="aten::matmul", pid=2, tid=1, cat="cpu_op", ts=20, dur=10,
             args={"Input Dims": dims}),
        dict(ph="X", name="aclnnMatmul_MatMulCommon_MatMulV2", pid=1, tid=1,
             ts=5, dur=5, args={"Task Type": "KERNEL_AICORE"}),
        # CUDA/CPU can instead expose a nested mm with structured dimensions.
        dict(ph="X", name="aten::mm", pid=2, tid=1, cat="cpu_op", ts=1, dur=8,
             args={"Input Shapes": [[32, 4096], [4096, 160]]}),
        dict(ph="X", name="aten::bmm", pid=2, tid=1, cat="cpu_op", ts=30, dur=10,
             args={"Input Dims": "2,32,4096;\r\n2,4096,1"}),
    ]
    path = tmp_path/"trace.json"
    before = json.dumps(events if as_list else dict(traceEvents=events)).encode()
    path.write_bytes(before)
    summary = inspect_trace(path)
    shapes = {row["operator"]: row for row in summary["matrix_shapes"]}
    assert shapes["aten::matmul"] == dict(operator="aten::matmul", input_shapes=dims, calls=2)
    assert shapes["aten::mm"]["input_shapes"] == [[32, 4096], [4096, 160]]
    assert shapes["aten::bmm"]["calls"] == 1
    assert summary["kernel_events"] == 1
    assert "overlap" in summary["matrix_scope"]
    assert summary["trace_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
