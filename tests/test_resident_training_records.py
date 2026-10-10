"""Optional diagnostic exports must not control VJP/optimizer correctness."""
import pytest
import torch
from resident_test_target import owner_devices
from tidegraph import ResidentPlacement
from test_resident_training import target, training_case


@pytest.mark.parametrize("implementation", ["native", "libtorch"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
@pytest.mark.parametrize("memory", ["add", "attention"])
def test_consumer_training_record_modes(implementation, dtype_name, memory, tmp_path):
    import os
    from test_online_consumer import packet
    from test_online_consumer_npu import target as consumer_target
    from test_online_resident_consumer import standalone
    from tools.online_bench.host import run
    device = consumer_target()
    p = packet(memory)
    results = []
    for diagnostics in (True, False):
        path = tmp_path/str(diagnostics)
        path.mkdir()
        if implementation == "libtorch":
            result, _ = standalone(p, device, "settle", "prefill", True, "adamw", path,
                                   devices=2, dtype_name=dtype_name, diagnostics=diagnostics)
        else:
            result = run(p, family="settle", implementation="native", device=device, dtype=dtype_name,
                schedule="prefill", preset="resident", training=True, optimizer="adamw",
                steps=2, warmup=0, windows_per_step=2, diagnostics=diagnostics,
                native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
                resident_placement=ResidentPlacement(devices=owner_devices(device, 2)))
        assert result["diagnostics"] == diagnostics
        assert result["memory_admission"]["allocator_within_estimate"]
        results.append(result)
    for key in ("outputs", "final_cut", "statistics", "losses", "precision"):
        assert results[0][key] == results[1][key], key


@pytest.mark.parametrize("family,schedule,mode,cards", [
    ("pdg", "streaming", "hard", 1),
    ("pdg", "greedy", "hst", 2),
    ("timed-dag", "streaming", "softp", 2),
    ("timed-dag", "greedy", "hard", 1),
    ("settle", "streaming", "hst", 1),
    ("settle", "greedy", "softp", 2),
])
def test_training_without_diagnostic_exports(target, family, schedule, mode, cards, tmp_path):
    owners = ResidentPlacement(devices=owner_devices(target, cards)) if cards>1 else None
    # Existing independent CPU autograd checks compare full continuation, roots,
    # None/connected-zero flags, three updates and the saved/restored suffix.
    training_case(target, family, schedule, "adamw", tmp_path,
                  emit_mode=mode, placement=owners, model_device="cpu", trace=False)


@pytest.mark.parametrize("training", [False, True])
def test_consumer_observer_requests_records(training):
    import os
    from test_online_consumer import packet
    from test_online_consumer_npu import target as consumer_target
    from online_consumer_support import observer, same
    from tools.online_bench.host import run
    device = consumer_target()
    p = packet("attention")
    expected, actual = [], []
    kwargs = dict(family="settle", training=training, optimizer="adamw",
                  steps=2, warmup=0, windows_per_step=2)
    reference = run(p, implementation="python", device="cpu", schedule="streaming",
                    diagnostics=True, observer=observer(expected), **kwargs)
    candidate = run(p, implementation="native", device=device, schedule="prefill", preset="resident",
                    diagnostics=False, observer=observer(actual), native_library=os.environ["TIDE_BUILD_DIR"],
                    resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
                    resident_placement=ResidentPlacement(devices=owner_devices(device, 2)),
                    **kwargs)
    same(actual, expected)
    assert candidate["outputs"] == reference["outputs"]
    assert candidate["memory_admission"]["allocator_within_estimate"]
