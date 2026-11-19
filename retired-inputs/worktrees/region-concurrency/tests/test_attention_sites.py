import torch

from benchmarks.attention_sites import evaluate, make_fixture


def test_kv_site_uses_original_causal_window_and_backpropagates_to_history():
    fixture, spec = make_fixture("kv-unit", batch=1, length=3, nodes=2, d=64, window=4)
    assert fixture["mask"][0, 0, 0].tolist() == [
        [True, True, True, True, False, False],
        [False, True, True, True, True, False],
        [False, False, True, True, True, True]]
    assert spec["old_history_keys"] == 3
    row = evaluate(fixture, spec, device=torch.device("cpu"), precision="fp32", mode="forward_backward",
                   backend="auto", warmup=0, steps=1)
    assert row["state"] == "passed"
    assert row["audit"]["gradients"]["retained_history_k"]["nonzero_entries"] > 0
    assert row["audit"]["gradients"]["retained_history_v"]["nonzero_entries"] > 0


def test_region_zero_prior_audit_is_separate_and_mode_invariant():
    fixture, spec = make_fixture("region-zero", batch=1, length=3, nodes=4, d=64)
    rows = [evaluate(fixture, spec, device=torch.device("cpu"), precision="bf16", mode=mode,
                     backend="auto", warmup=0, steps=1) for mode in ("inference", "forward_backward")]
    assert all(row["state"] == "passed" and row["audit"]["dispatch"]["effective"] == "tiled" for row in rows)
    assert rows[0]["output_sha256"] == rows[1]["output_sha256"]
    assert rows[1]["audit"]["gradients"]["weights"]["dtype"] == "torch.float32"
    assert any("cpu:float32" in op["inputs"] for op in rows[0]["audit"]["operators"])


def test_reception_has_independent_node_batch_time_axes():
    fixture, spec = make_fixture("reception-positive", batch=2, length=3, nodes=4, edges=5, d=128, reception_width=64)
    assert fixture["q"].shape == (4, 2, 3, 2, 5, 32)
    assert fixture["weights"].shape == (4, 2, 3, 5)
    assert spec["zero_prior_entries"] == 0 and fixture["mask"] is None
